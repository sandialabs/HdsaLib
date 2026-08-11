/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_CONTINUATION_SENSITIVITY_OPERATORS_HPP
#define HDSA_MD_CONTINUATION_SENSITIVITY_OPERATORS_HPP

#include <functional>
#include <limits>
#include <algorithm>
#include <cmath>
#include "HDSA_MD_Hessian_Analysis.hpp"
#include "HDSA_MD_Posterior_Sampling.hpp"
#include "HDSA_MD_Discrepancy_Parameter_Trajectory.hpp"
#include "HDSA_PC_Sensitivity_Operator_Interface.hpp"

namespace HDSA
{

  template <class RealT>
  class Discrepancy_Ops
  {
  public:
    std::function<void(HDSA::Vector<RealT> &, const HDSA::Vector<RealT> &, RealT)> Eval;
    std::function<void(HDSA::Vector<RealT> &, const HDSA::Vector<RealT> &, const HDSA::Vector<RealT> &, RealT)> Apply_z_Jacobian;
    std::function<void(HDSA::Vector<RealT> &, const HDSA::Vector<RealT> &, const HDSA::Vector<RealT> &, RealT)> Apply_z_Jacobian_Transpose;
    std::function<void(HDSA::Vector<RealT> &, const HDSA::Vector<RealT> &)> Apply_theta_Jacobian;
    std::function<void(HDSA::Vector<RealT> &, const HDSA::Vector<RealT> &, const HDSA::Vector<RealT> &)> Apply_z_theta_Hessian;
  };

  template <class RealT>
  class MD_Continuation_Sensitivity_Operators : public PC_Sensitivity_Operator_Interface<RealT>
  {

  private:
    HDSA::Ptr<HDSA::MD_Data_Interface<RealT>> data_interface_;
    HDSA::Ptr<HDSA::MD_z_Prior_Interface<RealT>> z_prior_interface_;
    HDSA::Ptr<HDSA::MD_Opt_Prob_Interface<RealT>> opt_prob_interface_;
    HDSA::Ptr<HDSA::MD_Hessian_Analysis<RealT>> hessian_analysis_;
    HDSA::Ptr<HDSA::MD_Posterior_Data<RealT>> post_data_;
    HDSA::Ptr<HDSA::Vector<RealT>> u_opt_;
    HDSA::Ptr<HDSA::Vector<RealT>> z_opt_;

    HDSA::Ptr<HDSA::MultiVector<RealT>> Mz_Wz_inv_Mz_Z_minus_z_opt_;
    HDSA::Ptr<HDSA::MultiVector<RealT>> Mz_Wz_inv_Mz_yi_;
    std::vector<RealT> si_;

    RealT current_t_;
    HDSA::Ptr<HDSA::Vector<RealT>> current_u_;
    HDSA::Ptr<HDSA::Vector<RealT>> current_beta_;
    HDSA::Ptr<HDSA::Vector<RealT>> current_z_;
    HDSA::Ptr<HDSA::Discrepancy_Ops<RealT>> current_disc_ops_;
    bool initialized_;

    void Discrepancy_Evaluation_Mean(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &z) const
    {
      int N = post_data_->N;
      u_out.Zeros();
      for (int ell = 0; ell < N; ell++)
      {
        RealT coeff = (*post_data_->a_ell)(ell, 0);
        HDSA::Ptr<HDSA::Vector<RealT>> col = (*Mz_Wz_inv_Mz_Z_minus_z_opt_)[ell];
        coeff += z.Dot(*col);
        u_out.Scaled_Plus(coeff, *(*post_data_->u_ell)[ell]);
        for (int i = 0; i < N; i++)
        {
          RealT si_val = si_[i] + z.Dot(*(*Mz_Wz_inv_Mz_yi_)[i]);
          RealT biell = (*post_data_->b_i_ell)(i, ell);
          u_out.Scaled_Plus(-biell * si_val, *(*post_data_->u_i_ell[i])[ell]);
        }
      }
      u_out.Scale(1.0 / post_data_->alpha_d);
    }

    void Apply_Discrepancy_z_Jacobian_Mean(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &z_in) const
    {
      int N = post_data_->N;
      u_out.Zeros();
      for (int ell = 0; ell < N; ell++)
      {
        RealT dot_val = (*Mz_Wz_inv_Mz_Z_minus_z_opt_)[ell]->Dot(z_in);
        u_out.Scaled_Plus(dot_val, *(*post_data_->u_ell)[ell]);
        for (int i = 0; i < N; i++)
        {
          RealT coeff = (*post_data_->b_i_ell)(i, ell) * ((*Mz_Wz_inv_Mz_yi_)[i]->Dot(z_in));
          u_out.Scaled_Plus(-coeff, *(*post_data_->u_i_ell[i])[ell]);
        }
      }
      u_out.Scale(1.0 / post_data_->alpha_d);
    }

    void Apply_Discrepancy_z_Jacobian_Transpose_Mean(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &u_in) const
    {
      int N = post_data_->N;
      z_out.Zeros();
      for (int ell = 0; ell < N; ell++)
      {
        RealT dot_val = (*post_data_->u_ell)[ell]->Dot(u_in);
        z_out.Scaled_Plus(dot_val, *(*Mz_Wz_inv_Mz_Z_minus_z_opt_)[ell]);
        for (int i = 0; i < N; i++)
        {
          RealT coeff = (*post_data_->b_i_ell)(i, ell) * ((*post_data_->u_i_ell[i])[ell]->Dot(u_in));
          z_out.Scaled_Plus(-coeff, *(*Mz_Wz_inv_Mz_yi_)[i]);
        }
      }
      z_out.Scale(1.0 / post_data_->alpha_d);
    }

    void Discrepancy_Evaluation_Sample(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &z, int sample_idx) const
    {
      Discrepancy_Evaluation_Mean(u_out, z);

      HDSA::Ptr<HDSA::Vector<RealT>> dz = z.Clone();
      dz->Set(z);
      dz->Scaled_Plus(-1.0, *z_opt_);

      HDSA::Ptr<HDSA::Vector<RealT>> Mz_dz = z.Clone();
      z_prior_interface_->Apply_M_z(*Mz_dz, *dz);
      HDSA::Ptr<HDSA::Vector<RealT>> Wz_inv_Mz_dz = z.Clone();
      z_prior_interface_->Apply_W_z_Inverse(*Wz_inv_Mz_dz, *Mz_dz);

      HDSA::Ptr<HDSA::Vector<RealT>> delta_sample = u_out.Clone();
      delta_sample->Zeros();
      for (int i = 0; i < post_data_->N; i++)
      {
        RealT sgi = post_data_->sum_g_vecs[i];
        RealT coeff = (1.0 / std::sqrt((*post_data_->Mu)(i, 0))) * (sgi + (*Mz_Wz_inv_Mz_yi_)[i]->Dot(*dz));
        delta_sample->Scaled_Plus(coeff, *(*post_data_->u_i_hat[i])[sample_idx]);
      }
      delta_sample->Scale(std::sqrt(post_data_->alpha_d));

      HDSA::Ptr<HDSA::Vector<RealT>> tmp_rhs = z.Clone();
      tmp_rhs->Set(*Wz_inv_Mz_dz);
      if (post_data_->N > 1)
      {
        HDSA::Ptr<HDSA::Dense_Matrix<RealT>> b = post_data_->M_z_Zc->MatVec(*Wz_inv_Mz_dz);
        HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(post_data_->N - 1, 1);
        HDSA::Linear_Algebra::Symmetric_Direct_Linear_Solve<RealT>(*post_data_->Zc_M_z_W_z_inv_M_z_Zc, *x, *b);
        for (int j = 0; j < post_data_->N - 1; j++)
        {
          tmp_rhs->Scaled_Plus(-(*x)(j, 0), *(*post_data_->W_z_inv_M_z_Zc)[j]);
        }
      }

      RealT tmp = Mz_dz->Dot(*tmp_rhs);
      if (tmp < -1.e-11)
      {
        std::cout << "Error in Posterior Discrepancy Sample: delta breve coeff < 0" << std::endl;
      }
      RealT breve_coeff = std::sqrt(std::abs(tmp));
      delta_sample->Scaled_Plus(breve_coeff, *(*post_data_->u_breve)[sample_idx]);

      u_out.Plus(*delta_sample);
    }

    void Apply_Discrepancy_z_Jacobian_Sample(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &z_in, const HDSA::Vector<RealT> &z, int sample_idx) const
    {
      // Note: z is needed since sampling is nonlinear in z via gamma(z) in delta_breve
      Apply_Discrepancy_z_Jacobian_Mean(u_out, z_in);

      HDSA::Ptr<HDSA::Vector<RealT>> u = u_out.Clone();
      u->Zeros();
      for (int i = 0; i < post_data_->N; i++)
      {
        RealT coeff = (1.0 / std::sqrt((*post_data_->Mu)(i, 0))) * ((*Mz_Wz_inv_Mz_yi_)[i]->Dot(z_in));
        u->Scaled_Plus(coeff, *(*post_data_->u_i_hat[i])[sample_idx]);
      }
      u->Scale(std::sqrt(post_data_->alpha_d));

      HDSA::Ptr<HDSA::Vector<RealT>> dz = z.Clone();
      dz->Set(z);
      dz->Scaled_Plus(-1.0, *z_opt_);
      HDSA::Ptr<HDSA::Vector<RealT>> Mz_dz = z.Clone();
      z_prior_interface_->Apply_M_z(*Mz_dz, *dz);
      HDSA::Ptr<HDSA::Vector<RealT>> Wz_inv_Mz_dz = z.Clone();
      z_prior_interface_->Apply_W_z_Inverse(*Wz_inv_Mz_dz, *Mz_dz);

      HDSA::Ptr<HDSA::Vector<RealT>> tmp_rhs = z.Clone();
      tmp_rhs->Set(*Wz_inv_Mz_dz);
      if (post_data_->N > 1)
      {
        HDSA::Ptr<HDSA::Dense_Matrix<RealT>> b = post_data_->M_z_Zc->MatVec(*Wz_inv_Mz_dz);
        HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(post_data_->N - 1, 1);
        HDSA::Linear_Algebra::Symmetric_Direct_Linear_Solve<RealT>(*post_data_->Zc_M_z_W_z_inv_M_z_Zc, *x, *b);
        for (int j = 0; j < post_data_->N - 1; j++)
        {
          tmp_rhs->Scaled_Plus(-(*x)(j, 0), *(*post_data_->W_z_inv_M_z_Zc)[j]);
        }
      }

      RealT tmp = Mz_dz->Dot(*tmp_rhs);
      if (tmp < -1.e-11)
      {
        std::cout << "Error in Posterior Discrepancy Samples: delta breve coeff < 0" << std::endl;
      }

      HDSA::Ptr<HDSA::Vector<RealT>> Mz_z_in = z.Clone();
      z_prior_interface_->Apply_M_z(*Mz_z_in, z_in);
      RealT denom = std::sqrt(std::abs(tmp) + (1e-15) * (1e-15));
      RealT breve_coeff_deriv = Mz_z_in->Dot(*tmp_rhs) / denom;
      u->Scaled_Plus(breve_coeff_deriv, *(*post_data_->u_breve)[sample_idx]);

      u_out.Plus(*u);
    }

    void Apply_Discrepancy_z_Jacobian_Transpose_Sample(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &u_in, const HDSA::Vector<RealT> &z, int sample_idx) const
    {
      // Note: z is needed since sampling is nonlinear in z via gamma(z) in delta_breve
      Apply_Discrepancy_z_Jacobian_Transpose_Mean(z_out, u_in);

      HDSA::Ptr<HDSA::Vector<RealT>> z_out_sample = z_out.Clone();
      z_out_sample->Zeros();
      for (int i = 0; i < post_data_->N; i++)
      {
        HDSA::Ptr<HDSA::Vector<RealT>> ui_hat_idx = (*post_data_->u_i_hat[i])[sample_idx];
        RealT coeff = (1.0 / std::sqrt((*post_data_->Mu)(i, 0))) * (ui_hat_idx->Dot(u_in));
        z_out_sample->Scaled_Plus(coeff, *(*Mz_Wz_inv_Mz_yi_)[i]);
      }
      z_out_sample->Scale(std::sqrt(post_data_->alpha_d));

      HDSA::Ptr<HDSA::Vector<RealT>> dz = z.Clone();
      dz->Set(z);
      dz->Scaled_Plus(-1.0, *z_opt_);
      HDSA::Ptr<HDSA::Vector<RealT>> Mz_dz = z.Clone();
      z_prior_interface_->Apply_M_z(*Mz_dz, *dz);
      HDSA::Ptr<HDSA::Vector<RealT>> Wz_inv_Mz_dz = z.Clone();
      z_prior_interface_->Apply_W_z_Inverse(*Wz_inv_Mz_dz, *Mz_dz);

      HDSA::Ptr<HDSA::Vector<RealT>> tmp_rhs = z.Clone();
      tmp_rhs->Set(*Wz_inv_Mz_dz);
      if (post_data_->N > 1)
      {
        HDSA::Ptr<HDSA::Dense_Matrix<RealT>> b = post_data_->M_z_Zc->MatVec(*Wz_inv_Mz_dz);
        HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(post_data_->N - 1, 1);
        HDSA::Linear_Algebra::Symmetric_Direct_Linear_Solve<RealT>(*post_data_->Zc_M_z_W_z_inv_M_z_Zc, *x, *b);
        for (int j = 0; j < post_data_->N - 1; j++)
        {
          tmp_rhs->Scaled_Plus(-(*x)(j, 0), *(*post_data_->W_z_inv_M_z_Zc)[j]);
        }
      }

      RealT tmp = Mz_dz->Dot(*tmp_rhs);
      if (tmp < -1.e-11)
      {
        std::cout << "Error in Posterior Discrepancy Samples: delta breve coeff < 0" << std::endl;
      }

      RealT denom = std::sqrt(std::abs(tmp) + (1e-15) * (1e-15));
      HDSA::Ptr<HDSA::Vector<RealT>> breve_coeff_grad = z.Clone();
      z_prior_interface_->Apply_M_z(*breve_coeff_grad, *tmp_rhs);
      breve_coeff_grad->Scale(1.0 / denom);

      HDSA::Ptr<HDSA::Vector<RealT>> u_breve_idx = (*post_data_->u_breve)[sample_idx];
      z_out_sample->Scaled_Plus(u_breve_idx->Dot(u_in), *breve_coeff_grad);

      z_out.Plus(*z_out_sample);
    }

    HDSA::Ptr<HDSA::Discrepancy_Ops<RealT>> Get_Discrepancy_Ops(int sample_idx) const
    {
      int num_samples = post_data_->num_samples;
      HDSA_TEST_FOR_EXCEPTION(sample_idx < 0 || sample_idx > num_samples, std::logic_error,
                              "sample_idx must be an integer in [0, num_samples].");

      HDSA::Ptr<HDSA::Discrepancy_Ops<RealT>> ops = HDSA::makePtr<HDSA::Discrepancy_Ops<RealT>>();

      if (sample_idx == 0)
      {
        // Mean
        ops->Eval = [this](HDSA::Vector<RealT> &out, const HDSA::Vector<RealT> &z, RealT t)
        {
          Discrepancy_Evaluation_Mean(out, z);
          out.Scale(t);
        };
        ops->Apply_z_Jacobian = [this](HDSA::Vector<RealT> &out, const HDSA::Vector<RealT> &z_in, const HDSA::Vector<RealT> &z, RealT t)
        {
          Apply_Discrepancy_z_Jacobian_Mean(out, z_in);
          out.Scale(t);
        };
        ops->Apply_z_Jacobian_Transpose = [this](HDSA::Vector<RealT> &out, const HDSA::Vector<RealT> &u_in, const HDSA::Vector<RealT> &z, RealT t)
        {
          Apply_Discrepancy_z_Jacobian_Transpose_Mean(out, u_in);
          out.Scale(t);
        };
        ops->Apply_theta_Jacobian = [this](HDSA::Vector<RealT> &out, const HDSA::Vector<RealT> &z)
        {
          Discrepancy_Evaluation_Mean(out, z);
        };
        ops->Apply_z_theta_Hessian = [this](HDSA::Vector<RealT> &out, const HDSA::Vector<RealT> &u_in, const HDSA::Vector<RealT> &z)
        {
          Apply_Discrepancy_z_Jacobian_Transpose_Mean(out, u_in);
        };
      }
      else
      {
        // Sample: Convert 1-based sample_idx to a 0-based index (sample_idx - 1)
        ops->Eval = [this, sample_idx](HDSA::Vector<RealT> &out, const HDSA::Vector<RealT> &z, RealT t)
        {
          Discrepancy_Evaluation_Sample(out, z, sample_idx - 1);
          out.Scale(t);
        };
        ops->Apply_z_Jacobian = [this, sample_idx](HDSA::Vector<RealT> &out, const HDSA::Vector<RealT> &z_in, const HDSA::Vector<RealT> &z, RealT t)
        {
          Apply_Discrepancy_z_Jacobian_Sample(out, z_in, z, sample_idx - 1);
          out.Scale(t);
        };
        ops->Apply_z_Jacobian_Transpose = [this, sample_idx](HDSA::Vector<RealT> &out, const HDSA::Vector<RealT> &u_in, const HDSA::Vector<RealT> &z, RealT t)
        {
          Apply_Discrepancy_z_Jacobian_Transpose_Sample(out, u_in, z, sample_idx - 1);
          out.Scale(t);
        };
        ops->Apply_theta_Jacobian = [this, sample_idx](HDSA::Vector<RealT> &out, const HDSA::Vector<RealT> &z)
        {
          Discrepancy_Evaluation_Sample(out, z, sample_idx - 1);
        };
        ops->Apply_z_theta_Hessian = [this, sample_idx](HDSA::Vector<RealT> &out, const HDSA::Vector<RealT> &u_in, const HDSA::Vector<RealT> &z)
        {
          Apply_Discrepancy_z_Jacobian_Transpose_Sample(out, u_in, z, sample_idx - 1);
        };
      }
      return ops;
    }

  public:
    MD_Continuation_Sensitivity_Operators(const HDSA::Ptr<HDSA::MD_Data_Interface<RealT>> &data_interface, const HDSA::Ptr<HDSA::MD_z_Prior_Interface<RealT>> &z_prior_interface,
                                          const HDSA::Ptr<HDSA::MD_Opt_Prob_Interface<RealT>> &opt_prob_interface, const HDSA::Ptr<HDSA::MD_Posterior_Sampling<RealT>> &post_sampling,
                                          const HDSA::Ptr<HDSA::MD_Hessian_Analysis<RealT>> &hessian_analysis) : data_interface_(data_interface), z_prior_interface_(z_prior_interface), opt_prob_interface_(opt_prob_interface), hessian_analysis_(hessian_analysis)
    {
      post_data_ = post_sampling->post_data;
      u_opt_ = data_interface_->Get_u_opt()->Clone();
      u_opt_->Set(*data_interface_->Get_u_opt());
      z_opt_ = data_interface_->Get_z_opt()->Clone();
      z_opt_->Set(*data_interface_->Get_z_opt());

      current_t_ = std::numeric_limits<RealT>::infinity();
      current_u_ = HDSA::nullPtr;
      current_z_ = HDSA::nullPtr;
      current_beta_ = HDSA::nullPtr;
      initialized_ = false;

      // Mz_Wz_inv_Mz_Z_minus_z_opt = Mz_Wz_inv_Mz_Z - Mz_Wz_inv_Mz_z_opt
      Mz_Wz_inv_Mz_Z_minus_z_opt_ = HDSA::makePtr<HDSA::MultiVector<RealT>>(post_data_->N, *z_opt_);
      for (int j = 0; j < post_data_->N; j++)
      {
        (*Mz_Wz_inv_Mz_Z_minus_z_opt_)[j]->Set(*(*post_data_->M_z_W_z_inv_M_z_Z)[j]);
        (*Mz_Wz_inv_Mz_Z_minus_z_opt_)[j]->Scaled_Plus(-1.0, *post_data_->M_z_W_z_inv_M_z_z_opt);
      }

      // Mz_Wz_inv_Mz_yi and si
      Mz_Wz_inv_Mz_yi_ = HDSA::makePtr<HDSA::MultiVector<RealT>>(post_data_->N, *z_opt_);
      si_.resize(post_data_->N);
      for (int i = 0; i < post_data_->N; i++)
      {
        HDSA::Ptr<HDSA::Vector<RealT>> yi = (*Mz_Wz_inv_Mz_yi_)[i];
        yi->Zeros();
        for (int j = 0; j < post_data_->N; j++)
        {
          yi->Scaled_Plus((*post_data_->g_vecs)(j, i), *(*post_data_->M_z_W_z_inv_M_z_Z)[j]);
        }
        yi->Scaled_Plus(-post_data_->sum_g_vecs[i], *post_data_->M_z_W_z_inv_M_z_z_opt);

        si_[i] = post_data_->sum_g_vecs[i] - z_opt_->Dot(*yi);
      }
    }

    virtual ~MD_Continuation_Sensitivity_Operators()
    {
    }

    HDSA::Ptr<HDSA::Vector<RealT>> Get_Current_u(void) const { return current_u_; }
    HDSA::Ptr<HDSA::Vector<RealT>> Get_Current_z(void) const { return current_z_; }

    void State_Evaluation(const HDSA::Vector<RealT> &beta, const HDSA::PC_Auxillary_Parameter_Trajectory<RealT> &theta_traj, RealT time_index)
    {
      RealT t = static_cast<const HDSA::MD_Discrepancy_Parameter_Trajectory<RealT> &>(theta_traj).Get_Time(time_index);

      bool needs_update = !initialized_;
      if (!needs_update)
      {
        RealT t_diff = std::abs(t - current_t_);
        if (t_diff > 1.e-15)
        {
          needs_update = true;
        }
        else
        {
          HDSA::Ptr<HDSA::Vector<RealT>> beta_diff = beta.Clone();
          beta_diff->Set(beta);
          beta_diff->Scaled_Plus(-1.0, *current_beta_);
          if (beta_diff->Norm() > 1.e-15)
          {
            needs_update = true;
          }
        }
      }

      if (needs_update)
      {
        current_t_ = t;
        current_beta_ = beta.Clone();
        current_beta_->Set(beta);
        current_z_ = z_opt_->Clone();
        current_z_->Set(*z_opt_);
        HDSA::Ptr<Vector<RealT>> dz = z_opt_->Clone();
        hessian_analysis_->Apply_V(*dz, beta);
        current_z_->Plus(*dz);
        current_u_ = u_opt_->Clone();
        opt_prob_interface_->State_Solve(*current_u_, *current_z_);
        current_disc_ops_ = Get_Discrepancy_Ops(static_cast<const HDSA::MD_Discrepancy_Parameter_Trajectory<RealT> &>(theta_traj).Get_Sample_Index());
        initialized_ = true;
      }
    }

    void Gradient(HDSA::Vector<RealT> &grad, const HDSA::Vector<RealT> &beta, const HDSA::PC_Auxillary_Parameter_Trajectory<RealT> &theta_traj, RealT &time_index) const override
    {
      // Note: const_cast needed because State_Evaluation modifies mutable state
      const_cast<MD_Continuation_Sensitivity_Operators<RealT> *>(this)->State_Evaluation(beta, theta_traj, time_index);

      HDSA::Ptr<HDSA::Vector<RealT>> delta = current_u_->Clone();
      current_disc_ops_->Eval(*delta, *current_z_, current_t_);

      HDSA::Ptr<HDSA::Vector<RealT>> u_plus_delta = current_u_->Clone();
      u_plus_delta->Set(*current_u_);
      u_plus_delta->Plus(*delta);

      HDSA::Ptr<HDSA::Vector<RealT>> grad_u = u_opt_->Clone();
      HDSA::Ptr<HDSA::Vector<RealT>> grad_z = z_opt_->Clone();
      opt_prob_interface_->Misfit_Gradient(*grad_u, *u_plus_delta, *current_z_);
      opt_prob_interface_->Regularization_Gradient(*grad_z, *u_plus_delta, *current_z_);

      HDSA::Ptr<HDSA::Vector<RealT>> z_tmp1 = z_opt_->Clone();
      current_disc_ops_->Apply_z_Jacobian_Transpose(*z_tmp1, *grad_u, *current_z_, current_t_);

      HDSA::Ptr<HDSA::Vector<RealT>> z_tmp2 = z_opt_->Clone();
      opt_prob_interface_->Apply_Solution_Operator_z_Jacobian_Transpose(*z_tmp2, *grad_u, *current_z_);

      HDSA::Ptr<Vector<RealT>> z_grad = grad_z->Clone();
      z_grad->Set(*grad_z);
      z_grad->Plus(*z_tmp1);
      z_grad->Plus(*z_tmp2);
      hessian_analysis_->Apply_V_Transpose(grad, *z_grad);
    }

    void Apply_Hessian(HDSA::Vector<RealT> &beta_out, const HDSA::Vector<RealT> &beta_in, const HDSA::Vector<RealT> &beta, const HDSA::PC_Auxillary_Parameter_Trajectory<RealT> &theta_traj, RealT &time_index) const override
    {
      const_cast<MD_Continuation_Sensitivity_Operators<RealT> *>(this)->State_Evaluation(beta, theta_traj, time_index);

      HDSA::Ptr<HDSA::Vector<RealT>> z_in = z_opt_->Clone();
      hessian_analysis_->Apply_V(*z_in, beta_in);

      HDSA::Ptr<HDSA::Vector<RealT>> delta = current_u_->Clone();
      current_disc_ops_->Eval(*delta, *current_z_, current_t_);

      HDSA::Ptr<HDSA::Vector<RealT>> u_plus_delta = current_u_->Clone();
      u_plus_delta->Set(*current_u_);
      u_plus_delta->Plus(*delta);

      HDSA::Ptr<HDSA::Vector<RealT>> z_out = z_opt_->Clone();
      opt_prob_interface_->Apply_RS_Hessian(*z_out, *z_in, *current_z_);

      // NOTE: Apply_RS_Hessian corresponds to the reduced Hessian of the
      // low-fidelity objective evaluated using S(z).  The continuation
      // gradient, however, needs to be evaluated at the discrepancy-corrected state:
      
      HDSA::Ptr<HDSA::Vector<RealT>> grad_u_corrected = current_u_->Clone();
      HDSA::Ptr<HDSA::Vector<RealT>> grad_u_low_fidelity = current_u_->Clone();
      HDSA::Ptr<HDSA::Vector<RealT>> z_correction = z_opt_->Clone();

      opt_prob_interface_->Misfit_Gradient(*grad_u_corrected, *u_plus_delta, *current_z_);
      opt_prob_interface_->Misfit_Gradient(*grad_u_low_fidelity, *current_u_, *current_z_);
      grad_u_corrected->Scaled_Plus(static_cast<RealT>(-1.0), *grad_u_low_fidelity);
      opt_prob_interface_->Apply_Solution_Operator_z_Hessian_Adjoint(*z_correction, *z_in, *grad_u_corrected, *current_z_);
      z_out->Plus(*z_correction);

      HDSA::Ptr<HDSA::Vector<RealT>> u_tmp = current_u_->Clone();
      current_disc_ops_->Apply_z_Jacobian(*u_tmp, *z_in, *current_z_, current_t_);
      opt_prob_interface_->Apply_Misfit_Hessian(*u_tmp, *u_tmp, *u_plus_delta, *current_z_);

      HDSA::Ptr<HDSA::Vector<RealT>> z_tmp1 = z_opt_->Clone();
      opt_prob_interface_->Apply_Solution_Operator_z_Jacobian_Transpose(*z_tmp1, *u_tmp, *current_z_);
      z_out->Plus(*z_tmp1);

      HDSA::Ptr<HDSA::Vector<RealT>> z_tmp2 = z_opt_->Clone();
      current_disc_ops_->Apply_z_Jacobian_Transpose(*z_tmp2, *u_tmp, *current_z_, current_t_);
      z_out->Plus(*z_tmp2);

      HDSA::Ptr<HDSA::Vector<RealT>> u_tmp2 = current_u_->Clone();
      opt_prob_interface_->Apply_Solution_Operator_z_Jacobian(*u_tmp2, *z_in, *current_z_);
      opt_prob_interface_->Apply_Misfit_Hessian(*u_tmp2, *u_tmp2, *u_plus_delta, *current_z_);

      HDSA::Ptr<HDSA::Vector<RealT>> z_tmp3 = z_opt_->Clone();
      current_disc_ops_->Apply_z_Jacobian_Transpose(*z_tmp3, *u_tmp2, *current_z_, current_t_);
      z_out->Plus(*z_tmp3);

      hessian_analysis_->Apply_V_Transpose(beta_out, *z_out);
    }

    void Apply_B(HDSA::Vector<RealT> &beta_out, const HDSA::Vector<RealT> &beta, const HDSA::PC_Auxillary_Parameter_Trajectory<RealT> &theta_traj, RealT &time_index) const override
    {
      const_cast<MD_Continuation_Sensitivity_Operators<RealT> *>(this)->State_Evaluation(beta, theta_traj, time_index);

      HDSA::Ptr<HDSA::Vector<RealT>> delta = current_u_->Clone();
      current_disc_ops_->Eval(*delta, *current_z_, current_t_);

      HDSA::Ptr<HDSA::Vector<RealT>> u_plus_delta = current_u_->Clone();
      u_plus_delta->Set(*current_u_);
      u_plus_delta->Plus(*delta);

      HDSA::Ptr<HDSA::Vector<RealT>> u_tmp = current_u_->Clone();
      current_disc_ops_->Apply_theta_Jacobian(*u_tmp, *current_z_);
      opt_prob_interface_->Apply_Misfit_Hessian(*u_tmp, *u_tmp, *u_plus_delta, *current_z_);

      HDSA::Ptr<HDSA::Vector<RealT>> z_out = z_opt_->Clone();
      opt_prob_interface_->Apply_Solution_Operator_z_Jacobian_Transpose(*z_out, *u_tmp, *current_z_);

      HDSA::Ptr<HDSA::Vector<RealT>> z_tmp1 = z_opt_->Clone();
      current_disc_ops_->Apply_z_Jacobian_Transpose(*z_tmp1, *u_tmp, *current_z_, current_t_);
      z_out->Plus(*z_tmp1);

      HDSA::Ptr<HDSA::Vector<RealT>> state_grad = u_opt_->Clone();
      opt_prob_interface_->Misfit_Gradient(*state_grad, *u_plus_delta, *current_z_);

      HDSA::Ptr<HDSA::Vector<RealT>> z_tmp2 = z_opt_->Clone();
      current_disc_ops_->Apply_z_theta_Hessian(*z_tmp2, *state_grad, *current_z_);
      z_out->Plus(*z_tmp2);

      hessian_analysis_->Apply_V_Transpose(beta_out, *z_out);
    }
  };

}

#endif