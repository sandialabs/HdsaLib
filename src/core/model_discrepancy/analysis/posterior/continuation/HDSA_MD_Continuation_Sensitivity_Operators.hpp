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
#include "HDSA_Std_Vector.hpp"
#include "HDSA_MD_Hessian_Analysis.hpp"
#include "HDSA_MD_Posterior_Sampling.hpp"
#include "HDSA_MD_Discrepancy_Parameter_Trajectory.hpp"
#include "HDSA_MD_Breve_Beta_Sampler.hpp"
#include "HDSA_PC_Sensitivity_Operator_Interface.hpp"

namespace HDSA
{

  template <class RealT>
  class Discrepancy_Ops
  {
  public:
    std::function<void(HDSA::Vector<RealT> &, const HDSA::Vector<RealT> &, RealT)> Eval;
    std::function<void(HDSA::Vector<RealT> &, const HDSA::Vector<RealT> &, RealT)> Apply_Beta_Jacobian;
    std::function<void(HDSA::Vector<RealT> &, const HDSA::Vector<RealT> &, RealT)> Apply_Beta_Jacobian_Transpose;
    std::function<void(HDSA::Vector<RealT> &, const HDSA::Vector<RealT> &)> Apply_theta_Jacobian;
    std::function<void(HDSA::Vector<RealT> &, const HDSA::Vector<RealT> &)> Apply_Beta_Theta_Hessian;
  };

  template <class RealT>
  class MD_Continuation_Sensitivity_Operators : public PC_Sensitivity_Operator_Interface<RealT>
  {

  private:
    HDSA::Ptr<HDSA::MD_Data_Interface<RealT>> data_interface_;
    HDSA::Ptr<HDSA::MD_u_Prior_Interface<RealT>> u_prior_interface_;
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
    int current_sample_idx_;
    bool initialized_;
    bool discard_cache_;
    RealT lazy_sampling_tol_;
    std::vector<HDSA::Ptr<HDSA::MD_Breve_Beta_Sampler<RealT>>> breve_samplers_;

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

    HDSA::Ptr<HDSA::MD_Breve_Beta_Sampler<RealT>> Get_Breve_Sampler(int sample_idx) const
    {
      HDSA_TEST_FOR_EXCEPTION(sample_idx <= 0 || sample_idx > post_data_->num_samples, std::logic_error,
                              "sample_idx must be an integer in [1, num_samples]." << std::endl);

      MD_Continuation_Sensitivity_Operators<RealT> *self = const_cast<MD_Continuation_Sensitivity_Operators<RealT> *>(this);
      if (self->breve_samplers_[sample_idx - 1] == HDSA::nullPtr)
      {
        if (discard_cache_)
        {
          for (int i = 0; i < post_data_->num_samples; ++i)
          {
            if (i != sample_idx - 1)
              self->breve_samplers_[i] = HDSA::nullPtr;
          }
        }

        HDSA::Ptr<HDSA::Vector<RealT>> beta_prototype;
        HDSA::Ptr<HDSA::Dense_Matrix<RealT>> evals = hessian_analysis_->Get_Evals();
        if (evals == HDSA::nullPtr || evals->Number_of_Rows() == 0 || evals->Number_of_Rows() == z_opt_->Dimension())
        {
          beta_prototype = z_opt_->Clone();
        }
        else
        {
          beta_prototype = HDSA::makePtr<HDSA::Std_Vector<RealT>>(evals->Number_of_Rows());
        }

        self->breve_samplers_[sample_idx - 1] =
            HDSA::makePtr<HDSA::MD_Breve_Beta_Sampler<RealT>>(hessian_analysis_, z_prior_interface_, post_data_, u_prior_interface_, *beta_prototype, *u_opt_, lazy_sampling_tol_);
      }

      return breve_samplers_[sample_idx - 1];
    }

    void Discrepancy_Evaluation_Sample_Beta(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &beta, int sample_idx) const
    {
      HDSA::Ptr<HDSA::Vector<RealT>> dz = z_opt_->Clone();
      hessian_analysis_->Apply_V(*dz, beta);

      HDSA::Ptr<HDSA::Vector<RealT>> z = z_opt_->Clone();
      z->Set(*z_opt_);
      z->Plus(*dz);

      Discrepancy_Evaluation_Mean(u_out, *z);

      HDSA::Ptr<HDSA::Vector<RealT>> u_hat = u_out.Clone();
      u_hat->Zeros();
      for (int i = 0; i < post_data_->N; ++i)
      {
        RealT coeff = (static_cast<RealT>(1.0) / std::sqrt((*post_data_->Mu)(i, 0))) *
                      (post_data_->sum_g_vecs[i] + (*Mz_Wz_inv_Mz_yi_)[i]->Dot(*dz));
        u_hat->Scaled_Plus(coeff, *(*post_data_->u_i_hat[i])[sample_idx - 1]);
      }
      u_hat->Scale(std::sqrt(post_data_->alpha_d));
      u_out.Plus(*u_hat);

      HDSA::Ptr<HDSA::Vector<RealT>> u_breve = u_out.Clone();
      Get_Breve_Sampler(sample_idx)->Eval(*u_breve, beta);
      u_out.Plus(*u_breve);
    }

    void Apply_Discrepancy_Beta_Jacobian_Sample(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &beta_in, int sample_idx) const
    {
      HDSA::Ptr<HDSA::Vector<RealT>> z_in = z_opt_->Clone();
      hessian_analysis_->Apply_V(*z_in, beta_in);

      Apply_Discrepancy_z_Jacobian_Mean(u_out, *z_in);

      HDSA::Ptr<HDSA::Vector<RealT>> u_hat = u_out.Clone();
      u_hat->Zeros();
      for (int i = 0; i < post_data_->N; ++i)
      {
        RealT coeff = (static_cast<RealT>(1.0) / std::sqrt((*post_data_->Mu)(i, 0))) *
                      ((*Mz_Wz_inv_Mz_yi_)[i]->Dot(*z_in));
        u_hat->Scaled_Plus(coeff, *(*post_data_->u_i_hat[i])[sample_idx - 1]);
      }
      u_hat->Scale(std::sqrt(post_data_->alpha_d));
      u_out.Plus(*u_hat);

      HDSA::Ptr<HDSA::Vector<RealT>> u_breve = u_out.Clone();
      Get_Breve_Sampler(sample_idx)->Apply_Jacobian(*u_breve, beta_in);
      u_out.Plus(*u_breve);
    }

    void Apply_Discrepancy_Beta_Jacobian_Transpose_Sample(HDSA::Vector<RealT> &beta_out, const HDSA::Vector<RealT> &u_in, int sample_idx) const
    {
      HDSA::Ptr<HDSA::Vector<RealT>> z_out = z_opt_->Clone();
      Apply_Discrepancy_z_Jacobian_Transpose_Mean(*z_out, u_in);

      HDSA::Ptr<HDSA::Vector<RealT>> z_hat = z_opt_->Clone();
      z_hat->Zeros();
      for (int i = 0; i < post_data_->N; ++i)
      {
        HDSA::Ptr<HDSA::Vector<RealT>> ui_hat_idx = (*post_data_->u_i_hat[i])[sample_idx - 1];
        RealT coeff = (static_cast<RealT>(1.0) / std::sqrt((*post_data_->Mu)(i, 0))) * ui_hat_idx->Dot(u_in);
        z_hat->Scaled_Plus(coeff, *(*Mz_Wz_inv_Mz_yi_)[i]);
      }
      z_hat->Scale(std::sqrt(post_data_->alpha_d));
      z_out->Plus(*z_hat);

      hessian_analysis_->Apply_V_Transpose(beta_out, *z_out);

      HDSA::Ptr<HDSA::Vector<RealT>> beta_breve = beta_out.Clone();
      Get_Breve_Sampler(sample_idx)->Apply_Jacobian_Transpose(*beta_breve, u_in);
      beta_out.Plus(*beta_breve);
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
          HDSA::Ptr<HDSA::Vector<RealT>> z_eval = z_opt_->Clone();
          z_eval->Set(*z_opt_);
          HDSA::Ptr<HDSA::Vector<RealT>> dz = z_opt_->Clone();
          hessian_analysis_->Apply_V(*dz, z);
          z_eval->Plus(*dz);
          Discrepancy_Evaluation_Mean(out, *z_eval);
          out.Scale(t);
        };
        ops->Apply_Beta_Jacobian = [this](HDSA::Vector<RealT> &out, const HDSA::Vector<RealT> &beta_in, RealT t)
        {
          HDSA::Ptr<HDSA::Vector<RealT>> z_in = z_opt_->Clone();
          hessian_analysis_->Apply_V(*z_in, beta_in);
          Apply_Discrepancy_z_Jacobian_Mean(out, *z_in);
          out.Scale(t);
        };
        ops->Apply_Beta_Jacobian_Transpose = [this](HDSA::Vector<RealT> &out, const HDSA::Vector<RealT> &u_in, RealT t)
        {
          HDSA::Ptr<HDSA::Vector<RealT>> z_out = z_opt_->Clone();
          Apply_Discrepancy_z_Jacobian_Transpose_Mean(*z_out, u_in);
          hessian_analysis_->Apply_V_Transpose(out, *z_out);
          out.Scale(t);
        };
        ops->Apply_theta_Jacobian = [this](HDSA::Vector<RealT> &out, const HDSA::Vector<RealT> &beta)
        {
          HDSA::Ptr<HDSA::Vector<RealT>> z_eval = z_opt_->Clone();
          z_eval->Set(*z_opt_);
          HDSA::Ptr<HDSA::Vector<RealT>> dz = z_opt_->Clone();
          hessian_analysis_->Apply_V(*dz, beta);
          z_eval->Plus(*dz);
          Discrepancy_Evaluation_Mean(out, *z_eval);
        };
        ops->Apply_Beta_Theta_Hessian = [this](HDSA::Vector<RealT> &out, const HDSA::Vector<RealT> &u_in)
        {
          HDSA::Ptr<HDSA::Vector<RealT>> z_out = z_opt_->Clone();
          Apply_Discrepancy_z_Jacobian_Transpose_Mean(*z_out, u_in);
          hessian_analysis_->Apply_V_Transpose(out, *z_out);
        };
      }
      else
      {
        // Sample: Convert 1-based sample_idx to a 0-based index (sample_idx - 1)
        ops->Eval = [this, sample_idx](HDSA::Vector<RealT> &out, const HDSA::Vector<RealT> &z, RealT t)
        {
          Discrepancy_Evaluation_Sample_Beta(out, z, sample_idx);
          out.Scale(t);
        };
        ops->Apply_Beta_Jacobian = [this, sample_idx](HDSA::Vector<RealT> &out, const HDSA::Vector<RealT> &beta_in, RealT t)
        {
          Apply_Discrepancy_Beta_Jacobian_Sample(out, beta_in, sample_idx);
          out.Scale(t);
        };
        ops->Apply_Beta_Jacobian_Transpose = [this, sample_idx](HDSA::Vector<RealT> &out, const HDSA::Vector<RealT> &u_in, RealT t)
        {
          Apply_Discrepancy_Beta_Jacobian_Transpose_Sample(out, u_in, sample_idx);
          out.Scale(t);
        };
        ops->Apply_theta_Jacobian = [this, sample_idx](HDSA::Vector<RealT> &out, const HDSA::Vector<RealT> &beta)
        {
          Discrepancy_Evaluation_Sample_Beta(out, beta, sample_idx);
        };
        ops->Apply_Beta_Theta_Hessian = [this, sample_idx](HDSA::Vector<RealT> &out, const HDSA::Vector<RealT> &u_in)
        {
          Apply_Discrepancy_Beta_Jacobian_Transpose_Sample(out, u_in, sample_idx);
        };
      }
      return ops;
    }

  public:
    MD_Continuation_Sensitivity_Operators(const HDSA::Ptr<HDSA::MD_Data_Interface<RealT>> &data_interface, const HDSA::Ptr<HDSA::MD_u_Prior_Interface<RealT>> &u_prior_interface,
                                          const HDSA::Ptr<HDSA::MD_z_Prior_Interface<RealT>> &z_prior_interface,
                                          const HDSA::Ptr<HDSA::MD_Opt_Prob_Interface<RealT>> &opt_prob_interface, const HDSA::Ptr<HDSA::MD_Posterior_Sampling<RealT>> &post_sampling,
                                          const HDSA::Ptr<HDSA::MD_Hessian_Analysis<RealT>> &hessian_analysis, bool discard_cache = true) : data_interface_(data_interface), u_prior_interface_(u_prior_interface), z_prior_interface_(z_prior_interface), opt_prob_interface_(opt_prob_interface), hessian_analysis_(hessian_analysis), discard_cache_(discard_cache)
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
      current_sample_idx_ = -1;
      initialized_ = false;
      lazy_sampling_tol_ = static_cast<RealT>(1.e-10);
      breve_samplers_.resize(post_data_->num_samples);

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
      int sample_idx = static_cast<const HDSA::MD_Discrepancy_Parameter_Trajectory<RealT> &>(theta_traj).Get_Sample_Index();

      bool needs_update = !initialized_;
      if (!needs_update)
      {
        RealT t_diff = std::abs(t - current_t_);
        if (t_diff > 1.e-15 || sample_idx != current_sample_idx_)
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
        current_sample_idx_ = sample_idx;
        current_beta_ = beta.Clone();
        current_beta_->Set(beta);
        current_z_ = z_opt_->Clone();
        current_z_->Set(*z_opt_);
        HDSA::Ptr<Vector<RealT>> dz = z_opt_->Clone();
        hessian_analysis_->Apply_V(*dz, beta);
        current_z_->Plus(*dz);
        current_u_ = u_opt_->Clone();
        opt_prob_interface_->State_Solve(*current_u_, *current_z_);
        current_disc_ops_ = Get_Discrepancy_Ops(sample_idx);
        initialized_ = true;
      }
      if (sample_idx > 0)
        Get_Breve_Sampler(sample_idx);
    }

    void Gradient(HDSA::Vector<RealT> &grad, const HDSA::Vector<RealT> &beta, const HDSA::PC_Auxillary_Parameter_Trajectory<RealT> &theta_traj, RealT &time_index) const override
    {
      // Note: const_cast needed because State_Evaluation modifies mutable state
      const_cast<MD_Continuation_Sensitivity_Operators<RealT> *>(this)->State_Evaluation(beta, theta_traj, time_index);

      HDSA::Ptr<HDSA::Vector<RealT>> delta = current_u_->Clone();
      current_disc_ops_->Eval(*delta, beta, current_t_);

      HDSA::Ptr<HDSA::Vector<RealT>> u_plus_delta = current_u_->Clone();
      u_plus_delta->Set(*current_u_);
      u_plus_delta->Plus(*delta);

      HDSA::Ptr<HDSA::Vector<RealT>> grad_u = u_opt_->Clone();
      HDSA::Ptr<HDSA::Vector<RealT>> grad_z = z_opt_->Clone();
      opt_prob_interface_->Misfit_Gradient(*grad_u, *u_plus_delta, *current_z_);
      opt_prob_interface_->Regularization_Gradient(*grad_z, *u_plus_delta, *current_z_);

      HDSA::Ptr<HDSA::Vector<RealT>> beta_tmp1 = grad.Clone();
      current_disc_ops_->Apply_Beta_Jacobian_Transpose(*beta_tmp1, *grad_u, current_t_);

      HDSA::Ptr<HDSA::Vector<RealT>> z_tmp2 = z_opt_->Clone();
      opt_prob_interface_->Apply_Solution_Operator_z_Jacobian_Transpose(*z_tmp2, *grad_u, *current_z_);

      HDSA::Ptr<Vector<RealT>> z_grad = grad_z->Clone();
      z_grad->Set(*grad_z);
      z_grad->Plus(*z_tmp2);
      hessian_analysis_->Apply_V_Transpose(grad, *z_grad);
      grad.Plus(*beta_tmp1);
    }

    void Apply_Hessian(HDSA::Vector<RealT> &beta_out, const HDSA::Vector<RealT> &beta_in, const HDSA::Vector<RealT> &beta, const HDSA::PC_Auxillary_Parameter_Trajectory<RealT> &theta_traj, RealT &time_index) const override
    {
      const_cast<MD_Continuation_Sensitivity_Operators<RealT> *>(this)->State_Evaluation(beta, theta_traj, time_index);

      HDSA::Ptr<HDSA::Vector<RealT>> z_in = z_opt_->Clone();
      hessian_analysis_->Apply_V(*z_in, beta_in);

      HDSA::Ptr<HDSA::Vector<RealT>> delta = current_u_->Clone();
      current_disc_ops_->Eval(*delta, beta, current_t_);

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
      current_disc_ops_->Apply_Beta_Jacobian(*u_tmp, beta_in, current_t_);

      HDSA::Ptr<HDSA::Vector<RealT>> u_tmp2 = current_u_->Clone();
      opt_prob_interface_->Apply_Misfit_Hessian(*u_tmp2, *u_tmp, *u_plus_delta, *current_z_);

      HDSA::Ptr<HDSA::Vector<RealT>> z_tmp1 = z_opt_->Clone();
      opt_prob_interface_->Apply_Solution_Operator_z_Jacobian_Transpose(*z_tmp1, *u_tmp2, *current_z_);
      z_out->Plus(*z_tmp1);

      HDSA::Ptr<HDSA::Vector<RealT>> beta_tmp2 = beta_out.Clone();
      current_disc_ops_->Apply_Beta_Jacobian_Transpose(*beta_tmp2, *u_tmp2, current_t_);

      opt_prob_interface_->Apply_Solution_Operator_z_Jacobian(*u_tmp, *z_in, *current_z_);
      opt_prob_interface_->Apply_Misfit_Hessian(*u_tmp2, *u_tmp, *u_plus_delta, *current_z_);

      hessian_analysis_->Apply_V_Transpose(beta_out, *z_out);
      beta_out.Plus(*beta_tmp2);

      HDSA::Ptr<HDSA::Vector<RealT>> beta_tmp3 = beta_out.Clone();
      current_disc_ops_->Apply_Beta_Jacobian_Transpose(*beta_tmp3, *u_tmp2, current_t_);
      beta_out.Plus(*beta_tmp3);
    }

    void Apply_B(HDSA::Vector<RealT> &beta_out, const HDSA::Vector<RealT> &beta, const HDSA::PC_Auxillary_Parameter_Trajectory<RealT> &theta_traj, RealT &time_index) const override
    {
      const_cast<MD_Continuation_Sensitivity_Operators<RealT> *>(this)->State_Evaluation(beta, theta_traj, time_index);

      HDSA::Ptr<HDSA::Vector<RealT>> delta = current_u_->Clone();
      current_disc_ops_->Eval(*delta, beta, current_t_);

      HDSA::Ptr<HDSA::Vector<RealT>> u_plus_delta = current_u_->Clone();
      u_plus_delta->Set(*current_u_);
      u_plus_delta->Plus(*delta);

      HDSA::Ptr<HDSA::Vector<RealT>> u_tmp = current_u_->Clone();
      current_disc_ops_->Apply_theta_Jacobian(*u_tmp, beta);

      HDSA::Ptr<HDSA::Vector<RealT>> u_tmp2 = current_u_->Clone();
      opt_prob_interface_->Apply_Misfit_Hessian(*u_tmp2, *u_tmp, *u_plus_delta, *current_z_);

      HDSA::Ptr<HDSA::Vector<RealT>> z_out = z_opt_->Clone();
      opt_prob_interface_->Apply_Solution_Operator_z_Jacobian_Transpose(*z_out, *u_tmp2, *current_z_);

      HDSA::Ptr<HDSA::Vector<RealT>> beta_tmp1 = beta_out.Clone();
      current_disc_ops_->Apply_Beta_Jacobian_Transpose(*beta_tmp1, *u_tmp2, current_t_);

      HDSA::Ptr<HDSA::Vector<RealT>> state_grad = u_opt_->Clone();
      opt_prob_interface_->Misfit_Gradient(*state_grad, *u_plus_delta, *current_z_);

      hessian_analysis_->Apply_V_Transpose(beta_out, *z_out);
      beta_out.Plus(*beta_tmp1);

      HDSA::Ptr<HDSA::Vector<RealT>> beta_tmp2 = beta_out.Clone();
      current_disc_ops_->Apply_Beta_Theta_Hessian(*beta_tmp2, *state_grad);
      beta_out.Plus(*beta_tmp2);
    }
  };

}

#endif
