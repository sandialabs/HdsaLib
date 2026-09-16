/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_BREVE_BETA_SAMPLER_HPP
#define HDSA_MD_BREVE_BETA_SAMPLER_HPP

#include "HDSA_Linear_Algebra.hpp"
#include "HDSA_MD_Hessian_Analysis.hpp"
#include "HDSA_MD_Lazy_Matrix_Normal_Operator.hpp"
#include "HDSA_MD_Posterior_Data.hpp"
#include "HDSA_MD_u_Prior_Interface.hpp"
#include "HDSA_MD_z_Prior_Interface.hpp"

namespace HDSA
{

  template <class RealT>
  class MD_Breve_Beta_Sampler
  {
  private:
    HDSA::Ptr<HDSA::MD_Hessian_Analysis<RealT>> hessian_analysis_;
    HDSA::Ptr<HDSA::MD_z_Prior_Interface<RealT>> z_prior_interface_;
    HDSA::Ptr<HDSA::MD_Posterior_Data<RealT>> post_data_;
    HDSA::Ptr<HDSA::MD_Lazy_Matrix_Normal_Operator<RealT>> lazy_X_;
    HDSA::Ptr<HDSA::Vector<RealT>> z_prototype_;

    void Project_Away_Zc(HDSA::Vector<RealT> &z_vec) const
    {
      if (post_data_->N <= 1)
        return;

      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> b = post_data_->M_z_Zc->MatVec(z_vec);
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(post_data_->N - 1, 1);
      HDSA::Linear_Algebra::Symmetric_Direct_Linear_Solve<RealT>(*post_data_->Zc_M_z_W_z_inv_M_z_Zc, *x, *b);
      for (int j = 0; j < post_data_->N - 1; ++j)
      {
        z_vec.Scaled_Plus(-(*x)(j, 0), *(*post_data_->W_z_inv_M_z_Zc)[j]);
      }
    }

  public:
    MD_Breve_Beta_Sampler(const HDSA::Ptr<HDSA::MD_Hessian_Analysis<RealT>> &hessian_analysis,
                          const HDSA::Ptr<HDSA::MD_z_Prior_Interface<RealT>> &z_prior_interface,
                          const HDSA::Ptr<HDSA::MD_Posterior_Data<RealT>> &post_data,
                          const HDSA::Ptr<HDSA::MD_u_Prior_Interface<RealT>> &u_prior_interface,
                          const HDSA::Vector<RealT> &beta_prototype,
                          const HDSA::Vector<RealT> &u_prototype,
                          const RealT tol = static_cast<RealT>(1.e-10))
        : hessian_analysis_(hessian_analysis), z_prior_interface_(z_prior_interface), post_data_(post_data)
    {
      z_prototype_ = post_data_->M_z_z_opt->Clone();

      auto sigma_apply = [this](HDSA::Vector<RealT> &beta_out, const HDSA::Vector<RealT> &beta_in)
      {
        this->Apply_Sigma_Beta(beta_out, beta_in);
      };
      auto sigma_sample = [this](HDSA::MultiVector<RealT> &beta_samps)
      {
        this->Sample_Sigma_Beta(beta_samps);
      };

      lazy_X_ = HDSA::makePtr<HDSA::MD_Lazy_Matrix_Normal_Operator<RealT>>(u_prior_interface, sigma_apply, sigma_sample, beta_prototype, u_prototype, tol);
    }

    virtual ~MD_Breve_Beta_Sampler()
    {
    }

    void Eval(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &beta)
    {
      lazy_X_->Forward_Apply(u_out, beta);
    }

    void Apply_Jacobian(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &beta_in)
    {
      lazy_X_->Forward_Apply(u_out, beta_in);
    }

    void Apply_Jacobian_Transpose(HDSA::Vector<RealT> &beta_out, const HDSA::Vector<RealT> &u_in)
    {
      lazy_X_->Adjoint_Apply(beta_out, u_in);
    }

    void Apply_Sigma_Beta(HDSA::Vector<RealT> &beta_out, const HDSA::Vector<RealT> &beta_in) const
    {
      HDSA::Ptr<HDSA::Vector<RealT>> z_in = z_prototype_->Clone();
      hessian_analysis_->Apply_V(*z_in, beta_in);

      HDSA::Ptr<HDSA::Vector<RealT>> Mz_z_in = z_prototype_->Clone();
      z_prior_interface_->Apply_M_z(*Mz_z_in, *z_in);

      HDSA::Ptr<HDSA::Vector<RealT>> tmp_rhs = z_prototype_->Clone();
      z_prior_interface_->Apply_W_z_Inverse(*tmp_rhs, *Mz_z_in);
      Project_Away_Zc(*tmp_rhs);

      HDSA::Ptr<HDSA::Vector<RealT>> Mz_tmp = z_prototype_->Clone();
      z_prior_interface_->Apply_M_z(*Mz_tmp, *tmp_rhs);
      hessian_analysis_->Apply_V_Transpose(beta_out, *Mz_tmp);
    }

    void Sample_Sigma_Beta(HDSA::MultiVector<RealT> &beta_samps) const
    {
      const int num_samples = beta_samps.Number_of_Vectors();
      HDSA::MultiVector<RealT> z_samps(num_samples, *z_prototype_);
      z_prior_interface_->Sample_with_Covariance_W_z_Inverse(z_samps);

      for (int k = 0; k < num_samples; ++k)
      {
        HDSA::Ptr<HDSA::Vector<RealT>> z_samp_k = z_samps[k];
        Project_Away_Zc(*z_samp_k);

        HDSA::Ptr<HDSA::Vector<RealT>> Mz_z_samp_k = z_prototype_->Clone();
        z_prior_interface_->Apply_M_z(*Mz_z_samp_k, *z_samp_k);
        hessian_analysis_->Apply_V_Transpose(*beta_samps[k], *Mz_z_samp_k);
      }
    }
  };

}

#endif
