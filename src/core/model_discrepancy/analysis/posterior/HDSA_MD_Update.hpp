/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_UPDATE_HPP
#define HDSA_MD_UPDATE_HPP

#include "HDSA_MD_Linearization_Update.hpp"
#include "HDSA_MD_Continuation_Update.hpp"

namespace HDSA
{

  template <class RealT>
  class MD_Update
  {

  private:
    int num_continuation_steps_;
    HDSA::Ptr<HDSA::MD_Continuation_Update<RealT>> continuation_update_;
    HDSA::Ptr<HDSA::MD_Linearization_Update<RealT>> linearization_update_;

  public:
    MD_Update(const HDSA::Ptr<HDSA::MD_Data_Interface<RealT>> &data_interface, const HDSA::Ptr<HDSA::MD_u_Prior_Interface<RealT>> &u_prior_interface,
              const HDSA::Ptr<HDSA::MD_z_Prior_Interface<RealT>> &z_prior_interface, const HDSA::Ptr<HDSA::MD_Opt_Prob_Interface<RealT>> &opt_prob_interface,
              const HDSA::Ptr<HDSA::MD_Posterior_Sampling<RealT>> &post_sampling, const HDSA::Ptr<HDSA::MD_Hessian_Analysis<RealT>> &hessian_analysis, 
              const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> &random_number_generator, int num_continuation_steps = 0, const RealT grad_tol = 1e-5) : num_continuation_steps_(num_continuation_steps)
    {
      if (num_continuation_steps_ > 0)
      {
        continuation_update_ = HDSA::makePtr<HDSA::MD_Continuation_Update<RealT>>(data_interface, z_prior_interface, opt_prob_interface, post_sampling, hessian_analysis, random_number_generator, num_continuation_steps, grad_tol);
      }
      else
      {
        linearization_update_ = HDSA::makePtr<HDSA::MD_Linearization_Update<RealT>>(data_interface, u_prior_interface, z_prior_interface, opt_prob_interface, post_sampling, hessian_analysis);
      }
    }

    ~MD_Update(void)
    {
    }

    HDSA::Ptr<HDSA::MD_Posterior_Vectors<RealT>> Posterior_Update_Samples(void) const
    {
      HDSA::Ptr<HDSA::MD_Posterior_Vectors<RealT>> posterior_samples;
      if (num_continuation_steps_ > 0)
      {
        std::cout << "Warning: The current implementation of posterior sampling with continuation is incomplete and should not be used. An improved version is currently in development." << std::endl;
        posterior_samples = continuation_update_->Posterior_Update_Samples();
      }
      else
      {
        posterior_samples = linearization_update_->Posterior_Update_Samples();
      }
      return posterior_samples;
    }

    HDSA::Ptr<HDSA::Vector<RealT>> Posterior_Update_Mean(void) const
    {
      HDSA::Ptr<HDSA::Vector<RealT>> z_update_mean;
      if (num_continuation_steps_ > 0)
      {
        z_update_mean = continuation_update_->Posterior_Update_Mean();
      }
      else
      {
        z_update_mean = linearization_update_->Posterior_Update_Mean();
      }
      return z_update_mean;
    }
  };

}

#endif
