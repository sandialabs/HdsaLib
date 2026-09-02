/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_SCALED_U_PRIOR_INTERFACE_HPP
#define HDSA_MD_SCALED_U_PRIOR_INTERFACE_HPP

#include "HDSA_MD_u_Prior_Interface.hpp"

namespace HDSA
{

  template <class RealT>
  class MD_Scaled_u_Prior_Interface : public HDSA::MD_u_Prior_Interface<RealT>
  {

  private:
    RealT alpha_u_;

  public:
    MD_Scaled_u_Prior_Interface(RealT alpha_u) : alpha_u_(alpha_u)
    {
    }

    virtual ~MD_Scaled_u_Prior_Interface()
    {
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Pure virtual functions
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    virtual void Apply_M_u(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in) const = 0;

    virtual void Apply_W_u_Acute_Plus_scalar_M_u_Inverse(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in, const RealT &scalar) const = 0;

    virtual void Apply_W_u_Acute_Inverse(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in) const = 0;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Virtual functions which must be implemented to enable posterior sampling
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Compute samples from a mean zero Gaussian with covariance \acute{W}_u^{-1}
    virtual void Sample_with_Covariance_W_u_Acute_Inverse(HDSA::MultiVector<RealT> &samples) const
    {
      (void) samples;
      HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                              "Error in HDSA::MD_Scaled_u_Prior_Interface::Sample_with_Covariance_W_u_Acute_Inverse: Method must be implemented to use sampling algorithms" << std::endl);
    }

    // Compute samples from a mean zero Gaussian with covariance \acute{W}_u^{-1}
    virtual void Sample_with_Covariance_W_u_Acute_Plus_scalar_M_u_Inverse(HDSA::MultiVector<RealT> &samples, const RealT &scalar) const
    {
      (void) samples;
      (void) scalar;
      HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                              "Error in HDSA::MD_Scaled_u_Prior_Interface::Sample_with_Covariance_W_u_Acute_Plus_scalar_M_u_Inverse: Method must be implemented to use sampling algorithms" << std::endl);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Implementation of base class functions
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    virtual void Apply_W_u_Inverse(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in) const
    {
      Apply_W_u_Acute_Inverse(u_out, u_in);
      u_out.Scale(alpha_u_);
    }

    virtual void Apply_W_u_Plus_scalar_M_u_Inverse(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in, const RealT &scalar) const
    {
      Apply_W_u_Acute_Plus_scalar_M_u_Inverse(u_out, u_in, scalar * alpha_u_);
      u_out.Scale(alpha_u_);
    }

    virtual void Sample_with_Covariance_W_u_Inverse(HDSA::MultiVector<RealT> &samples) const
    {
      Sample_with_Covariance_W_u_Acute_Inverse(samples);
      samples.Scale(std::sqrt(alpha_u_));
    }

    virtual void Sample_with_Covariance_W_u_Plus_scalar_M_u_Inverse(HDSA::MultiVector<RealT> &samples, const RealT &scalar) const
    {
      Sample_with_Covariance_W_u_Acute_Plus_scalar_M_u_Inverse(samples, alpha_u_ * scalar);
      samples.Scale(std::sqrt(alpha_u_));
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Helper functions
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void Set_alpha_u(RealT alpha_u_new)
    {
      alpha_u_ = alpha_u_new;
    }

    RealT Get_alpha_u() const {
      return alpha_u_;
    }
  };

}

#endif
