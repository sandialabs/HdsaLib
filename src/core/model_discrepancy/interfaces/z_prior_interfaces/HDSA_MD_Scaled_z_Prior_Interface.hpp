/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_SCALED_Z_PRIOR_INTERFACE_HPP
#define HDSA_MD_SCALED_Z_PRIOR_INTERFACE_HPP

#include "HDSA_MD_z_Prior_Interface.hpp"

namespace HDSA
{

  template <class RealT>
  class MD_Scaled_z_Prior_Interface : public HDSA::MD_z_Prior_Interface<RealT>
  {

  private:
    RealT alpha_z_;

  public:
    MD_Scaled_z_Prior_Interface(RealT alpha_z) : alpha_z_(alpha_z)
    {
    }

    virtual ~MD_Scaled_z_Prior_Interface()
    {
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Pure virtual functions
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    virtual void Apply_M_z(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in) const = 0;

    virtual void Apply_W_z_Acute_Inverse(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in) const = 0;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Virtual functions which must be implemented to enable posterior sampling
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Compute samples from a mean zero Gaussian with covariance \acute{W}_z^{-1}
    virtual void Sample_with_Covariance_W_z_Acute_Inverse(HDSA::MultiVector<RealT> &samples) const
    {
      (void) samples;
      HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                              "Error in HDSA::MD_Scaled_z_Prior_Interface::Sample_with_Covariance_W_z_Acute_Inverse: Method must be implemented to use sampling algorithms" << std::endl);
    }

    virtual void Apply_W_z_Acute(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in) const
    {
      (void) z_in;
      (void) z_out;
      HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                              "Error in HDSA::MD_Scaled_z_Prior_Interface::MD_z_Prior_Interface::Apply_W_z_Acute: Method must be implemented to use the Hessian GEVP" << std::endl);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Implementation of base class functions
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    virtual void Apply_W_z_Inverse(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in) const
    {
      Apply_W_z_Acute_Inverse(z_out, z_in);
      z_out.Scale(alpha_z_);
    }

    virtual void Sample_with_Covariance_W_z_Inverse(HDSA::MultiVector<RealT> &samples) const
    {
      Sample_with_Covariance_W_z_Acute_Inverse(samples);
      samples.Scale(std::sqrt(alpha_z_));
    }

    virtual void Apply_W_z(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in) const
    {
      Apply_W_z_Acute(z_out, z_in);
      z_out.Scale(1.0 / alpha_z_);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Helper functions
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void Set_alpha_z(RealT alpha_z_new)
    {
      alpha_z_ = alpha_z_new;
    }
  };

}

#endif
