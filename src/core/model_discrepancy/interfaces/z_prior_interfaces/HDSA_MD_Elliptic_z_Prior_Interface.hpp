/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_ELLIPTIC_Z_PRIOR_INTERFACE_HPP
#define HDSA_MD_ELLIPTIC_Z_PRIOR_INTERFACE_HPP

#include "HDSA_MD_Scaled_z_Prior_Interface.hpp"

namespace HDSA
{

  template <class RealT>
  class MD_Elliptic_z_Prior_Interface : public HDSA::MD_Scaled_z_Prior_Interface<RealT>
  {

  private:
  public:
    MD_Elliptic_z_Prior_Interface(RealT alpha_z) : HDSA::MD_Scaled_z_Prior_Interface<RealT>(alpha_z)
    {
    }

    virtual ~MD_Elliptic_z_Prior_Interface()
    {
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Pure virtual functions
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    virtual void Apply_E_z_Inverse(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in) const = 0;

    virtual void Apply_E_z_Inverse_Transpose(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in) const = 0;

    virtual void Apply_M_z(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in) const = 0;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Virtual functions which must be implemented to enable the Hessian GEVP
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Compute samples from a mean zero Gaussian with covariance W_z^{-1}
    virtual void Sample_with_Covariance_W_z_Acute_Inverse(HDSA::MultiVector<RealT> &samples) const
    {
      (void) samples;
      HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                              "Error in HDSA::MD_Elliptic_z_Prior_Interface::Sample_with_Covariance_W_z_Acute_Inverse: Method must be implemented to use sampling algorithms" << std::endl);
    }

    virtual void Apply_E_z(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in) const
    {
      (void) z_out;
      (void) z_in;
      HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                              "Error in HDSA::MD_Elliptic_z_Prior_Interface::Apply_E_z: Method must be implemented to use the Hessian GEVP" << std::endl);
    }

    virtual void Apply_E_z_Transpose(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in) const
    {
      (void) z_out;
      (void) z_in;
      HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                              "Error in HDSA::MD_Elliptic_z_Prior_Interface::Apply_E_z_Transpose: Method must be implemented to use the Hessian GEVP" << std::endl);
    }

    virtual void Apply_M_z_Inverse(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in) const
    {
      (void) z_out;
      (void) z_in;
      HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                              "Error in HDSA::MD_Elliptic_z_Prior_Interface::Apply_M_z_Inverse: Method must be implemented to use the Hessian GEVP" << std::endl);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Implementation of base class Virtual functions
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    virtual void Apply_W_z_Acute_Inverse(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in) const
    {
      HDSA::Ptr<HDSA::Vector<RealT>> z_tmp1 = z_in.Clone();
      HDSA::Ptr<HDSA::Vector<RealT>> z_tmp2 = z_in.Clone();
      Apply_E_z_Inverse_Transpose(*z_tmp1, z_in);
      Apply_M_z(*z_tmp2, *z_tmp1);
      Apply_E_z_Inverse(z_out, *z_tmp2);
    }

    virtual void Apply_W_z_Acute(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in) const
    {
      HDSA::Ptr<HDSA::Vector<RealT>> z_tmp1 = z_in.Clone();
      HDSA::Ptr<HDSA::Vector<RealT>> z_tmp2 = z_in.Clone();
      Apply_E_z(*z_tmp1, z_in);
      Apply_M_z_Inverse(*z_tmp2, *z_tmp1);
      Apply_E_z_Transpose(z_out, *z_tmp2);
    }
  };

}

#endif
