/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_PC_LIS_INTERFACE_HPP
#define HDSA_PC_LIS_INTERFACE_HPP

namespace HDSA
{

  template <class RealT>
  class PC_LIS_Interface
  {

  private:
  public:
    PC_LIS_Interface()
    {
    }

    virtual ~PC_LIS_Interface()
    {
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Pure virtual functions
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    virtual void Apply_Misfit_Hessian(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in, const HDSA::Vector<RealT> &z, const HDSA::Vector<RealT> &theta) const = 0;

    virtual void Apply_Prior_Precision(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in) const = 0;

    virtual void Apply_Prior_Covariance(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in) const = 0;

    virtual void Generate_Prior_Samples(HDSA::MultiVector<RealT> &samples) const = 0;
  };

}

#endif
