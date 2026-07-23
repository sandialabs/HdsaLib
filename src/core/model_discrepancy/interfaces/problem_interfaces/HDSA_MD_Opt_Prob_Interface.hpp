/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_OPT_PROB_INTERFACE_HPP
#define HDSA_MD_OPT_PROB_INTERFACE_HPP

namespace HDSA
{

  template <class RealT>
  class MD_Opt_Prob_Interface
  {

  private:
  public:
    MD_Opt_Prob_Interface()
    {
    }

    virtual ~MD_Opt_Prob_Interface()
    {
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Pure virtual functions
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    virtual void Apply_Solution_Operator_z_Jacobian_Transpose(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &u_in, const HDSA::Vector<RealT> &z) const = 0;

    virtual void Apply_RS_Hessian(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in, const HDSA::Vector<RealT> &z) const = 0;

    virtual void Misfit_Gradient(HDSA::Vector<RealT> &u_grad, const HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &z) const = 0;

    virtual void Apply_Misfit_Hessian(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in, const HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &z) const = 0;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// The following virtual functions are needed for the continuation algorithm
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    virtual void Apply_Solution_Operator_z_Jacobian(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &z_in, const HDSA::Vector<RealT> &z) const {};
    
    virtual void State_Solve(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &z) const {};
    
    virtual void Regularization_Gradient(HDSA::Vector<RealT> &grad_z, const HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &z) const {};
  };

}

#endif
