/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_BF_SOL_OP_INTERFACE_HPP
#define HDSA_BF_SOL_OP_INTERFACE_HPP

#include "HDSA_Vector.hpp"

namespace HDSA
{

  template <class RealT>
  class BF_Sol_Op_Interface
  {

  private:
  public:
    BF_Sol_Op_Interface()
    {
    }

    virtual ~BF_Sol_Op_Interface()
    {
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Pure virtual functions
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    virtual void State_Solve(HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &z) const = 0;

    virtual void Apply_Solution_Operator_z_Jacobian_Transpose(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &u_in, const HDSA::Vector<RealT> &z) const = 0;
  };

}

#endif
