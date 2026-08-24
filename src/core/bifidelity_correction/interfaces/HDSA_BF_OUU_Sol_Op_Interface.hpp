/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_BF_OUU_SOL_OP_INTERFACE_HPP
#define HDSA_BF_OUU_SOL_OP_INTERFACE_HPP

namespace HDSA
{

  template <class RealT>
  class BF_OUU_Sol_Op_Interface
  {

  private:
  public:
    BF_OUU_Sol_Op_Interface()
    {
    }

    virtual ~BF_OUU_Sol_Op_Interface()
    {
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Pure virtual functions
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    virtual void State_Solve_Per_Sample(HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &z, int s) const = 0;

    virtual void Apply_Solution_Operator_z_Jacobian_Transpose_Per_Sample(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &u_in, const HDSA::Vector<RealT> &z, int s) const = 0;
  };

}

#endif
