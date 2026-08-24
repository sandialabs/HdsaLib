/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_BF_SOL_OP_INTERFACE_SYNTHETIC_TEST_OUU_HPP
#define HDSA_BF_SOL_OP_INTERFACE_SYNTHETIC_TEST_OUU_HPP

#include "HDSA_BF_OUU_Sol_Op_Interface.hpp"

template <class RealT>
class BF_Sol_Op_Interface_synthetic_test_OUU : public HDSA::BF_OUU_Sol_Op_Interface<RealT>
{

private:
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Xi_;
  int m_;                                  
  RealT alpha_;

public:
  BF_Sol_Op_Interface_synthetic_test_OUU(HDSA::Ptr<HDSA::Dense_Matrix<RealT>> &Xi): Xi_(Xi)
  {
    m_ = 51;
    alpha_ = 1.2;
  }

  virtual ~BF_Sol_Op_Interface_synthetic_test_OUU()
  {
  }

  // Assume a constraint u = Xi(0) * z^3 + Xi(1) nodewise on the mesh defined by nodes in x_
  // Assume an objective (1/2)*(u-T)^t*M*(u-T) where T = (x_+1.0)^3 and
  // Assume a high-fidelity model u = alpha * Xi(0) * z^3 + Xi(1)

  void State_Solve_Per_Sample(HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &z, int s) const
  {
    HDSA::Std_Vector<RealT> u_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(u);
    const HDSA::Std_Vector<RealT> z_std = dynamic_cast<const HDSA::Std_Vector<RealT> &>(z);
    for (int k = 0; k < m_; k++)
    {
      u_std.Set_Entry(k, alpha_ * (*Xi_)(0,s) * std::pow(z_std(k), 3.0) + (*Xi_)(1,s) );
    }
  }

  void Apply_Solution_Operator_z_Jacobian_Transpose_Per_Sample(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &u_in, const HDSA::Vector<RealT> &z, int s) const
  {
    const HDSA::Std_Vector<RealT> u_in_std = dynamic_cast<const HDSA::Std_Vector<RealT> &>(u_in);
    const HDSA::Std_Vector<RealT> z_std = dynamic_cast<const HDSA::Std_Vector<RealT> &>(z);
    HDSA::Std_Vector<RealT> z_out_std = dynamic_cast<const HDSA::Std_Vector<RealT> &>(z_out);
    for (int k = 0; k < m_; k++)
    {
      z_out_std.Set_Entry(k, alpha_ * (*Xi_)(0,s) * 3.0 * std::pow(z_std(k), 2.0) * u_in_std(k) );
    }
  }
};

#endif
