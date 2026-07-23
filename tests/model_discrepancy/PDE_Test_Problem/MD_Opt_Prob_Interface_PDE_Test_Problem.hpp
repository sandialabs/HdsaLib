/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_OPT_PROB_INTERFACE_PDE_TEST_PROBLEM_HPP
#define HDSA_MD_OPT_PROB_INTERFACE_PDE_TEST_PROBLEM_HPP

#include "HDSA_MD_Opt_Prob_Interface.hpp"

template <class RealT>
class MD_Opt_Prob_Interface_PDE_Test_Problem : public HDSA::MD_Opt_Prob_Interface<RealT>
{

private:
  int m_;                                            // Mesh resolution
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x_;           // Mesh nodes on [0,1]
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> target_;      // Target state solution in misfit
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> S_;           // Stiffness matrix
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> M_;           // Mass matrix
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> robin_bc_;    // Robin boundary condition matrix
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> sol_op_lofi_; // Low-fidelity solution operator
  RealT diff_coeff_;                                 // Diffusion coefficient
  RealT robin_coeff_;                                // Robin boundary condition coefficient
  RealT reg_coeff_;                                  // Regularization coefficient

public:
  MD_Opt_Prob_Interface_PDE_Test_Problem()
  {
    m_ = 200;
    RealT h = 1.0 / static_cast<RealT>(m_ - 1);
    x_ = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    target_ = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    for (int k = 0; k < m_; k++)
    {
      x_->Set_Entry(k, 0, static_cast<RealT>(k) / static_cast<RealT>(m_ - 1));
      target_->Set_Entry(k, 0, 50.0 - 30.0 * std::pow((*x_)(k, 0) - 0.5, 2.0));
    }

    S_ = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, m_);
    M_ = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, m_);
    robin_bc_ = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, m_);

    S_->Set_Entry(0, 0, 1.0 / h);
    S_->Set_Entry(0, 1, -1.0 / h);
    for (int i = 1; i < m_ - 1; i++)
    {
      S_->Set_Entry(i, i, 2.0 / h);
      S_->Set_Entry(i, i - 1, -1.0 / h);
      S_->Set_Entry(i, i + 1, -1.0 / h);
    }
    S_->Set_Entry(m_ - 1, m_ - 2, -1.0 / h);
    S_->Set_Entry(m_ - 1, m_ - 1, 1.0 / h);

    M_->Set_Entry(0, 0, (1.0 / 3.0) * h);
    M_->Set_Entry(0, 1, (1.0 / 6.0) * h);
    for (int i = 1; i < m_ - 1; i++)
    {
      M_->Set_Entry(i, i, (2.0 / 3.0) * h);
      M_->Set_Entry(i, i - 1, (1.0 / 6.0) * h);
      M_->Set_Entry(i, i + 1, (1.0 / 6.0) * h);
    }
    M_->Set_Entry(m_ - 1, m_ - 2, (1.0 / 6.0) * h);
    M_->Set_Entry(m_ - 1, m_ - 1, (1.0 / 3.0) * h);

    robin_bc_->Set_Entry(0, 0, 1.0);
    robin_bc_->Set_Entry(m_ - 1, m_ - 1, 1.0);

    diff_coeff_ = 1.0;
    robin_coeff_ = 2.0;
    reg_coeff_ = 10.0;

    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> tmp_lofi = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, m_);
    for (int i = 0; i < m_; i++)
    {
      for (int j = 0; j < m_; j++)
      {
        RealT val = diff_coeff_ * (*S_)(i, j) + robin_coeff_ * (*robin_bc_)(i, j);
        tmp_lofi->Set_Entry(i, j, (1.e-2) * val);
      }
    }
    sol_op_lofi_ = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, m_);
    HDSA::Linear_Algebra::Symmetric_Direct_Linear_Solve<RealT>(*tmp_lofi, *sol_op_lofi_, *M_);
  }

  virtual ~MD_Opt_Prob_Interface_PDE_Test_Problem()
  {
  }

  void Apply_Solution_Operator_z_Jacobian_Transpose(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &u_in, const HDSA::Vector<RealT> &z) const
  {
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> b = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    const HDSA::Std_Vector<RealT> &u_in_std = dynamic_cast<const HDSA::Std_Vector<RealT> &>(u_in);
    for (int k = 0; k < m_; k++)
    {
      b->Set_Entry(k, 0, u_in_std(k));
    }

    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> tmp = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    sol_op_lofi_->Multiply(*tmp, *b, true);

    HDSA::Std_Vector<RealT> z_out_std = dynamic_cast<const HDSA::Std_Vector<RealT> &>(z_out);
    for (int k = 0; k < m_; k++)
    {
      z_out_std.Set_Entry(k, (*tmp)(k, 0));
    }
  }

  void Apply_RS_Hessian(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in, const HDSA::Vector<RealT> &z) const
  {
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> b = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    const HDSA::Std_Vector<RealT> &z_in_std = dynamic_cast<const HDSA::Std_Vector<RealT> &>(z_in);
    HDSA::Std_Vector<RealT> &z_out_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(z_out);
    for (int k = 0; k < m_; k++)
    {
      b->Set_Entry(k, 0, z_in_std(k));
    }

    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> tmp1 = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    sol_op_lofi_->Multiply(*tmp1, *b);
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> tmp2 = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    M_->Multiply(*tmp2, *tmp1);
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> tmp3 = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    sol_op_lofi_->Multiply(*tmp3, *tmp2, true);

    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> tmp4 = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    M_->Multiply(*tmp4, *b);

    for (int k = 0; k < m_; k++)
    {
      z_out_std.Set_Entry(k, (*tmp3)(k, 0) + reg_coeff_ * (*tmp4)(k, 0));
    }
  }

  void Misfit_Gradient(HDSA::Vector<RealT> &u_grad, const HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &z) const
  {
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> v = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    const HDSA::Std_Vector<RealT> u_std = dynamic_cast<const HDSA::Std_Vector<RealT> &>(u);
    HDSA::Std_Vector<RealT> u_grad_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(u_grad);
    for (int k = 0; k < m_; k++)
    {
      v->Set_Entry(k, 0, u_std(k) - (*target_)(k, 0));
    }
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> grad = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    M_->Multiply(*grad, *v);
    for (int k = 0; k < m_; k++)
    {
      u_grad_std.Set_Entry(k, (*grad)(k, 0));
    }
  }

  void Apply_Misfit_Hessian(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in, const HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &z) const
  {
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> v = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    const HDSA::Std_Vector<RealT> u_in_std = dynamic_cast<const HDSA::Std_Vector<RealT> &>(u_in);
    HDSA::Std_Vector<RealT> u_out_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(u_out);
    for (int k = 0; k < m_; k++)
    {
      v->Set_Entry(k, 0, u_in_std(k));
    }
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Hv = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    M_->Multiply(*Hv, *v);
    for (int k = 0; k < m_; k++)
    {
      u_out_std.Set_Entry(k, (*Hv)(k, 0));
    }
  }
};

#endif
