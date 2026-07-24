/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_U_PRIOR_INTERFACE_PDE_TEST_PROBLEM_HPP
#define HDSA_MD_U_PRIOR_INTERFACE_PDE_TEST_PROBLEM_HPP

#include "HDSA_MD_Elliptic_u_Prior_Interface.hpp"

template <class RealT>
class MD_Elliptic_u_Prior_Interface_PDE_Test_Problem : public HDSA::MD_Elliptic_u_Prior_Interface<RealT>
{

private:
  int m_;                                    // Mesh resolution
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> S_;   // Stiffness matrix
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> M_;   // Mass matrix
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> E_u_; // State elliptic operator

public:
  MD_Elliptic_u_Prior_Interface_PDE_Test_Problem(RealT &alpha_u, const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> &random_number_generator) : HDSA::MD_Elliptic_u_Prior_Interface<RealT>(alpha_u, random_number_generator)
  {
    m_ = 200;
    RealT h = 1.0 / static_cast<RealT>(m_ - 1);

    S_ = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, m_);
    M_ = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, m_);

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

    E_u_ = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, m_);
    for (int i = 0; i < m_; i++)
    {
      for (int j = 0; j < m_; j++)
      {
        RealT val = (1.e-2) * (*S_)(i, j) + (*M_)(i, j);
        E_u_->Set_Entry(i, j, val);
      }
    }
  }

  virtual ~MD_Elliptic_u_Prior_Interface_PDE_Test_Problem()
  {
  }

  void Apply_M_u(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in) const
  {
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> b = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    const HDSA::Std_Vector<RealT> &u_in_std = dynamic_cast<const HDSA::Std_Vector<RealT> &>(u_in);
    HDSA::Std_Vector<RealT> &u_out_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(u_out);
    for (int k = 0; k < m_; k++)
    {
      b->Set_Entry(k, 0, u_in_std(k));
    }
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    M_->Multiply(*x, *b);
    for (int k = 0; k < m_; k++)
    {
      u_out_std.Set_Entry(k, (*x)(k, 0));
    }
  }

  void Apply_E_u_Inverse(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in) const
  {
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> b = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    const HDSA::Std_Vector<RealT> &u_in_std = dynamic_cast<const HDSA::Std_Vector<RealT> &>(u_in);
    HDSA::Std_Vector<RealT> &u_out_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(u_out);
    for (int k = 0; k < m_; k++)
    {
      b->Set_Entry(k, 0, u_in_std(k));
    }
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    HDSA::Linear_Algebra::Symmetric_Direct_Linear_Solve<RealT>(*E_u_, *x, *b);
    for (int k = 0; k < m_; k++)
    {
      u_out_std.Set_Entry(k, (*x)(k, 0));
    }
  }

  void Apply_E_u_Inverse_Transpose(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in) const
  {
    Apply_E_u_Inverse(u_out, u_in);
  }
};

#endif
