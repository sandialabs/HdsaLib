/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_U_PRIOR_INTERFACE_SYNTHETIC_TEST_CONTINUATION_HPP
#define HDSA_MD_U_PRIOR_INTERFACE_SYNTHETIC_TEST_CONTINUATION_HPP

#include "HDSA_MD_u_Prior_Interface.hpp"

template <class RealT>
class MD_u_Prior_Interface_synthetic_test_continuation : public HDSA::MD_u_Prior_Interface<RealT>
{

private:
  int m_;                                    // Mesh resolution
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x_;   // Mesh nodes on [0,1]
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> S_;   // Stiffness matrix
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> M_;   // Mass matrix
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> W_u_; // State weighting matrix
  const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> random_number_generator_;

public:
  MD_u_Prior_Interface_synthetic_test_continuation(const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> &random_number_generator) : random_number_generator_(random_number_generator)
  {
    m_ = 51;
    RealT h = 1.0 / static_cast<RealT>(m_ - 1);
    x_ = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    for (int k = 0; k < m_; k++)
    {
      x_->Set_Entry(k, 0, static_cast<RealT>(k) / static_cast<RealT>(m_ - 1));
    }

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

    W_u_ = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, m_);

    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> I = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, m_);
    for (int k = 0; k < m_; k++)
    {
      I->Set_Entry(k, k, 1.0);
    }
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Minv = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, m_);
    HDSA::Linear_Algebra::Symmetric_Direct_Linear_Solve<RealT>(*M_, *Minv, *I);

    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> E_u = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, m_);
    for (int i = 0; i < m_; i++)
    {
      for (int j = 0; j < m_; j++)
      {
        RealT val = 2.0 * ((5.e-2) * (*S_)(i, j) + (*M_)(i, j));
        E_u->Set_Entry(i, j, val);
      }
    }

    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> tmp = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, m_);
    Minv->Multiply(*tmp, *E_u);
    E_u->Multiply(*W_u_, *tmp);
  }

  virtual ~MD_u_Prior_Interface_synthetic_test_continuation()
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

  void Apply_W_u_Plus_scalar_M_u_Inverse(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in, const RealT &scalar) const
  {
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> b = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    const HDSA::Std_Vector<RealT> &u_in_std = dynamic_cast<const HDSA::Std_Vector<RealT> &>(u_in);
    HDSA::Std_Vector<RealT> &u_out_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(u_out);
    for (int k = 0; k < m_; k++)
    {
      b->Set_Entry(k, 0, u_in_std(k));
    }
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);

    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Wu_scalar_Mu = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, m_);
    for (int i = 0; i < m_; i++)
    {
      for (int j = 0; j < m_; j++)
      {
        Wu_scalar_Mu->Set_Entry(i, j, (*W_u_)(i, j) + scalar * (*M_)(i, j));
      }
    }
    HDSA::Linear_Algebra::Symmetric_Direct_Linear_Solve<RealT>(*Wu_scalar_Mu, *x, *b);
    for (int k = 0; k < m_; k++)
    {
      u_out_std.Set_Entry(k, (*x)(k, 0));
    }
  }

  void Apply_W_u_Inverse(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in) const
  {
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> b = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    const HDSA::Std_Vector<RealT> &u_in_std = dynamic_cast<const HDSA::Std_Vector<RealT> &>(u_in);
    HDSA::Std_Vector<RealT> &u_out_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(u_out);
    for (int k = 0; k < m_; k++)
    {
      b->Set_Entry(k, 0, u_in_std(k));
    }
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    HDSA::Linear_Algebra::Symmetric_Direct_Linear_Solve<RealT>(*W_u_, *x, *b);
    for (int k = 0; k < m_; k++)
    {
      u_out_std.Set_Entry(k, (*x)(k, 0));
    }
  }

  // Compute samples from a mean zero Gaussian with covariance W_u^{-1}
  virtual void Sample_with_Covariance_W_u_Inverse(HDSA::MultiVector<RealT> &samples) const
  {
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> R = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, m_);
    HDSA::Linear_Algebra::Cholesky_Factorization(*W_u_, *R);

    int num_samples = samples.Number_of_Vectors();
    for (int i = 0; i < num_samples; i++)
    {
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> b = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
      for (int k = 0; k < m_; k++)
      {
        b->Set_Entry(k, 0, random_number_generator_->Generate_Standard_Normal_Sample());
      }
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
      HDSA::Linear_Algebra::Upper_Tri_Solve<RealT>(*x, *b, *R);
      HDSA::Std_Vector<RealT> &vec_out_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(*samples[i]);
      for (int k = 0; k < m_; k++)
      {
        vec_out_std.Set_Entry(k, (*x)(k, 0));
      }
    }
  }

  // Compute samples from a mean zero Gaussian with covariance W_u^{-1}
  virtual void Sample_with_Covariance_W_u_Plus_scalar_M_u_Inverse(HDSA::MultiVector<RealT> &samples, const RealT &scalar) const
  {
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> A = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, m_);
    for (int i = 0; i < m_; i++)
    {
      for (int j = 0; j < m_; j++)
      {
        RealT val = (*W_u_)(i, j) + scalar * (*M_)(i, j);
        A->Set_Entry(i, j, val);
      }
    }

    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> R = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, m_);
    HDSA::Linear_Algebra::Cholesky_Factorization(*A, *R);

    int num_samples = samples.Number_of_Vectors();
    for (int i = 0; i < num_samples; i++)
    {
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> b = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
      for (int k = 0; k < m_; k++)
      {
        b->Set_Entry(k, 0, random_number_generator_->Generate_Standard_Normal_Sample());
      }
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
      HDSA::Linear_Algebra::Upper_Tri_Solve<RealT>(*x, *b, *R);
      HDSA::Std_Vector<RealT> &vec_out_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(*samples[i]);
      for (int k = 0; k < m_; k++)
      {
        vec_out_std.Set_Entry(k, (*x)(k, 0));
      }
    }
  }
};

#endif
