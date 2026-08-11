/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_DATA_INTERFACE_SYNTHETIC_TEST_CONTINUATION_HPP
#define HDSA_MD_DATA_INTERFACE_SYNTHETIC_TEST_CONTINUATION_HPP

#include "HDSA_MD_Data_Interface.hpp"
#include "HDSA_Std_Vector.hpp"

template <class RealT>
class MD_Data_Interface_synthetic_test_continuation : public HDSA::MD_Data_Interface<RealT>
{

private:
  int m_;                                  // Mesh resolution
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x_; // Mesh nodes on [0,1]
  const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> random_number_generator_;

public:
  MD_Data_Interface_synthetic_test_continuation(const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> &random_number_generator) : random_number_generator_(random_number_generator)
  {
    m_ = 51;
    x_ = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    for (int k = 0; k < m_; k++)
    {
      x_->Set_Entry(k, 0, static_cast<RealT>(k) / static_cast<RealT>(m_ - 1));
    }
  }

  virtual ~MD_Data_Interface_synthetic_test_continuation()
  {
  }

  HDSA::Ptr<HDSA::Vector<RealT>> Load_Optimal_u() const
  {
    HDSA::Ptr<HDSA::Std_Vector<RealT>> u_opt = HDSA::makePtr<HDSA::Std_Vector<RealT>>(m_, random_number_generator_);
    for (int k = 0; k < m_; k++)
    {
      u_opt->Set_Entry(k, std::pow((*x_)(k, 0) + 1.0, 3.0));
    }
    return u_opt;
  }

  HDSA::Ptr<HDSA::Vector<RealT>> Load_Optimal_z() const
  {
    HDSA::Ptr<HDSA::Std_Vector<RealT>> z_opt = HDSA::makePtr<HDSA::Std_Vector<RealT>>(m_, random_number_generator_);
    for (int k = 0; k < m_; k++)
    {
      z_opt->Set_Entry(k, (*x_)(k, 0) + 1.0);
    }
    return z_opt;
  }

  HDSA::Ptr<HDSA::MultiVector<RealT>> Load_Z_Data() const
  {
    HDSA::Ptr<HDSA::Std_Vector<RealT>> z = HDSA::makePtr<HDSA::Std_Vector<RealT>>(m_, random_number_generator_);
    HDSA::Ptr<HDSA::MultiVector<RealT>> Z = HDSA::makePtr<HDSA::MultiVector<RealT>>(2, *z);

    HDSA::Ptr<HDSA::Vector<RealT>> z0 = (*Z)[0];
    HDSA::Ptr<HDSA::Vector<RealT>> z1 = (*Z)[1];
    HDSA::Std_Vector<RealT> z0_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(*z0);
    HDSA::Std_Vector<RealT> z1_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(*z1);

    for (int k = 0; k < m_; k++)
    {
      z0_std.Set_Entry(k, (*x_)(k, 0) + 1.0);
      z1_std.Set_Entry(k, (*x_)(k, 0) + std::pow((*x_)(k, 0), 2.0));
    }

    return Z;
  }

  HDSA::Ptr<HDSA::MultiVector<RealT>> Load_D_Data() const
  {
    HDSA::Ptr<HDSA::Std_Vector<RealT>> d = HDSA::makePtr<HDSA::Std_Vector<RealT>>(m_, random_number_generator_);
    HDSA::Ptr<HDSA::MultiVector<RealT>> D = HDSA::makePtr<HDSA::MultiVector<RealT>>(2, *d);

    HDSA::Ptr<HDSA::Vector<RealT>> d0 = (*D)[0];
    HDSA::Ptr<HDSA::Vector<RealT>> d1 = (*D)[1];
    HDSA::Std_Vector<RealT> d0_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(*d0);
    HDSA::Std_Vector<RealT> d1_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(*d1);

    for (int k = 0; k < m_; k++)
    {
      d0_std.Set_Entry(k, 0.2 * std::pow((*x_)(k, 0) + 1.0, 3.0));
      d1_std.Set_Entry(k, 0.2 * std::pow((*x_)(k, 0) + std::pow((*x_)(k, 0), 2.0), 3.0));
    }
    return D;
  }
};

#endif
