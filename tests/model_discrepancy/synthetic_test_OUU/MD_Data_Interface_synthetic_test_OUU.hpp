/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_DATA_INTERFACE_SYNTHETIC_TEST_OUU_HPP
#define HDSA_MD_DATA_INTERFACE_SYNTHETIC_TEST_OUU_HPP

#include "HDSA_MD_OUU_Data_Interface.hpp"
#include "HDSA_Std_Vector.hpp"

template <class RealT>
class MD_Data_Interface_synthetic_test_OUU : public HDSA::MD_OUU_Data_Interface<RealT>
{

private:
  int m_;                                  // Mesh resolution
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x_; // Mesh nodes on [0,1]
  const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> random_number_generator_;
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Xi_;

public:
  MD_Data_Interface_synthetic_test_OUU(const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> &random_number_generator, int ens_size, HDSA::Ptr<HDSA::Dense_Matrix<RealT>> &Xi) : HDSA::MD_OUU_Data_Interface<RealT>(ens_size), random_number_generator_(random_number_generator), Xi_(Xi)
  {
    m_ = 51;
    x_ = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    for (int k = 0; k < m_; k++)
    {
      x_->Set_Entry(k, 0, static_cast<RealT>(k) / static_cast<RealT>(m_ - 1));
    }
  }

  virtual ~MD_Data_Interface_synthetic_test_OUU()
  {
  }

  HDSA::Ptr<HDSA::Vector<RealT>> Load_Optimal_us(int s) const
  {
    HDSA::Ptr<HDSA::Std_Vector<RealT>> u_opt = HDSA::makePtr<HDSA::Std_Vector<RealT>>(m_, random_number_generator_);

    HDSA::Ptr<HDSA::Vector<RealT>> z_opt = Load_Optimal_z();
    HDSA::Std_Vector<RealT> z_opt_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(*z_opt);

    for (int k = 0; k < m_; k++)
    {
      RealT val = (*Xi_)(0, s) * std::pow(z_opt_std(k), 3.0) + (*Xi_)(1, s);
      u_opt->Set_Entry(k, val);
    }
    return u_opt;
  }

  HDSA::Ptr<HDSA::Vector<RealT>> Load_Optimal_z() const
  {
    HDSA::Ptr<HDSA::Std_Vector<RealT>> z_opt = HDSA::makePtr<HDSA::Std_Vector<RealT>>(m_, random_number_generator_);

    RealT val = 0.0;
    // read in data
    std::ifstream in("z_opt.txt");
    // read the elements in the file into a vector
    // test file open
    if (in)
    {
      for (int i = 0; i < m_; i++)
      {
        in >> val;
        z_opt->Set_Entry(i, val);
      }
    }
    else
    {
      std::cout << "Error loading the data from z_opt.txt" << std::endl;
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

    // read in data
    std::ifstream in_Z("Z.txt");
    // read the elements in the file into a vector
    // test file open
    RealT val;
    if (in_Z)
    {
      for (int i = 0; i < m_; i++)
      {
        in_Z >> val;
        z0_std.Set_Entry(i, val);

        in_Z >> val;
        z1_std.Set_Entry(i, val);
      }
    }
    else
    {
      std::cout << "Error loading the data from Z.txt" << std::endl;
    }

    return Z;
  }

  HDSA::Ptr<HDSA::MultiVector<RealT>> Load_Ds_Data(int s) const
  {
    HDSA::Ptr<HDSA::Std_Vector<RealT>> d = HDSA::makePtr<HDSA::Std_Vector<RealT>>(m_, random_number_generator_);
    HDSA::Ptr<HDSA::MultiVector<RealT>> D = HDSA::makePtr<HDSA::MultiVector<RealT>>(2, *d);

    HDSA::Ptr<HDSA::Vector<RealT>> d0 = (*D)[0];
    HDSA::Ptr<HDSA::Vector<RealT>> d1 = (*D)[1];
    HDSA::Std_Vector<RealT> d0_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(*d0);
    HDSA::Std_Vector<RealT> d1_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(*d1);

    HDSA::Ptr<HDSA::MultiVector<RealT>> Z = Load_Z_Data();
    HDSA::Std_Vector<RealT> z0_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(*(*Z)[0]);
    HDSA::Std_Vector<RealT> z1_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(*(*Z)[1]);

    for (int k = 0; k < m_; k++)
    {
      RealT val = 0.2 * (*Xi_)(0, s) * std::pow(z0_std(k), 3.0);
      d0_std.Set_Entry(k, val);
      val = 0.2 * (*Xi_)(0, s) * std::pow(z1_std(k), 3.0);
      d1_std.Set_Entry(k, val);
    }

    return D;
  }
};

#endif
