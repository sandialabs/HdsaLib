/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_DATA_INTERFACE_PDE_TEST_PROBLEM_HPP
#define HDSA_MD_DATA_INTERFACE_PDE_TEST_PROBLEM_HPP

#include "HDSA_MD_Data_Interface.hpp"
#include "HDSA_Std_Vector.hpp"

template <class RealT>
class MD_Data_Interface_PDE_Test_Problem : public HDSA::MD_Data_Interface<RealT>
{

private:
  int m_; // Mesh resolution
  const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> random_number_generator_;
  const HDSA::Ptr<const HDSA::Comm<int>> comm_;

public:
  MD_Data_Interface_PDE_Test_Problem(const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> &random_number_generator, const HDSA::Ptr<const HDSA::Comm<int>> &comm)
      : random_number_generator_(random_number_generator), comm_(comm)
  {
    m_ = 200;
  }

  virtual ~MD_Data_Interface_PDE_Test_Problem()
  {
  }

  HDSA::Ptr<HDSA::Vector<RealT>> Load_Optimal_u() const
  {
    HDSA::Ptr<HDSA::Std_Vector<RealT>> u_opt = HDSA::makePtr<HDSA::Std_Vector<RealT>>(m_, random_number_generator_, comm_);

    RealT val = 0.0;
    // read in data
    std::ifstream in("u_opt.txt");
    // read the elements in the file into a vector
    // test file open
    if (in)
    {
      for (int i = 0; i < m_; i++)
      {
        in >> val;
        u_opt->Set_Entry(i, val);
      }
    }
    else
    {
      std::cout << "Error loading the data from u_opt.txt" << std::endl;
    }
    return u_opt;
  }

  HDSA::Ptr<HDSA::Vector<RealT>> Load_Optimal_z() const
  {
    HDSA::Ptr<HDSA::Std_Vector<RealT>> z_opt = HDSA::makePtr<HDSA::Std_Vector<RealT>>(m_, random_number_generator_, comm_);

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
    HDSA::Ptr<HDSA::Std_Vector<RealT>> z = HDSA::makePtr<HDSA::Std_Vector<RealT>>(m_, random_number_generator_, comm_);
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

  HDSA::Ptr<HDSA::MultiVector<RealT>> Load_D_Data() const
  {
    HDSA::Ptr<HDSA::Std_Vector<RealT>> d = HDSA::makePtr<HDSA::Std_Vector<RealT>>(m_, random_number_generator_, comm_);
    HDSA::Ptr<HDSA::MultiVector<RealT>> D = HDSA::makePtr<HDSA::MultiVector<RealT>>(2, *d);

    HDSA::Ptr<HDSA::Vector<RealT>> d0 = (*D)[0];
    HDSA::Ptr<HDSA::Vector<RealT>> d1 = (*D)[1];
    HDSA::Std_Vector<RealT> d0_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(*d0);
    HDSA::Std_Vector<RealT> d1_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(*d1);

    // read in data
    std::ifstream in_D("D.txt");
    // read the elements in the file into a vector
    // test file open
    RealT val;
    if (in_D)
    {
      for (int i = 0; i < m_; i++)
      {
        in_D >> val;
        d0_std.Set_Entry(i, val);

        in_D >> val;
        d1_std.Set_Entry(i, val);
      }
    }
    else
    {
      std::cout << "Error loading the data from D.txt" << std::endl;
    }

    return D;
  }
};

#endif
