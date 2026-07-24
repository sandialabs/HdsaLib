/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef DATA_INTERFACE_SIMOPT_TEST_PROBLEM_HPP
#define DATA_INTERFACE_SIMOPT_TEST_PROBLEM_HPP

#include "HDSA_MD_Data_Interface.hpp"

template <class RealT>
class Data_Interface_SimOptTestProb : public HDSA::MD_Data_Interface<RealT>
{

private:
  int m_;                                  // Mesh resolution
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x_; // Mesh nodes on [0,1]
  HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> random_number_generator_;
  HDSA::Ptr<const HDSA::Comm<int>> comm_;

public:
  Data_Interface_SimOptTestProb(int &m, HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> &random_number_generator, HDSA::Ptr<const HDSA::Comm<int>> &comm) : random_number_generator_(random_number_generator), comm_(comm)
  {
    m_ = m;

    x_ = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    for (int k = 0; k < m_; k++)
    {
      x_->Set_Entry(k, 0, static_cast<RealT>(k) / static_cast<RealT>(m_ - 1));
    }
  }

  virtual ~Data_Interface_SimOptTestProb()
  {
  }

  HDSA::Ptr<HDSA::Vector<RealT>> Load_Optimal_u(void) const
  {
    ROL::Ptr<std::vector<RealT>> u_ptr = ROL::makePtr<std::vector<RealT>>(m_, 0.0);
    ROL::StdVector<RealT> u(u_ptr);
    ROL::Ptr<ROL::Vector<RealT>> up = ROL::makePtrFromRef(u);
    for (int i = 0; i < m_; i++)
    {
      (*u_ptr)[i] = std::pow(1.0 + (*x_)(i, 0), 3.0);
    }
    HDSA::Ptr<HDSA::Vector<RealT>> u_opt = HDSA::makePtr<HDSA::ROL_Vector<RealT>>(*up, random_number_generator_, comm_);
    HDSA::ROL_Vector<RealT> &u_opt_rol = dynamic_cast<HDSA::ROL_Vector<RealT> &>(*u_opt);
    u_opt_rol.rol_vec->set(*up);
    return u_opt;
  }

  HDSA::Ptr<HDSA::Vector<RealT>> Load_Optimal_z(void) const
  {
    ROL::Ptr<std::vector<RealT>> z_ptr = ROL::makePtr<std::vector<RealT>>(m_, 0.0);
    ROL::StdVector<RealT> z(z_ptr);
    ROL::Ptr<ROL::Vector<RealT>> zp = ROL::makePtrFromRef(z);
    for (int i = 0; i < m_; i++)
    {
      (*z_ptr)[i] = 1.0 + (*x_)(i, 0);
    }
    HDSA::Ptr<HDSA::Vector<RealT>> z_opt = HDSA::makePtr<HDSA::ROL_Vector<RealT>>(*zp, random_number_generator_, comm_);
    HDSA::ROL_Vector<RealT> &z_opt_rol = dynamic_cast<HDSA::ROL_Vector<RealT> &>(*z_opt);
    z_opt_rol.rol_vec->set(*zp);
    return z_opt;
  }

  HDSA::Ptr<HDSA::MultiVector<RealT>> Load_Z_Data(void) const
  {
    HDSA::Ptr<HDSA::Vector<RealT>> z_opt = Load_Optimal_z();
    HDSA::Ptr<HDSA::MultiVector<RealT>> Z = HDSA::makePtr<HDSA::MultiVector<RealT>>(2, *z_opt);
    (*Z)[0]->Set(*z_opt);

    ROL::Ptr<std::vector<RealT>> z_ptr = ROL::makePtr<std::vector<RealT>>(m_, 0.0);
    ROL::StdVector<RealT> z(z_ptr);
    ROL::Ptr<ROL::Vector<RealT>> zp = ROL::makePtrFromRef(z);
    for (int i = 0; i < m_; i++)
    {
      (*z_ptr)[i] = (*x_)(i, 0) + std::pow((*x_)(i, 0), 2.0);
    }
    HDSA::ROL_Vector<RealT> &z_opt_rol = dynamic_cast<HDSA::ROL_Vector<RealT> &>(*z_opt);
    z_opt_rol.rol_vec->set(*zp);
    (*Z)[1]->Set(*z_opt);

    return Z;
  }

  HDSA::Ptr<HDSA::MultiVector<RealT>> Load_D_Data(void) const
  {
    ROL::Ptr<std::vector<RealT>> u_ptr = ROL::makePtr<std::vector<RealT>>(m_, 0.0);
    ROL::StdVector<RealT> u(u_ptr);
    ROL::Ptr<ROL::Vector<RealT>> up = ROL::makePtrFromRef(u);
    for (int i = 0; i < m_; i++)
    {
      (*u_ptr)[i] = 0.2 * std::pow(1.0 + (*x_)(i, 0), 3.0);
    }
    HDSA::Ptr<HDSA::Vector<RealT>> u_hdsa = HDSA::makePtr<HDSA::ROL_Vector<RealT>>(*up, random_number_generator_, comm_);
    HDSA::ROL_Vector<RealT> &u_hdsa_rol = dynamic_cast<HDSA::ROL_Vector<RealT> &>(*u_hdsa);
    u_hdsa_rol.rol_vec->set(*up);
    HDSA::Ptr<HDSA::MultiVector<RealT>> Y = HDSA::makePtr<HDSA::MultiVector<RealT>>(2, *u_hdsa);
    (*Y)[0]->Set(*u_hdsa);

    for (int i = 0; i < m_; i++)
    {
      (*u_ptr)[i] = 0.2 * std::pow((*x_)(i, 0) + std::pow((*x_)(i, 0), 2.0), 3.0);
    }
    u_hdsa_rol.rol_vec->set(*up);
    (*Y)[1]->Set(*u_hdsa);

    return Y;
  }
};

#endif
