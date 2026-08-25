/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_DATA_INTERFACE_SYNTHETIC_TEST_HPP
#define HDSA_MD_DATA_INTERFACE_SYNTHETIC_TEST_HPP

#include "HDSA_MD_Data_Interface.hpp"
#include "HDSA_Std_Vector.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "HDSA_Tpetra_Vector.hpp"

template <class RealT>
class MD_Data_Interface_synthetic_test : public HDSA::MD_Data_Interface<RealT>
{

private:
  int n_y_; // Mesh resolution
  int n_t_; // Number of time steps
  RealT c_low_;
  RealT c_high_;
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x_; // Mesh nodes on [0,1]
  const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> Random_number_generator_;
  HDSA::Ptr<const HDSA::Comm<int>> comm_;

public:
  MD_Data_Interface_synthetic_test(const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> &Random_number_generator, HDSA::Ptr<const HDSA::Comm<int>> &comm, int n_y, int n_t, RealT c_low, RealT c_high) : Random_number_generator_(Random_number_generator), comm_(comm)
  {
    n_y_ = n_y;
    n_t_ = n_t;
    c_low_ = c_low;
    c_high_ = c_high;
    x_ = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_y_, 1);
    for (int k = 0; k < n_y_; k++)
    {
      x_->Set_Entry(k, 0, static_cast<RealT>(k) / static_cast<RealT>(n_y_ - 1));
    }
  }

  virtual ~MD_Data_Interface_synthetic_test()
  {
  }

  HDSA::Ptr<HDSA::Vector<RealT>> Load_Optimal_u() const
  {
    auto map = Tpetra::createUniformContigMap<Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type>(n_y_, comm_->Get_Teuchos_Communicator());
    HDSA::Ptr<Tpetra::MultiVector<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type, Tpetra::Map<>::node_type>> tpetra_vec = Tpetra::createMultiVector<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type, Tpetra::Map<>::node_type>(map, 1);
    HDSA::Ptr<HDSA::Vector<RealT>> spatial_vec = HDSA::makePtr<HDSA::Tpetra_Vector<RealT>>(tpetra_vec, Random_number_generator_);
    HDSA::Ptr<HDSA::Vector<RealT>> u_opt = HDSA::makePtr<HDSA::Transient_Vector<RealT>>(n_t_, spatial_vec);
    HDSA::Transient_Vector<RealT> u_opt_trans = dynamic_cast<HDSA::Transient_Vector<RealT> &>(*u_opt);
    RealT coeff = 1.0;
    for (int j = 0; j < n_t_; j++)
    {
      HDSA::Ptr<HDSA::Vector<RealT>> uj = u_opt_trans[j];
      HDSA::Tpetra_Vector<RealT> uj_tpetra = dynamic_cast<HDSA::Tpetra_Vector<RealT> &>(*uj);
      for (int k = 0; k < n_y_; k++)
      {
        uj_tpetra.getVector()->replaceGlobalValue(k, 0, coeff * std::pow((*x_)(k, 0) + 1.0, 3.0));
      }
      coeff = c_low_ * coeff;
    }
    return u_opt;
  }

  HDSA::Ptr<HDSA::Vector<RealT>> Load_Optimal_z() const
  {
    auto map = Tpetra::createUniformContigMap<Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type>(n_y_, comm_->Get_Teuchos_Communicator());
    HDSA::Ptr<Tpetra::MultiVector<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type, Tpetra::Map<>::node_type>> tpetra_vec = Tpetra::createMultiVector<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type, Tpetra::Map<>::node_type>(map, 1);
    for (int k = 0; k < n_y_; k++)
    {
      tpetra_vec->replaceGlobalValue(k, 0, (*x_)(k, 0) + 1.0);
    }
    HDSA::Ptr<HDSA::Vector<RealT>> z_opt = HDSA::makePtr<HDSA::Tpetra_Vector<RealT>>(tpetra_vec, Random_number_generator_);
    return z_opt;
  }

  HDSA::Ptr<HDSA::MultiVector<RealT>> Load_Z_Data() const
  {
    auto map = Tpetra::createUniformContigMap<Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type>(n_y_, comm_->Get_Teuchos_Communicator());
    HDSA::Ptr<Tpetra::MultiVector<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type, Tpetra::Map<>::node_type>> tpetra_vec = Tpetra::createMultiVector<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type, Tpetra::Map<>::node_type>(map, 1);
    HDSA::Ptr<HDSA::Vector<RealT>> z = HDSA::makePtr<HDSA::Tpetra_Vector<RealT>>(tpetra_vec, Random_number_generator_);
    HDSA::Ptr<HDSA::MultiVector<RealT>> Z = HDSA::makePtr<HDSA::MultiVector<RealT>>(2, *z);

    HDSA::Ptr<HDSA::Vector<RealT>> z0 = (*Z)[0];
    HDSA::Ptr<HDSA::Vector<RealT>> z1 = (*Z)[1];
    HDSA::Tpetra_Vector<RealT> z0_tpetra = dynamic_cast<HDSA::Tpetra_Vector<RealT> &>(*z0);
    HDSA::Tpetra_Vector<RealT> z1_tpetra = dynamic_cast<HDSA::Tpetra_Vector<RealT> &>(*z1);

    for (int k = 0; k < n_y_; k++)
    {
      z0_tpetra.getVector()->replaceGlobalValue(k, 0, (*x_)(k, 0) + 1.0);
      z1_tpetra.getVector()->replaceGlobalValue(k, 0, (*x_)(k, 0) + std::pow((*x_)(k, 0), 2.0));
    }
    return Z;
  }

  HDSA::Ptr<HDSA::MultiVector<RealT>> Load_D_Data() const
  {
    auto map = Tpetra::createUniformContigMap<Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type>(n_y_, comm_->Get_Teuchos_Communicator());
    HDSA::Ptr<Tpetra::MultiVector<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type, Tpetra::Map<>::node_type>> tpetra_vec = Tpetra::createMultiVector<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type, Tpetra::Map<>::node_type>(map, 1);
    HDSA::Ptr<HDSA::Vector<RealT>> spatial_vec = HDSA::makePtr<HDSA::Tpetra_Vector<RealT>>(tpetra_vec, Random_number_generator_);
    HDSA::Ptr<HDSA::Vector<RealT>> d = HDSA::makePtr<HDSA::Transient_Vector<RealT>>(n_t_, spatial_vec);
    HDSA::Ptr<HDSA::MultiVector<RealT>> D = HDSA::makePtr<HDSA::MultiVector<RealT>>(2, *d);

    HDSA::Ptr<HDSA::Vector<RealT>> d0 = (*D)[0];
    HDSA::Ptr<HDSA::Vector<RealT>> d1 = (*D)[1];
    HDSA::Transient_Vector<RealT> d0_trans = dynamic_cast<HDSA::Transient_Vector<RealT> &>(*d0);
    HDSA::Transient_Vector<RealT> d1_trans = dynamic_cast<HDSA::Transient_Vector<RealT> &>(*d1);

    RealT coeff_high = 1.0;
    RealT coeff_low = 1.0;
    for (int j = 0; j < n_t_; j++)
    {
      HDSA::Ptr<HDSA::Vector<RealT>> d0_j = d0_trans[j];
      HDSA::Tpetra_Vector<RealT> d0_tpetra = dynamic_cast<HDSA::Tpetra_Vector<RealT> &>(*d0_j);
      HDSA::Ptr<HDSA::Vector<RealT>> d1_j = d1_trans[j];
      HDSA::Tpetra_Vector<RealT> d1_tpetra = dynamic_cast<HDSA::Tpetra_Vector<RealT> &>(*d1_j);
      for (int k = 0; k < n_y_; k++)
      {
        d0_tpetra.getVector()->replaceGlobalValue(k, 0, (coeff_high - coeff_low) * std::pow((*x_)(k, 0) + 1.0, 3.0));
        d1_tpetra.getVector()->replaceGlobalValue(k, 0, (coeff_high - coeff_low) * std::pow((*x_)(k, 0) + std::pow((*x_)(k, 0), 2.0), 3.0));
      }
      coeff_high = c_high_ * coeff_high;
      coeff_low = c_low_ * coeff_low;
    }
    return D;
  }
};

#endif
