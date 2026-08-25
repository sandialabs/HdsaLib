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
  int m_;                                  // Mesh resolution
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x_; // Mesh nodes on [0,1]
  const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> random_number_generator_;
  HDSA::Ptr<const HDSA::Comm<int>> comm_;

public:
  MD_Data_Interface_synthetic_test(const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> &random_number_generator, HDSA::Ptr<const HDSA::Comm<int>> &comm) : random_number_generator_(random_number_generator), comm_(comm)
  {
    m_ = 51;
    x_ = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_, 1);
    for (int k = 0; k < m_; k++)
    {
      x_->Set_Entry(k, 0, static_cast<RealT>(k) / static_cast<RealT>(m_ - 1));
    }
  }

  virtual ~MD_Data_Interface_synthetic_test()
  {
  }

  HDSA::Ptr<HDSA::Vector<RealT>> Generate_Spatial_Nodes() const
  {
    auto map = Tpetra::createUniformContigMap<Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type>(m_, comm_->Get_Teuchos_Communicator());
    HDSA::Ptr<Tpetra::MultiVector<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type, Tpetra::Map<>::node_type>> tpetra_vec = Tpetra::createMultiVector<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type, Tpetra::Map<>::node_type>(map, 1);
    for (int k = 0; k < m_; k++)
    {
      tpetra_vec->replaceGlobalValue(k, 0, (*x_)(k, 0));
    }
    HDSA::Ptr<HDSA::Vector<RealT>> spatial_nodes = HDSA::makePtr<HDSA::Tpetra_Vector<RealT>>(tpetra_vec, random_number_generator_);
    return spatial_nodes;
  }

  HDSA::Ptr<HDSA::Vector<RealT>> Load_Optimal_u() const
  {
    auto map = Tpetra::createUniformContigMap<Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type>(m_, comm_->Get_Teuchos_Communicator());
    HDSA::Ptr<Tpetra::MultiVector<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type, Tpetra::Map<>::node_type>> tpetra_vec = Tpetra::createMultiVector<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type, Tpetra::Map<>::node_type>(map, 1);
    for (int k = 0; k < m_; k++)
    {
      tpetra_vec->replaceGlobalValue(k, 0, std::pow((*x_)(k, 0) + 1.0, 3.0));
    }
    HDSA::Ptr<HDSA::Vector<RealT>> u_opt = HDSA::makePtr<HDSA::Tpetra_Vector<RealT>>(tpetra_vec, random_number_generator_);
    return u_opt;
  }

  HDSA::Ptr<HDSA::Vector<RealT>> Load_Optimal_z() const
  {
    auto map = Tpetra::createUniformContigMap<Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type>(m_, comm_->Get_Teuchos_Communicator());
    HDSA::Ptr<Tpetra::MultiVector<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type, Tpetra::Map<>::node_type>> tpetra_vec = Tpetra::createMultiVector<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type, Tpetra::Map<>::node_type>(map, 1);
    for (int k = 0; k < m_; k++)
    {
      tpetra_vec->replaceGlobalValue(k, 0, (*x_)(k, 0) + 1.0);
    }
    HDSA::Ptr<HDSA::Vector<RealT>> z_opt = HDSA::makePtr<HDSA::Tpetra_Vector<RealT>>(tpetra_vec, random_number_generator_);
    return z_opt;
  }

  HDSA::Ptr<HDSA::MultiVector<RealT>> Load_Z_Data() const
  {
    auto map = Tpetra::createUniformContigMap<Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type>(m_, comm_->Get_Teuchos_Communicator());
    HDSA::Ptr<Tpetra::MultiVector<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type, Tpetra::Map<>::node_type>> tpetra_vec = Tpetra::createMultiVector<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type, Tpetra::Map<>::node_type>(map, 1);
    HDSA::Ptr<HDSA::Vector<RealT>> z = HDSA::makePtr<HDSA::Tpetra_Vector<RealT>>(tpetra_vec, random_number_generator_);
    HDSA::Ptr<HDSA::MultiVector<RealT>> Z = HDSA::makePtr<HDSA::MultiVector<RealT>>(2, *z);

    HDSA::Ptr<HDSA::Vector<RealT>> z0 = (*Z)[0];
    HDSA::Ptr<HDSA::Vector<RealT>> z1 = (*Z)[1];
    HDSA::Tpetra_Vector<RealT> z0_tpetra = dynamic_cast<HDSA::Tpetra_Vector<RealT> &>(*z0);
    HDSA::Tpetra_Vector<RealT> z1_tpetra = dynamic_cast<HDSA::Tpetra_Vector<RealT> &>(*z1);

    for (int k = 0; k < m_; k++)
    {
      z0_tpetra.getVector()->replaceGlobalValue(k, 0, (*x_)(k, 0) + 1.0);
      z1_tpetra.getVector()->replaceGlobalValue(k, 0, (*x_)(k, 0) + std::pow((*x_)(k, 0), 2.0));
    }
    return Z;
  }

  HDSA::Ptr<HDSA::MultiVector<RealT>> Load_D_Data() const
  {
    auto map = Tpetra::createUniformContigMap<Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type>(m_, comm_->Get_Teuchos_Communicator());
    HDSA::Ptr<Tpetra::MultiVector<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type, Tpetra::Map<>::node_type>> tpetra_vec = Tpetra::createMultiVector<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type, Tpetra::Map<>::node_type>(map, 1);
    HDSA::Ptr<HDSA::Vector<RealT>> d = HDSA::makePtr<HDSA::Tpetra_Vector<RealT>>(tpetra_vec, random_number_generator_);
    HDSA::Ptr<HDSA::MultiVector<RealT>> D = HDSA::makePtr<HDSA::MultiVector<RealT>>(2, *d);

    HDSA::Ptr<HDSA::Vector<RealT>> d0 = (*D)[0];
    HDSA::Ptr<HDSA::Vector<RealT>> d1 = (*D)[1];
    HDSA::Tpetra_Vector<RealT> d0_tpetra = dynamic_cast<HDSA::Tpetra_Vector<RealT> &>(*d0);
    HDSA::Tpetra_Vector<RealT> d1_tpetra = dynamic_cast<HDSA::Tpetra_Vector<RealT> &>(*d1);

    for (int k = 0; k < m_; k++)
    {
      d0_tpetra.getVector()->replaceGlobalValue(k, 0, 0.2 * std::pow((*x_)(k, 0) + 1.0, 3.0));
      d1_tpetra.getVector()->replaceGlobalValue(k, 0, 0.2 * std::pow((*x_)(k, 0) + std::pow((*x_)(k, 0), 2.0), 3.0));
    }
    return D;
  }
};

#endif
