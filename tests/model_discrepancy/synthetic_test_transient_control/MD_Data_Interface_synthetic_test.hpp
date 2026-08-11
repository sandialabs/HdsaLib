/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_DATA_INTERFACE_SYNTHETIC_TEST_HPP
#define HDSA_MD_DATA_INTERFACE_SYNTHETIC_TEST_HPP

#include "HDSA_MD_Data_Interface.hpp"
#include "HDSA_Std_Vector.hpp"

template <class RealT>
class MD_Data_Interface_synthetic_test : public HDSA::MD_Data_Interface<RealT>
{

private:
  int n_y_;                                // Mesh resolution
  int n_t_;                                // Number of time steps
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x_; // Mesh nodes on [0,1]
  std::vector<RealT> t_;
  const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> Random_number_generator_;
  const HDSA::Ptr<const HDSA::Comm<int>> comm_;

public:
  MD_Data_Interface_synthetic_test(const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> &Random_number_generator, const HDSA::Ptr<const HDSA::Comm<int>> &comm, int n_y, int n_t) : Random_number_generator_(Random_number_generator), comm_(comm)
  {
    n_y_ = n_y;
    n_t_ = n_t;
    x_ = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_y_, 1);
    for (int k = 0; k < n_y_; k++)
    {
      x_->Set_Entry(k, 0, static_cast<RealT>(k) / static_cast<RealT>(n_y_ - 1));
    }
    t_ = std::vector<RealT>(n_t_);
    for (int k = 0; k < n_t_; k++)
    {
      t_[k] = static_cast<RealT>(k) / static_cast<RealT>(n_t_ - 1);
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
    for (int j = 0; j < n_t_; j++)
    {
      HDSA::Ptr<HDSA::Vector<RealT>> uj = u_opt_trans[j];
      HDSA::Tpetra_Vector<RealT> uj_tpetra = dynamic_cast<HDSA::Tpetra_Vector<RealT> &>(*uj);
      for (int k = 0; k < n_y_; k++)
      {
        uj_tpetra.getVector()->replaceGlobalValue(k, 0, t_[j] * (1.0 - (*x_)(k, 0)) + 2.0 * t_[j] * (*x_)(k, 0));
      }
    }
    return u_opt;
  }

  HDSA::Ptr<HDSA::Vector<RealT>> Load_Optimal_z() const
  {
    HDSA::Ptr<HDSA::Std_Vector<RealT>> tmp = HDSA::makePtr<HDSA::Std_Vector<RealT>>(2, Random_number_generator_);
    HDSA::Ptr<HDSA::Vector<RealT>> z_opt = HDSA::makePtr<HDSA::Transient_Vector<RealT>>(n_t_, tmp);
    HDSA::Transient_Vector<RealT> z_opt_trans = dynamic_cast<HDSA::Transient_Vector<RealT> &>(*z_opt);
    for (int j = 0; j < n_t_; j++)
    {
      HDSA::Ptr<HDSA::Vector<RealT>> zj = z_opt_trans[j];
      zj->Set_Entry(0, t_[j]);
      zj->Set_Entry(1, 2 * t_[j]);
    }
    return z_opt;
  }

  HDSA::Ptr<HDSA::MultiVector<RealT>> Load_Z_Data() const
  {
    std::vector<HDSA::Ptr<HDSA::Vector<RealT>>> z_opt;
    z_opt.resize(1);
    z_opt[0] = Load_Optimal_z();
    HDSA::Ptr<HDSA::MultiVector<RealT>> Z = HDSA::makePtr<HDSA::MultiVector<RealT>>(z_opt);
    return Z;
  }

  HDSA::Ptr<HDSA::MultiVector<RealT>> Load_D_Data() const
  {
    std::vector<HDSA::Ptr<HDSA::Vector<RealT>>> u;
    u.resize(1);
    u[0] = Load_Optimal_u()->Clone(); // This leverages Load_Optimal_u to instantiate the vector
    u[0]->Set_Scalar(1.0);             // We overload the values to Set them to one
    HDSA::Ptr<HDSA::MultiVector<RealT>> D = HDSA::makePtr<HDSA::MultiVector<RealT>>(u);
    return D;
  }
};

#endif
