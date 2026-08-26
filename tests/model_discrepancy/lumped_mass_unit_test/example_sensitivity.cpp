/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#include "Teuchos_GlobalMPISession.hpp"
#include <fstream>
#include <chrono>

#include "HDSA_Stream.hpp"
#include "HDSA_Comm.hpp"
#include "HDSA_Ptr.hpp"
#include "HDSA_Random_Number_Generator.hpp"
#include "HDSA_Vector.hpp"
#include "HDSA_MultiVector.hpp"
#include "HDSA_Sparse_Matrix.hpp"
#include "HDSA_MD_Lumped_Mass_u_Prior_Interface.hpp"
#include "MD_Data_Interface_synthetic_test.hpp"
#include "Assemble_Operators.hpp"
#include "MD_u_Hyperparameter_Interface_synthetic_test.hpp"

typedef double RealT;

int main(int argc, char *argv[])
{

  HDSA::nullstream bhs;
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, &bhs);
  HDSA::Ptr<const HDSA::Comm<int>> comm = HDSA::makePtr<HDSA::Comm<int>>();

  RealT T = 1.0;
  int n_y = 31;
  int n_t = 12;
  RealT c_low = 0.93;
  RealT c_high = 0.95;

  int num_random_numbers = 1.e6;
  std::string random_number_file = "random_numbers.txt";
  HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> random_number_generator = HDSA::makePtr<HDSA::Random_Number_Generator<RealT>>(num_random_numbers, random_number_file);

  RealT alpha_u = 0.048969233204560;
  RealT beta_u = 0.007702351792463;
  RealT beta_t = 0.1;

  HDSA::Ptr<Assemble_Operators<RealT>> assemble_spatial_op = HDSA::makePtr<Assemble_Operators<RealT>>(comm, n_y);
  HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> M = assemble_spatial_op->Get_Sparse_Mass_Matrix();
  HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> S = assemble_spatial_op->Get_Sparse_Stiffness_Matrix();
  HDSA::Ptr<HDSA::MD_Data_Interface<RealT>> data_interface = HDSA::makePtr<MD_Data_Interface_synthetic_test<RealT>>(random_number_generator, comm, n_y, n_t, c_low, c_high);
  HDSA::Ptr<HDSA::MD_u_Hyperparameter_Interface<RealT>> u_hyperparam_interface = HDSA::makePtr<MD_u_Hyperparameter_Interface_synthetic_test<RealT>>();
  u_hyperparam_interface->Set_alpha_u(alpha_u);
  u_hyperparam_interface->Set_beta_u(beta_u);
  u_hyperparam_interface->Set_beta_t(beta_t);
  HDSA::Ptr<HDSA::MD_u_Prior_Interface<RealT>> spatial_u_prior_interface = HDSA::makePtr<HDSA::MD_Lumped_Mass_u_Prior_Interface<RealT>>(S, M, data_interface, u_hyperparam_interface);
  HDSA::Ptr<HDSA::MD_Transient_Prior_Covariance<RealT>> transient_prior_cov = HDSA::makePtr<HDSA::MD_Transient_Prior_Covariance<RealT>>(data_interface, u_hyperparam_interface, T, n_t, n_y);
  HDSA::Ptr<HDSA::MD_u_Prior_Interface<RealT>> u_prior_interface = HDSA::makePtr<HDSA::MD_Transient_Elliptic_u_Prior_Interface<RealT>>(spatial_u_prior_interface, transient_prior_cov);

  ///////////////////// Assemble operators for verification tests ////////////////////////////////////////////////////////
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> M_s = assemble_spatial_op->Get_Dense_Mass_Matrix();
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> E_s = assemble_spatial_op->Get_Dense_E(beta_u);
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> E_s_inv = assemble_spatial_op->Inverse(E_s);
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> M_lumped = assemble_spatial_op->Get_Dense_M_Lumped();
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> M_lumped_inv = assemble_spatial_op->Inverse(M_lumped);

  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> tmp_y = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_y, n_y);
  E_s->Multiply(*tmp_y, *M_lumped_inv);
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> W_s = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_y, n_y);
  tmp_y->Multiply(*W_s, *E_s);
  W_s->Scale(1.0 / alpha_u);
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> W_s_inv = assemble_spatial_op->Inverse(W_s);

  HDSA::Ptr<Assemble_Operators<RealT>> assemble_time_op = HDSA::makePtr<Assemble_Operators<RealT>>(comm, n_t);
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> M_t = assemble_time_op->Get_Dense_Mass_Matrix();
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> W_t = assemble_time_op->Get_Dense_E(beta_t);
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> W_t_inv = assemble_time_op->Inverse(W_t);

  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> M_u = assemble_time_op->Kronecker(M_t, M_s);
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> W_u = assemble_time_op->Kronecker(W_t, W_s);
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> W_u_inv = assemble_time_op->Kronecker(W_t_inv, W_s_inv);

  std::vector<RealT> diffs;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////// Test W_t ////////////////////////////////////////////////////////////
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> V = transient_prior_cov->Get_Evecs();
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> evals = transient_prior_cov->Get_Evals();
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Lambda = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_t, n_t);
  for (int i = 0; i < n_t; i++)
  {
    Lambda->Set_Entry(i, i, 1.0 / (*evals)(i, 0));
  }
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> MV = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_t, n_t);
  M_t->Multiply(*MV, *V);
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> tmp_t = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_t, n_t);
  MV->Multiply(*tmp_t, *Lambda);
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> test_t = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_t, n_t);
  tmp_t->Multiply(*test_t, *MV, false, true);
  RealT local_diff = assemble_time_op->Matrix_Difference(test_t, W_t);
  local_diff = std::sqrt(local_diff);
  diffs.push_back(local_diff);

  ////////////////////////////////////////////////////// Test M_u ////////////////////////////////////////////////////////////
  HDSA::Ptr<HDSA::Vector<RealT>> u_hdsa = data_interface->Get_u_opt()->Clone();
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> u_mat = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_t * n_y, 1);
  for (int i = 0; i < n_t; i++)
  {
    RealT ti = static_cast<RealT>(i) / static_cast<RealT>(n_t - 1);
    HDSA::Ptr<HDSA::Transient_Vector<RealT>> u_hdsa_trans = HDSA::dynamicPtrCast<HDSA::Transient_Vector<RealT>>(u_hdsa);
    for (int j = 0; j < n_y; j++)
    {
      RealT xj = static_cast<RealT>(j) / static_cast<RealT>(n_y - 1);
      RealT val = ti * xj;
      (*u_hdsa_trans)[i]->Set_Entry(j, val);
      u_mat->Set_Entry(i * n_y + j, 0, val);
    }
  }

  HDSA::Ptr<HDSA::Vector<RealT>> u_tmp = data_interface->Get_u_opt()->Clone();
  u_prior_interface->Apply_M_u(*u_tmp, *u_hdsa);
  RealT val_test = u_hdsa->Dot(*u_tmp);
  local_diff = std::abs(val_test - 1.0 / 9.0);
  diffs.push_back(local_diff);

  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> u_mat_tmp = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_t * n_y, 1);
  M_u->Multiply(*u_mat_tmp, *u_mat);
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> val_tmp = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(1, 1);
  u_mat_tmp->Multiply(*val_tmp, *u_mat, true, false);
  val_test = (*val_tmp)(0, 0);
  local_diff = std::abs(val_test - 1.0 / 9.0);
  diffs.push_back(local_diff);

  ////////////////////////////////////////////////////// Test W_u_inv ////////////////////////////////////////////////////////////
  u_hdsa->Randomize_Standard_Normal();
  for (int i = 0; i < n_t; i++)
  {
    HDSA::Ptr<HDSA::Transient_Vector<RealT>> u_hdsa_trans = HDSA::dynamicPtrCast<HDSA::Transient_Vector<RealT>>(u_hdsa);
    for (int j = 0; j < n_y; j++)
    {
      RealT val = (*u_hdsa_trans)[i]->Get_Entry(j);
      u_mat->Set_Entry(i * n_y + j, 0, val);
    }
  }

  u_tmp->Zeros();
  u_prior_interface->Apply_W_u_Inverse(*u_tmp, *u_hdsa);
  RealT val_test1 = u_hdsa->Dot(*u_tmp);

  u_mat_tmp->Zeros();
  W_u_inv->Multiply(*u_mat_tmp, *u_mat);
  val_tmp->Zeros();
  u_mat_tmp->Multiply(*val_tmp, *u_mat, true, false);
  RealT val_test2 = (*val_tmp)(0, 0);
  local_diff = std::abs(val_test1 - val_test2);
  diffs.push_back(local_diff);

  ////////////////////////////////////////////////////// Test W_u_Plus_scalar_M_u_inv ////////////////////////////////////////////////////////////
  u_hdsa->Randomize_Standard_Normal();
  RealT scalar = 1.e5;
  u_prior_interface->Precompute_W_u_Plus_scalar_M_u_Data(scalar);
  for (int i = 0; i < n_t; i++)
  {
    HDSA::Ptr<HDSA::Transient_Vector<RealT>> u_hdsa_trans = HDSA::dynamicPtrCast<HDSA::Transient_Vector<RealT>>(u_hdsa);
    for (int j = 0; j < n_y; j++)
    {
      RealT val = (*u_hdsa_trans)[i]->Get_Entry(j);
      u_mat->Set_Entry(i * n_y + j, 0, val);
    }
  }

  u_tmp->Zeros();
  u_prior_interface->Apply_W_u_Plus_scalar_M_u_Inverse(*u_tmp, *u_hdsa, scalar);
  val_test1 = u_hdsa->Dot(*u_tmp);

  u_mat_tmp->Zeros();
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> A = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_y * n_t, n_y * n_t);
  for (int i = 0; i < n_y * n_t; i++)
  {
    for (int j = 0; j < n_y * n_t; j++)
    {
      RealT val = (*W_u)(i, j) + scalar * (*M_u)(i, j);
      A->Set_Entry(i, j, val);
    }
  }
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> A_inv = assemble_spatial_op->Inverse(A);
  A_inv->Multiply(*u_mat_tmp, *u_mat);
  val_tmp->Zeros();
  u_mat_tmp->Multiply(*val_tmp, *u_mat, true, false);
  val_test2 = (*val_tmp)(0, 0);
  local_diff = std::abs(val_test1 - val_test2);
  diffs.push_back(local_diff);

  ////////////////////////////////////////////////////// Test Sample_with_Covariance_W_u_Inverse ////////////////////////////////////////////////////////////

  HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> random_number_generator_rep1 = HDSA::makePtr<HDSA::Random_Number_Generator<RealT>>(num_random_numbers, random_number_file);
  HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> random_number_generator_rep2 = HDSA::makePtr<HDSA::Random_Number_Generator<RealT>>(num_random_numbers, random_number_file);

  auto map = Tpetra::createUniformContigMap<Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type>(n_y, comm->Get_Teuchos_Communicator());
  HDSA::Ptr<Tpetra::MultiVector<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type, Tpetra::Map<>::node_type>> tpetra_vec = Tpetra::createMultiVector<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type, Tpetra::Map<>::node_type>(map, 1);
  HDSA::Ptr<HDSA::Vector<RealT>> spatial_vec = HDSA::makePtr<HDSA::Tpetra_Vector<RealT>>(tpetra_vec, random_number_generator_rep1);
  HDSA::Ptr<HDSA::Vector<RealT>> u_rep1 = HDSA::makePtr<HDSA::Transient_Vector<RealT>>(n_t, spatial_vec);

  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Lambda_inv_sqrt = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_t, n_t);
  for (int i = 0; i < n_t; i++)
  {
    Lambda_inv_sqrt->Set_Entry(i, i, std::sqrt((*evals)(i, 0)));
  }
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> V_Lambda_inv_sqrt = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_t, n_t);
  V->Multiply(*V_Lambda_inv_sqrt, *Lambda_inv_sqrt);

  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> I_y = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_y, n_y);
  for (int i = 0; i < n_y; i++)
  {
    I_y->Set_Entry(i, i, 1.0);
  }
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> T_mat = assemble_spatial_op->Kronecker(V_Lambda_inv_sqrt, I_y);

  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> M_lumped_sqrt = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_y, n_y);
  for (int i = 0; i < n_y; i++)
  {
    M_lumped_sqrt->Set_Entry(i, i, std::sqrt((*M_lumped)(i, i)));
  }
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> E_s_inv_M_lumped_sqrt = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_y, n_y);
  E_s_inv->Multiply(*E_s_inv_M_lumped_sqrt, *M_lumped_sqrt);

  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> I_t = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_t, n_t);
  for (int i = 0; i < n_t; i++)
  {
    I_t->Set_Entry(i, i, 1.0);
  }
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> L_mat = assemble_spatial_op->Kronecker(I_t, E_s_inv_M_lumped_sqrt);
  L_mat->Scale(std::sqrt(alpha_u));

  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> S_mat = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_t * n_y, n_t * n_y);
  T_mat->Multiply(*S_mat, *L_mat);

  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> tmp = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_t * n_y, n_t * n_y);
  S_mat->Multiply(*tmp, *S_mat, false, true);

  local_diff = assemble_spatial_op->Matrix_Difference(W_u_inv, tmp);
  local_diff = std::sqrt(local_diff);
  diffs.push_back(local_diff);

  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> omega = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_t * n_y, 1);
  for (int i = 0; i < n_t * n_y; i++)
  {
    RealT val = random_number_generator_rep2->Generate_Standard_Normal_Sample();
    omega->Set_Entry(i, 0, val);
  }
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> sample_test = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_t * n_y, 1);
  S_mat->Multiply(*sample_test, *omega);

  HDSA::Ptr<HDSA::MultiVector<RealT>> samples = HDSA::makePtr<HDSA::MultiVector<RealT>>(1, *u_rep1);
  u_prior_interface->Sample_with_Covariance_W_u_Inverse(*samples);

  RealT running_diff = 0.0;
  RealT normalize = 0.0;
  int count = 0;
  HDSA::Ptr<HDSA::Transient_Vector<RealT>> sample_trans = HDSA::dynamicPtrCast<HDSA::Transient_Vector<RealT>>((*samples)[0]);
  for (int i = 0; i < n_t; i++)
  {
    for (int j = 0; j < n_y; j++)
    {
      RealT val = (*sample_test)(count, 0) - (*sample_trans)[i]->Get_Entry(j);
      running_diff += std::pow(val, 2.0);
      normalize += std::pow((*sample_test)(count, 0), 2.0);
      count += 1;
    }
  }
  local_diff = std::sqrt(running_diff / normalize);
  diffs.push_back(local_diff);

  ////////////////////////////////////////////////////// Test Sample_with_Covariance_W_u_Plus_scalar_M_u_Inverse ////////////////////////////////////////////////////////////

  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> L_tmp1 = assemble_spatial_op->Kronecker(I_t, W_s);
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Lambda_inv = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_t, n_t);
  for (int i = 0; i < n_t; i++)
  {
    Lambda_inv->Set_Entry(i, i, (*evals)(i, 0));
  }
  Lambda_inv->Scale(scalar);
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> L_tmp2 = assemble_spatial_op->Kronecker(Lambda_inv, M_s);
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> L_tmp3 = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_t * n_y, n_t * n_y);
  for (int i = 0; i < n_t * n_y; i++)
  {
    for (int j = 0; j < n_t * n_y; j++)
    {
      RealT val = (*L_tmp1)(i, j) + (*L_tmp2)(i, j);
      L_tmp3->Set_Entry(i, j, val);
    }
  }
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> L_tmp4 = assemble_spatial_op->Inverse(L_tmp3);
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> L_mat_2 = assemble_spatial_op->Matrix_Sqrt(L_tmp4);

  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> S_mat_2 = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_t * n_y, n_t * n_y);
  T_mat->Multiply(*S_mat_2, *L_mat_2);

  omega->Zeros();
  for (int i = 0; i < n_t * n_y; i++)
  {
    RealT val = random_number_generator_rep2->Generate_Standard_Normal_Sample();
    omega->Set_Entry(i, 0, val);
  }
  sample_test->Zeros();
  S_mat_2->Multiply(*sample_test, *omega);

  samples->Zeros();
  HDSA::Ptr<HDSA::MD_Lumped_Mass_u_Prior_Interface<RealT>> spatial_u_prior_interface_cast = HDSA::dynamicPtrCast<HDSA::MD_Lumped_Mass_u_Prior_Interface<RealT>>(spatial_u_prior_interface);
  spatial_u_prior_interface_cast->Disable_Sampling_Preconditioner();
  u_prior_interface->Sample_with_Covariance_W_u_Plus_scalar_M_u_Inverse(*samples, scalar);

  running_diff = 0.0;
  normalize = 0.0;
  count = 0;
  for (int i = 0; i < n_t; i++)
  {
    for (int j = 0; j < n_y; j++)
    {
      RealT val = (*sample_test)(count, 0) - (*sample_trans)[i]->Get_Entry(j);
      running_diff += std::pow(val, 2.0);
      normalize += std::pow((*sample_test)(count, 0), 2.0);
      count += 1;
    }
  }
  local_diff = std::sqrt(running_diff / normalize);
  diffs.push_back(local_diff);

  ////////////////////////////////////////////////////// Write diffs to file ////////////////////////////////////////////////////////////
  int dim = diffs.size();
  std::string name = "diffs.txt";
  std::ofstream fout;
  fout.open(name);
  for (int i = 0; i < dim; i++)
  {
    fout << std::setprecision(16) << diffs[i] << std::endl;
  }
  fout.close();

  return 0;
}
