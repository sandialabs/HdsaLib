/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#include "Teuchos_GlobalMPISession.hpp"
#include <fstream>

#include "HDSA_Stream.hpp"
#include "HDSA_Comm.hpp"
#include "HDSA_Ptr.hpp"
#include "HDSA_Random_Number_Generator.hpp"
#include "HDSA_Vector.hpp"
#include "HDSA_MultiVector.hpp"
#include "HDSA_Sparse_Matrix.hpp"
#include "HDSA_MD_Lumped_Mass_u_Prior_Interface.hpp"
#include "HDSA_MD_Numeric_Laplacian_z_Prior_Interface.hpp"
#include "HDSA_MD_Prior_Sampling.hpp"
#include "HDSA_MD_Posterior_Data.hpp"
#include "HDSA_MD_Posterior_Sampling.hpp"
#include "HDSA_MD_Posterior_Vectors.hpp"
#include "HDSA_MD_Hessian_Analysis.hpp"
#include "HDSA_MD_Update.hpp"
#include "MD_Data_Interface_synthetic_test.hpp"
#include "MD_Opt_Prob_Interface_synthetic_test.hpp"
#include "MD_u_Hyperparameter_Interface_synthetic_test.hpp"
#include "MD_z_Hyperparameter_Interface_synthetic_test.hpp"

typedef double RealT;

int main(int argc, char *argv[])
{

  HDSA::nullstream bhs;
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, &bhs);
  HDSA::Ptr<const HDSA::Comm<int>> comm = HDSA::makePtr<HDSA::Comm<int>>();

  int num_random_numbers = 1.e6;
  std::string random_number_file = "random_numbers.txt";
  HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> random_number_generator = HDSA::makePtr<HDSA::Random_Number_Generator<RealT>>(num_random_numbers, random_number_file);

  HDSA::Ptr<HDSA::MD_Data_Interface<RealT>> data_interface = HDSA::makePtr<MD_Data_Interface_synthetic_test<RealT>>(random_number_generator, comm);
  HDSA::Ptr<HDSA::MD_Opt_Prob_Interface<RealT>> opt_prob_interface = HDSA::makePtr<MD_Opt_Prob_Interface_synthetic_test<RealT>>(comm);
  HDSA::Ptr<HDSA::MD_u_Hyperparameter_Interface<RealT>> u_hyperparam_interface = HDSA::makePtr<MD_u_Hyperparameter_Interface_synthetic_test<RealT>>();
  u_hyperparam_interface->Set_trace_estimator_sample_size(100);
  HDSA::Ptr<HDSA::MD_z_Hyperparameter_Interface<RealT>> z_hyperparam_interface = HDSA::makePtr<MD_z_Hyperparameter_Interface_synthetic_test<RealT>>(random_number_generator);

  u_hyperparam_interface->Set_beta_u(0.007702351792463);
  u_hyperparam_interface->Set_alpha_d(2.177109166165424e-07);
  z_hyperparam_interface->Set_beta_z(0.009305846653704);

  HDSA::Ptr<MD_Opt_Prob_Interface_synthetic_test<RealT>> opt_prob_interface_st = HDSA::dynamicPtrCast<MD_Opt_Prob_Interface_synthetic_test<RealT>>(opt_prob_interface);
  HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> M = opt_prob_interface_st->Get_Mass_Matrix();
  HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> S = opt_prob_interface_st->Get_Stiffness_Matrix();

  HDSA::Ptr<HDSA::MD_Lumped_Mass_u_Prior_Interface<RealT>> u_prior_interface = HDSA::makePtr<HDSA::MD_Lumped_Mass_u_Prior_Interface<RealT>>(S, M, data_interface, u_hyperparam_interface, comm, random_number_generator);
  HDSA::Ptr<HDSA::MD_Numeric_Laplacian_z_Prior_Interface<RealT>> z_prior_interface = HDSA::makePtr<HDSA::MD_Numeric_Laplacian_z_Prior_Interface<RealT>>(S, M, data_interface, z_hyperparam_interface, u_prior_interface);

  // This block of code is necessary to synchronize the hyperparameters when running in parallel
  // because the random number stream causes differences in the hyperparameters on difference processors.
  // The issue is that reading the random numbers from a text file is not compatiable with parallel executation.
  // u_hyperparam_interface->Set_alpha_u(0.05058967788039152);
  // z_hyperparam_interface->Set_alpha_z(4.231621091754468);
  // u_prior_interface->Set_alpha_u(0.05058967788039152);
  // z_prior_interface->Set_alpha_z(4.231621091754468);

  HDSA::Ptr<HDSA::MD_Prior_Sampling<RealT>> prior_sampling = HDSA::makePtr<HDSA::MD_Prior_Sampling<RealT>>(data_interface, u_prior_interface, z_prior_interface);

  int num_prior_samples = 100;
  HDSA::Ptr<HDSA::MultiVector<RealT>> prior_samples_at_z_opt = prior_sampling->Prior_Discrepancy_Samples_at_z_opt(num_prior_samples);
  std::string name = "prior_discrepancy_evaluated_at_z_opt";
  prior_samples_at_z_opt->Write_to_File(name);

  HDSA::Ptr<HDSA::MultiVector<RealT>> z = HDSA::makePtr<HDSA::MultiVector<RealT>>(3, *data_interface->Get_z_opt());
  HDSA::Tpetra_Vector<RealT> z0_tpetra = dynamic_cast<HDSA::Tpetra_Vector<RealT> &>(*(*z)[0]);
  HDSA::Tpetra_Vector<RealT> z1_tpetra = dynamic_cast<HDSA::Tpetra_Vector<RealT> &>(*(*z)[1]);
  HDSA::Tpetra_Vector<RealT> z2_tpetra = dynamic_cast<HDSA::Tpetra_Vector<RealT> &>(*(*z)[2]);
  int m = data_interface->Get_z_opt()->Dimension();
  RealT pi = 3.14159265358979323846;
  for (int k = 0; k < m; k++)
  {
    RealT x = static_cast<RealT>(k) / static_cast<RealT>(m - 1);
    z0_tpetra.getVector()->replaceGlobalValue(k, 0, x);
    z1_tpetra.getVector()->replaceGlobalValue(k, 0, x * x + 1.0);
    z2_tpetra.getVector()->replaceGlobalValue(k, 0, std::sin(2 * pi * x));
  }
  std::vector<HDSA::Ptr<HDSA::MultiVector<RealT>>> prior_samples = prior_sampling->Prior_Discrepancy_Samples(*z, num_prior_samples);
  for (int i = 0; i < num_prior_samples; i++)
  {
    std::string name = "prior_discrepancy_sample_" + std::to_string(i + 1);
    prior_samples[i]->Write_to_File(name);
  }

  HDSA::Ptr<HDSA::MD_Posterior_Data<RealT>> post_data = HDSA::makePtr<HDSA::MD_Posterior_Data<RealT>>();

  HDSA::Ptr<HDSA::MD_Posterior_Sampling<RealT>> post_sampling = HDSA::makePtr<HDSA::MD_Posterior_Sampling<RealT>>(data_interface, u_prior_interface, z_prior_interface);
  RealT alpha_d = u_hyperparam_interface->Get_alpha_d();
  int num_post_samples = 100;
  post_sampling->Compute_Posterior_Data(alpha_d, num_post_samples);

  std::vector<HDSA::Ptr<HDSA::Vector<RealT>>> z_test;
  z_test.resize(3);
  z_test[0] = data_interface->Get_z_opt()->Clone();
  z_test[0]->Set(*(*data_interface->Get_Z())[0]);
  z_test[1] = data_interface->Get_z_opt()->Clone();
  z_test[1]->Set(*(*data_interface->Get_Z())[1]);
  z_test[2] = data_interface->Get_z_opt()->Clone();
  HDSA::Tpetra_Vector<RealT> ztest2_tpetra = dynamic_cast<HDSA::Tpetra_Vector<RealT> &>(*z_test[2]);
  for (int k = 0; k < m; k++)
  {
    ztest2_tpetra.getVector()->replaceGlobalValue(k, 0, 1.5);
  }

  std::vector<HDSA::Ptr<HDSA::MD_Posterior_Vectors<RealT>>> post_discrepancy_samples = post_sampling->Posterior_Discrepancy_Samples(z_test);

  name = "posterior_discrepancy_mean_1.txt";
  post_discrepancy_samples[0]->mean->Write_to_File(name);
  name = "posterior_discrepancy_mean_2.txt";
  post_discrepancy_samples[1]->mean->Write_to_File(name);
  name = "posterior_discrepancy_mean_3.txt";
  post_discrepancy_samples[2]->mean->Write_to_File(name);
  name = "posterior_discrepancy_samples_1";
  post_discrepancy_samples[0]->samples->Write_to_File(name);
  name = "posterior_discrepancy_samples_2";
  post_discrepancy_samples[1]->samples->Write_to_File(name);
  name = "posterior_discrepancy_samples_3";
  post_discrepancy_samples[2]->samples->Write_to_File(name);

  HDSA::Ptr<HDSA::MD_Hessian_Analysis<RealT>> hessian_analysis = HDSA::makePtr<HDSA::MD_Hessian_Analysis<RealT>>(opt_prob_interface, z_prior_interface);

  int num_evals = 20;
  int oversampling = 10;
  hessian_analysis->Compute_Hessian_GEVP(data_interface->Get_z_opt(), num_evals, oversampling);

  HDSA::Ptr<HDSA::MD_Update<RealT>> update = HDSA::makePtr<HDSA::MD_Update<RealT>>(data_interface, u_prior_interface, z_prior_interface, opt_prob_interface, post_sampling, hessian_analysis, random_number_generator);

  HDSA::Ptr<HDSA::MD_Posterior_Vectors<RealT>> posterior_update_samples = update->Posterior_Update_Samples();
  name = "posterior_update_mean.txt";
  posterior_update_samples->mean->Write_to_File(name);
  name = "posterior_update_samples";
  posterior_update_samples->samples->Write_to_File(name);

  return 0;
}
