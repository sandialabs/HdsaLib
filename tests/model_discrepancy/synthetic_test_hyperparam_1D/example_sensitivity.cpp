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
#include "HDSA_MD_Numeric_Laplacian_u_Prior_Interface.hpp"
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
  HDSA::Ptr<HDSA::MD_z_Hyperparameter_Interface<RealT>> z_hyperparam_interface = HDSA::makePtr<MD_z_Hyperparameter_Interface_synthetic_test<RealT>>(random_number_generator);

  u_hyperparam_interface->Set_beta_u(0.007702351792463);
  u_hyperparam_interface->Set_alpha_d(2.177109166165424e-07);
  z_hyperparam_interface->Set_beta_z(0.009305846653704);

  HDSA::Ptr<MD_Opt_Prob_Interface_synthetic_test<RealT>> opt_prob_interface_st = HDSA::dynamicPtrCast<MD_Opt_Prob_Interface_synthetic_test<RealT>>(opt_prob_interface);
  HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> M = opt_prob_interface_st->Get_Mass_Matrix();
  HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> S = opt_prob_interface_st->Get_Stiffness_Matrix();

  HDSA::Ptr<HDSA::MD_Numeric_Laplacian_u_Prior_Interface<RealT>> u_prior_interface = HDSA::makePtr<HDSA::MD_Numeric_Laplacian_u_Prior_Interface<RealT>>(S, M, data_interface, u_hyperparam_interface, random_number_generator);
  HDSA::Ptr<HDSA::MD_Numeric_Laplacian_z_Prior_Interface<RealT>> z_prior_interface = HDSA::makePtr<HDSA::MD_Numeric_Laplacian_z_Prior_Interface<RealT>>(S, M, data_interface, z_hyperparam_interface, u_prior_interface);

  HDSA::Ptr<HDSA::MD_Prior_Sampling<RealT>> prior_sampling = HDSA::makePtr<HDSA::MD_Prior_Sampling<RealT>>(data_interface, u_prior_interface, z_prior_interface);

  int num_prior_samples = 100;

  HDSA::Ptr<MD_Data_Interface_synthetic_test<RealT>> data_interface_st = HDSA::dynamicPtrCast<MD_Data_Interface_synthetic_test<RealT>>(data_interface);
  HDSA::Ptr<HDSA::Vector<RealT>> x_coords = data_interface_st->Generate_Spatial_Nodes();
  HDSA::Ptr<HDSA::MultiVector<RealT>> spatial_coords = HDSA::makePtr<HDSA::MultiVector<RealT>>();
  spatial_coords->push_back(x_coords);
  prior_sampling->Generate_Prior_Discrepancy_Sample_Data(num_prior_samples, u_hyperparam_interface, z_hyperparam_interface, spatial_coords);
  HDSA::Ptr<HDSA::MultiVector<RealT>> prior_delta_z_opt = prior_sampling->Get_prior_delta_z_opt();
  std::vector<HDSA::Ptr<HDSA::Vector<RealT>>> prior_z_pert = prior_sampling->Get_prior_z_pert();
  std::vector<HDSA::Ptr<HDSA::MultiVector<RealT>>> prior_delta_z_pert = prior_sampling->Get_prior_delta_z_pert();
  std::string name = "prior_delta_z_opt";
  prior_delta_z_opt->Write_to_File(name);
  name = "prior_z_pert_1.txt";
  prior_z_pert[0]->Write_to_File(name);
  name = "prior_z_pert_2.txt";
  prior_z_pert[1]->Write_to_File(name);
  name = "prior_delta_z_pert_1";
  prior_delta_z_pert[0]->Write_to_File(name);
  name = "prior_delta_z_pert_2";
  prior_delta_z_pert[1]->Write_to_File(name);

  HDSA::Ptr<HDSA::MD_Posterior_Data<RealT>> post_data = HDSA::makePtr<HDSA::MD_Posterior_Data<RealT>>();

  HDSA::Ptr<HDSA::MD_Posterior_Sampling<RealT>> post_sampling = HDSA::makePtr<HDSA::MD_Posterior_Sampling<RealT>>(data_interface, u_prior_interface, z_prior_interface);
  RealT alpha_d = u_hyperparam_interface->Get_alpha_d();
  int num_post_samples = 100;
  post_sampling->Compute_Posterior_Data(alpha_d, num_post_samples);

  std::vector<HDSA::Ptr<HDSA::Vector<RealT>>> z_test;
  z_test.resize(3);
  z_test[0] = prior_z_pert[0]->Clone();
  z_test[0]->Set(*(*data_interface->Get_Z())[0]);
  z_test[1] = prior_z_pert[0]->Clone();
  z_test[1]->Set(*(*data_interface->Get_Z())[1]);
  z_test[2] = prior_z_pert[0]->Clone();
  HDSA::Tpetra_Vector<RealT> ztest2_tpetra = dynamic_cast<HDSA::Tpetra_Vector<RealT> &>(*z_test[2]);
  int m = prior_z_pert[0]->Dimension();
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
