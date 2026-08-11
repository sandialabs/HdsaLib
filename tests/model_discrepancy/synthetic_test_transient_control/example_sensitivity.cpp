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
#include "HDSA_MD_Transient_Vector_z_Prior_Interface.hpp"
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

  RealT T = 1.0;
  int n_y = 50;
  int n_t = 20;
  int num_controls = 2;

  int num_random_numbers = 1.e6;
  std::string random_number_file = "random_numbers.txt";
  HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> random_number_generator = HDSA::makePtr<HDSA::Random_Number_Generator<RealT>>(num_random_numbers, random_number_file);

  HDSA::Ptr<HDSA::MD_Data_Interface<RealT>> data_interface = HDSA::makePtr<MD_Data_Interface_synthetic_test<RealT>>(random_number_generator, comm, n_y, n_t);
  HDSA::Ptr<HDSA::MD_Opt_Prob_Interface<RealT>> opt_prob_interface = HDSA::makePtr<MD_Opt_Prob_Interface_synthetic_test<RealT>>(comm, data_interface, n_y, n_t);
  HDSA::Ptr<HDSA::MD_u_Hyperparameter_Interface<RealT>> u_hyperparam_interface = HDSA::makePtr<MD_u_Hyperparameter_Interface_synthetic_test<RealT>>();
  HDSA::Ptr<HDSA::MD_z_Hyperparameter_Interface<RealT>> z_hyperparam_interface = HDSA::makePtr<MD_z_Hyperparameter_Interface_synthetic_test<RealT>>(random_number_generator);

  u_hyperparam_interface->Set_beta_u(2.104323964883042e-06);
  u_hyperparam_interface->Set_beta_t(0.004105009974496);
  u_hyperparam_interface->Set_GSVD_Hyperparameters(50, 0, 1);
  z_hyperparam_interface->Set_beta_t(0.028338930835080);

  HDSA::Ptr<MD_Opt_Prob_Interface_synthetic_test<RealT>> opt_prob_interface_st = HDSA::dynamicPtrCast<MD_Opt_Prob_Interface_synthetic_test<RealT>>(opt_prob_interface);
  HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> M = opt_prob_interface_st->Get_Mass_Matrix();
  HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> S = opt_prob_interface_st->Get_Stiffness_Matrix();

  HDSA::Ptr<HDSA::MD_u_Prior_Interface<RealT>> spatial_u_prior_interface = HDSA::makePtr<HDSA::MD_Numeric_Laplacian_u_Prior_Interface<RealT>>(S, M, data_interface, u_hyperparam_interface, random_number_generator);
  HDSA::Ptr<HDSA::MD_Transient_Prior_Covariance<RealT>> transient_prior_cov = HDSA::makePtr<HDSA::MD_Transient_Prior_Covariance<RealT>>(data_interface, u_hyperparam_interface, T, n_t, n_y);
  HDSA::Ptr<HDSA::MD_u_Prior_Interface<RealT>> u_prior_interface = HDSA::makePtr<HDSA::MD_Transient_Elliptic_u_Prior_Interface<RealT>>(spatial_u_prior_interface, transient_prior_cov);

  HDSA::Ptr<HDSA::MD_z_Prior_Interface<RealT>> z_prior_interface = HDSA::makePtr<HDSA::MD_Transient_Vector_z_Prior_Interface<RealT>>(data_interface, z_hyperparam_interface, u_prior_interface, n_t, T, num_controls);

  HDSA::Ptr<HDSA::MD_Prior_Sampling<RealT>> prior_sampling = HDSA::makePtr<HDSA::MD_Prior_Sampling<RealT>>(data_interface, u_prior_interface, z_prior_interface);

  HDSA::Ptr<HDSA::MD_Posterior_Data<RealT>> post_data = HDSA::makePtr<HDSA::MD_Posterior_Data<RealT>>();

  HDSA::Ptr<HDSA::MD_Posterior_Sampling<RealT>> post_sampling = HDSA::makePtr<HDSA::MD_Posterior_Sampling<RealT>>(data_interface, u_prior_interface, z_prior_interface);
  RealT alpha_d = 1.000000000000000e-06;
  int num_post_samples = 100;
  post_sampling->Compute_Posterior_Data(alpha_d, num_post_samples);

  HDSA::Ptr<HDSA::MD_Hessian_Analysis<RealT>> hessian_analysis = HDSA::makePtr<HDSA::MD_Hessian_Analysis<RealT>>(opt_prob_interface, z_prior_interface);

  int num_evals = 30;
  int oversampling = 9;
  hessian_analysis->Compute_Hessian_GEVP(data_interface->Get_z_opt(), num_evals, oversampling);

  HDSA::Ptr<HDSA::MD_Update<RealT>> update = HDSA::makePtr<HDSA::MD_Update<RealT>>(data_interface, u_prior_interface, z_prior_interface, opt_prob_interface, post_sampling, hessian_analysis, random_number_generator);

  HDSA::Ptr<HDSA::MD_Posterior_Vectors<RealT>> posterior_update_samples = update->Posterior_Update_Samples();
  std::string name = "posterior_update_mean.txt";
  posterior_update_samples->mean->Write_to_File(name);
  name = "posterior_update_samples";
  posterior_update_samples->samples->Write_to_File(name);

  return 0;
}
