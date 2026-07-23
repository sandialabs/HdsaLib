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
#include "HDSA_MD_Prior_Sampling.hpp"
#include "HDSA_MD_Posterior_Data.hpp"
#include "HDSA_MD_Posterior_Sampling.hpp"
#include "HDSA_MD_Posterior_Vectors.hpp"
#include "HDSA_MD_Hessian_Analysis.hpp"
#include "HDSA_MD_Update.hpp"
#include "MD_Data_Interface_PDE_Test_Problem.hpp"
#include "MD_Opt_Prob_Interface_PDE_Test_Problem.hpp"
#include "MD_Elliptic_u_Prior_Interface_PDE_Test_Problem.hpp"
#include "MD_Elliptic_z_Prior_Interface_PDE_Test_Problem.hpp"

typedef double RealT;

int main(int argc, char *argv[])
{

  HDSA::nullstream bhs;
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, &bhs);
  HDSA::Ptr<const HDSA::Comm<int>> comm = HDSA::makePtr<HDSA::Comm<int>>();

  int num_random_numbers = 4.e5;
  std::string random_number_file = "random_numbers.txt";
  HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> random_number_generator = HDSA::makePtr<HDSA::Random_Number_Generator<RealT>>(num_random_numbers, random_number_file);

  HDSA::Ptr<HDSA::MD_Data_Interface<RealT>> data_interface = HDSA::makePtr<MD_Data_Interface_PDE_Test_Problem<RealT>>(random_number_generator, comm);
  HDSA::Ptr<HDSA::MD_Opt_Prob_Interface<RealT>> opt_prob_interface = HDSA::makePtr<MD_Opt_Prob_Interface_PDE_Test_Problem<RealT>>();
  RealT alpha_u = 1.0 / std::pow(2.0, 2.0);
  RealT alpha_z = 1.0 / std::pow(3.0, 2.0);
  HDSA::Ptr<HDSA::MD_u_Prior_Interface<RealT>> u_prior_interface = HDSA::makePtr<MD_Elliptic_u_Prior_Interface_PDE_Test_Problem<RealT>>(alpha_u, random_number_generator);
  HDSA::Ptr<HDSA::MD_z_Prior_Interface<RealT>> z_prior_interface = HDSA::makePtr<MD_Elliptic_z_Prior_Interface_PDE_Test_Problem<RealT>>(alpha_z, random_number_generator);

  int num_sing_vals = 200;
  int oversampling = 0;
  int num_subspace_iters = 1;
  HDSA::Ptr<HDSA::Vector<RealT>> u_vec = data_interface->Get_u_opt()->Clone();
  HDSA::Ptr<HDSA::MD_Elliptic_u_Prior_Interface<RealT>> elliptic_u_prior_interface = HDSA::dynamicPtrCast<HDSA::MD_Elliptic_u_Prior_Interface<RealT>>(u_prior_interface);
  elliptic_u_prior_interface->Compute_E_u_Inverse_GSVD(num_sing_vals, oversampling, num_subspace_iters, *u_vec);

  HDSA::Ptr<HDSA::MD_Prior_Sampling<RealT>> prior_sampling = HDSA::makePtr<HDSA::MD_Prior_Sampling<RealT>>(data_interface, u_prior_interface, z_prior_interface);

  int num_prior_samples = 100;
  HDSA::Ptr<HDSA::MultiVector<RealT>> prior_samples_at_z_opt = prior_sampling->Prior_Discrepancy_Samples_at_z_opt(num_prior_samples);
  std::string name = "prior_discrepancy_evaluated_at_z_opt";
  prior_samples_at_z_opt->Write_to_File(name);

  HDSA::Ptr<HDSA::MultiVector<RealT>> z = HDSA::makePtr<HDSA::MultiVector<RealT>>(3, *data_interface->Get_z_opt());
  HDSA::Ptr<HDSA::Vector<RealT>> z0 = (*z)[0];
  HDSA::Ptr<HDSA::Vector<RealT>> z1 = (*z)[1];
  HDSA::Ptr<HDSA::Vector<RealT>> z2 = (*z)[2];
  HDSA::Std_Vector<RealT> z0_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(*z0);
  HDSA::Std_Vector<RealT> z1_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(*z1);
  HDSA::Std_Vector<RealT> z2_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(*z2);
  int m = z0->Dimension();
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m, 1);
  for (int k = 0; k < m; k++)
  {
    x->Set_Entry(k, 0, static_cast<RealT>(k) / static_cast<RealT>(m - 1));
  }
  RealT pi = 3.14159265358979323846;
  for (int k = 0; k < m; k++)
  {
    z0_std.Set_Entry(k, 1.0 + std::sin(2 * pi * (*x)(k, 0)));
    z1_std.Set_Entry(k, 1.0 + std::cos(2 * pi * (*x)(k, 0)));
    z2_std.Set_Entry(k, 1.0 + std::sin(20 * pi * (*x)(k, 0)));
  }

  std::vector<HDSA::Ptr<HDSA::MultiVector<RealT>>> prior_samples = prior_sampling->Prior_Discrepancy_Samples(*z, num_prior_samples);

  for (int i = 0; i < num_prior_samples; i++)
  {
    std::string name = "prior_discrepancy_sample_" + std::to_string(i + 1);
    prior_samples[i]->Write_to_File(name);
  }

  HDSA::Ptr<HDSA::MD_Posterior_Data<RealT>> post_data = HDSA::makePtr<HDSA::MD_Posterior_Data<RealT>>();

  HDSA::Ptr<HDSA::MD_Posterior_Sampling<RealT>> post_sampling = HDSA::makePtr<HDSA::MD_Posterior_Sampling<RealT>>(data_interface, u_prior_interface, z_prior_interface);
  RealT alpha_d = 1.e-5;
  int num_post_samples = 100;
  post_sampling->Compute_Posterior_Data(alpha_d, num_post_samples);

  std::vector<HDSA::Ptr<HDSA::Vector<RealT>>> z_test;
  z_test.resize(3);
  z_test[0] = z0->Clone();
  z_test[0]->Set(*(*data_interface->Get_Z())[0]);
  z_test[1] = z0->Clone();
  z_test[1]->Set(*(*data_interface->Get_Z())[1]);
  z_test[2] = z0->Clone();
  HDSA::Std_Vector<RealT> ztest2_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(*z_test[2]);
  for (int k = 0; k < m; k++)
  {
    ztest2_std.Set_Entry(k, 1.5);
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

  HDSA::Ptr<HDSA::MD_Update<RealT>> update = HDSA::makePtr<HDSA::MD_Update<RealT>>(data_interface, u_prior_interface, z_prior_interface, opt_prob_interface, post_sampling, hessian_analysis);

  HDSA::Ptr<HDSA::MD_Posterior_Vectors<RealT>> posterior_update_samples = update->Posterior_Update_Samples();
  name = "posterior_update_mean.txt";
  posterior_update_samples->mean->Write_to_File(name);
  name = "posterior_update_samples";
  posterior_update_samples->samples->Write_to_File(name);

  return 0;
}
