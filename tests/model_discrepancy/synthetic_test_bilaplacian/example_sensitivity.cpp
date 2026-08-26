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
#include "HDSA_MD_Bilaplacian_u_Prior_Interface.hpp"
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
  u_hyperparam_interface->Set_alpha_u(0.048969233204560);
  u_hyperparam_interface->Set_beta_u(0.007702351792463);
  u_hyperparam_interface->Set_alpha_d(2.177109166165424e-07);

  HDSA::Ptr<MD_Opt_Prob_Interface_synthetic_test<RealT>> opt_prob_interface_st = HDSA::dynamicPtrCast<MD_Opt_Prob_Interface_synthetic_test<RealT>>(opt_prob_interface);
  HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> M = opt_prob_interface_st->Get_Mass_Matrix();
  HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> S = opt_prob_interface_st->Get_Stiffness_Matrix();

  HDSA::Ptr<HDSA::MD_Bilaplacian_u_Prior_Interface<RealT>> u_prior_interface = HDSA::makePtr<HDSA::MD_Bilaplacian_u_Prior_Interface<RealT>>(S, M, data_interface, u_hyperparam_interface);

  RealT scalar = 0.5;
  RealT scalar_shift = scalar / 0.048969233204560;
  u_prior_interface->Precompute_W_u_Plus_scalar_M_u_Data(scalar_shift);

  HDSA::Ptr<HDSA::Vector<RealT>> u_in = data_interface->Get_u_opt()->Clone();
  int n_y = u_in->Dimension();
  for (int j = 0; j < n_y; j++)
  {
    RealT val = static_cast<RealT>(j) / static_cast<RealT>(n_y - 1);
    u_in->Set_Entry(j, val);
  }

  std::vector<RealT> diff = std::vector<RealT>(5);

  HDSA::Ptr<HDSA::Vector<RealT>> u_test1 = data_interface->Get_u_opt()->Clone();
  u_prior_interface->Apply_M_u(*u_test1, *u_in);
  diff[0] = std::abs(u_test1->Norm() - 0.081035026720274);

  HDSA::Ptr<HDSA::Vector<RealT>> u_test2 = data_interface->Get_u_opt()->Clone();
  u_prior_interface->Apply_W_u_Acute_Inverse(*u_test2, *u_in);
  diff[1] = std::abs(u_test2->Norm() - 2.051786899060638e+02);

  HDSA::Ptr<HDSA::Vector<RealT>> u_test3 = data_interface->Get_u_opt()->Clone();
  u_prior_interface->Apply_W_u_Acute_Plus_scalar_M_u_Inverse(*u_test3, *u_in, scalar);
  diff[2] = std::abs(u_test3->Norm() - 1.382677020366670e+02);

  std::vector<HDSA::Ptr<HDSA::Vector<RealT>>> u_sample_test1;
  u_sample_test1.resize(1);
  u_sample_test1[0] = data_interface->Get_u_opt()->Clone();
  HDSA::Ptr<HDSA::MultiVector<RealT>> samples_test1 = HDSA::makePtr<HDSA::MultiVector<RealT>>(u_sample_test1);
  u_prior_interface->Sample_with_Covariance_W_u_Acute_Inverse(*samples_test1);
  diff[3] = std::abs(u_sample_test1[0]->Norm() - 8.764533181985763);

  std::vector<HDSA::Ptr<HDSA::Vector<RealT>>> u_sample_test2;
  u_sample_test2.resize(1);
  u_sample_test2[0] = data_interface->Get_u_opt()->Clone();
  HDSA::Ptr<HDSA::MultiVector<RealT>> samples_test2 = HDSA::makePtr<HDSA::MultiVector<RealT>>(u_sample_test2);
  u_prior_interface->Sample_with_Covariance_W_u_Acute_Plus_scalar_M_u_Inverse(*samples_test2, scalar);
  diff[4] = std::abs(u_sample_test2[0]->Norm() - 13.680907759863270);

  RealT max_diff = *std::max_element(diff.begin(), diff.end());
  std::cout << "max_diff = " << max_diff << std::endl;

  std::ofstream out("diffs.txt");
  for (const RealT &x : diff)
  {
    out << x << '\n';
  }
  out.close();

  return 0;
}
