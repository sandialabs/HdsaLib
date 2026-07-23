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
#include "HDSA_MD_Continuation_Update.hpp"
#include "MD_Data_Interface_synthetic_test_continuation.hpp"
#include "MD_Opt_Prob_Interface_synthetic_test_continuation.hpp"
#include "MD_u_Prior_Interface_synthetic_test_continuation.hpp"
#include "MD_z_Prior_Interface_synthetic_test_continuation.hpp"

typedef double RealT;

int main(int argc, char *argv[])
{

  HDSA::nullstream bhs;
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, &bhs);
  HDSA::Ptr<const HDSA::Comm<int>> comm = HDSA::makePtr<HDSA::Comm<int>>();

  int num_random_numbers = 1.e5;
  std::string random_number_file = "random_numbers.txt";
  HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> random_number_generator = HDSA::makePtr<HDSA::Random_Number_Generator<RealT>>(num_random_numbers, random_number_file);

  HDSA::Ptr<HDSA::MD_Data_Interface<RealT>> data_interface = HDSA::makePtr<MD_Data_Interface_synthetic_test_continuation<RealT>>(random_number_generator, comm);
  HDSA::Ptr<HDSA::MD_Opt_Prob_Interface<RealT>> opt_prob_interface = HDSA::makePtr<MD_Opt_Prob_Interface_synthetic_test_continuation<RealT>>();
  HDSA::Ptr<HDSA::MD_u_Prior_Interface<RealT>> u_prior_interface = HDSA::makePtr<MD_u_Prior_Interface_synthetic_test_continuation<RealT>>(random_number_generator);
  HDSA::Ptr<HDSA::MD_z_Prior_Interface<RealT>> z_prior_interface = HDSA::makePtr<MD_z_Prior_Interface_synthetic_test_continuation<RealT>>(random_number_generator);

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
    z0_std.Set_Entry(k, (*x)(k, 0));
    z1_std.Set_Entry(k, 1.0 + std::pow((*x)(k, 0), 2.0));
    z2_std.Set_Entry(k, std::sin(2 * pi * (*x)(k, 0)));
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

  int num_evals = 20;
  int oversampling = 10;
  hessian_analysis->Compute_Hessian_GEVP(data_interface->Get_z_opt(), num_evals, oversampling);

  // Standard Update
  // HDSA::Ptr<HDSA::MD_Update<RealT>> update = HDSA::makePtr<HDSA::MD_Update<RealT>>(data_interface, u_prior_interface, z_prior_interface, opt_prob_interface, post_sampling, hessian_analysis);
  // HDSA::Ptr<HDSA::Vector<RealT>> z_k = update->Posterior_Update_Mean(); 

  // Continuation Update
  int num_continuation_steps = 3;
  HDSA::Ptr<HDSA::MD_Continuation_Update<RealT>> update = HDSA::makePtr<HDSA::MD_Continuation_Update<RealT>>(data_interface, z_prior_interface, opt_prob_interface, post_sampling, hessian_analysis, num_continuation_steps);
  HDSA::Ptr<HDSA::Vector<RealT>> u_k = HDSA::makePtr<HDSA::Std_Vector<RealT>>(m, random_number_generator, comm);
  HDSA::Ptr<HDSA::Vector<RealT>> z_k = HDSA::makePtr<HDSA::Std_Vector<RealT>>(m, random_number_generator, comm);
  HDSA::Ptr<HDSA::Vector<RealT>> beta_k = HDSA::makePtr<HDSA::Std_Vector<RealT>>(num_evals, random_number_generator, comm);
  update->Posterior_Update_Mean(*u_k, *z_k, *beta_k);
  name = "posterior_update_mean.txt";
  z_k->Write_to_File(name);

  // std::cout << std::scientific << std::setprecision(3);

  auto objective_given_state = [&](const HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &z) -> RealT {
    HDSA::Ptr<HDSA::Vector<RealT>> grad_u = u.Clone();
    opt_prob_interface->Misfit_Gradient(*grad_u, u, z);

    const HDSA::Std_Vector<RealT> u_std = dynamic_cast<const HDSA::Std_Vector<RealT> &>(u);
    HDSA::Std_Vector<RealT> grad_u_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(*grad_u);

    RealT value = static_cast<RealT>(0);
    int m = u.Dimension();
    for (int k = 0; k < m; ++k) {
      const RealT diff = u_std(k) - std::pow(static_cast<RealT>(k) / static_cast<RealT>(m - 1) + static_cast<RealT>(1), static_cast<RealT>(3));
      value += static_cast<RealT>(0.5) * diff * grad_u_std(k);
    }
    return value;
  };

  auto low_fidelity_state = [&](const HDSA::Vector<RealT> &z_in) -> HDSA::Ptr<HDSA::Vector<RealT>> {
    HDSA::Ptr<HDSA::Vector<RealT>> u_lf = data_interface->Get_u_opt()->Clone();
    opt_prob_interface->State_Solve(*u_lf, z_in);
    return u_lf;
  };

  auto high_fidelity_state = [&](const HDSA::Vector<RealT> &z_in) -> HDSA::Ptr<HDSA::Vector<RealT>> {
    HDSA::Ptr<HDSA::Vector<RealT>> u_hf = data_interface->Get_u_opt()->Clone();
    const HDSA::Std_Vector<RealT> &z_std = dynamic_cast<const HDSA::Std_Vector<RealT> &>(z_in);
    HDSA::Std_Vector<RealT> &u_hf_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(*u_hf);

    for (int k = 0; k < z_in.Dimension(); ++k) {
      u_hf_std.Set_Entry(k, static_cast<RealT>(1.2) * std::pow(z_std(k), static_cast<RealT>(3)));
    }
    return u_hf;
  };

  auto hf_optimal_z_for_current_synthetic_model = [&](const HDSA::Vector<RealT> &z_template) -> HDSA::Ptr<HDSA::Vector<RealT>> {
    HDSA::Ptr<HDSA::Vector<RealT>> z_hf_opt = z_template.Clone();
    const HDSA::Std_Vector<RealT> &z_lf_opt_std = dynamic_cast<const HDSA::Std_Vector<RealT> &>(*data_interface->Get_z_opt());
    HDSA::Std_Vector<RealT> &z_hf_opt_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(*z_hf_opt);
    const int m = z_template.Dimension();
    const RealT scale = (1.0) / std::cbrt((1.2));
    for (int k = 0; k < m; ++k) {
      // Note that z_HF_opt = (x + 1) / cbrt(1.2)
      z_hf_opt_std.Set_Entry(k, scale * z_lf_opt_std(k));
    }

    return z_hf_opt;
  };

  HDSA::Ptr<const HDSA::Vector<RealT>> z_lf_opt = data_interface->Get_z_opt();

  HDSA::Ptr<HDSA::Vector<RealT>> u_lf_at_lf_opt = low_fidelity_state(*z_lf_opt);

  HDSA::Ptr<HDSA::Vector<RealT>> u_lf_at_updated = low_fidelity_state(*z_k);
  HDSA::Ptr<HDSA::Vector<RealT>> u_hf_at_lf_opt = high_fidelity_state(*z_lf_opt);
  HDSA::Ptr<HDSA::Vector<RealT>> u_hf_at_updated = high_fidelity_state(*z_k);

  RealT J_lf_at_lf_opt = objective_given_state(*u_lf_at_lf_opt, *z_lf_opt);
  RealT J_lf_at_updated = objective_given_state(*u_lf_at_updated, *z_k);
  RealT J_hf_at_lf_opt = objective_given_state(*u_hf_at_lf_opt, *z_lf_opt);
  RealT J_hf_at_updated = objective_given_state(*u_hf_at_updated, *z_k);

  HDSA::Ptr<HDSA::Vector<RealT>> z_hf_opt = hf_optimal_z_for_current_synthetic_model(*z_lf_opt);
  HDSA::Ptr<HDSA::Vector<RealT>> u_hf_at_hf_opt = high_fidelity_state(*z_hf_opt);
  RealT J_hf_at_hf_opt = objective_given_state(*u_hf_at_hf_opt, *z_hf_opt);
  HDSA::Ptr<HDSA::Vector<RealT>> diff_lf_to_hf_opt = z_lf_opt->Clone();
  diff_lf_to_hf_opt->Set(*z_lf_opt);
  diff_lf_to_hf_opt->Scaled_Plus(-1.0, *z_hf_opt);

  HDSA::Ptr<HDSA::Vector<RealT>> diff_updated_to_hf_opt = z_k->Clone();
  diff_updated_to_hf_opt->Set(*z_k);
  diff_updated_to_hf_opt->Scaled_Plus(-1.0, *z_hf_opt);

  std::cout << "-----------------------------------------------------" << std::endl;
  std::cout << "Actual objective comparison" << std::endl;
  std::cout << "\nJ_LF at low-fidelity optimum:       "
            << J_lf_at_lf_opt << std::endl;
  std::cout << "J_LF at updated solution:           "
            << J_lf_at_updated << std::endl;
  std::cout << "\nJ_HF at low-fidelity optimum:       "
            << J_hf_at_lf_opt << std::endl;
  std::cout << "J_HF at updated solution:           "
            << J_hf_at_updated << std::endl;
  std::cout << "J_HF at exact HF optimum:           "
            << J_hf_at_hf_opt << std::endl;

  std::cout << "\nHF improvement factor:              "
    << 100.0 * (1.0 - J_hf_at_updated / J_hf_at_lf_opt) << "%" << std::endl;

  std::cout << "\n||z_LF_opt - z_HF_opt||:            "
            << diff_lf_to_hf_opt->Norm() << std::endl;
  std::cout << "||z_updated - z_HF_opt||:           "
            << diff_updated_to_hf_opt->Norm() << std::endl;
  
  return 0;
}
