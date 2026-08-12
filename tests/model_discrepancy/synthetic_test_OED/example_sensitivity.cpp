/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#include "Teuchos_GlobalMPISession.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "HDSA_Comm.hpp"
#include "HDSA_Dense_Matrix.hpp"
#include "HDSA_Linear_Algebra.hpp"
#include "HDSA_MultiVector.hpp"
#include "HDSA_Ptr.hpp"
#include "HDSA_Random_Number_Generator.hpp"
#include "HDSA_Std_Vector.hpp"
#include "HDSA_Stream.hpp"
#include "HDSA_Vector.hpp"

#include "HDSA_MD_Continuation_Update.hpp"
#include "HDSA_MD_Hessian_Analysis.hpp"
#include "HDSA_MD_OED.hpp"
#include "HDSA_MD_Posterior_Data.hpp"
#include "HDSA_MD_Posterior_Sampling.hpp"

#include "MD_Data_Interface_synthetic_test_OED.hpp"
#include "MD_Opt_Prob_Interface_synthetic_test_OED.hpp"
#include "MD_u_Prior_Interface_synthetic_test_OED.hpp"
#include "MD_z_Prior_Interface_synthetic_test_OED.hpp"

typedef double RealT;

namespace {
int Dense_Vector_Length(const HDSA::Dense_Matrix<RealT>& x) {
  return x.Number_of_Rows() * x.Number_of_Columns();
}

RealT Dense_Vector_Entry_Column_Major(const HDSA::Dense_Matrix<RealT>& x, const int idx) {
  const int rows = x.Number_of_Rows();
  return x(idx % rows, idx / rows);
}

HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Vector_To_Dense(const HDSA::Vector<RealT>& x) {
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> y = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(x.Dimension(), 1);

  y->Zeros();

  for (int i = 0; i < x.Dimension(); ++i) {
    y->Set_Entry(i, 0, x.Get_Entry(i));
  }

  return y;
}

HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Append_Betas(const HDSA::Dense_Matrix<RealT>& old_betas,
                                                  const HDSA::Dense_Matrix<RealT>& new_betas) {
  const int old_len = Dense_Vector_Length(old_betas);
  const int new_len = Dense_Vector_Length(new_betas);

  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> all_betas = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(old_len + new_len, 1);

  all_betas->Zeros();

  for (int i = 0; i < old_len; ++i) {
    all_betas->Set_Entry(i, 0, Dense_Vector_Entry_Column_Major(old_betas, i));
  }

  for (int i = 0; i < new_len; ++i) {
    all_betas->Set_Entry(old_len + i, 0, Dense_Vector_Entry_Column_Major(new_betas, i));
  }

  return all_betas;
}

void Evaluate_Synthetic_Discrepancy(HDSA::Vector<RealT>& D_out, const HDSA::Vector<RealT>& Z_in) {
  HDSA_TEST_FOR_EXCEPTION(D_out.Dimension() != Z_in.Dimension(), std::logic_error,
                          "Error in example_sensitivity::Evaluate_Synthetic_Discrepancy: "
                          "D_out and Z_in dimensions are inconsistent."
                              << std::endl);

  for (int i = 0; i < Z_in.Dimension(); ++i) {
    const RealT z_i = Z_in.Get_Entry(i);
    D_out.Set_Entry(i, static_cast<RealT>(0.2) * z_i * z_i * z_i);
  }
}

RealT M_z_Norm_Difference(const HDSA::Vector<RealT>& a, const HDSA::Vector<RealT>& b,
                          const HDSA::MD_z_Prior_Interface<RealT>& z_prior_interface) {
  HDSA::Ptr<HDSA::Vector<RealT>> diff = a.Clone();
  diff->Set(a);
  diff->Scaled_Plus(static_cast<RealT>(-1), b);

  HDSA::Ptr<HDSA::Vector<RealT>> Mz_diff = a.Clone();
  z_prior_interface.Apply_M_z(*Mz_diff, *diff);

  const RealT norm_sq = diff->Dot(*Mz_diff);

  return std::sqrt(std::max(static_cast<RealT>(0), norm_sq));
}

RealT Trace_Wz_Inverse_Mz(const HDSA::Vector<RealT>& prototype,
                          const HDSA::MD_z_Prior_Interface<RealT>& z_prior_interface) {
  const int n = prototype.Dimension();

  RealT trace_val = static_cast<RealT>(0);

  for (int i = 0; i < n; ++i) {
    HDSA::Ptr<HDSA::Vector<RealT>> e_i = prototype.Clone();
    e_i->Zeros();
    e_i->Set_Entry(i, static_cast<RealT>(1));

    HDSA::Ptr<HDSA::Vector<RealT>> Mz_e_i = prototype.Clone();
    z_prior_interface.Apply_M_z(*Mz_e_i, *e_i);

    HDSA::Ptr<HDSA::Vector<RealT>> Wz_inv_Mz_e_i = prototype.Clone();
    z_prior_interface.Apply_W_z_Inverse(*Wz_inv_Mz_e_i, *Mz_e_i);

    trace_val += Wz_inv_Mz_e_i->Get_Entry(i);
  }

  return trace_val;
}

RealT Vector_Difference_Norm(const HDSA::Vector<RealT>& a, const HDSA::Vector<RealT>& b) {
  HDSA::Ptr<HDSA::Vector<RealT>> diff = a.Clone();
  diff->Set(a);
  diff->Scaled_Plus(static_cast<RealT>(-1), b);
  return diff->Norm();
}

HDSA::Ptr<HDSA::Vector<RealT>> Low_Fidelity_State(const HDSA::MD_Opt_Prob_Interface<RealT>& opt_prob_interface,
                                                  const HDSA::Vector<RealT>& z,
                                                  const HDSA::Vector<RealT>& u_prototype) {
  HDSA::Ptr<HDSA::Vector<RealT>> u = u_prototype.Clone();
  opt_prob_interface.State_Solve(*u, z);
  return u;
}

HDSA::Ptr<HDSA::Vector<RealT>> High_Fidelity_State(const HDSA::Vector<RealT>& z,
                                                   const HDSA::Vector<RealT>& u_prototype) {
  HDSA::Ptr<HDSA::Vector<RealT>> u = u_prototype.Clone();

  for (int i = 0; i < z.Dimension(); ++i) {
    const RealT z_i = z.Get_Entry(i);
    u->Set_Entry(i, static_cast<RealT>(1.2) * z_i * z_i * z_i);
  }

  return u;
}

HDSA::Ptr<HDSA::Vector<RealT>> Synthetic_HF_Optimal_z(const HDSA::Vector<RealT>& z_lf_opt) {
  HDSA::Ptr<HDSA::Vector<RealT>> z_hf = z_lf_opt.Clone();

  const RealT scale = static_cast<RealT>(1) / std::cbrt(static_cast<RealT>(1.2));

  for (int i = 0; i < z_lf_opt.Dimension(); ++i) {
    z_hf->Set_Entry(i, scale * z_lf_opt.Get_Entry(i));
  }

  return z_hf;
}

RealT Objective_Given_State(const HDSA::MD_Opt_Prob_Interface<RealT>& opt_prob_interface, const HDSA::Vector<RealT>& u,
                            const HDSA::Vector<RealT>& z) {
  HDSA::Ptr<HDSA::Vector<RealT>> grad_u = u.Clone();
  opt_prob_interface.Misfit_Gradient(*grad_u, u, z);

  RealT value = static_cast<RealT>(0);

  for (int i = 0; i < u.Dimension(); ++i) {
    const RealT x_i = static_cast<RealT>(i) / static_cast<RealT>(u.Dimension() - 1);

    const RealT target_i = std::pow(x_i + static_cast<RealT>(1), static_cast<RealT>(3));

    const RealT diff = u.Get_Entry(i) - target_i;

    value += static_cast<RealT>(0.5) * diff * grad_u->Get_Entry(i);
  }

  return value;
}
} // namespace

int main(int argc, char* argv[]) {
  HDSA::nullstream bhs;
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, &bhs);

  HDSA::Ptr<const HDSA::Comm<int>> comm = HDSA::makePtr<HDSA::Comm<int>>();

  const int num_random_numbers = 100000;
  const std::string random_number_file = "random_numbers.txt";

  HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> random_number_generator =
      HDSA::makePtr<HDSA::Random_Number_Generator<RealT>>(num_random_numbers, random_number_file);

  HDSA::Ptr<HDSA::MD_Data_Interface<RealT>> data_interface =
      HDSA::makePtr<MD_Data_Interface_synthetic_test_OED<RealT>>();

  HDSA::Ptr<HDSA::MD_Opt_Prob_Interface<RealT>> opt_prob_interface =
      HDSA::makePtr<MD_Opt_Prob_Interface_synthetic_test_OED<RealT>>();

  HDSA::Ptr<HDSA::MD_u_Prior_Interface<RealT>> u_prior_interface =
      HDSA::makePtr<MD_u_Prior_Interface_synthetic_test_OED<RealT>>(random_number_generator);

  HDSA::Ptr<HDSA::MD_z_Prior_Interface<RealT>> z_prior_interface =
      HDSA::makePtr<MD_z_Prior_Interface_synthetic_test_OED<RealT>>(random_number_generator);

  /*
    Hessian analysis provides the reduced design basis V used by OED.
  */
  HDSA::Ptr<HDSA::MD_Hessian_Analysis<RealT>> hessian_analysis =
      HDSA::makePtr<HDSA::MD_Hessian_Analysis<RealT>>(opt_prob_interface, z_prior_interface);

  const int num_evals = 10;
  const int oversampling = 10;

  hessian_analysis->Compute_Hessian_GEVP(data_interface->Get_z_opt(), num_evals, oversampling);

  /*
    OED setup and offline reduced matrix construction.
  */
  HDSA::Ptr<HDSA::MD_OED<RealT>> md_oed =
      HDSA::makePtr<HDSA::MD_OED<RealT>>(data_interface, u_prior_interface, z_prior_interface, hessian_analysis);

  md_oed->Offline_Computation();

  const int r = md_oed->Get_Reduced_Dimension();

  const typename HDSA::MD_OED<RealT>::Offline_Data& offline = md_oed->Get_Offline_Data();

  HDSA_TEST_FOR_EXCEPTION(!offline.Is_Initialized(), std::logic_error,
                          "Error in example_sensitivity: OED offline data were not initialized." << std::endl);

  /*
    Sequential OED parameters.
  */
  const int num_oed_steps = 5;
  const RealT alpha_d = static_cast<RealT>(1e-5);

  const RealT alpha_k_denom = Trace_Wz_Inverse_Mz(*data_interface->Get_z_opt(), *z_prior_interface);

  typename HDSA::MD_OED<RealT>::SPG_Options spg_options;
  spg_options.max_iter = 10;
  spg_options.pg_tol = static_cast<RealT>(1e-8);
  spg_options.armijo_c = static_cast<RealT>(1e-4);
  spg_options.backtrack_factor = static_cast<RealT>(0.5);
  spg_options.max_backtracks = 30;
  spg_options.nonmonotone_window = 5;
  spg_options.verbosity = false;

  HDSA::Dense_Matrix<RealT> beta_0(r, 1);
  beta_0.Zeros();

  for (int i = 0; i < r; ++i) {
    const RealT sign = (i % 2 == 0) ? static_cast<RealT>(1) : static_cast<RealT>(-1);

    beta_0.Set_Entry(i, 0, sign * static_cast<RealT>(3e-2) * static_cast<RealT>(i + 1) / static_cast<RealT>(r));
  }

  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> betas = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(0, 1);

  HDSA::Ptr<HDSA::MultiVector<RealT>> Z_accum =
      HDSA::makePtr<HDSA::MultiVector<RealT>>(0, *data_interface->Get_z_opt());

  HDSA::Ptr<HDSA::MultiVector<RealT>> D_accum =
      HDSA::makePtr<HDSA::MultiVector<RealT>>(0, *data_interface->Get_u_opt());

  std::vector<HDSA::Ptr<HDSA::Vector<RealT>>> z_bars;
  std::vector<HDSA::Ptr<HDSA::Vector<RealT>>> selected_designs;

  HDSA::Ptr<HDSA::Vector<RealT>> z_lofi = data_interface->Get_z_opt()->Clone();

  z_lofi->Set(*data_interface->Get_z_opt());

  HDSA::Ptr<HDSA::Vector<RealT>> current_z_bar = HDSA::nullPtr;
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> current_beta_bar = HDSA::nullPtr;

  std::cout << std::scientific << std::setprecision(6);
  std::cout << "\n=====================================================" << std::endl;
  std::cout << "Beginning sequential OED workflow" << std::endl;
  std::cout << "Reduced dimension r = " << r << std::endl;
  std::cout << "Number of OED steps = " << num_oed_steps << std::endl;
  std::cout << "=====================================================" << std::endl;

  for (int step = 0; step < num_oed_steps; ++step) {
    std::cout << "\nOED step " << step + 1 << " / " << num_oed_steps << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;

    HDSA::Ptr<HDSA::Vector<RealT>> z_p;

    if (step == 0) {
      /*
        Start at the low-fidelity optimum.
      */
      z_p = z_lofi->Clone();
      z_p->Set(*z_lofi);

      std::cout << "Using initial design z_lofi = z_opt." << std::endl;
    } else {
      HDSA::Ptr<HDSA::Vector<RealT>> radius_reference;

      if (step == 1) {
        radius_reference = z_lofi;
      } else {
        radius_reference = z_bars[step - 2];
      }

      const RealT prev_z_distance = M_z_Norm_Difference(*current_z_bar, *radius_reference, *z_prior_interface);
      const RealT alpha_k = prev_z_distance * prev_z_distance / alpha_k_denom;
      const RealT constr_radius = prev_z_distance;

      std::cout << "Previous posterior movement radius = " << constr_radius << std::endl;
      std::cout << "OED covariance coefficient alpha_k = " << alpha_k << std::endl;
      md_oed->Set_Covariance_Coefficient(alpha_k);

      typename HDSA::MD_OED<RealT>::Seq_Design_Result seq_result =
          md_oed->Generate_Seq_Optimal_Design(beta_0, alpha_d, *betas, *current_beta_bar, constr_radius, spg_options);

      betas = Append_Betas(*betas, *seq_result.beta_new);

      std::cout << "Sequential OED final objective = " << seq_result.optimizer_info.final_objective << std::endl;
      std::cout << "Sequential OED projected-gradient norm = " << seq_result.optimizer_info.projected_gradient_norm
                << std::endl;

      z_p = (*seq_result.Z_new)[0]->Clone();
      z_p->Set(*(*seq_result.Z_new)[0]);
    }

    selected_designs.push_back(z_p);

    /*
      Evaluate discrepancy and append data.
    */
    HDSA::Ptr<HDSA::Vector<RealT>> D_p = data_interface->Get_u_opt()->Clone();
    Evaluate_Synthetic_Discrepancy(*D_p, *z_p);
    Z_accum->push_back(z_p);
    D_accum->push_back(D_p);
    data_interface->Set_Z_and_D(Z_accum, D_accum);

    /*
      Recompute posterior data for the accumulated design/discrepancy set.
    */
    HDSA::Ptr<HDSA::MD_Posterior_Sampling<RealT>> post_sampling =
        HDSA::makePtr<HDSA::MD_Posterior_Sampling<RealT>>(data_interface, u_prior_interface, z_prior_interface);
    int num_post_samples = 0;
    post_sampling->Compute_Posterior_Data(alpha_d, num_post_samples);

    /*
      Continuation update of the optimization solution.
    */
    const int num_continuation_steps = 3;

    HDSA::Ptr<HDSA::MD_Continuation_Update<RealT>> cont_update = HDSA::makePtr<HDSA::MD_Continuation_Update<RealT>>(
        data_interface, z_prior_interface, opt_prob_interface, post_sampling, hessian_analysis, random_number_generator, num_continuation_steps);

    HDSA::Ptr<HDSA::Vector<RealT>> u_k = data_interface->Get_u_opt()->Clone();
    HDSA::Ptr<HDSA::Vector<RealT>> z_k = data_interface->Get_z_opt()->Clone();
    HDSA::Ptr<HDSA::Vector<RealT>> beta_k = HDSA::makePtr<HDSA::Std_Vector<RealT>>(r);

    cont_update->Posterior_Update_Mean(*u_k, *z_k, *beta_k);

    current_z_bar = z_k;
    current_beta_bar = Vector_To_Dense(*beta_k);
    z_bars.push_back(current_z_bar);

    const std::string selected_name = "sequential_oed_design_" + std::to_string(step + 1) + ".txt";
    const std::string posterior_name = "sequential_oed_posterior_z_bar_" + std::to_string(step + 1) + ".txt";

    z_p->Write_to_File(selected_name);
    current_z_bar->Write_to_File(posterior_name);
  }

  current_z_bar->Write_to_File("sequential_oed_final_posterior_z_bar.txt");

  /*
    Compare low-fidelity, sequentially updated, and synthetic high-fidelity
    objective values.
  */
  HDSA::Ptr<const HDSA::Vector<RealT>> z_lf_opt = data_interface->Get_z_opt();
  HDSA::Ptr<HDSA::Vector<RealT>> z_hf_opt = Synthetic_HF_Optimal_z(*z_lf_opt);

  HDSA::Ptr<HDSA::Vector<RealT>> u_lf_at_lf_opt =
      Low_Fidelity_State(*opt_prob_interface, *z_lf_opt, *data_interface->Get_u_opt());
  HDSA::Ptr<HDSA::Vector<RealT>> u_lf_at_updated =
      Low_Fidelity_State(*opt_prob_interface, *current_z_bar, *data_interface->Get_u_opt());
  HDSA::Ptr<HDSA::Vector<RealT>> u_hf_at_lf_opt = High_Fidelity_State(*z_lf_opt, *data_interface->Get_u_opt());
  HDSA::Ptr<HDSA::Vector<RealT>> u_hf_at_updated = High_Fidelity_State(*current_z_bar, *data_interface->Get_u_opt());
  HDSA::Ptr<HDSA::Vector<RealT>> u_hf_at_hf_opt = High_Fidelity_State(*z_hf_opt, *data_interface->Get_u_opt());

  const RealT J_lf_at_lf_opt = Objective_Given_State(*opt_prob_interface, *u_lf_at_lf_opt, *z_lf_opt);
  const RealT J_lf_at_updated = Objective_Given_State(*opt_prob_interface, *u_lf_at_updated, *current_z_bar);
  const RealT J_hf_at_lf_opt = Objective_Given_State(*opt_prob_interface, *u_hf_at_lf_opt, *z_lf_opt);
  const RealT J_hf_at_updated = Objective_Given_State(*opt_prob_interface, *u_hf_at_updated, *current_z_bar);
  const RealT J_hf_at_hf_opt = Objective_Given_State(*opt_prob_interface, *u_hf_at_hf_opt, *z_hf_opt);
  const RealT dist_lf_to_hf = Vector_Difference_Norm(*z_lf_opt, *z_hf_opt);
  const RealT dist_updated_to_hf = Vector_Difference_Norm(*current_z_bar, *z_hf_opt);

  std::cout << "\n=====================================================" << std::endl;
  std::cout << "Sequential OED objective comparison" << std::endl;
  std::cout << "=====================================================" << std::endl;

  std::cout << "\nJ_LF at low-fidelity optimum:       " << J_lf_at_lf_opt << std::endl;
  std::cout << "J_LF at sequential OED update:      " << J_lf_at_updated << std::endl;
  std::cout << "\nJ_HF at low-fidelity optimum:       " << J_hf_at_lf_opt << std::endl;
  std::cout << "J_HF at sequential OED update:      " << J_hf_at_updated << std::endl;
  std::cout << "J_HF at exact synthetic HF optimum: " << J_hf_at_hf_opt << std::endl;

  std::cout << "\nHF improvement factor:              "
            << static_cast<RealT>(100) * (static_cast<RealT>(1) - J_hf_at_updated / J_hf_at_lf_opt) << "%" << std::endl;
  std::cout << "\n||z_LF_opt - z_HF_opt||:            " << dist_lf_to_hf << std::endl;
  std::cout << "||z_updated - z_HF_opt||:           " << dist_updated_to_hf << std::endl;

  std::cout << "\nSelected OED design points written to:" << std::endl;
  for (int step = 0; step < num_oed_steps; ++step) {
    std::cout << "  sequential_oed_design_" << step + 1 << ".txt" << std::endl;
  }

  std::cout << "\nPosterior z_bar history written to:" << std::endl;
  for (int step = 0; step < num_oed_steps; ++step) {
    std::cout << "  sequential_oed_posterior_z_bar_" << step + 1 << ".txt" << std::endl;
  }

  std::cout << "  sequential_oed_final_posterior_z_bar.txt" << std::endl;
  std::cout << "=====================================================" << std::endl;

  return 0;
}