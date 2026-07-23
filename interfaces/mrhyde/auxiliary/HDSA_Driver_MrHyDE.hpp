/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_DRIVER_MRHYDE_HPP
#define HDSA_DRIVER_MRHYDE_HPP

#include "HDSA_Stream.hpp"
#include "HDSA_Ptr.hpp"
#include "HDSA_Comm.hpp"
#include "HDSA_Random_Number_Generator.hpp"
#include "HDSA_MD_Data_Interface_MrHyDE.hpp"
#include "HDSA_MD_Opt_Prob_Interface_MrHyDE.hpp"
#include "HDSA_Output_Writer_MrHyDE.hpp"
#include "HDSA_Sparse_Matrix.hpp"
#include "HDSA_MD_Multi_State_u_Hyperparameter_Interface.hpp"
#include "HDSA_MD_u_Hyperparameter_Interface_MrHyDE.hpp"
#include "HDSA_MD_z_Hyperparameter_Interface_MrHyDE.hpp"
#include "HDSA_Prior_Operators_Interface_MrHyDE.hpp"
#include "HDSA_MD_u_Prior_Interface.hpp"
#include "HDSA_MD_Numeric_Laplacian_u_Prior_Interface.hpp"
#include "HDSA_MD_Bilaplacian_u_Prior_Interface.hpp"
#include "HDSA_MD_Lumped_Mass_u_Prior_Interface.hpp"
#include "HDSA_MD_Multi_State_u_Prior_Interface.hpp"
#include "HDSA_MD_z_Prior_Interface.hpp"
#include "HDSA_MD_Numeric_Laplacian_z_Prior_Interface.hpp"
#include "HDSA_MD_Vector_z_Prior_Interface.hpp"
#include "HDSA_MD_Prior_Sampling.hpp"
#include "HDSA_MD_Posterior_Sampling.hpp"
#include "HDSA_MD_Hessian_Analysis.hpp"
#include "HDSA_MD_Update.hpp"
#include "HDSA_MD_Continuation_Update.hpp"
#include "HDSA_MD_OUU_Data_Interface_MrHyDE.hpp"
#include "HDSA_MD_OUU_Opt_Prob_Interface_MrHyDE.hpp"
#include "HDSA_MD_OUU_Ensemble_Weighting_Matrix.hpp"
#include "HDSA_MD_OUU_Hyperparameter_Data_Interface.hpp"
#include "HDSA_MD_OUU_u_Prior_Interface.hpp"
#include "HDSA_BF_Update.hpp"
#include "HDSA_BF_Sol_Op_Interface_MrHyDE.hpp"

template <class RealT,
          class LO = Tpetra::Map<>::local_ordinal_type,
          class GO = Tpetra::Map<>::global_ordinal_type,
          class Node = Tpetra::Map<>::node_type>
class Driver_MrHyDE
{

private:
  Teuchos::RCP<MpiComm> comm_;
  Teuchos::RCP<Teuchos::ParameterList> settings_;
  Teuchos::RCP<MrHyDE::SolverManager<SolverNode>> solver_;
  Teuchos::RCP<MrHyDE::PostprocessManager<SolverNode>> postproc_;
  Teuchos::RCP<MrHyDE::ParameterManager<SolverNode>> params_;

public:
  Driver_MrHyDE(Teuchos::RCP<MpiComm> &comm, Teuchos::RCP<Teuchos::ParameterList> &settings, Teuchos::RCP<MrHyDE::SolverManager<SolverNode>> &solver,
                Teuchos::RCP<MrHyDE::PostprocessManager<SolverNode>> &postproc, Teuchos::RCP<MrHyDE::ParameterManager<SolverNode>> &params)
      : comm_(comm), settings_(settings), solver_(solver), postproc_(postproc), params_(params)
  {
  }

  virtual ~Driver_MrHyDE()
  {
  }

  void HDSA_Solve(void)
  {
    Teuchos::ParameterList HDSAsettings;

    if (settings_->sublist("Analysis").isSublist("HDSA"))
      HDSAsettings = settings_->sublist("Analysis").sublist("HDSA");
    else
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: MrHyDE could not find the HDSA sublist in the input file!  Abort!");

    bool do_bifidelity_correction = HDSAsettings.sublist("Configuration").get<bool>("do_bifidelity_correction", false);
    if (do_bifidelity_correction)
    {
      BF_Solve(HDSAsettings);
    }
    else
    {
      MD_Solve(HDSAsettings);
    }
  }

  void BF_Solve(Teuchos::ParameterList &HDSAsettings)
  {
    postproc_->write_solution = false;
    postproc_->write_optimization_solution = false;

    HDSA::Ptr<std::ostream> outStream;
    HDSA::nullstream bhs; // outputs nothing
    if (comm_->getRank() == 0)
    {
      outStream = HDSA::makePtrFromRef(std::cout);
    }
    else
    {
      outStream = HDSA::makePtrFromRef(bhs);
    }

    int hdsa_verbosity = HDSAsettings.sublist("Configuration").get<int>("verbosity", 0);
    Teuchos::ParameterList data_load_list = HDSAsettings.sublist("DataLoadParameters");
    std::string random_number_file = data_load_list.get<std::string>("random_number_file", "error");
    int num_random_numbers = data_load_list.get<int>("num_random_numbers", 0);

    HDSA::Ptr<HDSA::Random_Number_Generator<ScalarT>> random_number_generator;
    HDSA::Ptr<HDSA::Comm<int>> hdsa_comm = HDSA::makePtr<HDSA::Comm<int>>(comm_);
    if (random_number_file == "error")
    {
      random_number_generator = HDSA::makePtr<HDSA::Random_Number_Generator<ScalarT>>(hdsa_comm);
    }
    else
    {
      if (num_random_numbers == 0)
      {
        *outStream << " Error: number of random numbers not specified" << std::endl;
      }
      random_number_generator = HDSA::makePtr<HDSA::Random_Number_Generator<ScalarT>>(num_random_numbers, random_number_file);
    }

    if (hdsa_verbosity > 1)
    {
      *outStream << "Beginning Data_Interface instantiation" << std::endl;
    }
    HDSA::Ptr<HDSA::MD_Data_Interface<ScalarT>> data_interface = HDSA::makePtr<MD_Data_Interface_MrHyDE<ScalarT>>(comm_, solver_, params_, random_number_generator, data_load_list);

    if (hdsa_verbosity > 1)
    {
      HDSA::Ptr<const HDSA::Vector<ScalarT>> u_opt = data_interface->Get_u_opt();
      HDSA::Ptr<const HDSA::Vector<ScalarT>> z_opt = data_interface->Get_z_opt();
      *outStream << "u_opt->norm() = " << u_opt->Norm() << std::endl;
      *outStream << "z_opt->norm() = " << z_opt->Norm() << std::endl;
    }

    if (hdsa_verbosity > 1)
    {
      *outStream << "Beginning Sol_Op_Interface instantiation" << std::endl;
    }
    HDSA::Ptr<HDSA::BF_Sol_Op_Interface<ScalarT>> sol_op_interface = HDSA::makePtr<BF_Sol_Op_Interface_MrHyDE<ScalarT>>(comm_, solver_->varlist, random_number_generator);

    if (hdsa_verbosity > 1)
    {
      *outStream << "Beginning Opt_Prob_Interface instantiation" << std::endl;
    }
    HDSA::Ptr<HDSA::MD_Opt_Prob_Interface<ScalarT>> opt_prob_interface = HDSA::makePtr<MD_Opt_Prob_Interface_MrHyDE<ScalarT>>(solver_, postproc_, params_, data_interface);

    if (hdsa_verbosity > 1)
    {
      *outStream << "Beginning optimization solution update" << std::endl;
    }
    RealT hessian_tol = HDSAsettings.sublist("Configuration").get<RealT>("Hessian Solve Tolerance", 1.e-4);
    HDSA::Ptr<HDSA::BF_Update<RealT>> bf_update = HDSA::makePtr<HDSA::BF_Update<RealT>>(sol_op_interface, opt_prob_interface, hdsa_verbosity, hessian_tol, "GMRES");
    HDSA::Ptr<HDSA::Vector<RealT>> z_update = bf_update->Update(*data_interface->Get_u_opt(), *data_interface->Get_z_opt());

    if (hdsa_verbosity > 1)
    {
      *outStream << "z_update->norm() = " << z_update->Norm() << std::endl;
    }

    std::string opt_solution_exo_file_ = data_load_list.get<std::string>("OptimalSolutionExoFile", "error");
    bool write_exo = true;
    if (opt_solution_exo_file_ == "error")
    {
      write_exo = false;
    }
    HDSA::Ptr<Output_Writer_MrHyDE<ScalarT>> output_writer = HDSA::makePtr<Output_Writer_MrHyDE<ScalarT>>(postproc_, solver_, write_exo);

    std::string filename = "./hdsa_output/z_update";
    output_writer->Write_to_File(z_update, filename, false);
  }

  void MD_Solve(Teuchos::ParameterList &HDSAsettings)
  {
    postproc_->write_solution = false;
    postproc_->write_optimization_solution = false;

    HDSA::Ptr<std::ostream> outStream;
    HDSA::nullstream bhs; // outputs nothing
    if (comm_->getRank() == 0)
    {
      outStream = HDSA::makePtrFromRef(std::cout);
    }
    else
    {
      outStream = HDSA::makePtrFromRef(bhs);
    }

    int stoch_dim = params_->getNumParams("stochastic");
    bool is_stoch = false;
    if (stoch_dim > 0)
    {
      is_stoch = true;
    }

    int num_prior_samples = HDSAsettings.sublist("Configuration").get<int>("num_prior_samples", 0);
    int num_posterior_samples = HDSAsettings.sublist("Configuration").get<int>("num_posterior_samples", 0);
    int prior_num_state_solves = HDSAsettings.sublist("Configuration").get<int>("prior_num_state_solves", 0);
    int hdsa_verbosity = HDSAsettings.sublist("Configuration").get<int>("verbosity", 0);
    bool execute_prior_discrepancy_sampling = HDSAsettings.sublist("Configuration").get<bool>("execute_prior_discrepancy_sampling", false);
    bool execute_posterior_discrepancy_sampling = HDSAsettings.sublist("Configuration").get<bool>("execute_posterior_discrepancy_sampling", false);
    bool execute_optimal_solution_update = HDSAsettings.sublist("Configuration").get<bool>("execute_optimal_solution_update", false);
    bool use_continuation = HDSAsettings.sublist("Configuration").get<bool>("use_continuation", false);

    std::string prior_computation = HDSAsettings.sublist("Prior Computation").get<std::string>("State Prior", "Numeric_Laplacian");
    bool use_direct_solvers = HDSAsettings.sublist("Prior Computation").get<bool>("use_direct_solvers", false);
    bool use_incomplete_prec = HDSAsettings.sublist("Prior Computation").get<bool>("use_incomplete_prec", true);

    int num_states = solver_->varlist[0][0].size();
    std::vector<ScalarT> alpha_u = std::vector<ScalarT>(num_states, 0.0);
    std::vector<ScalarT> beta_u = std::vector<ScalarT>(num_states, 0.0);
    std::vector<ScalarT> beta_t = std::vector<ScalarT>(num_states, 0.0);
    std::vector<ScalarT> alpha_d = std::vector<ScalarT>(num_states, 0.0);
    std::vector<int> prior_num_sing_vals = std::vector<int>(num_states, 0);
    std::vector<int> prior_oversampling = std::vector<int>(num_states, 0);
    std::vector<int> prior_num_subspace_iter = std::vector<int>(num_states, 0);
    for (int k = 0; k < num_states; k++)
    {
      std::string state_var_name = solver_->varlist[0][0][k];
      alpha_u[k] = HDSAsettings.sublist("HyperParameters").sublist(state_var_name).get<ScalarT>("alpha_u", 0.0);
      beta_u[k] = HDSAsettings.sublist("HyperParameters").sublist(state_var_name).get<ScalarT>("beta_u", 0.0);
      beta_t[k] = HDSAsettings.sublist("HyperParameters").sublist(state_var_name).get<ScalarT>("beta_t", 0.0);
      alpha_d[k] = HDSAsettings.sublist("HyperParameters").sublist(state_var_name).get<ScalarT>("alpha_d", 0.0);
      prior_num_sing_vals[k] = HDSAsettings.sublist("HyperParameters").sublist(state_var_name).get<int>("prior_num_sing_vals", 200);
      prior_oversampling[k] = HDSAsettings.sublist("HyperParameters").sublist(state_var_name).get<int>("prior_oversampling", 20);
      prior_num_subspace_iter[k] = HDSAsettings.sublist("HyperParameters").sublist(state_var_name).get<int>("prior_num_subspace_iter", 1);
    }

    ScalarT alpha_z = HDSAsettings.sublist("HyperParameters").sublist("z").get<ScalarT>("alpha_z", 0.0);
    ScalarT beta_z = HDSAsettings.sublist("HyperParameters").sublist("z").get<ScalarT>("beta_z", 0.0);
    ScalarT beta_t_z_prior = HDSAsettings.sublist("HyperParameters").sublist("z").get<ScalarT>("beta_t", 0.0);

    ScalarT max_marginal_var_percent = HDSAsettings.sublist("HyperParameters").sublist("OUU").get<ScalarT>("max_marginal_var_percent", 1.0);
    ScalarT min_cond_variance_percent = HDSAsettings.sublist("HyperParameters").sublist("OUU").get<ScalarT>("min_cond_variance_percent", 0.1);
    bool assume_independent_ensembles = HDSAsettings.sublist("HyperParameters").sublist("OUU").get<bool>("assume_independent_ensembles", false);

    int hessian_num_eig_vals = HDSAsettings.sublist("HyperParameters").get<int>("hessian_num_eig_vals", 5);
    int hessian_oversampling = HDSAsettings.sublist("HyperParameters").get<int>("hessian_oversampling", 3);

    bool center_data = HDSAsettings.sublist("HyperParameters").get<bool>("center_data", false);
    bool adapt_time_variance = HDSAsettings.sublist("HyperParameters").get<bool>("adapt_time_variance", false);

    Teuchos::ParameterList data_load_list = HDSAsettings.sublist("DataLoadParameters");
    std::string random_number_file = data_load_list.get<std::string>("random_number_file", "error");
    int num_random_numbers = data_load_list.get<int>("num_random_numbers", 0);

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////// Random number and sampler instantiation //////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////

    HDSA::Ptr<HDSA::Random_Number_Generator<ScalarT>> random_number_generator;
    HDSA::Ptr<HDSA::Comm<int>> hdsa_comm = HDSA::makePtr<HDSA::Comm<int>>(comm_);
    if (random_number_file == "error")
    {
      random_number_generator = HDSA::makePtr<HDSA::Random_Number_Generator<ScalarT>>(hdsa_comm);
    }
    else
    {
      if (num_random_numbers == 0)
      {
        *outStream << " Error: number of random numbers not specified" << std::endl;
      }
      random_number_generator = HDSA::makePtr<HDSA::Random_Number_Generator<ScalarT>>(num_random_numbers, random_number_file);
    }

    HDSA::Ptr<ROL::SampleGenerator<ScalarT>> sampler;
    int ens_size = data_load_list.get<int>("Ensemble Size", 0);
    if (is_stoch)
    {
      if (ens_size == 0)
      {
        *outStream << "Error: the ensemble size was not specified" << std::endl;
      }
      HDSA::Ptr<ROL::BatchManager<ScalarT>> bman = HDSA::makePtr<ROL::MrHyDETeuchosBatchManager<ScalarT, int>>(comm_);
      std::string sample_pt_file = data_load_list.get("Sample Set File", "error");
      std::string sample_wt_file = data_load_list.get("Sample Weight File", "error");
      sampler = HDSA::makePtr<ROL::Sample_Set_Reader<ScalarT>>(ens_size, stoch_dim, bman, sample_pt_file, sample_wt_file);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////// Data_Interface ///////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    if (hdsa_verbosity > 1)
    {
      *outStream << "Beginning Data_Interface instantiation" << std::endl;
    }

    HDSA::Ptr<HDSA::MD_Data_Interface<ScalarT>> data_interface;
    if (is_stoch)
    {
      std::vector<HDSA::Ptr<MD_Data_Interface_MrHyDE<ScalarT>>> data_interface_ens;
      data_interface_ens.resize(ens_size);
      for (int s = 0; s < ens_size; s++)
      {
        data_interface_ens[s] = HDSA::makePtr<MD_Data_Interface_MrHyDE<ScalarT>>(comm_, solver_, params_, random_number_generator, data_load_list);

        std::string exo_file_base = data_interface_ens[s]->Get_Opt_Solution_Exo_File();
        std::string exo_file = exo_file_base + std::to_string(s) + ".exo";
        data_interface_ens[s]->Overwrite_Opt_Solution_Exo_File(exo_file);

        std::vector<std::string> exo_files_base = data_interface_ens[s]->Get_HiFi_Exo_Files();
        std::vector<std::string> exo_files;
        exo_files.resize(exo_files_base.size());
        for (int k = 0; k < exo_files.size(); k++)
        {
          exo_files[k] = exo_files_base[k] + std::to_string(s) + ".exo";
        }
        data_interface_ens[s]->Overwrite_HiFi_Exo_Files(exo_files);
      }
      data_interface = HDSA::makePtr<MD_OUU_Data_Interface_MrHyDE<ScalarT>>(data_interface_ens, sampler, ens_size);
    }
    else
    {
      data_interface = HDSA::makePtr<MD_Data_Interface_MrHyDE<ScalarT>>(comm_, solver_, params_, random_number_generator, data_load_list);
    }

    if (hdsa_verbosity > 0)
    {
      HDSA::Ptr<const HDSA::Vector<ScalarT>> u_opt = data_interface->Get_u_opt();
      HDSA::Ptr<const HDSA::Vector<ScalarT>> z_opt = data_interface->Get_z_opt();
      HDSA::Ptr<const HDSA::MultiVector<ScalarT>> Z = data_interface->Get_Z();
      HDSA::Ptr<const HDSA::MultiVector<ScalarT>> D = data_interface->Get_D();
      *outStream << "u_opt->norm() = " << u_opt->Norm() << std::endl;
      *outStream << "z_opt->norm() = " << z_opt->Norm() << std::endl;
      int N = Z->Number_of_Vectors();
      for (int k = 0; k < N; k++)
      {
        *outStream << "Z[" << k << "]->norm() = " << (*Z)[k]->Norm() << std::endl;
        *outStream << "D[" << k << "]->norm() = " << (*D)[k]->Norm() << std::endl;
      }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////// Opt_Prob_Interface ///////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////

    if (hdsa_verbosity > 1)
    {
      *outStream << "Beginning Opt_Prob_Interface instantiation" << std::endl;
    }

    HDSA::Ptr<HDSA::MD_Opt_Prob_Interface<ScalarT>> opt_prob_interface;
    if (is_stoch)
    {
      std::vector<ScalarT> ens_weights = std::vector<ScalarT>(ens_size, 0.0);
      for (int s = 0; s < ens_size; s++)
      {
        ens_weights[s] = sampler->getMyWeight(s);
      }
      opt_prob_interface = HDSA::makePtr<MD_OUU_Opt_Prob_Interface_MrHyDE<ScalarT>>(solver_, postproc_, params_, data_interface, sampler, ens_weights);
    }
    else
    {
      opt_prob_interface = HDSA::makePtr<MD_Opt_Prob_Interface_MrHyDE<ScalarT>>(solver_, postproc_, params_, data_interface);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////// u_Prior_Interface ////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////

    if (hdsa_verbosity > 1)
    {
      *outStream << "Beginning u_Prior_Interface instantiation" << std::endl;
    }

    vector<string> blockNames = solver_->mesh->getBlockNames();
    HDSA::Ptr<Prior_Operators_Interface_MrHyDE<ScalarT>> prior_operator_interface = HDSA::makePtr<Prior_Operators_Interface_MrHyDE<ScalarT>>(comm_, settings_, blockNames);
    HDSA::Ptr<HDSA::Sparse_Matrix<ScalarT>> M = HDSA::makePtr<HDSA::Sparse_Matrix<ScalarT>>(prior_operator_interface->M);
    HDSA::Ptr<HDSA::Sparse_Matrix<ScalarT>> S = HDSA::makePtr<HDSA::Sparse_Matrix<ScalarT>>(prior_operator_interface->S);

    HDSA::Ptr<HDSA::MD_u_Prior_Interface<ScalarT>> u_prior_interface;
    HDSA::Ptr<HDSA::MD_u_Hyperparameter_Interface<ScalarT>> u_hyperparam_interface;

    std::vector<HDSA::Ptr<HDSA::MD_u_Hyperparameter_Interface<ScalarT>>> u_hyperparam_interface_std;
    std::vector<HDSA::Ptr<HDSA::MD_u_Prior_Interface<ScalarT>>> u_prior_interface_std;
    u_hyperparam_interface_std.resize(num_states);
    u_prior_interface_std.resize(num_states);

    bool is_transient = solver_->isTransient;
    for (int k = 0; k < num_states; k++)
    {
      u_hyperparam_interface_std[k] = HDSA::makePtr<MD_u_Hyperparameter_Interface_MrHyDE<ScalarT>>(comm_, data_interface, is_transient, center_data, adapt_time_variance, k);
      u_hyperparam_interface_std[k]->Set_alpha_d(alpha_d[k]);
      u_hyperparam_interface_std[k]->Set_alpha_u(alpha_u[k]);
      u_hyperparam_interface_std[k]->Set_beta_u(beta_u[k]);
      u_hyperparam_interface_std[k]->Set_beta_t(beta_t[k]);
      if ((prior_computation != "Lumped_Mass") && (prior_computation != "Bilaplacian"))
      {
        u_hyperparam_interface_std[k]->Set_GSVD_Hyperparameters(prior_num_sing_vals[k], prior_oversampling[k], prior_num_subspace_iter[k]);
      }

      if (is_transient)
      {
        HDSA::Ptr<HDSA::MD_u_Prior_Interface<ScalarT>> spatial_u_prior_interface_k;
        HDSA::Ptr<HDSA::MD_Transient_Prior_Covariance<ScalarT>> transient_prior_cov_k;
        ScalarT T = solver_->final_time;
        int n_t = solver_->settings->sublist("Solver").get<int>("number of steps", 0) + 1;

        if (is_stoch)
        {
          HDSA::Ptr<HDSA::MD_OUU_Data_Interface<ScalarT>> ouu_data_interface = Teuchos::rcp_dynamic_cast<HDSA::MD_OUU_Data_Interface<ScalarT>>(data_interface);
          HDSA::Ptr<HDSA::MD_OUU_Hyperparameter_Data_Interface<ScalarT>> data_interface_hyperparam = HDSA::makePtr<HDSA::MD_OUU_Hyperparameter_Data_Interface<ScalarT>>(ouu_data_interface);
          if (prior_computation == "Lumped_Mass")
          {
            spatial_u_prior_interface_k = HDSA::makePtr<HDSA::MD_Lumped_Mass_u_Prior_Interface<ScalarT>>(S, M, data_interface_hyperparam, u_hyperparam_interface_std[k], hdsa_comm, random_number_generator, use_direct_solvers, hdsa_verbosity, use_incomplete_prec, *outStream);
          }
          else if (prior_computation == "Bilaplacian")
          {
            spatial_u_prior_interface_k = HDSA::makePtr<HDSA::MD_Bilaplacian_u_Prior_Interface<ScalarT>>(S, M, data_interface_hyperparam, u_hyperparam_interface_std[k], hdsa_comm, random_number_generator, use_direct_solvers, hdsa_verbosity, use_incomplete_prec, *outStream);
          }
          else
          {
            spatial_u_prior_interface_k = HDSA::makePtr<HDSA::MD_Numeric_Laplacian_u_Prior_Interface<ScalarT>>(S, M, data_interface_hyperparam, u_hyperparam_interface_std[k], random_number_generator);
          }
          int n_y = data_interface_hyperparam->Get_u_opt()->Dimension() / n_t;
          transient_prior_cov_k = HDSA::makePtr<HDSA::MD_Transient_Prior_Covariance<ScalarT>>(data_interface_hyperparam, u_hyperparam_interface_std[k], T, n_t, n_y);
        }
        else
        {
          int n_y = data_interface->Get_u_opt()->Dimension() / n_t;
          if (prior_computation == "Lumped_Mass")
          {
            spatial_u_prior_interface_k = HDSA::makePtr<HDSA::MD_Lumped_Mass_u_Prior_Interface<ScalarT>>(S, M, data_interface, u_hyperparam_interface_std[k], hdsa_comm, random_number_generator, use_direct_solvers, hdsa_verbosity, use_incomplete_prec, *outStream);
          }
          if (prior_computation == "Bilaplacian")
          {
            spatial_u_prior_interface_k = HDSA::makePtr<HDSA::MD_Bilaplacian_u_Prior_Interface<ScalarT>>(S, M, data_interface, u_hyperparam_interface_std[k], hdsa_comm, random_number_generator, use_direct_solvers, hdsa_verbosity, use_incomplete_prec, *outStream);
          }
          else
          {
            spatial_u_prior_interface_k = HDSA::makePtr<HDSA::MD_Numeric_Laplacian_u_Prior_Interface<ScalarT>>(S, M, data_interface, u_hyperparam_interface_std[k], random_number_generator);
          }
          transient_prior_cov_k = HDSA::makePtr<HDSA::MD_Transient_Prior_Covariance<ScalarT>>(data_interface, u_hyperparam_interface_std[k], T, n_t, n_y);
        }

        u_prior_interface_std[k] = HDSA::makePtr<HDSA::MD_Transient_Elliptic_u_Prior_Interface<ScalarT>>(spatial_u_prior_interface_k, transient_prior_cov_k);
      }
      else
      {
        if (is_stoch)
        {
          HDSA::Ptr<HDSA::MD_OUU_Data_Interface<ScalarT>> ouu_data_interface = Teuchos::rcp_dynamic_cast<HDSA::MD_OUU_Data_Interface<ScalarT>>(data_interface);
          HDSA::Ptr<HDSA::MD_OUU_Hyperparameter_Data_Interface<ScalarT>> data_interface_hyperparam = HDSA::makePtr<HDSA::MD_OUU_Hyperparameter_Data_Interface<ScalarT>>(ouu_data_interface);
          if (prior_computation == "Lumped_Mass")
          {
            u_prior_interface_std[k] = HDSA::makePtr<HDSA::MD_Lumped_Mass_u_Prior_Interface<ScalarT>>(S, M, data_interface_hyperparam, u_hyperparam_interface_std[k], hdsa_comm, random_number_generator, use_direct_solvers, hdsa_verbosity, use_incomplete_prec, *outStream);
          }
          else if (prior_computation == "Bilaplacian")
          {
            u_prior_interface_std[k] = HDSA::makePtr<HDSA::MD_Bilaplacian_u_Prior_Interface<ScalarT>>(S, M, data_interface_hyperparam, u_hyperparam_interface_std[k], hdsa_comm, random_number_generator, use_direct_solvers, hdsa_verbosity, use_incomplete_prec, *outStream);
          }
          else
          {
            u_prior_interface_std[k] = HDSA::makePtr<HDSA::MD_Numeric_Laplacian_u_Prior_Interface<ScalarT>>(S, M, data_interface_hyperparam, u_hyperparam_interface_std[k], random_number_generator);
          }
        }
        else
        {
          if (prior_computation == "Lumped_Mass")
          {
            u_prior_interface_std[k] = HDSA::makePtr<HDSA::MD_Lumped_Mass_u_Prior_Interface<ScalarT>>(S, M, data_interface, u_hyperparam_interface_std[k], hdsa_comm, random_number_generator, use_direct_solvers, hdsa_verbosity, use_incomplete_prec, *outStream);
          }
          else if (prior_computation == "Bilaplacian")
          {
            u_prior_interface_std[k] = HDSA::makePtr<HDSA::MD_Bilaplacian_u_Prior_Interface<ScalarT>>(S, M, data_interface, u_hyperparam_interface_std[k], hdsa_comm, random_number_generator, use_direct_solvers, hdsa_verbosity, use_incomplete_prec, *outStream);
          }
          else
          {
            u_prior_interface_std[k] = HDSA::makePtr<HDSA::MD_Numeric_Laplacian_u_Prior_Interface<ScalarT>>(S, M, data_interface, u_hyperparam_interface_std[k], random_number_generator);
          }
        }
      }
    }

    if (is_stoch)
    {
      HDSA::Ptr<HDSA::MD_u_Prior_Interface<ScalarT>> us_prior_interface;
      if (num_states > 1)
      {
        u_hyperparam_interface = HDSA::makePtr<HDSA::MD_Multi_State_u_Hyperparameter_Interface<ScalarT>>(u_hyperparam_interface_std);
        us_prior_interface = HDSA::makePtr<HDSA::MD_Multi_State_u_Prior_Interface<ScalarT>>(data_interface, u_prior_interface_std);
      }
      else
      {
        u_hyperparam_interface = u_hyperparam_interface_std[0];
        us_prior_interface = u_prior_interface_std[0];
      }

      HDSA::Ptr<HDSA::MD_OUU_Ensemble_Weighting_Matrix<ScalarT>> ensemble_weighting = HDSA::makePtr<HDSA::MD_OUU_Ensemble_Weighting_Matrix<ScalarT>>(data_interface, us_prior_interface, ens_size, max_marginal_var_percent, min_cond_variance_percent, assume_independent_ensembles);
      u_prior_interface = HDSA::makePtr<HDSA::MD_OUU_u_Prior_Interface<ScalarT>>(us_prior_interface, ensemble_weighting);
    }
    else
    {
      if (num_states > 1)
      {
        u_hyperparam_interface = HDSA::makePtr<HDSA::MD_Multi_State_u_Hyperparameter_Interface<ScalarT>>(u_hyperparam_interface_std);
        u_prior_interface = HDSA::makePtr<HDSA::MD_Multi_State_u_Prior_Interface<ScalarT>>(data_interface, u_prior_interface_std);
      }
      else
      {
        u_hyperparam_interface = u_hyperparam_interface_std[0];
        u_prior_interface = u_prior_interface_std[0];
      }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////// z_Prior_Interface ////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////

    if (hdsa_verbosity > 1)
    {
      *outStream << "Beginning z_Prior_Interface instantiation" << std::endl;
    }

    std::string z_type;
    if (params_->getNumParams("discretized") > 0)
    {
      z_type = "spatial field";
    }
    else if (params_->have_dynamic_scalar)
    {
      z_type = "transient vector";
    }
    else
    {
      z_type = "vector";
    }
    HDSA::Ptr<HDSA::MD_z_Hyperparameter_Interface<ScalarT>> z_hyperparam_interface = HDSA::makePtr<MD_z_Hyperparameter_Interface_MrHyDE<ScalarT>>(solver_, params_, comm_, data_interface, random_number_generator, z_type, prior_num_state_solves);
    z_hyperparam_interface->Set_alpha_z(alpha_z);

    HDSA::Ptr<HDSA::MD_z_Prior_Interface<ScalarT>> z_prior_interface;
    if (z_type == "spatial field")
    {
      z_hyperparam_interface->Set_beta_z(beta_z);
      z_prior_interface = HDSA::makePtr<HDSA::MD_Numeric_Laplacian_z_Prior_Interface<ScalarT>>(S, M, data_interface, z_hyperparam_interface, u_prior_interface, use_direct_solvers, hdsa_verbosity, use_incomplete_prec, *outStream);
    }
    else if (z_type == "transient vector")
    {
      ScalarT T = solver_->final_time;
      int n_t = solver_->settings->sublist("Solver").get<int>("number of steps", 0);
      int num_controls = params_->getNumParams("active");
      z_hyperparam_interface->Set_beta_t(beta_t_z_prior);
      z_prior_interface = HDSA::makePtr<HDSA::MD_Transient_Vector_z_Prior_Interface<ScalarT>>(data_interface, z_hyperparam_interface, u_prior_interface, n_t, T, num_controls);
    }
    else if (z_type == "vector")
    {
      z_prior_interface = HDSA::makePtr<HDSA::MD_Vector_z_Prior_Interface<ScalarT>>(data_interface, z_hyperparam_interface, u_prior_interface);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////// Output_Writer ////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    std::string opt_solution_exo_file_ = data_load_list.get<std::string>("OptimalSolutionExoFile", "error");
    bool write_exo = true;
    if (opt_solution_exo_file_ == "error")
    {
      write_exo = false;
    }
    HDSA::Ptr<Output_Writer_MrHyDE<ScalarT>> output_writer = HDSA::makePtr<Output_Writer_MrHyDE<ScalarT>>(postproc_, solver_, write_exo);
    output_writer->Write_Hyperparameters(u_hyperparam_interface, z_hyperparam_interface);

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////// Prior Discrepancy Analysis ///////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////

    if ((num_prior_samples > 0) & execute_prior_discrepancy_sampling)
    {
      if (hdsa_verbosity > 1)
      {
        *outStream << "Beginning prior discrepancy analysis" << std::endl;
      }

      HDSA::Ptr<HDSA::MD_Prior_Sampling<ScalarT>> prior_sampling = HDSA::makePtr<HDSA::MD_Prior_Sampling<ScalarT>>(data_interface, u_prior_interface, z_prior_interface);
      HDSA::Ptr<HDSA::MultiVector<ScalarT>> spatial_coords = data_interface->Read_Spatial_Node_Data();
      prior_sampling->Generate_Prior_Discrepancy_Sample_Data(num_prior_samples, u_hyperparam_interface, z_hyperparam_interface, spatial_coords);

      HDSA::Ptr<HDSA::MultiVector<ScalarT>> prior_delta_z_opt = prior_sampling->Get_prior_delta_z_opt();
      std::vector<HDSA::Ptr<HDSA::Vector<ScalarT>>> prior_z_pert = prior_sampling->Get_prior_z_pert();
      std::vector<HDSA::Ptr<HDSA::MultiVector<ScalarT>>> prior_delta_z_pert = prior_sampling->Get_prior_delta_z_pert();
      output_writer->Write_Prior_Discrepancy_Samples(prior_delta_z_opt, prior_z_pert, prior_delta_z_pert);

      if (is_transient)
      {
        std::vector<std::vector<std::vector<ScalarT>>> prior_delta_z_opt_time_evol = prior_sampling->Get_prior_delta_z_opt_time_evol();
        std::vector<std::vector<ScalarT>> prior_discrep_data_time_evol = prior_sampling->Get_prior_discrep_data_time_evol();
        output_writer->Write_Prior_Discrepancy_Time_Evolution(prior_delta_z_opt_time_evol, prior_discrep_data_time_evol);
      }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////// Posterior Discrepancy Analysis ///////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////

    HDSA::Ptr<HDSA::MD_Posterior_Sampling<ScalarT>> post_sampling = HDSA::makePtr<HDSA::MD_Posterior_Sampling<ScalarT>>(data_interface, u_prior_interface, z_prior_interface);
    if (execute_posterior_discrepancy_sampling || execute_optimal_solution_update)
    {
      if (hdsa_verbosity > 1)
      {
        *outStream << "Beginning posterior data computation" << std::endl;
      }

      post_sampling->Compute_Posterior_Data(u_hyperparam_interface->Get_alpha_d(), num_posterior_samples);
    }

    if ((num_posterior_samples > 0) & execute_posterior_discrepancy_sampling)
    {
      if (hdsa_verbosity > 1)
      {
        *outStream << "Beginning posterior discrepancy analysis" << std::endl;
      }

      std::vector<HDSA::Ptr<HDSA::Vector<ScalarT>>> z_in;
      int N = data_interface->Get_Z()->Number_of_Vectors();
      z_in.resize(N);
      for (int k = 0; k < N; k++)
      {
        z_in[k] = (*data_interface->Get_Z())[k];
      }
      std::vector<HDSA::Ptr<HDSA::MD_Posterior_Vectors<ScalarT>>> post_delta = post_sampling->Posterior_Discrepancy_Samples(z_in);

      for (int k = 0; k < N; k++)
      {
        HDSA::Ptr<HDSA::Vector<ScalarT>> dk = (*data_interface->Get_D())[k];
        HDSA::Ptr<HDSA::Vector<ScalarT>> tmp1 = dk->Clone();
        HDSA::Ptr<HDSA::Vector<ScalarT>> tmp2 = dk->Clone();

        u_prior_interface->Apply_M_u(*tmp1, *dk);
        RealT normalization = std::sqrt(dk->Dot(*tmp1));

        tmp1->Set(*dk);
        tmp1->Scaled_Plus(-1.0, *post_delta[k]->mean);
        u_prior_interface->Apply_M_u(*tmp2, *tmp1);
        post_delta[k]->ref_mean_diff = std::sqrt(tmp2->Dot(*tmp1)) / normalization;

        for (int i = 0; i < num_posterior_samples; i++)
        {
          tmp1->Set(*dk);
          tmp1->Scaled_Plus(-1.0, *(*post_delta[k]->samples)[i]);
          u_prior_interface->Apply_M_u(*tmp2, *tmp1);
          post_delta[k]->ref_samples_diff[i] = std::sqrt(tmp2->Dot(*tmp1)) / normalization;
        }
      }
      output_writer->Write_Posterior_Discrepancy_Samples(post_delta);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////// Posterior Optimal Solution Analysis //////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////

    if (execute_optimal_solution_update)
    {
      HDSA::Ptr<HDSA::MD_Hessian_Analysis<ScalarT>> hessian_analysis = HDSA::makePtr<HDSA::MD_Hessian_Analysis<ScalarT>>(opt_prob_interface, z_prior_interface);
      if (hessian_num_eig_vals > 0)
      {
        if (hdsa_verbosity > 1)
        {
          *outStream << "Beginning Hessian analysis" << std::endl;
        }

        hessian_analysis->Compute_Hessian_GEVP(data_interface->Get_z_opt(), hessian_num_eig_vals, hessian_oversampling, false);
        HDSA::Ptr<HDSA::Dense_Matrix<ScalarT>> evals = hessian_analysis->Get_Evals();
        output_writer->Write_Hessian_Eigenvalues(evals);
      }

      if (num_posterior_samples > 0)
      {
        if (hdsa_verbosity > 1)
        {
          *outStream << "Beginning posterior optimal solution analysis" << std::endl;
        }

        HDSA::Ptr<HDSA::MD_Posterior_Vectors<ScalarT>> posterior_update_samples;
        if (use_continuation)
        {
          *outStream << "Performing continuation optimal solution update is now yet supported in the MrHyDE interface..." << std::endl;
          //*outStream << "Performing continuation optimal solution update..." << std::endl;
          HDSA::Ptr<HDSA::MD_Continuation_Update<ScalarT>> update = HDSA::makePtr<HDSA::MD_Continuation_Update<ScalarT>>(data_interface, z_prior_interface, opt_prob_interface, post_sampling, hessian_analysis, 3);
          //posterior_update_samples = update->Posterior_Update_Samples();
        }
        else
        {
          //*outStream << "Performing linearization optimal solution update..." << std::endl;
          HDSA::Ptr<HDSA::MD_Update<ScalarT>> update = HDSA::makePtr<HDSA::MD_Update<ScalarT>>(data_interface, u_prior_interface, z_prior_interface, opt_prob_interface, post_sampling, hessian_analysis);
          posterior_update_samples = update->Posterior_Update_Samples();
        }

        output_writer->Write_Optimal_Solution_Update(posterior_update_samples);

        if (hdsa_verbosity > 0)
        {
          *outStream << "z_update_mean norm = " << posterior_update_samples->mean->Norm() << std::endl;
        }
      }
      else
      {
        HDSA::Ptr<HDSA::Vector<ScalarT>> z_update_mean;
        if (use_continuation)
        {
          *outStream << "Performing continuation optimal solution update is now yet supported in the MrHyDE interface..." << std::endl;
          //*outStream << "Performing continuation optimal solution update..." << std::endl;
          HDSA::Ptr<HDSA::MD_Continuation_Update<ScalarT>> update = HDSA::makePtr<HDSA::MD_Continuation_Update<ScalarT>>(data_interface, z_prior_interface, opt_prob_interface, post_sampling, hessian_analysis, 3);
          //z_update_mean = update->Posterior_Update_Mean();
        }
        else
        {
          //*outStream << "Performing linearization optimal solution update..." << std::endl;
          HDSA::Ptr<HDSA::MD_Update<ScalarT>> update = HDSA::makePtr<HDSA::MD_Update<ScalarT>>(data_interface, u_prior_interface, z_prior_interface, opt_prob_interface, post_sampling, hessian_analysis);
          z_update_mean = update->Posterior_Update_Mean();
        }

        output_writer->Write_Optimal_Solution_Update(z_update_mean);

        if (hdsa_verbosity > 0)
        {
          *outStream << "z_update_mean norm = " << z_update_mean->Norm() << std::endl;
        }
      }
    }
  }
};
#endif
