/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_DATA_INTERFACE_MRHYDE_HPP
#define HDSA_MD_DATA_INTERFACE_MRHYDE_HPP

#include "HDSA_Data_Loader_MrHyDE.hpp"
#include "HDSA_MD_Data_Interface.hpp"
#include "HDSA_Tpetra_Vector.hpp"
#include "HDSA_Std_Vector.hpp"
#include "HDSA_Solver_Interface_MrHyDE.hpp"

template <class RealT,
          class LO = Tpetra::Map<>::local_ordinal_type,
          class GO = Tpetra::Map<>::global_ordinal_type,
          class Node = Tpetra::Map<>::node_type>
class MD_Data_Interface_MrHyDE : public HDSA::MD_Data_Interface<RealT>
{

private:
  Teuchos::RCP<Teuchos::MpiComm<int>> comm_;
  Teuchos::RCP<MrHyDE::SolverManager<SolverNode>> solve_;
  HDSA::Ptr<MrHyDE::ParameterManager<SolverNode>> params_;
  const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> random_number_generator_;
  Teuchos::ParameterList data_load_list_;

  HDSA::Ptr<HDSA::Comm<int>> hdsa_comm_;
  HDSA::Ptr<Data_Loader_MrHyDE<RealT, LO, GO, Node>> data_loader_;
  HDSA::Ptr<Solver_Interface_MrHyDE<RealT>> solver_interface_;

  int num_hifi_;
  std::string opt_solution_exo_file_;
  std::vector<std::string> hifi_exo_files_;
  std::string opt_solution_txt_file_u_;
  std::string opt_solution_txt_file_z_;
  std::vector<std::string> hifi_txt_files_u_;
  std::vector<std::string> txt_files_z_;

public:
  MD_Data_Interface_MrHyDE(Teuchos::RCP<Teuchos::MpiComm<int>> &comm, Teuchos::RCP<MrHyDE::SolverManager<SolverNode>> &solve, HDSA::Ptr<MrHyDE::ParameterManager<SolverNode>> &params, const HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> &random_number_generator, Teuchos::ParameterList &data_load_list) : comm_(comm), solve_(solve), params_(params), random_number_generator_(random_number_generator), data_load_list_(data_load_list)
  {
    hdsa_comm_ = HDSA::makePtr<HDSA::Comm<int>>(comm_);

    data_loader_ = HDSA::makePtr<Data_Loader_MrHyDE<RealT, LO, GO, Node>>(comm_, solve_, params_, random_number_generator_);
    solver_interface_ = HDSA::makePtr<Solver_Interface_MrHyDE<RealT>>(solve_, params_);

    num_hifi_ = data_load_list_.get<int>("NumHifi", 1);
    opt_solution_exo_file_ = data_load_list_.get<std::string>("OptimalSolutionExoFile", "error");

    opt_solution_txt_file_u_ = data_load_list_.get<std::string>("OptimalSolutionTxtFileU", "error");
    opt_solution_txt_file_z_ = data_load_list_.get<std::string>("OptimalSolutionTxtFileZ", "error");

    hifi_exo_files_.resize(num_hifi_);

    hifi_txt_files_u_.resize(num_hifi_);
    txt_files_z_.resize(num_hifi_);
    for (int k = 0; k < num_hifi_; k++)
    {
      hifi_exo_files_[k] = data_load_list_.get<std::string>("HifiExoFile" + std::to_string(k + 1), "error");

      hifi_txt_files_u_[k] = data_load_list_.get<std::string>("HifiTxtFileU" + std::to_string(k + 1), "error");
      txt_files_z_[k] = data_load_list_.get<std::string>("TxtFileZ" + std::to_string(k + 1), "error");
    }
  }

  virtual ~MD_Data_Interface_MrHyDE()
  {
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Implementation of base class pure virtual functions: Load_Optimal_u, Load_Optimal_z, Load_Z_Data, Load_D_Data
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  HDSA::Ptr<HDSA::Vector<RealT>> Load_Optimal_u(void) const override
  {

    int num_time_nodes = solve_->settings->sublist("Solver").get<int>("number of steps", 0) + 1;

    HDSA::Ptr<HDSA::Vector<RealT>> u_opt;
    if (num_time_nodes > 1)
    {
      std::vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>>> u_tpetra;
      std::vector<HDSA::Ptr<HDSA::Vector<ScalarT>>> u_hdsa;
      u_tpetra.resize(num_time_nodes);
      u_hdsa.resize(num_time_nodes);
      for (int i = 0; i < num_time_nodes; i++)
      {
        if (opt_solution_exo_file_ != "error")
        {
          u_tpetra[i] = data_loader_->Read_Exodus_Data(opt_solution_exo_file_, true, i + 1);
        }
        else if (opt_solution_txt_file_u_ != "error")
        {
          u_tpetra[i] = data_loader_->Read_Text_Data(opt_solution_txt_file_u_, true, i + 1);
        }
        else
        {
          HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                                  "Error in HDSA_Data_Interface_MrHyDE: no valid input file given for Load_Optimal_u" << std::endl);
        }
        u_hdsa[i] = HDSA::makePtr<HDSA::Tpetra_Vector<RealT>>(u_tpetra[i], random_number_generator_);
      }
      u_opt = HDSA::makePtr<HDSA::Transient_Vector<ScalarT>>(u_hdsa);
    }
    else
    {
      Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> u_tpetra;
      if (opt_solution_exo_file_.substr(opt_solution_exo_file_.length() - 3, opt_solution_exo_file_.length()) == "exo")
      {
        u_tpetra = data_loader_->Read_Exodus_Data(opt_solution_exo_file_);
      }
      else if (opt_solution_txt_file_u_ != "error")
      {
        u_tpetra = data_loader_->Read_Text_Data(opt_solution_txt_file_u_);
      }
      else
      {
        HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                                "Error in HDSA_Data_Interface_MrHyDE: no valid input file given for Load_Optimal_u" << std::endl);
      }

      u_opt = HDSA::makePtr<HDSA::Tpetra_Vector<RealT>>(u_tpetra, random_number_generator_);
    }
    return u_opt;
  }

  HDSA::Ptr<HDSA::Vector<RealT>> Load_Optimal_z(void) const override
  {
    HDSA::Ptr<HDSA::Vector<RealT>> z_opt_hdsa;

    if (opt_solution_txt_file_z_ != "error")
    {
      if (params_->getNumParams("discretized") > 0)
      {
        Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> z_tpetra = data_loader_->Read_Text_Data(opt_solution_txt_file_z_, false);
        z_opt_hdsa = HDSA::makePtr<HDSA::Tpetra_Vector<RealT>>(z_tpetra, random_number_generator_);
      }
      else if (params_->have_dynamic_scalar)
      {
        std::vector<std::vector<RealT>> z_vec = data_loader_->Read_Text_Data_Dynamic_std(opt_solution_txt_file_z_);
        std::vector<HDSA::Ptr<HDSA::Vector<RealT>>> trans_vec;
        int num_time_steps = z_vec.size();
        trans_vec.resize(num_time_steps);
        for (int i = 0; i < num_time_steps; i++)
        {
          trans_vec[i] = HDSA::makePtr<HDSA::Std_Vector<RealT>>(z_vec[i], random_number_generator_);
        }
        z_opt_hdsa = HDSA::makePtr<HDSA::Transient_Vector<RealT>>(trans_vec);
      }
      else
      {
        std::vector<RealT> z_vec = data_loader_->Read_Text_Data_std(opt_solution_txt_file_z_);
        z_opt_hdsa = HDSA::makePtr<HDSA::Std_Vector<RealT>>(z_vec, random_number_generator_);
      }
    }
    else if (opt_solution_exo_file_ != "error")
    {
      Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> z_tpetra = data_loader_->Read_Exodus_Data(opt_solution_exo_file_, false);
      z_opt_hdsa = HDSA::makePtr<HDSA::Tpetra_Vector<RealT>>(z_tpetra, random_number_generator_);
    }
    else
    {
      HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                              "Error in HDSA_Data_Interface_MrHyDE: no valid input file given for Load_Optimal_z" << std::endl);
    }

    return z_opt_hdsa;
  }

  HDSA::Ptr<HDSA::MultiVector<RealT>> Load_Z_Data(void) const override
  {
    HDSA::Ptr<HDSA::MultiVector<RealT>> Z = HDSA::makePtr<HDSA::MultiVector<RealT>>();

    for (int k = 0; k < num_hifi_; k++)
    {
      HDSA::Ptr<HDSA::Vector<RealT>> z_hdsa;
      if (txt_files_z_[k] != "error")
      {
        if (params_->getNumParams("discretized") > 0)
        {
          Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> z_tpetra = data_loader_->Read_Text_Data(txt_files_z_[k], false);
          z_hdsa = HDSA::makePtr<HDSA::Tpetra_Vector<RealT>>(z_tpetra, random_number_generator_);
        }
        else if (params_->have_dynamic_scalar)
        {
          std::vector<std::vector<RealT>> z_vec = data_loader_->Read_Text_Data_Dynamic_std(opt_solution_txt_file_z_);
          std::vector<HDSA::Ptr<HDSA::Vector<RealT>>> trans_vec;
          int num_time_steps = z_vec.size();
          trans_vec.resize(num_time_steps);
          for (int i = 0; i < num_time_steps; i++)
          {
            trans_vec[i] = HDSA::makePtr<HDSA::Std_Vector<RealT>>(z_vec[i], random_number_generator_);
          }
          z_hdsa = HDSA::makePtr<HDSA::Transient_Vector<RealT>>(trans_vec);
        }
        else
        {
          std::vector<RealT> z_vec = data_loader_->Read_Text_Data_std(txt_files_z_[k]);
          z_hdsa = HDSA::makePtr<HDSA::Std_Vector<RealT>>(z_vec, random_number_generator_);
        }
      }
      else if (hifi_exo_files_[k] != "error")
      {
        Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> z_tpetra = data_loader_->Read_Exodus_Data(hifi_exo_files_[k], false);
        z_hdsa = HDSA::makePtr<HDSA::Tpetra_Vector<RealT>>(z_tpetra, random_number_generator_);
      }
      else
      {
        HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                                "Error in HDSA_Data_Interface_MrHyDE: no valid input file given for Load_Z_Data" << std::endl);
      }
      Z->push_back(z_hdsa);
    }
    return Z;
  }

  HDSA::Ptr<HDSA::MultiVector<RealT>> Load_D_Data(void) const override
  {
    int num_time_nodes = solve_->settings->sublist("Solver").get<int>("number of steps", 0) + 1;
    std::vector<HDSA::Ptr<HDSA::Vector<RealT>>> u_vecs;
    HDSA::Ptr<HDSA::MultiVector<RealT>> Z = Load_Z_Data();
    for (int k = 0; k < num_hifi_; k++)
    {
      HDSA::Ptr<HDSA::Vector<RealT>> u_k_lofi = Load_Optimal_u()->Clone();
      solver_interface_->State_Solve(*u_k_lofi, *(*Z)[k]);

      if (num_time_nodes > 1)
      {
        std::vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>>> u_tpetra_hifi;
        std::vector<HDSA::Ptr<HDSA::Vector<ScalarT>>> u_hdsa_hifi;
        u_tpetra_hifi.resize(num_time_nodes);
        u_hdsa_hifi.resize(num_time_nodes);

        for (int i = 0; i < num_time_nodes; i++)
        {
          if (hifi_exo_files_[k] != "error")
          {
            u_tpetra_hifi[i] = data_loader_->Read_Exodus_Data(hifi_exo_files_[k], true, i + 1);
          }
          else if (hifi_txt_files_u_[k] != "error")
          {
            u_tpetra_hifi[i] = data_loader_->Read_Text_Data(hifi_txt_files_u_[k], true, i + 1);
          }
          else
          {
            HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                                    "Error in HDSA_Data_Interface_MrHyDE: no valid input file given for Load_D_Data" << std::endl);
          }
          u_hdsa_hifi[i] = HDSA::makePtr<HDSA::Tpetra_Vector<RealT>>(u_tpetra_hifi[i], random_number_generator_);
        }
        HDSA::Ptr<HDSA::Vector<RealT>> u_k_hifi = HDSA::makePtr<HDSA::Transient_Vector<ScalarT>>(u_hdsa_hifi);
        u_k_hifi->Scaled_Plus(-1.0, *u_k_lofi);
        u_vecs.push_back(u_k_hifi);
      }
      else
      {
        Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> u_tpetra_hifi;
        if (hifi_exo_files_[k] != "error")
        {
          u_tpetra_hifi = data_loader_->Read_Exodus_Data(hifi_exo_files_[k]);
        }
        else if (hifi_txt_files_u_[k] != "error")
        {
          u_tpetra_hifi = data_loader_->Read_Text_Data(hifi_txt_files_u_[k]);
        }
        else
        {
          HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                                  "Error in HDSA_Data_Interface_MrHyDE: no valid input file given for Load_D_Data" << std::endl);
        }
        HDSA::Ptr<HDSA::Vector<RealT>> u_k = HDSA::makePtr<HDSA::Tpetra_Vector<RealT>>(u_tpetra_hifi, random_number_generator_);
        u_k->Scaled_Plus(-1.0, *u_k_lofi);
        u_vecs.push_back(u_k);
      }
    }
    HDSA::Ptr<HDSA::MultiVector<RealT>> D = HDSA::makePtr<HDSA::MultiVector<RealT>>(u_vecs);
    return D;
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Implementation of base class virtual functions: Read_Spatial_Node_Data, Extract_State_Component, Set_State_Component
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  HDSA::Ptr<HDSA::MultiVector<RealT>> Read_Spatial_Node_Data() const override
  {
    HDSA::Ptr<HDSA::MultiVector<RealT>> spatial_nodes;
    if (opt_solution_exo_file_ != "error")
    {
      spatial_nodes = data_loader_->Read_Exodus_Spatial_Node_Data(opt_solution_exo_file_);
    }
    else
    {
      std::cout << "Error: Read_Spatial_Node_Data is currently only supported for Exodus file reading" << std::endl;

      // Default to assume 1D on the domain [0,1] so that test problems will run
      Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> vec = solve_->linalg->getNewVector(0);
      int dim = vec->getGlobalLength();
      for (int k = 0; k < dim; k++)
      {
        RealT val = double(k) / double(dim - 1);
        vec->replaceGlobalValue(k, 0, val);
      }
      std::vector<HDSA::Ptr<HDSA::Vector<RealT>>> coord_vecs;
      coord_vecs.resize(1);
      coord_vecs[0] = HDSA::makePtr<HDSA::Tpetra_Vector<RealT>>(vec, random_number_generator_);
      spatial_nodes = HDSA::makePtr<HDSA::MultiVector<RealT>>(coord_vecs);
    }
    return spatial_nodes;
  }

  HDSA::Ptr<const HDSA::Vector<RealT>> Extract_State_Component(const HDSA::Vector<RealT> &u, int component_id) const override
  {

    HDSA::Ptr<const HDSA::Vector<RealT>> u_component;
    int num_states = solve_->varlist[0][0].size();
    if (num_states == 1)
    {
      u_component = HDSA::makePtrFromRef(u);
    }
    else
    {
      const HDSA::Tpetra_Vector<RealT> &eu = dynamic_cast<const HDSA::Tpetra_Vector<RealT> &>(u);
      HDSA::Ptr<Tpetra::MultiVector<RealT>> eu_tpetra = eu.getVector();
      Teuchos::ArrayRCP<const RealT> u_view = eu_tpetra->get1dView();
      Teuchos::RCP<const Tpetra::Map<LO, GO>> map = eu_tpetra->getMap();

      int num_local_elements = map->getLocalNumElements() / num_states;
      int init_index = map->getMinGlobalIndex() / num_states;
      int num_global_element = map->getGlobalNumElements() / num_states;
      Teuchos::Array<GO> component_ids(num_local_elements);
      for (int i = 0; i < num_local_elements; ++i)
      {
        component_ids[i] = init_index + i;
      }
      Teuchos::RCP<const Tpetra::Map<LO, GO>> component_map = HDSA::makePtr<Tpetra::Map<LO, GO>>(num_global_element, component_ids, 0, solve_->Comm);

      HDSA::Ptr<Tpetra::MultiVector<ScalarT, LO, GO, Node>> tpetra_vec = HDSA::makePtr<Tpetra::MultiVector<ScalarT, LO, GO, Node>>(component_map, 1);
      for (int k = 0; k < num_local_elements; k++)
      {
        tpetra_vec->replaceLocalValue(k, 0, u_view[num_states * k + component_id]);
      }
      u_component = HDSA::makePtr<HDSA::Tpetra_Vector<RealT>>(tpetra_vec, random_number_generator_);
    }
    return u_component;
  }

  void Set_State_Component(HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &u_component, int component_id) const override
  {
    int num_states = solve_->varlist[0][0].size();
    if (num_states == 1)
    {
      u.Set(u_component);
    }
    else
    {
      const HDSA::Tpetra_Vector<RealT> u_tpetra = dynamic_cast<const HDSA::Tpetra_Vector<RealT> &>(u);
      const HDSA::Tpetra_Vector<RealT> u_component_tpetra = dynamic_cast<const HDSA::Tpetra_Vector<RealT> &>(u_component);
      Teuchos::ArrayRCP<const RealT> u_component_view = u_component_tpetra.getVector()->get1dView();
      int local_dim = u_component_tpetra.getVector()->getLocalLength();
      for (int k = 0; k < local_dim; k++)
      {
        u_tpetra.getVector()->replaceLocalValue(num_states * k + component_id, 0, u_component_view[k]);
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Accessor functions to facilitate other classes in the MrHyDE interface
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  HDSA::Ptr<MrHyDE::ParameterManager<SolverNode>> Get_Parameter_Manager(void) const
  {
    return params_;
  }

  std::string Get_Opt_Solution_Exo_File(void) const
  {
    return opt_solution_exo_file_;
  }

  void Overwrite_Opt_Solution_Exo_File(std::string &exo_file)
  {
    opt_solution_exo_file_ = exo_file;
  }

  std::vector<std::string> Get_HiFi_Exo_Files(void) const
  {
    return hifi_exo_files_;
  }

  void Overwrite_HiFi_Exo_Files(std::vector<std::string> &exo_files)
  {
    for (int k = 0; k < exo_files.size(); k++)
    {
      hifi_exo_files_[k] = exo_files[k];
    }
  }
};
#endif
