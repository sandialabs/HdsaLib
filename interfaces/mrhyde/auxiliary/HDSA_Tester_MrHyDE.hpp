/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_TESTER_MRHYDE_HPP
#define HDSA_TESTER_MRHYDE_HPP

#include "HDSA_Comm.hpp"
#include "HDSA_MD_Bilaplacian_u_Prior_Interface.hpp"
#include "HDSA_MD_Continuation_Update.hpp"
#include "HDSA_MD_Data_Interface_MrHyDE.hpp"
#include "HDSA_MD_Hessian_Analysis.hpp"
#include "HDSA_MD_Lumped_Mass_u_Prior_Interface.hpp"
#include "HDSA_MD_Multi_State_u_Hyperparameter_Interface.hpp"
#include "HDSA_MD_Multi_State_u_Prior_Interface.hpp"
#include "HDSA_MD_Numeric_Laplacian_u_Prior_Interface.hpp"
#include "HDSA_MD_Numeric_Laplacian_z_Prior_Interface.hpp"
#include "HDSA_MD_OUU_Data_Interface_MrHyDE.hpp"
#include "HDSA_MD_OUU_Ensemble_Weighting_Matrix.hpp"
#include "HDSA_MD_OUU_Hyperparameter_Data_Interface.hpp"
#include "HDSA_MD_OUU_Opt_Prob_Interface_MrHyDE.hpp"
#include "HDSA_MD_OUU_u_Prior_Interface.hpp"
#include "HDSA_MD_Opt_Prob_Interface_MrHyDE.hpp"
#include "HDSA_MD_Posterior_Sampling.hpp"
#include "HDSA_MD_Prior_Sampling.hpp"
#include "HDSA_MD_Update.hpp"
#include "HDSA_MD_Vector_z_Prior_Interface.hpp"
#include "HDSA_MD_u_Hyperparameter_Interface_MrHyDE.hpp"
#include "HDSA_MD_u_Prior_Interface.hpp"
#include "HDSA_MD_z_Hyperparameter_Interface_MrHyDE.hpp"
#include "HDSA_MD_z_Prior_Interface.hpp"
#include "HDSA_Output_Writer_MrHyDE.hpp"
#include "HDSA_Prior_Operators_Interface_MrHyDE.hpp"
#include "HDSA_Ptr.hpp"
#include "HDSA_Random_Number_Generator.hpp"
#include "HDSA_Sparse_Matrix.hpp"
#include "HDSA_Sparse_Matrix_Trilinos.hpp"
#include "HDSA_Stream.hpp"
#include "HDSA_Vector.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

template <class RealT, class LO = Tpetra::Map<>::local_ordinal_type, class GO = Tpetra::Map<>::global_ordinal_type, class Node = Tpetra::Map<>::node_type> class Tester_MrHyDE
{

  private:
    Teuchos::RCP<MpiComm> comm_;
    Teuchos::RCP<Teuchos::ParameterList> settings_;
    Teuchos::RCP<MrHyDE::SolverManager<SolverNode>> solver_;
    Teuchos::RCP<MrHyDE::PostprocessManager<SolverNode>> postproc_;
    Teuchos::RCP<MrHyDE::ParameterManager<SolverNode>> params_;

  public:
    Tester_MrHyDE(Teuchos::RCP<MpiComm> &comm, Teuchos::RCP<Teuchos::ParameterList> &settings, Teuchos::RCP<MrHyDE::SolverManager<SolverNode>> &solver, Teuchos::RCP<MrHyDE::PostprocessManager<SolverNode>> &postproc,
                  Teuchos::RCP<MrHyDE::ParameterManager<SolverNode>> &params)
        : comm_(comm), settings_(settings), solver_(solver), postproc_(postproc), params_(params)
    {
        Teuchos::ParameterList HDSAsettings;
        if (settings_->sublist("Analysis").isSublist("HDSA"))
            HDSAsettings = settings_->sublist("Analysis").sublist("HDSA");
        else
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: MrHyDE could not find the HDSA sublist in the input file!  Abort!");

        bool check_solution_operator_z_jacobian = HDSAsettings.sublist("Tests").get<bool>("check_solution_operator_z_jacobian", false);
        bool check_prior_dirichlet_condition = HDSAsettings.sublist("Tests").get<bool>("check_prior_dirichlet_condition", false);

        if (check_solution_operator_z_jacobian)
        {
            Solution_Operator_z_Jacobian_Finite_Difference_Check(HDSAsettings);
        }
        if (check_prior_dirichlet_condition)
        {
            Prior_Dirichlet_Condition_Check(HDSAsettings);
        }
    }

    virtual ~Tester_MrHyDE()
    {
    }
   
    bool Prior_Dirichlet_Condition_Check(Teuchos::ParameterList &HDSAsettings)
    {
        bool passed = true;

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

        bool is_transient = solver_->isTransient;
        int num_states = solver_->varlist[0][0].size();
        Teuchos::ParameterList data_load_list = HDSAsettings.sublist("DataLoadParameters");
        std::string random_number_file = data_load_list.get<std::string>("random_number_file", "error");
       
        HDSA::Ptr<const HDSA::Comm<int>> hdsa_comm = HDSA::makePtr<HDSA::Comm<int>>(comm_);
        HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> random_number_generator = HDSA::makePtr<HDSA::Random_Number_Generator<RealT>>(hdsa_comm);;
      
        HDSA::Ptr<HDSA::MD_Data_Interface<RealT>> data_interface = HDSA::makePtr<MD_Data_Interface_MrHyDE<RealT>>(comm_, solver_, params_, random_number_generator, data_load_list);
        HDSA::Ptr<HDSA::MD_Opt_Prob_Interface<RealT>> opt_prob_interface= HDSA::makePtr<MD_Opt_Prob_Interface_MrHyDE<RealT>>(solver_, postproc_, params_, data_interface);

        HDSA::Ptr<HDSA::Vector<RealT>> u_opt = data_interface->Get_u_opt()->Clone();
        opt_prob_interface->State_Solve(*u_opt, *data_interface->Get_z_opt());
        HDSA::Ptr<HDSA::Vector<RealT>> u_tmp = data_interface->Get_u_opt()->Clone();
        u_tmp->Set(*u_opt);
        u_tmp->Scaled_Plus(-1.0,*data_interface->Get_u_opt());
        
        RealT tolerance = HDSAsettings.sublist("Tests").get<RealT>("load_optimal_u_check_tolerance", 1.0e-4);
        const RealT relative_error = u_tmp->Norm()/data_interface->Get_u_opt()->Norm();
        if (relative_error <= tolerance)
        {
            *outStream << "Load_Optimal_u check passed" << std::endl;
        }
        else
        {
            passed = false;
            *outStream << "Load_Optimal_u check failed" << std::endl;
            *outStream << "  relative_error = " << relative_error << std::endl;
            *outStream << "  tolerance      = " << tolerance << std::endl;
        }

        vector<string> blockNames = solver_->mesh->getBlockNames();
        HDSA::Ptr<Prior_Operators_Interface_MrHyDE<RealT>> prior_operator_interface = HDSA::makePtr<Prior_Operators_Interface_MrHyDE<RealT>>(comm_, settings_, blockNames);

        HDSA::Ptr<HDSA::Vector<RealT>> dirichlet_vec;
        if (is_transient)
        {
            HDSA::Ptr<const HDSA::Transient_Vector<RealT>> u_opt_trans = HDSA::dynamicPtrCast<const HDSA::Transient_Vector<RealT>>(data_interface->Get_u_opt());
            dirichlet_vec = (*u_opt_trans)[0]->Clone();
        }
        else
        {
            dirichlet_vec = data_interface->Get_u_opt()->Clone();
        }
        prior_operator_interface->Instantiate_Prior_Dirichlet_Operator(solver_, dirichlet_vec);

        for(int k = 0; k < num_states; k++)
        {
            HDSA::Ptr<HDSA::Vector<RealT>> dvk = data_interface->Extract_State_Component(*dirichlet_vec, k)->Clone();
            dvk->Set(*data_interface->Extract_State_Component(*dirichlet_vec, k));

            HDSA::Ptr<const HDSA::Tpetra_Vector<RealT>> dvk_tpetra = HDSA::dynamicPtrCast<const HDSA::Tpetra_Vector<RealT>>(dvk);
            HDSA::Ptr<const Tpetra::Map<LO,GO,Node>> vec_map = dvk_tpetra->getVector()->getMap();
            HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> Mk = HDSA::makePtr<HDSA::Sparse_Matrix_Trilinos<RealT>>(prior_operator_interface->M,vec_map);

            if (dvk->Norm() > 0.0)
            {
                *outStream << "Dirichlet vector passed" << std::endl;
            }
            else
            {
                passed = false;
                *outStream << "Dirichlet vector failed" << std::endl;
                *outStream << "The Dirichlet index vector is zero" << std::endl;
            }

            HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> Dk = Mk->Clone(1);
            Dk->Set_Diagonal(*dvk, false);

            HDSA::Ptr<HDSA::Vector<RealT>> tmp_in = dvk->Clone();
            tmp_in->Set_Scalar(1.0);
            tmp_in->Scaled_Plus(-1.0,*dvk);
            HDSA::Ptr<HDSA::Vector<RealT>> tmp_out = dvk->Clone();
            Dk->Apply(*tmp_out, *tmp_in);

            if (tmp_out->Norm() == 0.0)
            {
                *outStream << "Dirichlet matrix test 1 passed" << std::endl;
            }
            else
            {
                passed = false;
                *outStream << "Dirichlet matrix test 1 failed" << std::endl;
                *outStream << "The matrix Dk does not impose the Dirichlet condition" << std::endl;
            }

            tmp_in->Set(*data_interface->Extract_State_Component(*u_opt,k));
            tmp_out->Zeros();
            Dk->Apply(*tmp_out, *tmp_in);

            if (tmp_out->Norm() == 0.0)
            {
                *outStream << "Dirichlet matrix test 2 passed" << std::endl;
            }
            else
            {
                passed = false;
                *outStream << "Dirichlet matrix test 2 failed" << std::endl;
                *outStream << "The matrix Dk does not impose the Dirichlet condition" << std::endl;
            }

        }

        if (comm_->getRank() == 0)
        {
            try
            {
                std::filesystem::create_directory("test_vectors");
            }
            catch (const std::exception &e)
            {
            }
        }
        comm_->barrier();
        HDSA::Ptr<const HDSA::Vector<RealT>> u = data_interface->Get_u_opt();
        u->Write_to_File("test_vectors/u.txt");
        dirichlet_vec->Write_to_File("test_vectors/dv.txt");
        std::vector<HDSA::Ptr<const HDSA::Vector<RealT>>> uk, dirichlet_veck;
        uk.resize(num_states);
        dirichlet_veck.resize(num_states);
        for(int k = 0; k < num_states; k++)
        {
            uk[k] = data_interface->Extract_State_Component(*u,k);
            uk[k]->Write_to_File("test_vectors/u_"+std::to_string(k)+".txt");
            dirichlet_veck[k] = data_interface->Extract_State_Component(*dirichlet_vec,k);
            dirichlet_veck[k]->Write_to_File("test_vectors/dv_"+std::to_string(k)+".txt");
        }

        return passed;
    }

    bool Solution_Operator_z_Jacobian_Finite_Difference_Check(Teuchos::ParameterList &HDSAsettings)
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

        RealT h = HDSAsettings.sublist("Tests").get<RealT>("solution_operator_z_jacobian_check_step", 1.0e-6);
        RealT tolerance = HDSAsettings.sublist("Tests").get<RealT>("solution_operator_z_jacobian_check_tolerance", 1.0e-4);
        TEUCHOS_TEST_FOR_EXCEPTION(h <= static_cast<RealT>(0), std::logic_error, "Error: solution_operator_z_jacobian_check_step must be positive.");
        TEUCHOS_TEST_FOR_EXCEPTION(tolerance <= static_cast<RealT>(0), std::logic_error, "Error: solution_operator_z_jacobian_check_tolerance must be positive.");

        HDSA::Ptr<const HDSA::Comm<int>> hdsa_comm = HDSA::makePtr<HDSA::Comm<int>>(comm_);
        HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> random_number_generator = HDSA::makePtr<HDSA::Random_Number_Generator<RealT>>(hdsa_comm);
        Teuchos::ParameterList data_load_list = HDSAsettings.sublist("DataLoadParameters");
        HDSA::Ptr<HDSA::MD_Data_Interface<RealT>> data_interface = HDSA::makePtr<MD_Data_Interface_MrHyDE<RealT>>(comm_, solver_, params_, random_number_generator, data_load_list);
        HDSA::Ptr<HDSA::MD_Opt_Prob_Interface<RealT>> opt_prob_interface = HDSA::makePtr<MD_Opt_Prob_Interface_MrHyDE<RealT>>(solver_, postproc_, params_, data_interface);

        HDSA::Ptr<const HDSA::Vector<RealT>> z = data_interface->Get_z_opt();
        HDSA::Ptr<const HDSA::Vector<RealT>> u_shape = data_interface->Get_u_opt();

        HDSA::Ptr<HDSA::Vector<RealT>> u_base = u_shape->Clone();
        opt_prob_interface->State_Solve(*u_base, *z);

        HDSA::Ptr<HDSA::Vector<RealT>> z_direction = z->Clone();
        z_direction->Set_Scalar(1.0);

        HDSA::Ptr<HDSA::Vector<RealT>> u_jacobian = u_shape->Clone();
        opt_prob_interface->Apply_Solution_Operator_z_Jacobian(*u_jacobian, *z_direction, *z);

        HDSA::Ptr<HDSA::Vector<RealT>> z_plus = z->Clone();
        HDSA::Ptr<HDSA::Vector<RealT>> z_minus = z->Clone();
        z_plus->Set(*z);
        z_minus->Set(*z);
        z_plus->Scaled_Plus(h, *z_direction);
        z_minus->Scaled_Plus(-h, *z_direction);

        HDSA::Ptr<HDSA::Vector<RealT>> u_plus = u_shape->Clone();
        HDSA::Ptr<HDSA::Vector<RealT>> u_minus = u_shape->Clone();
        opt_prob_interface->State_Solve(*u_plus, *z_plus);
        opt_prob_interface->State_Solve(*u_minus, *z_minus);

        HDSA::Ptr<HDSA::Vector<RealT>> u_fd = u_shape->Clone();
        u_fd->Set(*u_plus);
        u_fd->Scaled_Plus(-1.0, *u_minus);
        u_fd->Scale(1.0 / (2.0 * h));

        HDSA::Ptr<HDSA::Vector<RealT>> error = u_shape->Clone();
        error->Set(*u_jacobian);
        error->Scaled_Plus(-1.0, *u_fd);

        const RealT fd_norm = u_fd->Norm();
        const RealT error_norm = error->Norm();
        const RealT relative_error = error_norm / std::max(static_cast<RealT>(1.0), fd_norm);

        bool passed = true;
        if (relative_error <= tolerance)
        {
            *outStream << "Apply_Solution_Operator_z_Jacobian finite-difference check passed" << std::endl;
        }
        else
        {
            passed = false;
            *outStream << "Apply_Solution_Operator_z_Jacobian finite-difference check failed" << std::endl;
            *outStream << "  relative error = " << relative_error << std::endl;
            *outStream << "  tolerance      = " << tolerance << std::endl;
        }
        return passed;
    }

};
#endif
