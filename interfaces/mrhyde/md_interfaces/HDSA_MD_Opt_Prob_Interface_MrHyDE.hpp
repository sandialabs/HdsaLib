/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_OPT_PROB_INTERFACE_MRHYDE_HPP
#define HDSA_MD_OPT_PROB_INTERFACE_MRHYDE_HPP

#include "HDSA_MD_Opt_Prob_Interface.hpp"
#include "HDSA_Tpetra_Vector.hpp"
#include "HDSA_Std_Vector.hpp"
#include "HDSA_Solver_Interface_MrHyDE.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

template <class RealT>
class MD_Opt_Prob_Interface_MrHyDE : public HDSA::MD_Opt_Prob_Interface<RealT>
{

private:

  HDSA::Ptr<MrHyDE::SolverManager<SolverNode>> solver_;
  HDSA::Ptr<MrHyDE::PostprocessManager<SolverNode>> postproc_;
  HDSA::Ptr<MrHyDE::ParameterManager<SolverNode>> params_;
  
  HDSA::Ptr<Solver_Interface_MrHyDE<RealT>> solver_interface_;
  HDSA::Ptr<HDSA::Vector<RealT>> z_base_;
  HDSA::Ptr<HDSA::Vector<RealT>> grad_base_;

public:
  MD_Opt_Prob_Interface_MrHyDE(HDSA::Ptr<MrHyDE::SolverManager<SolverNode>> &solver, HDSA::Ptr<MrHyDE::PostprocessManager<SolverNode>> &postproc, HDSA::Ptr<MrHyDE::ParameterManager<SolverNode>> &params, const HDSA::Ptr<HDSA::MD_Data_Interface<RealT>> &data_interface)
  {
    postproc_ = postproc;
    solver_ = solver;
    params_ = params;

    solver_interface_ = HDSA::makePtr<Solver_Interface_MrHyDE<RealT>>(solver_, params_);

    postproc_->hdsa_solop_data.resize(solver_->setnames.size());
    for (int set = 0; set < solver_->setnames.size(); set++)
    {
      postproc_->hdsa_solop_data[set] = HDSA::makePtr<MrHyDE::SolutionStorage<SolverNode>>(solver_->settings);
    }

    z_base_ = data_interface->Get_z_opt()->Clone();
    z_base_->Set(*data_interface->Get_z_opt());
    grad_base_ = data_interface->Get_z_opt()->Clone();
    RS_Gradient(*grad_base_, *z_base_);
  }

  virtual ~MD_Opt_Prob_Interface_MrHyDE()
  {
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Implementation of base class pure virtual functions:
  // Apply_Solution_Operator_z_Jacobian_Transpose
  // Apply_RS_Hessian
  // Misfit_Gradient
  // Apply_Misfit_Hessian
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  void Apply_Solution_Operator_z_Jacobian_Transpose(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &u_in, const HDSA::Vector<RealT> &z) const override
  {
    Write_Data_Solution_Operator(u_in);
    Do_Solution_Operator(true);
    RS_Gradient(z_out, z);
    z_out.Scale(-1.0);
  }

  void Apply_RS_Hessian(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in, const HDSA::Vector<RealT> &z) const override
  {
    Do_Solution_Operator(false);
    Do_Base_Update(z);
    HDSA::Ptr<HDSA::Vector<RealT>> z_pert = z.Clone();
    z_pert->Set(z);
    RealT h = 1.e-4;
    z_pert->Scaled_Plus(h, z_in);

    RS_Gradient(z_out, *z_pert);
    z_out.Scaled_Plus(-1.0, *grad_base_);
    z_out.Scale(1.0 / h);
  }

  void Misfit_Gradient(HDSA::Vector<RealT> &u_grad, const HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &z) const override
  {
    Do_Solution_Operator(false);
    if (solver_->isTransient)
    {
      const HDSA::Transient_Vector<RealT> &eu = dynamic_cast<const HDSA::Transient_Vector<RealT> &>(u);
      HDSA::Transient_Vector<RealT> &eu_grad = dynamic_cast<HDSA::Transient_Vector<RealT> &>(u_grad);
      int n_t = solver_->settings->sublist("Solver").get<int>("number of steps", 0) + 1;

      if (postproc_->objectives[0].type == "integrated control")
      { // only works for one objective term
        eu_grad[0]->Zeros();
        for (int i = 0; i < n_t - 1; i++)
        { // exludes initial condition
          const HDSA::Tpetra_Vector<RealT> &eu_i = dynamic_cast<const HDSA::Tpetra_Vector<RealT> &>(*eu[i + 1]);
          HDSA::Tpetra_Vector<RealT> &eu_grad_i = dynamic_cast<HDSA::Tpetra_Vector<RealT> &>(*eu_grad[i + 1]);
          RealT currenttime = solver_->initial_time + (double)i * solver_->deltat;

          // the gradient should be a non-overlapping vector, but the state should be an overlapping vector
          HDSA::Ptr<Tpetra::MultiVector<RealT>> eu_grad_i_tpetra = eu_grad_i.getVector();
          HDSA::Ptr<Tpetra::MultiVector<RealT>> ui_over = solver_->linalg->getNewOverlappedVector(0);
          solver_->linalg->importVectorToOverlapped(0, ui_over, eu_i.getVector());

          postproc_->setTimeIndex(i);
          solver_->assembler->updateStage(0, currenttime, solver_->deltat);
          postproc_->computeObjectiveGradState(0, ui_over, currenttime, solver_->deltat, eu_grad_i_tpetra);
          if (i == 0)
          {
            eu_grad_i.Scale(solver_->deltat);
          }
        }
        u_grad.Scale(-1.0);
      }
    }
    else
    {
      const HDSA::Tpetra_Vector<RealT> &eu = dynamic_cast<const HDSA::Tpetra_Vector<RealT> &>(u);
      HDSA::Tpetra_Vector<RealT> &eu_grad = dynamic_cast<HDSA::Tpetra_Vector<RealT> &>(u_grad);

      // the gradient should be a non-overlapping vector, but the state should be an overlapping vector
      HDSA::Ptr<Tpetra::MultiVector<RealT>> grad = eu_grad.getVector();
      HDSA::Ptr<Tpetra::MultiVector<RealT>> u_over = solver_->linalg->getNewOverlappedVector(0);
      solver_->linalg->importVectorToOverlapped(0, u_over, eu.getVector());
      postproc_->computeObjectiveGradState(0, u_over, 0.0, solver_->deltat, grad);

      u_grad.Scale(-1.0);
    }
  }

  void Apply_Misfit_Hessian(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &u_in, const HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &z) const override
  {
    HDSA::Ptr<HDSA::Vector<RealT>> ugrad_nom = u_out.Clone();
    Misfit_Gradient(*ugrad_nom, u, z);

    HDSA::Ptr<HDSA::Vector<RealT>> u_pert = u_out.Clone();
    u_pert->Set(u);
    RealT h = 1.0e-4;
    u_pert->Scaled_Plus(h, u_in);
    Misfit_Gradient(u_out, *u_pert, z);

    u_out.Scaled_Plus(-1.0, *ugrad_nom);
    u_out.Scale(1.0 / h);
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Functions needed by continuation
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  void Apply_Solution_Operator_z_Jacobian(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &z_in, const HDSA::Vector<RealT> &z) const override
  {
    Do_Solution_Operator(false);
    u_out.Zeros();

    Ensure_Current_Params_And_State(z);

    if (solver_->isTransient)
    {
      Apply_Solution_Operator_z_Jacobian_Transient(u_out, z_in, z);
    }
    else
    {
      HDSA::Tpetra_Vector<RealT> &eu_out = dynamic_cast<HDSA::Tpetra_Vector<RealT> &>(u_out);
      Apply_Solution_Operator_z_Jacobian_Steady(eu_out.getVector(), z_in, 0.0, 0);
    }
  }

  void State_Solve(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &z) const override
  {
    Do_Solution_Operator(false);
    solver_interface_->State_Solve(u_out, z);
  }

  void Regularization_Gradient(HDSA::Vector<RealT> &grad_z, const HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &z) const override
  {
    Do_Solution_Operator(false);
    grad_z.Zeros();

    if (solver_->isTransient)
    {
      const HDSA::Transient_Vector<RealT> &eu = dynamic_cast<const HDSA::Transient_Vector<RealT> &>(u);
      const int n_t = eu.Get_n_t();

      if (HDSA::Transient_Vector<RealT> *egrad_z = dynamic_cast<HDSA::Transient_Vector<RealT> *>(&grad_z))
      {
        for (int i = 0; i < n_t - 1; i++)
        {
          params_->updateDynamicParams(i);
          postproc_->setTimeIndex(i);

          std::vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>>> current_soln;
          current_soln.push_back(Overlapped_State_Vector(*eu[i + 1], 0));

          DFAD obj_sens = 0.0;
          const RealT currenttime = solver_->initial_time + (RealT)i * solver_->deltat;
          postproc_->computeObjectiveGradParam(current_soln, currenttime, solver_->deltat, obj_sens);
          Copy_Objective_Param_Gradient(*(*egrad_z)[i], obj_sens);
        }
      }
      else
      {
        HDSA::Ptr<HDSA::Vector<RealT>> grad_step = grad_z.Clone();
        for (int i = 0; i < n_t - 1; i++)
        {
          params_->updateDynamicParams(i);
          postproc_->setTimeIndex(i);

          std::vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>>> current_soln;
          current_soln.push_back(Overlapped_State_Vector(*eu[i + 1], 0));

          DFAD obj_sens = 0.0;
          const RealT currenttime = solver_->initial_time + (RealT)i * solver_->deltat;
          postproc_->computeObjectiveGradParam(current_soln, currenttime, solver_->deltat, obj_sens);
          grad_step->Zeros();
          Copy_Objective_Param_Gradient(*grad_step, obj_sens);
          grad_z.Scaled_Plus(1.0, *grad_step);
        }
      }
    }
    else
    {
      std::vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>>> current_soln;
      current_soln.push_back(Overlapped_State_Vector(u, 0));

      DFAD obj_sens = 0.0;
      postproc_->computeObjectiveGradParam(current_soln, 0.0, solver_->deltat, obj_sens);
      Copy_Objective_Param_Gradient(grad_z, obj_sens);
    }
  }

  void Apply_Solution_Operator_z_Hessian_Adjoint(HDSA::Vector<RealT> &z_out, const HDSA::Vector<RealT> &z_in, const HDSA::Vector<RealT> &u_adj, const HDSA::Vector<RealT> &z) const override
  {
    // MrHyDE exposes the first derivative needed by
    // Apply_Solution_Operator_z_Jacobian_Transpose, but not the mixed second
    // residual derivatives needed for an exact S_zz(z)^* apply.  Use a finite
    // difference of the first-derivative adjoint apply for this second-order term.
    HDSA::Ptr<HDSA::Vector<RealT>> z_base_apply = z_out.Clone();
    HDSA::Ptr<HDSA::Vector<RealT>> z_pert = z.Clone();

    const RealT z_norm = z.Norm();
    const RealT dz_norm = z_in.Norm();
    if (dz_norm == 0.0)
    {
      z_out.Zeros();
      return;
    }

    const RealT h = 1.0e-5 * (1.0 + z_norm) / dz_norm;
    z_pert->Set(z);
    z_pert->Scaled_Plus(h, z_in);

    Apply_Solution_Operator_z_Jacobian_Transpose(*z_base_apply, u_adj, z);
    Apply_Solution_Operator_z_Jacobian_Transpose(z_out, u_adj, *z_pert);

    z_out.Scaled_Plus(-1.0, *z_base_apply);
    z_out.Scale(1.0 / h);
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Functions that aid in the implementation of the base class pure virtual function
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  void Write_Data_Solution_Operator(const HDSA::Vector<RealT> &u) const
  {
    if (solver_->isTransient)
    {
      const HDSA::Transient_Vector<RealT> &eu = dynamic_cast<const HDSA::Transient_Vector<RealT> &>(u);
      int n_t = solver_->settings->sublist("Solver").get<int>("number of steps", 0) + 1;
      for (int i = 0; i < n_t; i++)
      {
        const HDSA::Tpetra_Vector<RealT> &eu_i = dynamic_cast<const HDSA::Tpetra_Vector<RealT> &>(*eu[i]);
        RealT currenttime = solver_->initial_time + (double)i * solver_->deltat;
        postproc_->hdsa_solop_data[0]->store(eu_i.getVector(), currenttime, 0);
      }
    }
    else
    {
      const HDSA::Tpetra_Vector<RealT> &eu = dynamic_cast<const HDSA::Tpetra_Vector<RealT> &>(u);
      postproc_->hdsa_solop_data[0]->store(eu.getVector(), 0.0, 0);
    }
  }

  void Do_Solution_Operator(bool solop_flag) const
  {
    postproc_->hdsa_solop = solop_flag;
  }

  void Do_Base_Update(const HDSA::Vector<RealT> &z) const
  {
    HDSA::Ptr<HDSA::Vector<RealT>> z_tmp = z.Clone();
    z_tmp->Set(z);
    z_tmp->Scaled_Plus(-1.0, *z_base_);
    RealT diff = z_tmp->Norm();
    if(diff > 1.e-13)
    {
      z_base_->Set(z);
      RS_Gradient(*grad_base_, *z_base_);
    }
  }

  void RS_Gradient(HDSA::Vector<RealT> &grad_z, const HDSA::Vector<RealT> &z) const
  {
    bool new_z = Check_New_Params(z);
    if (new_z)
    {
      HDSA::Ptr<MrHyDE_OptVector> z_rol = solver_interface_->Map_HDSA_Vector_to_MrHyDE_OptVector(z);
      MrHyDE_OptVector curr_z = params_->getCurrentVector();
      ROL::Ptr<ROL::Vector<RealT>> z_tmp = curr_z.clone();
      MrHyDE_OptVector ez_tmp = Teuchos::dyn_cast<MrHyDE_OptVector>(dynamic_cast<ROL::Vector<RealT> &>(*z_tmp));
      ez_tmp.set(*z_rol);

      params_->updateParams(ez_tmp);
      ScalarT val = 0.0;
      solver_->forwardModel(val);
    }

    HDSA::Ptr<MrHyDE_OptVector> grad_z_rol = solver_interface_->Map_HDSA_Vector_to_MrHyDE_OptVector(grad_z);
    grad_z_rol->zero();

    solver_->adjointModel(*grad_z_rol);
  }

  bool Check_New_Params(const HDSA::Vector<RealT> &z) const
  {
    HDSA::Ptr<MrHyDE_OptVector> z_rol = solver_interface_->Map_HDSA_Vector_to_MrHyDE_OptVector(z);
    MrHyDE_OptVector curr_z = params_->getCurrentVector();

    ROL::Ptr<ROL::Vector<RealT>> diff = curr_z.clone();
    diff->zero();
    diff->set(curr_z);
    diff->axpy(-1.0, *z_rol);

    ScalarT dnorm = diff->norm();
    ScalarT refnorm = curr_z.norm();
    if (refnorm > 0.0)
    {
      dnorm = dnorm / refnorm;
    }
    ScalarT reltol = 1.0e-12;

    bool new_z = false;
    if (dnorm > reltol)
    {
      new_z = true;
    }
    return new_z;
  }

  void Ensure_Current_Params_And_State(const HDSA::Vector<RealT> &z) const
  {
    bool new_z = Check_New_Params(z);
    if (new_z)
    {
      HDSA::Ptr<MrHyDE_OptVector> z_rol = solver_interface_->Map_HDSA_Vector_to_MrHyDE_OptVector(z);
      params_->updateParams(*z_rol);
      ScalarT val = 0.0;
      solver_->forwardModel(val);
    }
  }

  Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> Owned_State_Vector(const HDSA::Vector<RealT> &u, const size_t set) const
  {
    const HDSA::Tpetra_Vector<RealT> &eu = dynamic_cast<const HDSA::Tpetra_Vector<RealT> &>(u);
    Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> u_owned = solver_->linalg->getNewVector(set);
    u_owned->update(1.0, *eu.getVector(), 0.0);
    return u_owned;
  }

  Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> Overlapped_State_Vector(const HDSA::Vector<RealT> &u, const size_t set) const
  {
    Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> u_over = solver_->linalg->getNewOverlappedVector(set);
    Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> u_owned = Owned_State_Vector(u, set);
    solver_->linalg->importVectorToOverlapped(set, u_over, u_owned);
    return u_over;
  }

  Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> Extract_Stored_State(const size_t set, const size_t timeindex) const
  {
    Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> u_vec;
    bool found = postproc_->soln[set]->extract(u_vec, timeindex);
    TEUCHOS_TEST_FOR_EXCEPTION(!found, std::runtime_error, "Error in HDSA MrHyDE interface: unable to find stored forward solution.");
    return u_vec;
  }

  Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> Extract_Stored_State_Overlapped(const size_t set, const size_t timeindex) const
  {
    return Extract_Stored_State(set, timeindex);
  }

  Teuchos::RCP<Tpetra::CrsMatrix<ScalarT, LO, GO, SolverNode>> Assemble_State_Jacobian(const size_t set,
                                     const size_t stage,
                                     std::vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>>> &sol,
                                     std::vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>>> &sol_stage,
                                     std::vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>>> &sol_prev,
                                     const bool isTD,
                                     const ScalarT current_time,
                                     const bool is_final_time) const
  {
    std::vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>>> zero_vec;
    Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> res_over = solver_->linalg->getNewOverlappedVector(set);
    Teuchos::RCP<Tpetra::CrsMatrix<ScalarT, LO, GO, SolverNode>> J = solver_->linalg->getNewMatrix(set);
    Teuchos::RCP<Tpetra::CrsMatrix<ScalarT, LO, GO, SolverNode>> J_over = solver_->linalg->getNewOverlappedMatrix(set);

    solver_->linalg->fillComplete(J_over);
    J_over->resumeFill();
    J_over->setAllToScalar(0.0);
    res_over->putScalar(0.0);

    auto paramvec = params_->getDiscretizedParamsOver();
    auto paramdot = params_->getDiscretizedParamsDotOver();
    solver_->assembler->assembleJacRes(set, stage, sol, sol_stage, sol_prev, zero_vec, zero_vec, zero_vec,
                                       true, false, false, false, 0, res_over, J_over, isTD, current_time,
                                       false, false, params_->num_active_params, paramvec, paramdot,
                                       is_final_time, solver_->deltat);

    solver_->linalg->fillComplete(J_over);
    J->resumeFill();
    solver_->linalg->exportMatrixFromOverlapped(set, J, J_over);
    solver_->linalg->fillComplete(J);
    return J;
  }

  Teuchos::RCP<Tpetra::CrsMatrix<ScalarT, LO, GO, SolverNode>> Assemble_Previous_State_Jacobian(const size_t set,
                                              const size_t stage,
                                              std::vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>>> &sol,
                                              std::vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>>> &sol_stage,
                                              std::vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>>> &sol_prev,
                                              const size_t previous_step,
                                              const ScalarT current_time,
                                              const bool is_final_time) const
  {
    std::vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>>> zero_vec;
    Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> res_over = solver_->linalg->getNewOverlappedVector(set);
    std::vector<Teuchos::RCP<Tpetra::CrsMatrix<ScalarT, LO, GO, SolverNode>>> Jprev = solver_->linalg->getNewPreviousMatrix(set, previous_step + 1);
    Teuchos::RCP<Tpetra::CrsMatrix<ScalarT, LO, GO, SolverNode>> J = Jprev[previous_step];
    Teuchos::RCP<Tpetra::CrsMatrix<ScalarT, LO, GO, SolverNode>> J_over = solver_->linalg->getNewOverlappedMatrix(set);

    solver_->linalg->fillComplete(J_over);
    J_over->resumeFill();
    J_over->setAllToScalar(0.0);
    res_over->putScalar(0.0);

    auto paramvec = params_->getDiscretizedParamsOver();
    auto paramdot = params_->getDiscretizedParamsDotOver();
    solver_->assembler->assembleJacRes(set, stage, sol, sol_stage, sol_prev, zero_vec, zero_vec, zero_vec,
                                       true, false, false, true, previous_step, res_over, J_over, true,
                                       current_time, false, false, params_->num_active_params, paramvec,
                                       paramdot, is_final_time, solver_->deltat);

    solver_->linalg->fillComplete(J_over);
    J->resumeFill();
    solver_->linalg->exportMatrixFromOverlapped(set, J, J_over);
    solver_->linalg->fillComplete(J);
    return J;
  }

  void Apply_Residual_z_Derivative(Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> &rhs,
                                   const HDSA::Vector<RealT> &z_in,
                                   const size_t set,
                                   const size_t stage,
                                   std::vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>>> &sol,
                                   std::vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>>> &sol_stage,
                                   std::vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>>> &sol_prev,
                                   const bool isTD,
                                   const ScalarT current_time,
                                   const bool is_final_time) const
  {
    rhs->putScalar(0.0);

    if (const HDSA::Std_Vector<RealT> *ez_in = dynamic_cast<const HDSA::Std_Vector<RealT> *>(&z_in))
    {
      Apply_Residual_Active_Param_Derivative(rhs, *ez_in, set, stage, sol, sol_stage, sol_prev, isTD, current_time, is_final_time);
    }
    else if (const HDSA::Tpetra_Vector<RealT> *ez_in = dynamic_cast<const HDSA::Tpetra_Vector<RealT> *>(&z_in))
    {
      Apply_Residual_Discretized_Param_Derivative(rhs, *ez_in, set, stage, sol, sol_stage, sol_prev, isTD, current_time, is_final_time);
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                                  "Error in HDSA MrHyDE interface: unsupported parameter-direction vector type.");
    }
  }

  void Apply_Residual_Active_Param_Derivative(Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> &rhs,
                                              const HDSA::Std_Vector<RealT> &z_in,
                                              const size_t set,
                                              const size_t stage,
                                              std::vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>>> &sol,
                                              std::vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>>> &sol_stage,
                                              std::vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>>> &sol_prev,
                                              const bool isTD,
                                              const ScalarT current_time,
                                              const bool is_final_time) const
  {
    TEUCHOS_TEST_FOR_EXCEPTION(z_in.Dimension() > static_cast<int>(params_->num_active_params), std::runtime_error,
                               "Error in HDSA MrHyDE interface: scalar direction has more entries than active parameters.");
    if (params_->num_active_params == 0)
    {
      return;
    }

    params_->sacadoizeParams(true);

    std::vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>>> zero_vec;
    Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> res = solver_->linalg->getNewVector(set, params_->num_active_params);
    Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> res_over = solver_->linalg->getNewOverlappedVector(set, params_->num_active_params);
    Teuchos::RCP<Tpetra::CrsMatrix<ScalarT, LO, GO, SolverNode>> J_over = solver_->linalg->getNewOverlappedMatrix(set);
    res_over->putScalar(0.0);

    auto paramvec = params_->getDiscretizedParamsOver();
    auto paramdot = params_->getDiscretizedParamsDotOver();
    solver_->assembler->assembleJacRes(set, stage, sol, sol_stage, sol_prev, zero_vec, zero_vec, zero_vec,
                                       false, true, false, false, 0, res_over, J_over, isTD, current_time,
                                       false, false, params_->num_active_params, paramvec, paramdot,
                                       is_final_time, solver_->deltat);

    solver_->linalg->exportVectorFromOverlapped(set, res, res_over);

    Teuchos::ArrayRCP<ScalarT> rhs_data = rhs->getDataNonConst(0);
    for (int p = 0; p < z_in.Dimension(); p++)
    {
      Teuchos::ArrayRCP<const ScalarT> res_data = res->getData(p);
      const ScalarT alpha = z_in.Get_Entry(p);
      for (size_t i = 0; i < rhs_data.size(); i++)
      {
        // assembleJacRes stores active-parameter sensitivities in the
        // residual multivector with the same sign convention as the nonlinear
        // residual used by MrHyDE's Newton solve, i.e. -F_p.  The tangent solve
        // below forms F_u du_tmp = F_p dz and then returns -du_tmp, so undo
        // that sign here.
        rhs_data[i] -= alpha * res_data[i];
      }
    }

    params_->sacadoizeParams(false);
  }

  void Apply_Residual_Discretized_Param_Derivative(Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> &rhs,
                                                   const HDSA::Tpetra_Vector<RealT> &z_in,
                                                   const size_t set,
                                                   const size_t stage,
                                                   std::vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>>> &sol,
                                                   std::vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>>> &sol_stage,
                                                   std::vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>>> &sol_prev,
                                                   const bool isTD,
                                                   const ScalarT current_time,
                                                   const bool is_final_time) const
  {
    params_->sacadoizeParams(false);

    std::vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>>> zero_vec;
    Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> res_over = solver_->linalg->getNewOverlappedVector(set);
    Teuchos::RCP<Tpetra::CrsMatrix<ScalarT, LO, GO, SolverNode>> J = solver_->linalg->getNewParamStateMatrix(set);
    Teuchos::RCP<Tpetra::CrsMatrix<ScalarT, LO, GO, SolverNode>> J_over = solver_->linalg->getNewOverlappedParamStateMatrix(set);

    J->setAllToScalar(0.0);
    J_over->setAllToScalar(0.0);
    res_over->putScalar(0.0);

    auto paramvec = params_->getDiscretizedParamsOver();
    auto paramdot = params_->getDiscretizedParamsDotOver();
    solver_->assembler->assembleJacRes(set, stage, sol, sol_stage, sol_prev, zero_vec, zero_vec, zero_vec,
                                       true, false, true, false, 0, res_over, J_over, isTD, current_time,
                                       false, false, params_->num_active_params, paramvec, paramdot,
                                       is_final_time, solver_->deltat);

    solver_->linalg->fillCompleteParamState(set, J_over);
    solver_->linalg->exportParamStateMatrixFromOverlapped(set, J, J_over);
    solver_->linalg->fillCompleteParamState(set, J);

    J->apply(*z_in.getVector(), *rhs, Teuchos::TRANS);
  }

  void Apply_Solution_Operator_z_Jacobian_Steady(const Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> &u_out, const HDSA::Vector<RealT> &z_in,
                                                 const ScalarT current_time, const size_t timeindex) const
  {
    const size_t set = 0;
    const size_t stage = 0;

    params_->updateDynamicParams(timeindex);
    solver_->assembler->updatePhysicsSet(set);
    solver_->assembler->updateStage(stage, current_time, solver_->deltat);

    std::vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>>> sol, sol_stage, sol_prev;
    for (size_t iset = 0; iset < solver_->setnames.size(); iset++)
    {
      sol.push_back(Extract_Stored_State_Overlapped(iset, timeindex));
    }

    Teuchos::RCP<Tpetra::CrsMatrix<ScalarT, LO, GO, SolverNode>> J = Assemble_State_Jacobian(set, stage, sol, sol_stage, sol_prev, false, current_time, true);
    Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> rhs = solver_->linalg->getNewVector(set);
    Apply_Residual_z_Derivative(rhs, z_in, set, stage, sol, sol_stage, sol_prev, false, current_time, true);

    Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> du = solver_->linalg->getNewVector(set);
    du->putScalar(0.0);
    solver_->linalg->linearSolver(set, J, rhs, du);

    u_out->update(-1.0, *du, 0.0);
  }

  void Apply_Solution_Operator_z_Jacobian_Transient(HDSA::Vector<RealT> &u_out, const HDSA::Vector<RealT> &z_in, const HDSA::Vector<RealT> &z) const
  {
    TEUCHOS_TEST_FOR_EXCEPTION(solver_->setnames.size() != 1, std::runtime_error,
                               "Error in HDSA MrHyDE interface: transient tangent solves are implemented only for one physics set.");
    TEUCHOS_TEST_FOR_EXCEPTION(solver_->maxnumstages[0] != 1, std::runtime_error,
                               "Error in HDSA MrHyDE interface: transient tangent solves are implemented only for one-stage time integrators.");

    HDSA::Transient_Vector<RealT> &eu_out = dynamic_cast<HDSA::Transient_Vector<RealT> &>(u_out);
    const HDSA::Transient_Vector<RealT> *ez_in_trans = dynamic_cast<const HDSA::Transient_Vector<RealT> *>(&z_in);

    const size_t set = 0;
    const size_t stage = 0;
    const int n_t = eu_out.Get_n_t();
    const size_t nprev = solver_->maxnumsteps[set];

    eu_out[0]->Zeros();

    for (int step = 0; step < n_t - 1; step++)
    {
      const ScalarT current_time = solver_->initial_time + (ScalarT)step * solver_->deltat;
      const bool is_final_time = (step == n_t - 2);

      params_->updateDynamicParams(step);
      solver_->assembler->updateTimeStep(step);
      solver_->assembler->updatePhysicsSet(set);
      solver_->assembler->updateStage(stage, current_time, solver_->deltat);

      std::vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>>> sol, sol_stage, sol_prev;
      sol.push_back(Extract_Stored_State_Overlapped(set, step + 1));
      sol_stage.push_back(sol[0]);
      for (size_t p = 0; p < nprev; p++)
      {
        const int state_index = step - static_cast<int>(p);
        sol_prev.push_back(Extract_Stored_State_Overlapped(set, state_index >= 0 ? state_index : 0));
      }

      Teuchos::RCP<Tpetra::CrsMatrix<ScalarT, LO, GO, SolverNode>> J = Assemble_State_Jacobian(set, stage, sol, sol_stage, sol_prev, true, current_time, is_final_time);
      Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> rhs = solver_->linalg->getNewVector(set);

      const HDSA::Vector<RealT> *z_dir = &z_in;
      if (ez_in_trans != NULL)
      {
        z_dir = &(*((*ez_in_trans)[step]));
      }
      Apply_Residual_z_Derivative(rhs, *z_dir, set, stage, sol, sol_stage, sol_prev, true, current_time, is_final_time);

      for (size_t p = 0; p < nprev; p++)
      {
        const int tangent_index = step - static_cast<int>(p);
        if (tangent_index >= 0)
        {
          Teuchos::RCP<Tpetra::CrsMatrix<ScalarT, LO, GO, SolverNode>> Jprev = Assemble_Previous_State_Jacobian(set, stage, sol, sol_stage, sol_prev, p, current_time, is_final_time);
          Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> du_prev_owned = Owned_State_Vector(*eu_out[tangent_index], set);
          Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> mvprod = solver_->linalg->getNewVector(set);
          Jprev->apply(*du_prev_owned, *mvprod);
          rhs->update(1.0, *mvprod, 1.0);
        }
      }

      Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> du = solver_->linalg->getNewVector(set);
      du->putScalar(0.0);
      solver_->linalg->linearSolver(set, J, rhs, du);

      HDSA::Tpetra_Vector<RealT> &eu_out_step = dynamic_cast<HDSA::Tpetra_Vector<RealT> &>(*eu_out[step + 1]);
      eu_out_step.getVector()->update(-1.0, *du, 0.0);
    }
  }

  void Copy_Objective_Param_Gradient(HDSA::Vector<RealT> &grad_z, const DFAD &obj_sens) const
  {
    if (HDSA::Std_Vector<RealT> *egrad_z = dynamic_cast<HDSA::Std_Vector<RealT> *>(&grad_z))
    {
      const int dim = egrad_z->Dimension();
      for (int k = 0; k < dim; k++)
      {
        RealT val = 0.0;
        if (k < obj_sens.size())
        {
          val = obj_sens.fastAccessDx(k);
        }
        egrad_z->Set_Entry(k, val);
      }
    }
    else if (HDSA::Tpetra_Vector<RealT> *egrad_z = dynamic_cast<HDSA::Tpetra_Vector<RealT> *>(&grad_z))
    {
      Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> grad_over = solver_->linalg->getNewOverlappedParamVector();
      grad_over->putScalar(0.0);
      Teuchos::ArrayRCP<ScalarT> grad_data = grad_over->getDataNonConst(0);
      for (size_t i = 0; i < grad_data.size(); i++)
      {
        const int deriv_index = static_cast<int>(i + params_->num_active_params);
        if (deriv_index < obj_sens.size())
        {
          grad_data[i] = obj_sens.fastAccessDx(deriv_index);
        }
      }
      Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> grad_owned = egrad_z->getVector();
      solver_->linalg->exportParamVectorFromOverlapped(grad_owned, grad_over);
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                                  "Error in HDSA MrHyDE interface: unsupported parameter-gradient vector type.");
    }
  }

};
#endif
