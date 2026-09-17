/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_SOLVER_INTERFACE_MRHYDE_HPP
#define HDSA_SOLVER_INTERFACE_MRHYDE_HPP

#include "HDSA_Transient_Vector.hpp"

template <class RealT,
          class LO = Tpetra::Map<>::local_ordinal_type,
          class GO = Tpetra::Map<>::global_ordinal_type,
          class Node = Tpetra::Map<>::node_type>
class Solver_Interface_MrHyDE
{

private:
  Teuchos::RCP<MrHyDE::SolverManager<SolverNode>> solve_;
  HDSA::Ptr<MrHyDE::ParameterManager<SolverNode>> params_;

public:
  Solver_Interface_MrHyDE(const Teuchos::RCP<MrHyDE::SolverManager<SolverNode>> &solve, const HDSA::Ptr<MrHyDE::ParameterManager<SolverNode>> &params) : solve_(solve), params_(params)
  {
  }

  virtual ~Solver_Interface_MrHyDE()
  {
  }

  void State_Solve(HDSA::Vector<RealT> &u, const HDSA::Vector<RealT> &z) const
  {

    ROL::Ptr<MrHyDE_OptVector> z_rol = Map_HDSA_Vector_to_MrHyDE_OptVector(z);
    params_->updateParams(*z_rol);
    ScalarT val = 0.0;
    solve_->forwardModel(val);

    if (solve_->isTransient)
    {
      HDSA::Transient_Vector<RealT> &u_trans = dynamic_cast<HDSA::Transient_Vector<RealT> &>(u);
      int n_t = solve_->settings->sublist("Solver").get<int>("number of steps", 0) + 1;
      for (int i = 0; i < n_t; i++)
      {
        HDSA::Ptr<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> u_vec;
        solve_->postproc->soln[0]->extract(u_vec, i);
        HDSA::Tpetra_Vector<RealT> &eu_i = dynamic_cast<HDSA::Tpetra_Vector<RealT> &>(*u_trans[i]);
        eu_i.getVector()->update(1.0, *u_vec, 0.0);
      }
    }
    else
    {
      HDSA::Ptr<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> u_vec;
      solve_->postproc->soln[0]->extract(u_vec, 0);
      HDSA::Tpetra_Vector<RealT> &eu = dynamic_cast<HDSA::Tpetra_Vector<RealT> &>(u);
      eu.getVector()->update(1.0, *u_vec, 0.0);
    }
  }

  HDSA::Ptr<MrHyDE_OptVector> Map_HDSA_Vector_to_MrHyDE_OptVector(const HDSA::Vector<RealT> &z) const
  {
    ROL::Ptr<MrHyDE_OptVector> z_rol;
    if (const HDSA::Tpetra_Vector<RealT> *ez = dynamic_cast<const HDSA::Tpetra_Vector<RealT> *>(&z))
    {
      HDSA::Ptr<Tpetra::MultiVector<ScalarT, LO, GO, SolverNode>> fvec = ez->getVector();
      ROL::Ptr<std::vector<ScalarT>> svec = ROL::makePtr<std::vector<RealT>>(0);
      z_rol = ROL::makePtr<MrHyDE_OptVector>(fvec, svec);
    }
    else if (const HDSA::Transient_Vector<RealT> *ez = dynamic_cast<const HDSA::Transient_Vector<RealT> *>(&z))
    {
      std::vector<ROL::Ptr<Tpetra::MultiVector<RealT, LO, GO, SolverNode>>> f_vec;
      std::vector<ROL::Ptr<std::vector<RealT>>> s_vec;
      int n_t = ez->Get_n_t();
      s_vec.resize(n_t);
      for (int k = 0; k < n_t; k++)
      {
        HDSA::Ptr<HDSA::Vector<RealT>> z_k = (*ez)[k];
        const HDSA::Ptr<HDSA::Std_Vector<RealT>> ez_k = HDSA::dynamicPtrCast<HDSA::Std_Vector<RealT>>(z_k);
        s_vec[k] = ez_k->get_std_vec();
      }
      RealT dt = solve_->deltat; // Assumes that z is discretized on the same time nodes as the state
      z_rol = ROL::makePtr<MrHyDE_OptVector>(f_vec, s_vec, dt);
    }
    else if (const HDSA::Std_Vector<RealT> *ez = dynamic_cast<const HDSA::Std_Vector<RealT> *>(&z))
    {
      ROL::Ptr<std::vector<ScalarT>> svec = ez->get_std_vec();
      z_rol = ROL::makePtr<MrHyDE_OptVector>(svec);
    }
    return z_rol;
  }
};
#endif
