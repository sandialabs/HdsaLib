/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_PRIOR_OPERATORS_INTERFACE_MRHYDE_HPP
#define HDSA_PRIOR_OPERATORS_INTERFACE_MRHYDE_HPP

template <class RealT,
          class LO = Tpetra::Map<>::local_ordinal_type,
          class GO = Tpetra::Map<>::global_ordinal_type,
          class Node = Tpetra::Map<>::node_type>
class Prior_Operators_Interface_MrHyDE
{

public:
  Teuchos::RCP<Tpetra::CrsMatrix<ScalarT, LO, GO, Node>> M;
  Teuchos::RCP<Tpetra::CrsMatrix<ScalarT, LO, GO, Node>> S;

  Prior_Operators_Interface_MrHyDE(Teuchos::RCP<Teuchos::MpiComm<int>> &comm, Teuchos::RCP<Teuchos::ParameterList> &Settings, std::vector<string> &blockNames)
  {
    Teuchos::RCP<Teuchos::ParameterList> Settings_prior = HDSA::makePtr<Teuchos::ParameterList>(*Settings);
    Settings_prior->remove("Physics");
    Settings_prior->sublist("Physics").set("modules", "ellipticPrior");
    Settings_prior->sublist("Solver").set("solver", "steady-state");
    Settings_prior->sublist("Solver").set("matrix free", true);
    Settings_prior->remove("Analysis");
    Settings_prior->sublist("Analysis").set("Analysis type", "forward");
    Settings_prior->remove("Functions");
    Settings_prior->sublist("Functions").set("ellipticPrior diffusion", "0.0");
    Settings_prior->sublist("Functions").set("ellipticPrior reaction", "1.0");
    Settings_prior->sublist("Functions").set("specific heat", "0.0");
    Settings_prior->remove("Postprocess");
    Settings_prior->sublist("Postprocess").set("write solution", false);
    Settings_prior->sublist("Postprocess").set("create optimization movie", false);

    M = Instantiate_Prior_Operators(comm, Settings_prior, blockNames);
    Settings_prior->sublist("Functions").set("ellipticPrior diffusion", "1.0");
    Settings_prior->sublist("Functions").set("ellipticPrior reaction", "0.0");
    S = Instantiate_Prior_Operators(comm, Settings_prior, blockNames);
  }

  virtual ~Prior_Operators_Interface_MrHyDE()
  {
  }

  Teuchos::RCP<Tpetra::CrsMatrix<ScalarT, LO, GO, Node>> Instantiate_Prior_Operators(Teuchos::RCP<Teuchos::MpiComm<int>> comm, Teuchos::RCP<Teuchos::ParameterList> &Settings, std::vector<string> &blockNames)
  {

    Teuchos::RCP<MrHyDE::MeshInterface> mesh = Teuchos::rcp(new MrHyDE::MeshInterface(Settings, comm));

    Teuchos::RCP<MrHyDE::PhysicsInterface> physics = Teuchos::rcp(new MrHyDE::PhysicsInterface(Settings, comm,
                                                                                               mesh->getBlockNames(),
                                                                                               mesh->getPhaseBlockNames(),
                                                                                               mesh->getSideNames(),
                                                                                               mesh->getDimension(),
                                                                                               mesh->getPhaseDimension()));

    mesh->finalize(physics->getVarList(), physics->getVarTypes(), physics->getDerivedList());

    Teuchos::RCP<MrHyDE::DiscretizationInterface> disc = Teuchos::rcp(new MrHyDE::DiscretizationInterface(Settings, comm,
                                                                                                          mesh, physics));

    Teuchos::RCP<MrHyDE::ParameterManager<SolverNode>> params = Teuchos::rcp(new MrHyDE::ParameterManager<SolverNode>(comm, Settings, mesh, physics, disc));

    Teuchos::RCP<MrHyDE::AssemblyManager<SolverNode>> assembler = Teuchos::rcp(new MrHyDE::AssemblyManager<SolverNode>(comm, Settings, mesh, disc, physics, params));

    assembler->setMeshData();

    Teuchos::RCP<MrHyDE::MultiscaleManager> multiscale_manager = Teuchos::rcp(new MrHyDE::MultiscaleManager(comm, mesh, Settings,
                                                                                                            assembler->groups,
#ifndef MrHyDE_NO_AD
                                                                                                            assembler->function_managers_AD));
#else
                                                                                                            assembler->function_managers));
#endif

    Teuchos::RCP<MrHyDE::PostprocessManager<SolverNode>>
        postproc = Teuchos::rcp(new MrHyDE::PostprocessManager<SolverNode>(comm, Settings, mesh,
                                                                           disc, physics,
                                                                           multiscale_manager,
                                                                           assembler, params));

    Teuchos::RCP<MrHyDE::SolverManager<SolverNode>> solve = Teuchos::rcp(new MrHyDE::SolverManager<SolverNode>(comm, Settings, mesh, disc, physics, assembler, params));

    solve->multiscale_manager = multiscale_manager;
    assembler->multiscale_manager = multiscale_manager;
    solve->postproc = postproc;

    mesh->allocateMeshDataStructures();
    assembler->allocateGroupStorage();
    solve->completeSetup();
    postproc->linalg = solve->linalg;
    solve->setupExplicitMass();
    assembler->finalizeFunctions();
    solve->finalizeMultiscale();

    Kokkos::fence();
    comm->barrier();

    int set = 0;
    int stage = 0;

    HDSA::Ptr<Tpetra::MultiVector<RealT, LO, GO, Node>> current_res, current_res_over;
    current_res = solve->linalg->getNewVector(set);
    current_res_over = solve->linalg->getNewOverlappedVector(set);

    Teuchos::RCP<Tpetra::CrsMatrix<ScalarT, LO, GO, Node>> J, J_over;
    J = solve->linalg->getNewMatrix(set);
    J_over = solve->linalg->getNewOverlappedMatrix(set);
    solve->linalg->fillComplete(J_over);
    J_over->resumeFill();
    J_over->setAllToScalar(0.0);

    auto paramvec = params->getDiscretizedParamsOver();
    auto paramdot = params->getDiscretizedParamsDotOver();

    std::vector<HDSA::Ptr<Tpetra::MultiVector<RealT, LO, GO, Node>>> zero_soln;
    assembler->assembleJacRes(set, stage, zero_soln, zero_soln, zero_soln, zero_soln, zero_soln, zero_soln, true, false, false, false, 0,
                              current_res_over, J_over, false, 0.0, false, false,
                              params->num_active_params, paramvec, paramdot, false, 0.0);
    solve->linalg->fillComplete(J_over);
    J->resumeFill();
    solve->linalg->exportMatrixFromOverlapped(set, J, J_over);
    solve->linalg->fillComplete(J);

    return J;
  }
};
#endif
