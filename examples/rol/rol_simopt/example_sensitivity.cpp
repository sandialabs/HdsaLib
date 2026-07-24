/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#include "rol_simopt_test_problem.hpp"
#include "ROL_Algorithm.hpp"
#include "ROL_TrustRegionStep.hpp"
#include "ROL_StatusTest.hpp"
#include "ROL_CompositeStep.hpp"
#include "ROL_ConstraintStatusTest.hpp"
#include "ROL_ParameterList.hpp"

#include "ROL_Stream.hpp"
#include "Teuchos_GlobalMPISession.hpp"

#include <iostream>
#include <fstream>
#include <math.h>
#include "HDSA_Stream.hpp"
#include "HDSA_Comm.hpp"
#include "HDSA_Ptr.hpp"
#include "HDSA_Random_Number_Generator.hpp"
#include "HDSA_Vector.hpp"
#include "HDSA_Std_Vector.hpp"
#include "HDSA_ROL_Vector.hpp"
#include "HDSA_MD_ROL_Opt_Prob_Interface.hpp"
#include "HDSA_MultiVector.hpp"
#include "HDSA_MD_Prior_Sampling.hpp"
#include "HDSA_MD_Posterior_Data.hpp"
#include "HDSA_MD_Posterior_Sampling.hpp"
#include "HDSA_MD_Posterior_Vectors.hpp"
#include "HDSA_MD_Hessian_Analysis.hpp"
#include "HDSA_MD_Update.hpp"
#include "HDSA_MD_Continuation_Update.hpp"
#include "Elliptic_u_Prior_Interface_rol_simopt_test_problem.hpp"
#include "Elliptic_z_Prior_Interface_rol_simopt_test_problem.hpp"
#include "Data_Interface_rol_simopt_test_problem.hpp"

typedef double RealT;

int main(int argc, char *argv[])
{

  Teuchos::GlobalMPISession mpiSession(&argc, &argv);

  HDSA::Ptr<const HDSA::Comm<int>> comm = HDSA::makePtr<HDSA::Comm<int>>();
  HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> random_number_generator = HDSA::makePtr<HDSA::Random_Number_Generator<RealT>>(comm);

  // This little trick lets us print to std::cout only if a (dummy) command-line argument is provided.
  int iprint = argc - 1;
  ROL::Ptr<std::ostream> outStream;
  ROL::nullstream bhs; // outputs nothing
  if (iprint > 0)
    outStream = ROL::makePtrFromRef(std::cout);
  else
    outStream = ROL::makePtrFromRef(bhs);

  std::string filename = "input.xml";
  auto parlist = ROL::getParametersFromXmlFile(filename);

  int m = 51;
  Constraint_SimOptTestProb<RealT> con(m);
  Objective_SimOptTestProb<RealT> obj(m);
  ROL::Ptr<ROL::Objective_SimOpt<RealT>> pobj = ROL::makePtrFromRef(obj);
  ROL::Ptr<ROL::Constraint_SimOpt<RealT>> pcon = ROL::makePtrFromRef(con);

  ROL::Ptr<std::vector<RealT>> z_ptr = ROL::makePtr<std::vector<RealT>>(m, 0.0);
  ROL::StdVector<RealT> z(z_ptr);
  ROL::Ptr<ROL::Vector<RealT>> zp = ROL::makePtrFromRef(z);
  ROL::Ptr<std::vector<RealT>> u_ptr = ROL::makePtr<std::vector<RealT>>(m, 0.0);
  ROL::StdVector<RealT> u(u_ptr);
  ROL::Ptr<ROL::Vector<RealT>> up = ROL::makePtrFromRef(u);

  HDSA::Ptr<HDSA::MD_Opt_Prob_Interface<RealT>> opt_prob_interface = HDSA::makePtr<HDSA::MD_ROL_Opt_Prob_Interface<RealT>>(pobj, pcon, up, zp);

  // Need to check hyper-parameter values
  RealT alpha_u = parlist->sublist("MD Prior").get("alpha_u", 1.0 / std::pow(2.0, 2.0));
  RealT beta_u = parlist->sublist("MD Prior").get("beta_u", 1.e-3);
  RealT alpha_z = parlist->sublist("MD Prior").get("alpha_z", 1.0 / std::pow(100.0, 2.0));
  RealT beta_z = parlist->sublist("MD Prior").get("beta_z", 1.e-2);
  RealT alpha_d = parlist->sublist("MD Prior").get("alpha_d", 1.e-3);
  HDSA::Ptr<HDSA::MD_Data_Interface<RealT>> data_interface = HDSA::makePtr<Data_Interface_SimOptTestProb<RealT>>(m, random_number_generator, comm);
  HDSA::Ptr<HDSA::MD_u_Prior_Interface<RealT>> u_prior_interface = HDSA::makePtr<Elliptic_u_Prior_Interface_SimOptTestProb<RealT>>(alpha_u, beta_u, m);
  HDSA::Ptr<HDSA::MD_z_Prior_Interface<RealT>> z_prior_interface = HDSA::makePtr<Elliptic_z_Prior_Interface_SimOptTestProb<RealT>>(alpha_z, beta_z, m, random_number_generator);

  int num_sing_vals = parlist->sublist("MD Prior").get("Number of Singular Values", m);
  int oversampling = parlist->sublist("MD Prior").get("Oversampling Factor", 10);
  int num_subspace_iters = parlist->sublist("MD Prior").get("Number of Subspace Iterations", 1);
  HDSA::Ptr<HDSA::Vector<RealT>> u_vec = data_interface->Get_u_opt()->Clone();
  HDSA::Ptr<HDSA::MD_Elliptic_u_Prior_Interface<RealT>> elliptic_u_prior_interface = HDSA::dynamicPtrCast<HDSA::MD_Elliptic_u_Prior_Interface<RealT>>(u_prior_interface);
  elliptic_u_prior_interface->Compute_E_u_Inverse_GSVD(num_sing_vals, oversampling, num_subspace_iters, *u_vec);

  HDSA::Ptr<HDSA::MD_Prior_Sampling<RealT>> prior_sampling = HDSA::makePtr<HDSA::MD_Prior_Sampling<RealT>>(data_interface, u_prior_interface, z_prior_interface);

  int num_prior_samples = parlist->sublist("MD Prior").get("Number of Prior Samples", 100);
  HDSA::Ptr<HDSA::MultiVector<RealT>> prior_samples_at_z_opt = prior_sampling->Prior_Discrepancy_Samples_at_z_opt(num_prior_samples);
  std::string name = "prior_discrepancy_evaluated_at_z_opt";
  prior_samples_at_z_opt->Write_to_File(name);

  HDSA::Ptr<HDSA::MultiVector<RealT>> z_test1 = HDSA::makePtr<HDSA::MultiVector<RealT>>(3, *data_interface->Get_z_opt());
  HDSA::Ptr<HDSA::Vector<RealT>> z0 = (*z_test1)[0];
  HDSA::Ptr<HDSA::ROL_Vector<RealT>> z0_rol = HDSA::dynamicPtrCast<HDSA::ROL_Vector<RealT>>(z0);
  HDSA::Ptr<HDSA::Vector<RealT>> z1 = (*z_test1)[1];
  HDSA::Ptr<HDSA::ROL_Vector<RealT>> z1_rol = HDSA::dynamicPtrCast<HDSA::ROL_Vector<RealT>>(z1);
  HDSA::Ptr<HDSA::Vector<RealT>> z2 = (*z_test1)[2];
  HDSA::Ptr<HDSA::ROL_Vector<RealT>> z2_rol = HDSA::dynamicPtrCast<HDSA::ROL_Vector<RealT>>(z2);
  ROL::StdVector<RealT> z0_std = dynamic_cast<ROL::StdVector<RealT> &>(*z0_rol->rol_vec);
  ROL::StdVector<RealT> z1_std = dynamic_cast<ROL::StdVector<RealT> &>(*z1_rol->rol_vec);
  ROL::StdVector<RealT> z2_std = dynamic_cast<ROL::StdVector<RealT> &>(*z2_rol->rol_vec);
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m, 1);
  for (int k = 0; k < m; k++)
  {
    x->Set_Entry(k, 0, static_cast<RealT>(k) / static_cast<RealT>(m - 1));
  }
  RealT pi = 3.14159265358979323846;
  for (int k = 0; k < m; k++)
  {
    (*z0_std.getVector())[k] = (*x)(k, 0);
    (*z1_std.getVector())[k] = 1.0 + std::pow((*x)(k, 0), 2.0);
    (*z2_std.getVector())[k] = std::sin(2 * pi * (*x)(k, 0));
  }

  std::vector<HDSA::Ptr<HDSA::MultiVector<RealT>>> prior_samples = prior_sampling->Prior_Discrepancy_Samples(*z_test1, num_prior_samples);
  for (int i = 0; i < num_prior_samples; i++)
  {
    std::string name = "prior_discrepancy_sample_" + std::to_string(i + 1);
    prior_samples[i]->Write_to_File(name);
  }

  HDSA::Ptr<HDSA::MD_Posterior_Data<RealT>> post_data = HDSA::makePtr<HDSA::MD_Posterior_Data<RealT>>();

  HDSA::Ptr<HDSA::MD_Posterior_Sampling<RealT>> post_sampling = HDSA::makePtr<HDSA::MD_Posterior_Sampling<RealT>>(data_interface, u_prior_interface, z_prior_interface);
  int num_post_samples = num_prior_samples;
  post_sampling->Compute_Posterior_Data(alpha_d, num_post_samples);

  std::vector<HDSA::Ptr<HDSA::Vector<RealT>>> z_test2;
  z_test2.resize(3);
  z_test2[0] = z0->Clone();
  z_test2[0]->Set(*(*data_interface->Get_Z())[0]);
  z_test2[1] = z0->Clone();
  z_test2[1]->Set(*(*data_interface->Get_Z())[1]);
  z_test2[2] = z0->Clone();
  HDSA::Ptr<HDSA::ROL_Vector<RealT>> ztest2_rol = HDSA::dynamicPtrCast<HDSA::ROL_Vector<RealT>>(z_test2[2]);
  ROL::StdVector<RealT> ztest2_std = dynamic_cast<ROL::StdVector<RealT> &>(*ztest2_rol->rol_vec);
  for (int k = 0; k < m; k++)
  {
    (*ztest2_std.getVector())[k] = 1.5;
  }

  std::vector<HDSA::Ptr<HDSA::MD_Posterior_Vectors<RealT>>> post_discrepancy_samples = post_sampling->Posterior_Discrepancy_Samples(z_test2);

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
  int num_evals = parlist->sublist("MD Hessian Analysis").get("Rank", 10);
  oversampling = parlist->sublist("MD Hessian Analysis").get("Oversampling Factor", 10);
  hessian_analysis->Compute_Hessian_GEVP(data_interface->Get_z_opt(), num_evals, oversampling);

  int num_continuation_steps = parlist->sublist("MD Continuation Update").get("Number of Continuation Steps", 0);
  HDSA::Ptr<HDSA::MD_Update<RealT>> update = HDSA::makePtr<HDSA::MD_Update<RealT>>(data_interface, u_prior_interface, z_prior_interface, opt_prob_interface, post_sampling, hessian_analysis, num_continuation_steps);
  HDSA::Ptr<HDSA::MD_Posterior_Vectors<RealT>> posterior_update_samples = update->Posterior_Update_Samples();
  name = "posterior_update_mean.txt";
  posterior_update_samples->mean->Write_to_File(name);
  name = "posterior_update_samples";
  posterior_update_samples->samples->Write_to_File(name);

  return 0;
}
