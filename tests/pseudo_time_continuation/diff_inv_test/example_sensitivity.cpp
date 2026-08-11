/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#include "Teuchos_GlobalMPISession.hpp"
#include <fstream>

#include "HDSA_Stream.hpp"
#include "HDSA_Comm.hpp"
#include "HDSA_Ptr.hpp"
#include "HDSA_Vector.hpp"
#include "HDSA_PC_Quasi_Newton_Preconditioner_LIS.hpp"
#include "HDSA_PC_Pseudo_Time_Continuation.hpp"
#include "Adv_Diff_Constraint.hpp"
#include "Prior_and_Likelihood.hpp"
#include "Reduced_Space_Objective.hpp"
#include "PC_Sensitivity_Operator_Interface_Adv_Diff.hpp"
#include "PC_LIS_Interface_Adv_Diff.hpp"

typedef double RealT;

int main(int argc, char *argv[])
{

  HDSA::nullstream bhs;
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, &bhs);
  HDSA::Ptr<const HDSA::Comm<int>> comm = HDSA::makePtr<HDSA::Comm<int>>();

  int m = 100;
  HDSA::Ptr<HDSA::Vector<RealT>> u = HDSA::makePtr<HDSA::Std_Vector<RealT>>(m);
  HDSA::Ptr<HDSA::Vector<RealT>> z_bar = HDSA::makePtr<HDSA::Std_Vector<RealT>>(m);
  HDSA::Ptr<HDSA::Vector<RealT>> theta_bar = HDSA::makePtr<HDSA::Std_Vector<RealT>>(m);

  HDSA::Std_Vector<RealT> &z_bar_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(*z_bar);
  RealT val = 0.0;
  // read in data
  std::ifstream in_z("z_bar.txt");
  // read the elements in the file into a vector
  // test file open
  if (in_z)
  {
    for (int i = 0; i < m; i++)
    {
      in_z >> val;
      z_bar_std.Set_Entry(i, val);
    }
  }
  else
  {
    std::cout << "Error loading the data from z_bar.txt" << std::endl;
  }

  HDSA::Std_Vector<RealT> &theta_bar_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(*theta_bar);
  // read in data
  std::ifstream in_theta("theta_bar.txt");
  // read the elements in the file into a vector
  // test file open
  if (in_theta)
  {
    for (int i = 0; i < m; i++)
    {
      in_theta >> val;
      theta_bar_std.Set_Entry(i, val);
    }
  }
  else
  {
    std::cout << "Error loading the data from theta_bar.txt" << std::endl;
  }

  HDSA::Ptr<Adv_Diff_Constraint<RealT>> con = HDSA::makePtr<Adv_Diff_Constraint<RealT>>(m);
  HDSA::Ptr<Prior_and_Likelihood<RealT>> prior_and_like = HDSA::makePtr<Prior_and_Likelihood<RealT>>(con);
  HDSA::Ptr<Reduced_Space_Objective<RealT>> obj = HDSA::makePtr<Reduced_Space_Objective<RealT>>(con, prior_and_like, u, z_bar, theta_bar);

  HDSA::Ptr<HDSA::PC_Sensitivity_Operator_Interface<RealT>> sen_op_interface = HDSA::makePtr<PC_Sensitivity_Operator_Interface_Adv_Diff<RealT>>(obj);
  HDSA::Ptr<HDSA::PC_LIS_Interface<RealT>> lis_interface = HDSA::makePtr<PC_LIS_Interface_Adv_Diff<RealT>>(obj);
  HDSA::Ptr<HDSA::PC_Quasi_Newton_Preconditioner_LIS<RealT>> qn_prec = HDSA::makePtr<HDSA::PC_Quasi_Newton_Preconditioner_LIS<RealT>>(z_bar, theta_bar, lis_interface);
  
  RealT grad_tol = 1.e-5;
  HDSA::Ptr<HDSA::PC_Pseudo_Time_Continuation<RealT>> sen = HDSA::makePtr<HDSA::PC_Pseudo_Time_Continuation<RealT>>(z_bar, sen_op_interface, qn_prec, grad_tol);

  int rank = 8;
  int oversampling = 10;
  qn_prec->Compute_Hessian_GEVP(z_bar, theta_bar, rank, oversampling);

  HDSA::Ptr<HDSA::Vector<RealT>> z_star = z_bar->Clone();
  HDSA::Ptr<HDSA::Vector<RealT>> grad_star = z_bar->Clone();
  HDSA::Ptr<HDSA::Vector<RealT>> theta_star = theta_bar->Clone();

  HDSA::Std_Vector<RealT> &theta_star_std = dynamic_cast<HDSA::Std_Vector<RealT> &>(*theta_star);
  // read in data
  std::ifstream in_theta_star("theta_star.txt");
  // read the elements in the file into a vector
  // test file open
  if (in_theta_star)
  {
    for (int i = 0; i < m; i++)
    {
      in_theta_star >> val;
      theta_star_std.Set_Entry(i, val);
    }
  }
  else
  {
    std::cout << "Error loading the data from theta_star.txt" << std::endl;
  }

  int N_fe = 40;
  HDSA::Ptr<HDSA::PC_Auxillary_Parameter_Trajectory<RealT>> theta_traj_fe = HDSA::makePtr<HDSA::PC_Euclidean_Auxillary_Parameter_Trajectory<RealT>>(N_fe, theta_bar, theta_star);
  sen->Pseudo_Time_Continuation_Forward_Euler(*z_star, *grad_star, *theta_traj_fe);
  std::string name = "z_star_fe.txt";
  z_star->Write_to_File(name);
  name = "grad_star_fe.txt";
  grad_star->Write_to_File(name);

  z_star->Zeros();
  grad_star->Zeros();
  int N_me = 40;
  HDSA::Ptr<HDSA::PC_Auxillary_Parameter_Trajectory<RealT>> theta_traj_me = HDSA::makePtr<HDSA::PC_Euclidean_Auxillary_Parameter_Trajectory<RealT>>(N_me, theta_bar, theta_star);
  sen->Pseudo_Time_Continuation_Modified_Euler(*z_star, *grad_star, *theta_traj_me);
  name = "z_star_me.txt";
  z_star->Write_to_File(name);
  name = "grad_star_me.txt";
  grad_star->Write_to_File(name);

  return 0;
}
