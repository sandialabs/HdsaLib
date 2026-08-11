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
#include "HDSA_PC_Quasi_Newton_Preconditioner.hpp"
#include "HDSA_PC_Pseudo_Time_Continuation.hpp"
#include "rosenbrock.hpp"
#include "PC_Sensitivity_Operator_Interface_Rosenbrock.hpp"

typedef double RealT;

int main(int argc, char *argv[])
{

  HDSA::nullstream bhs;
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, &bhs);
  HDSA::Ptr<const HDSA::Comm<int>> comm = HDSA::makePtr<HDSA::Comm<int>>();

  int d = 3;
  HDSA::Ptr<Rosenbrock<RealT>> rosenbrock = HDSA::makePtr<Rosenbrock<RealT>>(d);

  HDSA::Ptr<HDSA::Vector<RealT>> z_bar = HDSA::makePtr<HDSA::Std_Vector<RealT>>(d);
  z_bar->Set_Scalar(1.0);
  HDSA::Ptr<HDSA::Vector<RealT>> theta_bar = HDSA::makePtr<HDSA::Std_Vector<RealT>>(d - 1);
  theta_bar->Set_Scalar(1.0);

  HDSA::Ptr<HDSA::PC_Sensitivity_Operator_Interface<RealT>> sen_op = HDSA::makePtr<PC_Sensitivity_Operator_Interface_Rosenbrock<RealT>>(rosenbrock);
  HDSA::Ptr<HDSA::PC_Quasi_Newton_Preconditioner<RealT>> qn_prec = HDSA::makePtr<HDSA::PC_Quasi_Newton_Preconditioner<RealT>>();
  HDSA::Ptr<HDSA::PC_Pseudo_Time_Continuation<RealT>> sen = HDSA::makePtr<HDSA::PC_Pseudo_Time_Continuation<RealT>>(z_bar, sen_op, qn_prec);

  HDSA::Ptr<HDSA::Vector<RealT>> theta_star = theta_bar->Clone();
  theta_star->Set_Scalar(1.2);

  HDSA::Ptr<HDSA::Vector<RealT>> z_star = z_bar->Clone();
  HDSA::Ptr<HDSA::Vector<RealT>> grad_star = z_bar->Clone();
  int N = 30;
  HDSA::Ptr<HDSA::PC_Auxillary_Parameter_Trajectory<RealT>> theta_traj = HDSA::makePtr<HDSA::PC_Euclidean_Auxillary_Parameter_Trajectory<RealT>>(N, theta_bar, theta_star);
  sen->Pseudo_Time_Continuation_Modified_Euler(*z_star, *grad_star, *theta_traj);

  std::string name = "z_star.txt";
  z_star->Write_to_File(name);

  return 0;
}
