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
#include "HDSA_BF_Sol_Op_Interface.hpp"
#include "HDSA_MD_Opt_Prob_Interface.hpp"
#include "HDSA_BF_Update.hpp"
#include "MD_Opt_Prob_Interface_synthetic_test.hpp"
#include "BF_Sol_Op_Interface_synthetic_test.hpp"

typedef double RealT;

int main(int argc, char *argv[])
{

  HDSA::nullstream bhs;
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, &bhs);
  HDSA::Ptr<const HDSA::Comm<int>> comm = HDSA::makePtr<HDSA::Comm<int>>();

  int m = 51;
  HDSA::Ptr<HDSA::BF_Sol_Op_Interface<RealT>> sol_op_interface = HDSA::makePtr<BF_Sol_Op_Interface_synthetic_test<RealT>>();
  HDSA::Ptr<HDSA::MD_Opt_Prob_Interface<RealT>> opt_prob_interface = HDSA::makePtr<MD_Opt_Prob_Interface_synthetic_test<RealT>>();

  HDSA::Ptr<HDSA::BF_Update<RealT>> bf_update = HDSA::makePtr<HDSA::BF_Update<RealT>>(sol_op_interface, opt_prob_interface);

  HDSA::Ptr<HDSA::Std_Vector<RealT>> u_lofi = HDSA::makePtr<HDSA::Std_Vector<RealT>>(m);
  HDSA::Ptr<HDSA::Std_Vector<RealT>> z_lofi = HDSA::makePtr<HDSA::Std_Vector<RealT>>(m);
  for (int k = 0; k < m; k++)
  {
    u_lofi->Set_Entry(k, std::pow(static_cast<RealT>(k) / static_cast<RealT>(m - 1) + 1.0, 3.0));
    z_lofi->Set_Entry(k, static_cast<RealT>(k) / static_cast<RealT>(m - 1) + 1.0);
  }
  HDSA::Ptr<HDSA::Vector<RealT>> z_update = bf_update->Update(*u_lofi, *z_lofi);

  std::string file_name = "z_update.txt";
  z_update->Write_to_File(file_name);

  return 0;
}
