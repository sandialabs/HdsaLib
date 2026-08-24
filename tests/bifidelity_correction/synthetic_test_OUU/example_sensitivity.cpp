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
#include "HDSA_BF_OUU_Sol_Op_Interface.hpp"
#include "HDSA_MD_OUU_Opt_Prob_Interface.hpp"
#include "HDSA_BF_OUU_Update.hpp"
#include "MD_Opt_Prob_Interface_synthetic_test_OUU.hpp"
#include "BF_Sol_Op_Interface_synthetic_test_OUU.hpp"
#include "MD_Data_Interface_synthetic_test_OUU.hpp"
#include "HDSA_Ensemble_Vector.hpp"
#include "HDSA_Random_Number_Generator.hpp"


typedef double RealT;

int main(int argc, char *argv[])
{

  HDSA::nullstream bhs;
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, &bhs);
  HDSA::Ptr<const HDSA::Comm<int>> comm = HDSA::makePtr<HDSA::Comm<int>>();

  int num_random_numbers = 3.e6;
  std::string random_number_file = "random_numbers.txt";
  HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> random_number_generator = HDSA::makePtr<HDSA::Random_Number_Generator<RealT>>(num_random_numbers, random_number_file);

  int ens_size = 30;
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Xi = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(3, ens_size);
  for (int s = 0; s < ens_size; s++)
  {
    RealT val = static_cast<RealT>(s) / static_cast<RealT>(ens_size - 1);
    Xi->Set_Entry(0, s, val);
    Xi->Set_Entry(1, s, val);
    Xi->Set_Entry(2, s, val);
  }

  HDSA::Ptr<HDSA::BF_OUU_Sol_Op_Interface<RealT>> ouu_sol_op_interface = HDSA::makePtr<BF_Sol_Op_Interface_synthetic_test_OUU<RealT>>(Xi);
  HDSA::Ptr<HDSA::MD_OUU_Opt_Prob_Interface<RealT>> ouu_opt_prob_interface = HDSA::makePtr<MD_Opt_Prob_Interface_synthetic_test_OUU<RealT>>(ens_size, Xi);

  HDSA::Ptr<HDSA::BF_OUU_Update<RealT>> bf_update = HDSA::makePtr<HDSA::BF_OUU_Update<RealT>>(ouu_sol_op_interface, ouu_opt_prob_interface);

  HDSA::Ptr<HDSA::MD_OUU_Data_Interface<RealT>> ouu_data_interface = HDSA::makePtr<MD_Data_Interface_synthetic_test_OUU<RealT>>(random_number_generator, ens_size, Xi);
  std::vector<HDSA::Ptr<HDSA::Vector<RealT>>> u_lofi_ens;
  u_lofi_ens.resize(ens_size);
  for(int s = 0; s < ens_size; s++)
  {
    u_lofi_ens[s] = ouu_data_interface->Load_Optimal_us(s);
  }
  HDSA::Ptr<HDSA::Vector<RealT>> u_lofi = HDSA::makePtr<HDSA::Ensemble_Vector<RealT>>(u_lofi_ens);
  HDSA::Ptr<HDSA::Vector<RealT>> z_lofi = ouu_data_interface->Load_Optimal_z();

  HDSA::Ptr<HDSA::Vector<RealT>> z_update = bf_update->Update(*u_lofi, *z_lofi);

  std::string file_name = "z_update.txt";
  z_update->Write_to_File(file_name);

  return 0;
}
