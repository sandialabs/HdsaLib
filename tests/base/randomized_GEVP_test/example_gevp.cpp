/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#include "Teuchos_GlobalMPISession.hpp"
#include <fstream>

#include "HDSA_Stream.hpp"
#include "HDSA_Comm.hpp"
#include "HDSA_Ptr.hpp"
#include "HDSA_Random_Number_Generator.hpp"
#include "HDSA_Std_Vector.hpp"
#include "HDSA_MultiVector.hpp"
#include "HDSA_Dense_Matrix.hpp"
#include "Randomized_GEVP_test.hpp"

typedef double RealT;

int main(int argc, char *argv[])
{

  HDSA::nullstream bhs;
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, &bhs);
  HDSA::Ptr<const HDSA::Comm<int>> comm = HDSA::makePtr<HDSA::Comm<int>>();

  int num_random_numbers = 1.e5;
  std::string random_number_file = "random_numbers.txt";
  HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> random_number_generator = HDSA::makePtr<HDSA::Random_Number_Generator<RealT>>(num_random_numbers, random_number_file);

  int m = 50;
  HDSA::Ptr<HDSA::Vector<RealT>> vec = HDSA::makePtr<HDSA::Std_Vector<RealT>>(m, random_number_generator);
  HDSA::Ptr<HDSA::Randomized_GEVP<RealT>> gevp = HDSA::makePtr<Randomized_GEVP_test<RealT>>(*vec, random_number_generator);

  int num_evals = 20;
  int oversampling = 20;
  HDSA::Ptr<HDSA::MultiVector<RealT>> evecs = HDSA::makePtr<HDSA::MultiVector<RealT>>(num_evals, *vec);
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> evals = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(num_evals, 1);

  gevp->Compute_GEVP(*evecs, *evals, num_evals, oversampling);

  std::string name = "Evec";
  evecs->Write_to_File(name);
  name = "evals.txt";
  std::ofstream fout;
  fout.open(name);
  for (int i = 0; i < num_evals; i++)
  {
    fout << std::setprecision(16) << (*evals)(i, 0) << "  ";
  }
  fout.close();

  return 0;
}
