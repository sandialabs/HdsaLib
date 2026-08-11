/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#include "Teuchos_GlobalMPISession.hpp"
#include <fstream>

#include "HDSA_Stream.hpp"
#include "HDSA_Comm.hpp"
#include "HDSA_Ptr.hpp"
#include "HDSA_Std_Vector.hpp"
#include "Matrix_Sqrt_test.hpp"

typedef double RealT;

int main(int argc, char *argv[])
{

  HDSA::nullstream bhs;
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, &bhs);
  HDSA::Ptr<const HDSA::Comm<int>> comm = HDSA::makePtr<HDSA::Comm<int>>();

  int num_random_numbers = 1.e5;
  std::string random_number_file = "random_numbers.txt";
  HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> random_number_generator = HDSA::makePtr<HDSA::Random_Number_Generator<RealT>>(num_random_numbers, random_number_file);

  int m = 100;
  HDSA::Ptr<HDSA::Matrix_Sqrt<RealT>> mat_sqrt = HDSA::makePtr<Matrix_Sqrt_test<RealT>>(m);
  HDSA::Ptr<HDSA::Vector<RealT>> vec_in = HDSA::makePtr<HDSA::Std_Vector<RealT>>(m, random_number_generator);
  vec_in->Randomize_Standard_Normal();
  HDSA::Ptr<HDSA::Vector<RealT>> vec_out_1 = HDSA::makePtr<HDSA::Std_Vector<RealT>>(m, random_number_generator);
  HDSA::Ptr<HDSA::Vector<RealT>> vec_out_2 = HDSA::makePtr<HDSA::Std_Vector<RealT>>(m, random_number_generator);
  HDSA::Ptr<HDSA::Vector<RealT>> vec_out_3 = HDSA::makePtr<HDSA::Std_Vector<RealT>>(m, random_number_generator);

  mat_sqrt->Matrix_Sqrt_Apply(*vec_out_1, *vec_in);
  mat_sqrt->Matrix_Sqrt_Apply(*vec_out_2, *vec_out_1);
  mat_sqrt->Apply(*vec_out_3, *vec_in);
  vec_out_2->Scaled_Plus(-1.0, *vec_out_3);

  std::vector<RealT> error = std::vector<RealT>(1);
  error[0] = vec_out_2->Norm();

  std::cout << "Error in Matrix squart root = " << error[0] << std::endl;

  std::ofstream out("error.txt");
  for (const RealT &x : error)
  {
    out << x << '\n';
  }
  out.close();

  return 0;
}
