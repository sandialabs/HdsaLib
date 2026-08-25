/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#include "Teuchos_GlobalMPISession.hpp"
#include <fstream>

#include "HDSA_Stream.hpp"
#include "HDSA_Comm.hpp"
#include "HDSA_Ptr.hpp"
#include "HDSA_Sparse_Matrix.hpp"
#include "HDSA_Sparse_Matrix_Trilinos.hpp"
#include "HDSA_Incomplete_Chol_Factor.hpp"
#include "HDSA_Sparse_Matrix_Sqrt.hpp"

typedef double RealT;

int main(int argc, char *argv[])
{

  HDSA::nullstream bhs;
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, &bhs);
  HDSA::Ptr<const HDSA::Comm<int>> comm = HDSA::makePtr<HDSA::Comm<int>>();
  HDSA::Ptr<HDSA::Random_Number_Generator<RealT>> random_number_generator = HDSA::makePtr<HDSA::Random_Number_Generator<RealT>>();

  int m = 100;
  RealT h = 1.0 / static_cast<RealT>(m - 1);
  auto map = Tpetra::createUniformContigMap<Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type>(m, comm->Get_Teuchos_Communicator());
  HDSA::Ptr<Tpetra::CrsMatrix<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type>> M = HDSA::makePtr<Tpetra::CrsMatrix<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type>>(map, 3); // 3 is the maximum number of non-zero entries per row
  Teuchos::Array<Tpetra::Map<>::global_ordinal_type> cols0 = {0, 1};
  Teuchos::Array<RealT> vals0_M = {h / 3.0, h / 6.0};
  if (M->getRowMap()->isNodeGlobalElement(0))
  {
    M->insertGlobalValues(0, cols0(), vals0_M());
  }
  for (int i = 1; i < m - 1; ++i)
  {
    Teuchos::Array<Tpetra::Map<>::global_ordinal_type> cols = {i - 1, i, i + 1};
    Teuchos::Array<RealT> vals_M = {h / 6.0, 2.0 * h / 3.0, h / 6.0};
    if (M->getRowMap()->isNodeGlobalElement(i))
    {
      M->insertGlobalValues(i, cols(), vals_M());
    }
  }
  Teuchos::Array<Tpetra::Map<>::global_ordinal_type> colsm = {m - 2, m - 1};
  Teuchos::Array<RealT> valsm_M = {h / 6.0, h / 3.0};
  if (M->getRowMap()->isNodeGlobalElement(m - 1))
  {
    M->insertGlobalValues(m - 1, colsm(), valsm_M());
  }
  M->fillComplete();

  HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> M_sm = HDSA::makePtr<HDSA::Sparse_Matrix_Trilinos<RealT>>(M);
  M_sm->Set_Symmetric();
  HDSA::Ptr<HDSA::Incomplete_Chol_Factor<RealT>> L = M_sm->Get_Incomplete_Chol_Factor();
  HDSA::Ptr<HDSA::Sparse_Matrix_Sqrt<RealT>> mat_sqrt = HDSA::makePtr<HDSA::Sparse_Matrix_Sqrt<RealT>>(M_sm);
  mat_sqrt->Set_Incomplete_Factor(L);

  HDSA::Ptr<Tpetra::MultiVector<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type, Tpetra::Map<>::node_type>> tpetra_vec = Tpetra::createMultiVector<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type, Tpetra::Map<>::node_type>(map, 1);
  for (int k = 0; k < m; k++)
  {
    tpetra_vec->replaceGlobalValue(k, 0, static_cast<RealT>(k) / static_cast<RealT>(m - 1));
  }
  HDSA::Ptr<HDSA::Vector<RealT>> vec_in = HDSA::makePtr<HDSA::Tpetra_Vector<RealT>>(tpetra_vec, random_number_generator);
  HDSA::Ptr<HDSA::Vector<RealT>> vec_out = vec_in->Clone();

  std::string output_message = mat_sqrt->Matrix_Sqrt_Apply(*vec_out, *vec_in);

  std::vector<RealT> error = std::vector<RealT>(1);
  error[0] = std::abs(vec_out->Norm() - 0.576038);

  std::ofstream out("error.txt");
  for (const RealT &x : error)
  {
    out << x << '\n';
  }
  out.close();

  if (error[0] > 1.e-5)
  {
    std::cout << "Test failed" << std::endl;
    std::cout << output_message << std::endl;
    std::cout << "vec_out->Norm() = " << vec_out->Norm() << std::endl;
  }
  else
  {
    std::cout << "Test passed" << std::endl;
    std::cout << output_message << std::endl;
  }

  return 0;
}
