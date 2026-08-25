/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_SPARSE_MATRIX_TRILINOS_HPP
#define HDSA_SPARSE_MATRIX_TRILINOS_HPP

#include "HDSA_Incomplete_Chol_Factor_Trilinos.hpp"
#include "HDSA_Sparse_Matrix_Solver_Trilinos.hpp"
#include "HDSA_Tpetra_Vector.hpp"
#include "TpetraExt_MatrixMatrix.hpp"
#include "Tpetra_CrsMatrix_decl.hpp"

namespace HDSA {

template <class RealT, class LO = Tpetra::Map<>::local_ordinal_type, class GO = Tpetra::Map<>::global_ordinal_type,
          class Node = Tpetra::Map<>::node_type>
class Sparse_Matrix_Trilinos : public HDSA::Sparse_Matrix<RealT> {

private:
  HDSA::Ptr<Tpetra::CrsMatrix<RealT, LO, GO, Node>> A_;

public:
  Sparse_Matrix_Trilinos(HDSA::Ptr<Tpetra::CrsMatrix<RealT, LO, GO, Node>>& A, bool is_symmetric = false) : A_(A) {
    if (is_symmetric) { HDSA::Sparse_Matrix<RealT>::Set_Symmetric(); }
  }

  virtual ~Sparse_Matrix_Trilinos() {}

  HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> Clone(int max_entries_per_row = 0) const {
    if (max_entries_per_row == 0) { max_entries_per_row = A_->getGlobalMaxNumRowEntries(); }
    HDSA::Ptr<Tpetra::CrsMatrix<RealT, LO, GO, Node>> B =
        HDSA::makePtr<Tpetra::CrsMatrix<RealT, LO, GO, Node>>(A_->getRowMap(), max_entries_per_row);
    B->scale(0.0);
    HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> B_sm = HDSA::makePtr<HDSA::Sparse_Matrix_Trilinos<RealT>>(B);
    return B_sm;
  }

  // Compute C = this * B, with options for transposes
  void Matrix_Matrix_Multiply(HDSA::Sparse_Matrix<RealT>& C, const HDSA::Sparse_Matrix<RealT>& B, bool A_trans = false,
                              bool B_trans = false) const {
    HDSA::Ptr<Tpetra::CrsMatrix<RealT, LO, GO, Node>> B_tpetra =
        dynamic_cast<const HDSA::Sparse_Matrix_Trilinos<RealT>&>(B).Get_Tpetra_Matrix();
    HDSA::Ptr<Tpetra::CrsMatrix<RealT, LO, GO, Node>> C_tpetra =
        dynamic_cast<const HDSA::Sparse_Matrix_Trilinos<RealT>&>(C).Get_Tpetra_Matrix();
    Tpetra::MatrixMatrix::Multiply(*A_, A_trans, *B_tpetra, B_trans, *C_tpetra);
  }

  void Apply(HDSA::Vector<RealT>& x_out, const HDSA::Vector<RealT>& x_in) const {
    const HDSA::Tpetra_Vector<RealT>& ex_in = dynamic_cast<const HDSA::Tpetra_Vector<RealT>&>(x_in);
    HDSA::Tpetra_Vector<RealT>& ex_out = dynamic_cast<HDSA::Tpetra_Vector<RealT>&>(x_out);
    A_->apply(*ex_in.getVector(), *ex_out.getVector());
  }

  void Set(HDSA::Sparse_Matrix<RealT>& B) {
    // Prepare A for updates
    A_->resumeFill();

    HDSA::Sparse_Matrix_Trilinos<RealT> B_t = dynamic_cast<HDSA::Sparse_Matrix_Trilinos<RealT>&>(B);
    // Loop over each row of B and copy its entries to A
    for (Tpetra::global_size_t row = 0; row < B_t.Get_Tpetra_Matrix()->getGlobalNumRows(); ++row) {
      if (A_->getRowMap()->isNodeGlobalElement(row)) {
        size_t numEntries = B_t.Get_Tpetra_Matrix()->getNumEntriesInGlobalRow(row);
        typename Tpetra::CrsMatrix<>::nonconst_global_inds_host_view_type indices;
        typename Tpetra::CrsMatrix<>::nonconst_values_host_view_type values;
        Kokkos::resize(indices, numEntries);
        Kokkos::resize(values, numEntries);

        // Get the global row copy from B
        B_t.Get_Tpetra_Matrix()->getGlobalRowCopy(row, indices, values, numEntries);
        // Replace the entries in A with those from B
        A_->insertGlobalValues(row, numEntries, &values[0], &indices[0]);
      }
    }

    // Complete the fill process
    A_->fillComplete();
  }

  void Set_Diagonal(HDSA::Vector<RealT>& vec, bool reciprocate_diag) {
    HDSA::Tpetra_Vector<RealT> tpetra_vec = dynamic_cast<HDSA::Tpetra_Vector<RealT>&>(vec);
    HDSA::Ptr<Tpetra::MultiVector<RealT, LO, GO, Node>> d = tpetra_vec.getVector();
    auto view = d->getLocalViewHost(Tpetra::Access::ReadOnly);

    A_->resumeFill();
    for (Tpetra::global_size_t row = 0; row < A_->getGlobalNumRows(); ++row) {
      if (A_->getRowMap()->isNodeGlobalElement(row)) {
        typename Tpetra::CrsMatrix<>::nonconst_global_inds_host_view_type indices;
        typename Tpetra::CrsMatrix<>::nonconst_values_host_view_type values;
        Kokkos::resize(indices, 1);
        Kokkos::resize(values, 1);
        indices[0] = row;
        LO i = d->getMap()->getLocalElement(row);
        if (reciprocate_diag) {
          values[0] = 1.0 / view(i, 0);
        } else {
          values[0] = view(i, 0);
        }
        A_->insertGlobalValues(row, 1, &values[0], &indices[0]);
      }
    }

    // Complete the fill process
    A_->fillComplete();
  }

  void Scaled_Plus(const RealT& alpha, const HDSA::Sparse_Matrix<RealT>& B) {
    A_->resumeFill(); // Prepare A for updates
    // Loop over each row of B and copy its entries to A
    const HDSA::Sparse_Matrix_Trilinos<RealT> B_t = dynamic_cast<const HDSA::Sparse_Matrix_Trilinos<RealT>&>(B);
    for (Tpetra::global_size_t row = 0; row < B_t.Get_Tpetra_Matrix()->getGlobalNumRows(); ++row) {
      if (A_->getRowMap()->isNodeGlobalElement(row)) {
        size_t numEntries = B_t.Get_Tpetra_Matrix()->getNumEntriesInGlobalRow(row);
        typename Tpetra::CrsMatrix<>::nonconst_global_inds_host_view_type indices;
        typename Tpetra::CrsMatrix<>::nonconst_values_host_view_type values;
        Kokkos::resize(indices, numEntries);
        Kokkos::resize(values, numEntries);

        // Get the global row copy from B
        B_t.Get_Tpetra_Matrix()->getGlobalRowCopy(row, indices, values, numEntries);
        for (int k = 0; k < numEntries; k++) {
          values[k] = alpha * values[k];
        }
        // Replace the entries in A with those from B
        Teuchos::ArrayView<const GO> indicesView(indices.data(), numEntries);
        Teuchos::ArrayView<const RealT> valuesView(values.data(), numEntries);
        A_->sumIntoGlobalValues(row, indicesView, valuesView);
      }
    }
    A_->fillComplete(); // Complete the fill process
  }

  int Get_Max_Nonzeros_Per_Row(void) const { return A_->getGlobalMaxNumRowEntries(); }

  void Begin_Fill(void) { A_->resumeFill(); }

  void End_Fill(void) {
    A_->fillComplete(); // Complete the fill process
  }

  HDSA::Ptr<const Teuchos::Comm<int>> Get_Comm(void) const { return A_->getComm(); }

  HDSA::Ptr<Tpetra::CrsMatrix<RealT, LO, GO, Node>> Get_Tpetra_Matrix(void) const { return A_; }

  HDSA::Ptr<HDSA::Incomplete_Chol_Factor<RealT>> Get_Incomplete_Chol_Factor(void) const {
    if (!HDSA::Sparse_Matrix<RealT>::Is_Symmetric()) {
      HDSA_TEST_FOR_EXCEPTION(
          true, std::logic_error,
          "Error in HDSA::Sparse_Matrix_Trilinos: Get_Incomplete_Chol_Factor() was called on a non-symmetric matrix"
              << std::endl);
    }
    HDSA::Ptr<HDSA::Incomplete_Chol_Factor<RealT>> L = HDSA::makePtr<HDSA::Incomplete_Chol_Factor_Trilinos<RealT>>(A_);
    return L;
  }

  HDSA::Ptr<HDSA::Sparse_Matrix_Solver<RealT>>
  Get_Sparse_Matrix_Solver(bool use_direct = true, int verbosity = 0, std::ostream& out_stream = std::cout,
                           const std::string solver_type_message = "") const {
    HDSA::Ptr<HDSA::Sparse_Matrix_Solver<RealT>> solver = HDSA::makePtr<HDSA::Sparse_Matrix_Solver_Trilinos<RealT>>(
        A_, use_direct, verbosity, out_stream, solver_type_message);
    return solver;
  }
};

} // namespace HDSA

#endif
