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

namespace HDSA
{

template <class RealT, class LO = Tpetra::Map<>::local_ordinal_type, class GO = Tpetra::Map<>::global_ordinal_type, class Node = Tpetra::Map<>::node_type> class Sparse_Matrix_Trilinos : public HDSA::Sparse_Matrix<RealT>
{

  private:
    HDSA::Ptr<Tpetra::CrsMatrix<RealT, LO, GO, Node>> A_;

  public:
    Sparse_Matrix_Trilinos(HDSA::Ptr<Tpetra::CrsMatrix<RealT, LO, GO, Node>> &A, bool is_symmetric = false) : A_(A)
    {
        if (is_symmetric)
        {
            HDSA::Sparse_Matrix<RealT>::Set_Symmetric();
        }
    }

    Sparse_Matrix_Trilinos(HDSA::Ptr<Tpetra::CrsMatrix<RealT, LO, GO, Node>> &A, HDSA::Ptr<const Tpetra::Map<LO, GO, Node>> &target_map, bool is_symmetric = false)
    {
        if (is_symmetric)
        {
            HDSA::Sparse_Matrix<RealT>::Set_Symmetric();
        }

        TEUCHOS_TEST_FOR_EXCEPTION(A.is_null(), std::invalid_argument, "Input matrix A is null.");

        TEUCHOS_TEST_FOR_EXCEPTION(target_map.is_null(), std::invalid_argument, "Input target_map is null.");

        TEUCHOS_TEST_FOR_EXCEPTION(!A->isFillComplete(), std::runtime_error,
                                   "Input matrix A must be fillComplete before it can be safely "
                                   "reindexed using local row/column indices.");

        HDSA::Ptr<const Tpetra::Map<LO, GO, Node>> old_row_map = A->getRowMap();
        HDSA::Ptr<const Tpetra::Map<LO, GO, Node>> old_col_map = A->getColMap();
        HDSA::Ptr<const Tpetra::Map<LO, GO, Node>> old_domain_map = A->getDomainMap();
        HDSA::Ptr<const Tpetra::Map<LO, GO, Node>> old_range_map = A->getRangeMap();

        TEUCHOS_TEST_FOR_EXCEPTION(old_row_map.is_null() || old_col_map.is_null() || old_domain_map.is_null() || old_range_map.is_null(), std::runtime_error,
                                   "Input matrix A has a null row, column, domain, or range map.");

        /*
         * With only one target_map, this routine assumes A is being reindexed
         * as a square operator:
         *
         *   old row/range/domain space  ---> target_map
         *   old column space            ---> corresponding imported target GIDs
         *
         * Therefore target_map must have the same local and global number of
         * entries as the old row/domain/range maps.
         */

        TEUCHOS_TEST_FOR_EXCEPTION(old_row_map->getGlobalNumElements() != target_map->getGlobalNumElements(), std::runtime_error, "old_row_map and target_map have different global numbers of elements.");

        TEUCHOS_TEST_FOR_EXCEPTION(old_domain_map->getGlobalNumElements() != target_map->getGlobalNumElements(), std::runtime_error, "old_domain_map and target_map have different global numbers of elements.");

        TEUCHOS_TEST_FOR_EXCEPTION(old_range_map->getGlobalNumElements() != target_map->getGlobalNumElements(), std::runtime_error, "old_range_map and target_map have different global numbers of elements.");

        TEUCHOS_TEST_FOR_EXCEPTION(old_row_map->getLocalNumElements() != target_map->getLocalNumElements(), std::runtime_error,
                                   "old_row_map and target_map have different local numbers of elements. "
                                   "This constructor assumes local-LID correspondence.");

        TEUCHOS_TEST_FOR_EXCEPTION(old_domain_map->getLocalNumElements() != target_map->getLocalNumElements(), std::runtime_error,
                                   "old_domain_map and target_map have different local numbers of elements. "
                                   "This constructor assumes local-LID correspondence.");

        TEUCHOS_TEST_FOR_EXCEPTION(old_range_map->getLocalNumElements() != target_map->getLocalNumElements(), std::runtime_error,
                                   "old_range_map and target_map have different local numbers of elements. "
                                   "This constructor assumes local-LID correspondence.");

        /*
         * Build a vector on the old domain map whose value at local index lid is
         * the corresponding target-map GID:
         *
         *   old domain GID old_domain_map->getGlobalElement(lid)
         *   maps to
         *   target GID     target_map->getGlobalElement(lid)
         *
         * This avoids assuming any formula for the GID transformation.
         */

        HDSA::Ptr<Tpetra::Vector<RealT, LO, GO, Node>> domain_gid_translation = HDSA::makePtr<Tpetra::Vector<RealT, LO, GO, Node>>(old_domain_map);
        Teuchos::ArrayRCP<RealT> domain_translation_data = domain_gid_translation->getDataNonConst(0);
        const size_t num_local_domain = old_domain_map->getLocalNumElements();
        for (size_t lid = 0; lid < num_local_domain; ++lid)
        {
            const LO local_lid = static_cast<LO>(lid);
            domain_translation_data[lid] = target_map->getGlobalElement(local_lid);
        }

        /*
         * Import that translation vector to the old column map.
         *
         * After this import:
         *
         *   col_gid_translation[old_col_lid]
         *
         * gives the new target-map GID corresponding to the old column-map entry
         *
         *   old_col_map->getGlobalElement(old_col_lid).
         */

        HDSA::Ptr<Tpetra::Vector<RealT, LO, GO, Node>> col_gid_translation = HDSA::makePtr<Tpetra::Vector<RealT, LO, GO, Node>>(old_col_map);
        HDSA::Ptr<Tpetra::Import<LO, GO, Node>> domain_to_col_importer = HDSA::makePtr<Tpetra::Import<LO, GO, Node>>(old_domain_map, old_col_map);
        col_gid_translation->doImport(*domain_gid_translation, *domain_to_col_importer, Tpetra::INSERT);
        Teuchos::ArrayRCP<const RealT> col_translation_data = col_gid_translation->getData(0);

        /*
         * Construct the new matrix on target_map.
         *
         * We use the same maximum number of entries per row as the old matrix.
         */

        const size_t max_entries_per_row = A->getLocalMaxNumRowEntries();

        A_ = HDSA::makePtr<Tpetra::CrsMatrix<RealT, LO, GO, Node>>(target_map, max_entries_per_row);

        /*
         * Copy rows.  For each old row local ID, use the same local ID in
         * target_map to get the new row GID.
         *
         * For each old local column ID, use the imported column translation vector
         * to get the new column GID.
         */

        typename Tpetra::CrsMatrix<RealT, LO, GO, Node>::nonconst_local_inds_host_view_type old_local_cols("old_local_cols", max_entries_per_row);
        typename Tpetra::CrsMatrix<RealT, LO, GO, Node>::nonconst_values_host_view_type values("values", max_entries_per_row);
        const size_t num_local_rows = old_row_map->getLocalNumElements();
        for (size_t row_lid_size = 0; row_lid_size < num_local_rows; ++row_lid_size)
        {
            const LO old_row_lid = static_cast<LO>(row_lid_size);
            const GO new_row_gid = target_map->getGlobalElement(old_row_lid);

            size_t num_entries = 0;
            A->getLocalRowCopy(old_row_lid, old_local_cols, values, num_entries);
            if (num_entries == 0)
            {
                continue;
            }

            Teuchos::Array<GO> new_global_cols(num_entries);
            Teuchos::Array<RealT> new_values(num_entries);
            for (size_t k = 0; k < num_entries; ++k)
            {
                const LO old_col_lid = old_local_cols(k);

                TEUCHOS_TEST_FOR_EXCEPTION(old_col_lid < 0 || static_cast<size_t>(old_col_lid) >= old_col_map->getLocalNumElements(), std::runtime_error,
                                           "Encountered invalid old column local index while reindexing matrix.");

                new_global_cols[k] = col_translation_data[static_cast<size_t>(old_col_lid)];
                new_values[k] = values(k);
            }

            A_->insertGlobalValues(new_row_gid, new_global_cols.view(0, num_entries), new_values.view(0, num_entries));
        }

        /*
         * Complete the new matrix using target_map as both domain and range.
         * This is appropriate for square mass/stiffness/Dirichlet-style operators.
         */

        A_->fillComplete(target_map, target_map);
    }

    virtual ~Sparse_Matrix_Trilinos()
    {
    }

    HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> Clone(int max_entries_per_row = 0) const
    {
        if (max_entries_per_row == 0)
        {
            max_entries_per_row = A_->getGlobalMaxNumRowEntries();
        }
        HDSA::Ptr<Tpetra::CrsMatrix<RealT, LO, GO, Node>> B = HDSA::makePtr<Tpetra::CrsMatrix<RealT, LO, GO, Node>>(A_->getRowMap(), max_entries_per_row);
        B->scale(0.0);
        HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> B_sm = HDSA::makePtr<HDSA::Sparse_Matrix_Trilinos<RealT>>(B);
        return B_sm;
    }

    // Compute C = this * B, with options for transposes
    void Matrix_Matrix_Multiply(HDSA::Sparse_Matrix<RealT> &C, const HDSA::Sparse_Matrix<RealT> &B, bool A_trans = false, bool B_trans = false) const
    {
        HDSA::Ptr<Tpetra::CrsMatrix<RealT, LO, GO, Node>> B_tpetra = dynamic_cast<const HDSA::Sparse_Matrix_Trilinos<RealT> &>(B).Get_Tpetra_Matrix();
        HDSA::Ptr<Tpetra::CrsMatrix<RealT, LO, GO, Node>> C_tpetra = dynamic_cast<const HDSA::Sparse_Matrix_Trilinos<RealT> &>(C).Get_Tpetra_Matrix();
        Tpetra::MatrixMatrix::Multiply(*A_, A_trans, *B_tpetra, B_trans, *C_tpetra);
    }

    void Apply(HDSA::Vector<RealT> &x_out, const HDSA::Vector<RealT> &x_in) const
    {
        const HDSA::Tpetra_Vector<RealT> &ex_in = dynamic_cast<const HDSA::Tpetra_Vector<RealT> &>(x_in);
        HDSA::Tpetra_Vector<RealT> &ex_out = dynamic_cast<HDSA::Tpetra_Vector<RealT> &>(x_out);
        A_->apply(*ex_in.getVector(), *ex_out.getVector());
    }

    void Set(HDSA::Sparse_Matrix<RealT> &B)
    {
        // Prepare A for updates
        A_->resumeFill();

        HDSA::Sparse_Matrix_Trilinos<RealT> B_t = dynamic_cast<HDSA::Sparse_Matrix_Trilinos<RealT> &>(B);
        // Loop over each row of B and copy its entries to A
        for (Tpetra::global_size_t row = 0; row < B_t.Get_Tpetra_Matrix()->getGlobalNumRows(); ++row)
        {
            if (A_->getRowMap()->isNodeGlobalElement(row))
            {
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

    void Set_Diagonal(HDSA::Vector<RealT> &vec, bool reciprocate_diag)
    {
        HDSA::Tpetra_Vector<RealT> tpetra_vec = dynamic_cast<HDSA::Tpetra_Vector<RealT> &>(vec);
        HDSA::Ptr<Tpetra::MultiVector<RealT, LO, GO, Node>> d = tpetra_vec.getVector();
        auto view = d->getLocalViewHost(Tpetra::Access::ReadOnly);

        A_->resumeFill();
        for (Tpetra::global_size_t row = 0; row < A_->getGlobalNumRows(); ++row)
        {
            if (A_->getRowMap()->isNodeGlobalElement(row))
            {
                typename Tpetra::CrsMatrix<>::nonconst_global_inds_host_view_type indices;
                typename Tpetra::CrsMatrix<>::nonconst_values_host_view_type values;
                Kokkos::resize(indices, 1);
                Kokkos::resize(values, 1);
                indices[0] = row;
                LO i = d->getMap()->getLocalElement(row);
                if (reciprocate_diag)
                {
                    values[0] = 1.0 / view(i, 0);
                }
                else
                {
                    values[0] = view(i, 0);
                }
                A_->insertGlobalValues(row, 1, &values[0], &indices[0]);
            }
        }

        // Complete the fill process
        A_->fillComplete();
    }

    void Scaled_Plus(const RealT &alpha, const HDSA::Sparse_Matrix<RealT> &B)
    {
        A_->resumeFill(); // Prepare A for updates
        // Loop over each row of B and copy its entries to A
        const HDSA::Sparse_Matrix_Trilinos<RealT> B_t = dynamic_cast<const HDSA::Sparse_Matrix_Trilinos<RealT> &>(B);
        for (Tpetra::global_size_t row = 0; row < B_t.Get_Tpetra_Matrix()->getGlobalNumRows(); ++row)
        {
            if (A_->getRowMap()->isNodeGlobalElement(row))
            {
                size_t numEntries = B_t.Get_Tpetra_Matrix()->getNumEntriesInGlobalRow(row);
                typename Tpetra::CrsMatrix<>::nonconst_global_inds_host_view_type indices;
                typename Tpetra::CrsMatrix<>::nonconst_values_host_view_type values;
                Kokkos::resize(indices, numEntries);
                Kokkos::resize(values, numEntries);

                // Get the global row copy from B
                B_t.Get_Tpetra_Matrix()->getGlobalRowCopy(row, indices, values, numEntries);
                for (int k = 0; k < numEntries; k++)
                {
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

    int Get_Max_Nonzeros_Per_Row(void) const
    {
        return A_->getGlobalMaxNumRowEntries();
    }

    void Begin_Fill(void)
    {
        A_->resumeFill();
    }

    void End_Fill(void)
    {
        A_->fillComplete(); // Complete the fill process
    }

    HDSA::Ptr<const Teuchos::Comm<int>> Get_Comm(void) const
    {
        return A_->getComm();
    }

    HDSA::Ptr<Tpetra::CrsMatrix<RealT, LO, GO, Node>> Get_Tpetra_Matrix(void) const
    {
        return A_;
    }

    HDSA::Ptr<HDSA::Incomplete_Chol_Factor<RealT>> Get_Incomplete_Chol_Factor(void) const
    {
        if (!HDSA::Sparse_Matrix<RealT>::Is_Symmetric())
        {
            HDSA_TEST_FOR_EXCEPTION(true, std::logic_error, "Error in HDSA::Sparse_Matrix_Trilinos: Get_Incomplete_Chol_Factor() was called on a non-symmetric matrix" << std::endl);
        }
        HDSA::Ptr<HDSA::Incomplete_Chol_Factor<RealT>> L = HDSA::makePtr<HDSA::Incomplete_Chol_Factor_Trilinos<RealT>>(A_);
        return L;
    }

    HDSA::Ptr<HDSA::Sparse_Matrix_Solver<RealT>> Get_Sparse_Matrix_Solver(bool use_direct = true, int verbosity = 0, std::ostream &out_stream = std::cout, const std::string solver_type_message = "") const
    {
        HDSA::Ptr<HDSA::Sparse_Matrix_Solver<RealT>> solver = HDSA::makePtr<HDSA::Sparse_Matrix_Solver_Trilinos<RealT>>(A_, use_direct, verbosity, out_stream, solver_type_message);
        return solver;
    }
};

} // namespace HDSA

#endif
