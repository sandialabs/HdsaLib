/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_ASSEMBLE_OPERATORS_HPP
#define HDSA_ASSEMBLE_OPERATORS_HPP

#include "HDSA_Sparse_Matrix_Trilinos.hpp"

template <class RealT>
class Assemble_Operators
{

private:
  int n_y_;                                    // Mesh resolution
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x_;     // Mesh nodes on [0,1]
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> M_;     // Mass matrix
  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> S_;     // Stiffness matrix
  HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> M_sm_; // Mass matrix
  HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> S_sm_; // Stiffness matrix

public:
  Assemble_Operators(HDSA::Ptr<const HDSA::Comm<int>> &comm, int n_y)
  {
    n_y_ = n_y;

    RealT h = 1.0 / static_cast<RealT>(n_y_ - 1);
    x_ = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_y_, 1);
    for (int k = 0; k < n_y_; k++)
    {
      x_->Set_Entry(k, 0, static_cast<RealT>(k) / static_cast<RealT>(n_y_ - 1));
    }

    M_ = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_y_, n_y_);
    M_->Set_Entry(0, 0, (1.0 / 3.0) * h);
    M_->Set_Entry(0, 1, (1.0 / 6.0) * h);
    for (int i = 1; i < n_y_ - 1; i++)
    {
      M_->Set_Entry(i, i, (2.0 / 3.0) * h);
      M_->Set_Entry(i, i - 1, (1.0 / 6.0) * h);
      M_->Set_Entry(i, i + 1, (1.0 / 6.0) * h);
    }
    M_->Set_Entry(n_y_ - 1, n_y_ - 2, (1.0 / 6.0) * h);
    M_->Set_Entry(n_y_ - 1, n_y_ - 1, (1.0 / 3.0) * h);

    S_ = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_y_, n_y_);
    S_->Set_Entry(0, 0, 1.0 / h);
    S_->Set_Entry(0, 1, -1.0 / h);
    for (int i = 1; i < n_y_ - 1; i++)
    {
      S_->Set_Entry(i, i, 2.0 / h);
      S_->Set_Entry(i, i - 1, -1.0 / h);
      S_->Set_Entry(i, i + 1, -1.0 / h);
    }
    S_->Set_Entry(n_y_ - 1, n_y_ - 2, -1.0 / h);
    S_->Set_Entry(n_y_ - 1, n_y_ - 1, 1.0 / h);

    const int m = n_y_;
    auto map = Tpetra::createUniformContigMap<Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type>(m, comm->Get_Teuchos_Communicator());
    HDSA::Ptr<Tpetra::CrsMatrix<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type>> M = HDSA::makePtr<Tpetra::CrsMatrix<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type>>(map, 3); // 3 is the maximum number of non-zero entries per row
    HDSA::Ptr<Tpetra::CrsMatrix<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type>> S = HDSA::makePtr<Tpetra::CrsMatrix<RealT, Tpetra::Map<>::local_ordinal_type, Tpetra::Map<>::global_ordinal_type>>(map, 3); // 3 is the maximum number of non-zero entries per row
    Teuchos::Array<Tpetra::Map<>::global_ordinal_type> cols0 = {0, 1};
    Teuchos::Array<RealT> vals0_M = {h / 3.0, h / 6.0};
    M->insertGlobalValues(0, cols0(), vals0_M());
    Teuchos::Array<RealT> vals0_S = {1.0 / h, -1.0 / h};
    S->insertGlobalValues(0, cols0(), vals0_S());
    for (int i = 1; i < m - 1; ++i)
    {
      Teuchos::Array<Tpetra::Map<>::global_ordinal_type> cols = {i - 1, i, i + 1};
      Teuchos::Array<RealT> vals_M = {h / 6.0, 2.0 * h / 3.0, h / 6.0};
      M->insertGlobalValues(i, cols(), vals_M());
      Teuchos::Array<RealT> vals_S = {-1.0 / h, 2.0 / h, -1.0 / h};
      S->insertGlobalValues(i, cols(), vals_S());
    }
    Teuchos::Array<Tpetra::Map<>::global_ordinal_type> colsm = {m - 2, m - 1};
    Teuchos::Array<RealT> valsm_M = {h / 6.0, h / 3.0};
    M->insertGlobalValues(m - 1, colsm(), valsm_M());
    Teuchos::Array<RealT> valsm_S = {-1.0 / h, 1.0 / h};
    S->insertGlobalValues(m - 1, colsm(), valsm_S());
    M->fillComplete();
    S->fillComplete();

    M_sm_ = HDSA::makePtr<HDSA::Sparse_Matrix_Trilinos<RealT>>(M);
    S_sm_ = HDSA::makePtr<HDSA::Sparse_Matrix_Trilinos<RealT>>(S);
  }

  virtual ~Assemble_Operators()
  {
  }

  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Matrix_Sqrt(const HDSA::Ptr<const HDSA::Dense_Matrix<RealT>> &A) const
  {
    int n = A->Number_of_Columns();
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> V = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n, n);
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> S = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n, 1);
    HDSA::Linear_Algebra::Symmetric_Eig_Decomposition<RealT>(*A, *V, *S);
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> D = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n, n);
    for (int i = 0; i < n; i++)
    {
      D->Set_Entry(i, i, std::sqrt((*S)(i, 0)));
    }
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> tmp = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n, n);
    V->Multiply(*tmp, *D);
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> A_sqrt = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n, n);
    tmp->Multiply(*A_sqrt, *V, false, true);
    return A_sqrt;
  }

  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Kronecker(const HDSA::Ptr<HDSA::Dense_Matrix<RealT>> &A, const HDSA::Ptr<HDSA::Dense_Matrix<RealT>> &B) const
  {
    int m_A = A->Number_of_Rows();
    int n_A = A->Number_of_Columns();
    int m_B = B->Number_of_Rows();
    int n_B = B->Number_of_Columns();

    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> C = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(m_A * m_B, n_A * n_B);
    for (int i_A = 0; i_A < m_A; i_A++)
    {
      for (int j_A = 0; j_A < n_A; j_A++)
      {
        for (int i_B = 0; i_B < m_B; i_B++)
        {
          for (int j_B = 0; j_B < n_B; j_B++)
          {
            RealT val = (*A)(i_A, j_A) * (*B)(i_B, j_B);
            int i = i_A * m_B + i_B;
            int j = j_A * n_B + j_B;
            C->Set_Entry(i, j, val);
          }
        }
      }
    }
    return C;
  }

  RealT Matrix_Difference(const HDSA::Ptr<const HDSA::Dense_Matrix<RealT>> &A, const HDSA::Ptr<const HDSA::Dense_Matrix<RealT>> &B) const
  {
    int n = A->Number_of_Rows();
    RealT val = 0.0;
    RealT normal = 0.0;
    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < n; j++)
      {
        val += std::pow((*A)(i, j) - (*B)(i, j), 2.0);
        normal += std::pow((*A)(i, j), 2.0);
      }
    }
    return val / normal;
  }

  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Inverse(const HDSA::Ptr<const HDSA::Dense_Matrix<RealT>> &A) const
  {
    int n = A->Number_of_Rows();
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Ainv = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n, n);

    for (int j = 0; j < n; j++)
    {
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n, 1);
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> b = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n, 1);
      b->Set_Entry(j, 0, 1.0);
      HDSA::Linear_Algebra::Symmetric_Direct_Linear_Solve<RealT>(*A, *x, *b);
      for (int i = 0; i < n; i++)
      {
        Ainv->Set_Entry(i, j, (*x)(i, 0));
      }
    }
    return Ainv;
  }

  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Get_Dense_E(RealT beta) const
  {
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> E = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_y_, n_y_);
    for (int i = 0; i < n_y_; i++)
    {
      for (int j = 0; j < n_y_; j++)
      {
        RealT val = beta * (*S_)(i, j) + (*M_)(i, j);
        E->Set_Entry(i, j, val);
      }
    }
    return E;
  }

  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Get_Dense_M_Lumped(void) const
  {
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> M_lumped = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n_y_, n_y_);
    for (int i = 0; i < n_y_; i++)
    {
      RealT val = 0.0;
      for (int j = 0; j < n_y_; j++)
      {
        val += (*M_)(i, j);
      }
      M_lumped->Set_Entry(i, i, val);
    }
    return M_lumped;
  }

  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Get_Dense_Mass_Matrix(void) const
  {
    return M_;
  }

  HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Get_Dense_Stiffness_Matrix(void) const
  {
    return S_;
  }

  HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> Get_Sparse_Mass_Matrix(void) const
  {
    return M_sm_;
  }

  HDSA::Ptr<HDSA::Sparse_Matrix<RealT>> Get_Sparse_Stiffness_Matrix(void) const
  {
    return S_sm_;
  }
};

#endif
