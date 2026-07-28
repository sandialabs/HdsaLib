/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_DENSE_MATRIX_HPP
#define HDSA_DENSE_MATRIX_HPP

#include "Teuchos_SerialDenseMatrix.hpp"

namespace HDSA
{

  template <class RealT>
  class Dense_Matrix
  {

    HDSA::Ptr<Teuchos::SerialDenseMatrix<int, RealT>> A_;

  public:
    // Null constructor
    Dense_Matrix(void)
    {
    }

    // Constructor given matrix Dimensions
    Dense_Matrix(int m, int n)
    {
      A_ = HDSA::makePtr<Teuchos::SerialDenseMatrix<int, RealT>>(m, n);
    }

    ~Dense_Matrix()
    {
    }

    // Number of rows
    int Number_of_Rows(void) const
    {
      return A_->numRows();
    }

    // Number of columns
    int Number_of_Columns(void) const
    {
      return A_->numCols();
    }

    // Access the (i,j) element
    RealT operator()(int i, int j) const
    {
      return (*A_)(i, j);
    }

    // Overwrite the (i,j) element
    void Set_Entry(int i, int j, RealT val)
    {
      (*A_)(i, j) = val;
    }

    // Multiply this*B (a matrix multiply) with options to transpose this and/or B
    int Multiply(HDSA::Dense_Matrix<RealT> &C, const HDSA::Dense_Matrix<RealT> &B, bool A_Trans = false, bool B_Trans = false) const
    {
      int info;
      if (!A_Trans && !B_Trans)
      {
        // No transposes
        info = C.Get_Teuchos_Matrix()->multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, *A_, *B.Get_Teuchos_Matrix(), 0.0);
      }
      else if (A_Trans && !B_Trans)
      {
        // Transpose A and not B
        info = C.Get_Teuchos_Matrix()->multiply(Teuchos::TRANS, Teuchos::NO_TRANS, 1.0, *A_, *B.Get_Teuchos_Matrix(), 0.0);
      }
      else if (!A_Trans && B_Trans)
      {
        // Transpose B and not A
        info = C.Get_Teuchos_Matrix()->multiply(Teuchos::NO_TRANS, Teuchos::TRANS, 1.0, *A_, *B.Get_Teuchos_Matrix(), 0.0);
      }
      else
      {
        // Transpose both A and B
        info = C.Get_Teuchos_Matrix()->multiply(Teuchos::TRANS, Teuchos::TRANS, 1.0, *A_, *B.Get_Teuchos_Matrix(), 0.0);
      }
      return info;
    }

    void Assign(const HDSA::Dense_Matrix<RealT>& B) { A_->assign(*B.Get_Teuchos_Matrix()); }

    // Allocating matrix multiply convenience overload.
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Multiply(const HDSA::Dense_Matrix<RealT>& B, bool A_Trans = false,
                                                  bool B_Trans = false) const {
      const int C_rows = A_Trans ? Number_of_Columns() : Number_of_Rows();
      const int C_cols = B_Trans ? B.Number_of_Rows() : B.Number_of_Columns();
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> C = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(C_rows, C_cols);
      C->Zeros();
      Multiply(*C, B, A_Trans, B_Trans);
      return C;
    }

    void Zeros(void)
    {
      A_->putScalar(0.0);
    }

    void Scale(RealT alpha)
    {
      A_->scale(alpha);
    }

    HDSA::Ptr<Teuchos::SerialDenseMatrix<int, RealT>> Get_Teuchos_Matrix(void) const
    {
      return A_;
    }

    void Write_to_File(const std::string &name) const
    {
      int m = this->Number_of_Rows();
      int n = this->Number_of_Columns();
      std::ofstream fout;
      fout.open(name);
      for (int i = 0; i < m; i++)
      {
        for (int j = 0; j < n; j++)
        {
          fout << std::setprecision(16) << (*this)(i, j) << "  ";
        }
        fout << "  " << std::endl;
      }
      fout.close();
    }
  };

}

#endif
