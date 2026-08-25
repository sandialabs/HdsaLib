/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_LINEAR_ALGEBRA_HPP
#define HDSA_LINEAR_ALGEBRA_HPP

#include "BelosConfigDefs.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosBlockCGSolMgr.hpp"
#include "BelosBlockGmresSolMgr.hpp"

#include "Teuchos_SerialDenseMatrix.hpp"
#include "Teuchos_SerialDenseVector.hpp"
#include "Teuchos_LAPACK.hpp"
#include "Teuchos_SerialDenseSolver.hpp"
#include "Teuchos_SerialSpdDenseSolver.hpp"

#include "HDSA_Linear_Operator.hpp"
#include "HDSA_Dense_Matrix.hpp"
#include "HDSA_Vector.hpp"
#include "HDSA_Belos_Adapter.hpp"

namespace HDSA
{

  namespace Linear_Algebra
  {

    // Solve the linear system A*x = b
    template <class RealT>
    std::string Iterative_Linear_Solve(HDSA::Vector<RealT> &x, const HDSA::Vector<RealT> &b, const HDSA::Linear_Operator<RealT> &A,
                                       RealT tol, std::string solver = "CG", int verbosity = 0, std::ostream &out_stream = std::cout)
    {
      std::string output_message;
      // Build the problem matrix
      HDSA::Ptr<HDSA::Belos_Operator<RealT>> A_Belos = HDSA::makePtr<HDSA::Belos_Operator<RealT>>(&A);

      int frequency = 1; // how often residuals are printed by solver
      int blocksize = 1;
      int numrhs = 1;
      bool verbose = false;
      if (verbosity > 10)
      {
        verbose = true;
      }

      Teuchos::CommandLineProcessor cmdp(false, true);
      cmdp.setOption("verbose", "quiet", &verbose, "Print messages and results.");
      cmdp.setOption("frequency", &frequency, "Solvers frequency for printing residuals (#iters).");
      cmdp.setOption("tol", &tol, "Relative residual tolerance used by CG solver.");
      cmdp.setOption("num-rhs", &numrhs, "Number of right-hand sides to be solved for.");
      cmdp.setOption("blocksize", &blocksize, "Block size used by CG .");

      int maxits = b.Dimension();
      Teuchos::ParameterList belosList;
      belosList.set("Block Size", blocksize);      // Blocksize to be used by iterative solver
      belosList.set("Num Blocks", maxits);         // Number of blocks
      belosList.set("Maximum Iterations", maxits); // Maximum number of iterations allowed
      belosList.set("Convergence Tolerance", tol); // Relative convergence tolerance requested
      if (verbose)
      {
        belosList.set("Verbosity", Belos::Errors + Belos::Warnings + Belos::StatusTestDetails + Belos::IterationDetails);
        belosList.set("Output Style", Belos::Brief);
        belosList.set("Output Frequency", frequency);
        belosList.set("Explicit Residual Test", true);
      }
      else
        belosList.set("Verbosity", Belos::Errors + Belos::Warnings);

      HDSA::Ptr<HDSA::Belos_Vector<RealT>> soln = HDSA::makePtr<HDSA::Belos_Vector<RealT>>(b, numrhs);
      HDSA::Ptr<HDSA::Belos_Vector<RealT>> rhs = HDSA::makePtr<HDSA::Belos_Vector<RealT>>(b, numrhs);
      rhs->vec[0]->Set(b);

      RealT rhs_Norm = b.Norm();
      if (rhs_Norm != 0.0)
      {
        rhs->vec[0]->Scale(1.0 / rhs_Norm);
        soln->vec[0]->Scale(0.0);

        HDSA::Ptr<Belos::LinearProblem<RealT, Belos::MultiVec<RealT>, Belos::Operator<RealT>>> problem =
            HDSA::makePtr<Belos::LinearProblem<RealT, Belos::MultiVec<RealT>, Belos::Operator<RealT>>>(A_Belos, soln, rhs);
        bool set = problem->setProblem();
        if (set == false)
        {
          if (verbose)
          {
            HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                                    "Error in HDSA::Linear_Algebra: Belos::LinearProblem failed to Set up correctly" << std::endl);
          }
          verbose = true;
        }

        HDSA::Ptr<Belos::SolverManager<RealT, Belos::MultiVec<RealT>, Belos::Operator<RealT>>> belos_solver;
        if (solver == "CG")
        {
          belos_solver = HDSA::makePtr<Belos::BlockCGSolMgr<RealT, Belos::MultiVec<RealT>, Belos::Operator<RealT>>>(problem, HDSA::makePtrFromRef(belosList));
        }
        else if (solver == "GMRES")
        {
          belos_solver = HDSA::makePtr<Belos::BlockGmresSolMgr<RealT, Belos::MultiVec<RealT>, Belos::Operator<RealT>>>(problem, HDSA::makePtrFromRef(belosList));
        }
        else
        {
          HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                                  "Error in HDSA::Linear_Algebra: Linear solver is not specified correctly" << std::endl);
        }

        Belos::ReturnType ret = belos_solver->solve();

        if (ret != Belos::Converged)
        {
          out_stream << "Belos solver did not converge for linear solve" << std::endl;
        }

        x.Set(*soln->vec[0]);
        x.Scale(rhs_Norm);

        // Test achievedTol output
        if (verbosity > 2)
        {
          RealT ach_tol = belos_solver->achievedTol();
          int num_iter = belos_solver->getNumIters();
          std::ostringstream oss;
          oss << std::scientific << ach_tol;
          std::string tol_str = oss.str();
          output_message = "Iterative_Linear_Solve achieved the tolerance " + tol_str + " with " + std::to_string(num_iter) + " iterations";
        }
      }
      else
      {
        x.Zeros();
        output_message = "Iterative_Linear_Solve returned the null solution";
      }
      return output_message;
    }

    // Compute the SVD of A=U*S*V^T where U and V are orthogonal matrix and S is a diagaonal matrix, stored as a nx1 matrix
    template <class RealT>
    void SVD(const HDSA::Dense_Matrix<RealT> &A, HDSA::Dense_Matrix<RealT> &U, HDSA::Dense_Matrix<RealT> &VT, HDSA::Dense_Matrix<RealT> &S)
    {
      int m = A.Number_of_Rows();
      int n = A.Number_of_Columns();
      Teuchos::LAPACK<int, RealT> lapack;
      char JOBU = 'S';
      char JOBVT = 'S';
      HDSA::Ptr<Teuchos::SerialDenseVector<int, RealT>> S_vec = HDSA::makePtr<Teuchos::SerialDenseVector<int, RealT>>(n);
      int LWORK = std::max(1, std::max(3 * std::min(m, n) + std::max(m, n), 5 * std::min(m, n))) + 1;
      HDSA::Ptr<Teuchos::SerialDenseMatrix<int, RealT>> WORK = HDSA::makePtr<Teuchos::SerialDenseMatrix<int, RealT>>(LWORK, 1);
      HDSA::Ptr<Teuchos::SerialDenseMatrix<int, RealT>> RWORK = HDSA::makePtr<Teuchos::SerialDenseMatrix<int, RealT>>(LWORK, 1);
      int info;
      lapack.GESVD(JOBU, JOBVT, m, n, A.Get_Teuchos_Matrix()->values(), m, (*S_vec).values(), U.Get_Teuchos_Matrix()->values(), m, VT.Get_Teuchos_Matrix()->values(), n, (*WORK).values(), LWORK, (*RWORK).values(), &info);
      // Yields the decomposition A = U*diag(S)*VT, note VT is the transpose of V
      for (int k = 0; k < n; k++)
      {
        S.Set_Entry(k, 0, (*S_vec)(k));
      }
    }

    // Solve upper triangular system R*x=b for upper triangluar matrix R
    template <class RealT>
    void Upper_Tri_Solve(HDSA::Dense_Matrix<RealT> &x, const HDSA::Dense_Matrix<RealT> &b, const HDSA::Dense_Matrix<RealT> &R)
    {
      int n = R.Number_of_Rows();
      for (int c = 0; c < x.Number_of_Columns(); c++)
      {
        for (int k = n - 1; k >= 0; k--)
        {
          RealT val = b(k, c);
          for (int j = k + 1; j < n; j++)
          {
            val -= x(j, c) * R(k, j);
          }
          val = val / R(k, k);
          x.Set_Entry(k, c, val);
        }
      }
    }

    // Compute Cholesky factorization of A, A=R^T*R
    template <class RealT>
    int Cholesky_Factorization(const HDSA::Dense_Matrix<RealT> &A, HDSA::Dense_Matrix<RealT> &R)
    {
      int n = A.Number_of_Rows();
      Teuchos::SerialSpdDenseSolver<int, RealT> Chol_Solve;
      HDSA::Ptr<Teuchos::SerialSymDenseMatrix<int, RealT>> C = HDSA::makePtr<Teuchos::SerialSymDenseMatrix<int, RealT>>(n, n);
      for (int i = 0; i < n; i++)
      {
        for (int j = 0; j < n; j++)
        {
          (*C)(i, j) = A(i, j);
        }
      }
      Chol_Solve.setMatrix(C);
      int info = Chol_Solve.factor();
      HDSA::Ptr<Teuchos::SerialSymDenseMatrix<int, RealT>> Rc = Chol_Solve.getFactoredMatrix(); // R should be upper triangular. The R returned is symmetric, its upper half is what we need.
      for (int i = 0; i < n; i++)
      {
        for (int j = i; j < n; j++)
        {
          R.Set_Entry(i, j, (*Rc)(i, j));
        }
      }
      return info;
    }

    // Solve the symmetric linear system A x = b via a direct method
    template <class RealT>
    void Symmetric_Direct_Linear_Solve(const HDSA::Dense_Matrix<RealT> &A, HDSA::Dense_Matrix<RealT> &x, const HDSA::Dense_Matrix<RealT> &b)
    {
      int n = b.Number_of_Rows();
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> R = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n, n);
      HDSA::Linear_Algebra::Cholesky_Factorization<RealT>(A, *R);
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> y = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n, x.Number_of_Columns());
      for (int c = 0; c < x.Number_of_Columns(); c++)
      {
        for (int i = 0; i < n; i++)
        {
          RealT val = b(i, c);
          for (int j = 0; j < i; j++)
          {
            val -= (*y)(j, c) * (*R)(j, i);
          }
          val = val / (*R)(i, i);
          y->Set_Entry(i, c, val);
        }
      }
      HDSA::Linear_Algebra::Upper_Tri_Solve<RealT>(x, *y, *R);
    }

    // Solve the symmetric linear system R^T*R x = b via a direct method
    template <class RealT>
    void Symmetric_Direct_Linear_Solve_Prefactor(const HDSA::Dense_Matrix<RealT> &R, HDSA::Dense_Matrix<RealT> &x, const HDSA::Dense_Matrix<RealT> &b)
    {
      int n = b.Number_of_Rows();
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> y = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(n, x.Number_of_Columns());
      for (int c = 0; c < x.Number_of_Columns(); c++)
      {
        for (int i = 0; i < n; i++)
        {
          RealT val = b(i, c);
          for (int j = 0; j < i; j++)
          {
            val -= (*y)(j, c) * R(j, i);
          }
          val = val / R(i, i);
          y->Set_Entry(i, c, val);
        }
      }
      HDSA::Linear_Algebra::Upper_Tri_Solve<RealT>(x, *y, R);
    }

    // Compute eigenvalue decomposition A=V*S*V^T for a symmetric matrix A, store eigenvalues in a size nx1 matrix S
    template <class RealT>
    void Symmetric_Eig_Decomposition(const HDSA::Dense_Matrix<RealT> &A, HDSA::Dense_Matrix<RealT> &V, HDSA::Dense_Matrix<RealT> &S)
    {
      int n = A.Number_of_Rows();
      HDSA::Ptr<Teuchos::SerialDenseMatrix<int, RealT>> B = HDSA::makePtr<Teuchos::SerialDenseMatrix<int, RealT>>(n, n);
      for (int i = 0; i < n; i++)
      {
        for (int j = 0; j < n; j++)
        {
          (*B)(i, j) = A(i, j);
        }
      }
      HDSA::Ptr<Teuchos::SerialDenseVector<int, RealT>> S_rev = HDSA::makePtr<Teuchos::SerialDenseVector<int, RealT>>(n); // SYEV outputs eigenvalues if reverse order
      Teuchos::LAPACK<int, RealT> lapack;
      char JOBZ = 'V';
      char UPLO = 'U';
      HDSA::Ptr<Teuchos::SerialDenseMatrix<int, RealT>> WORK = HDSA::makePtr<Teuchos::SerialDenseMatrix<int, RealT>>(3 * n, 3 * n);
      int lwork = (*WORK).stride();
      int info;
      lapack.SYEV(JOBZ, UPLO, n, (*B).values(), n, (*S_rev).values(), (*WORK).values(), lwork, &info);
      for (int j = 0; j < n; j++)
      {
        S.Set_Entry(j, 0, (*S_rev)(n - 1 - j));
        RealT sign = 1.0;
        if ((*B)(0, n - 1 - j) < 0.0)
        {
          sign = -1.0;
        }
        for (int i = 0; i < n; i++)
        {
          V.Set_Entry(i, j, sign * (*B)(i, n - 1 - j));
        }
      }
    }

    // Compute eigenvalue decomposition A=M*V*S*V^T*M for a symmetric matrix A and symmetric positive definite matrix M, store eigenvalues in a size nx1 matrix S
    template <class RealT>
    void Symmetric_Gen_Eig_Decomposition(const HDSA::Dense_Matrix<RealT> &A, const HDSA::Dense_Matrix<RealT> &M, HDSA::Dense_Matrix<RealT> &V, HDSA::Dense_Matrix<RealT> &S)
    {
      int n = A.Number_of_Rows();
      HDSA::Ptr<Teuchos::SerialDenseMatrix<int, RealT>> B = HDSA::makePtr<Teuchos::SerialDenseMatrix<int, RealT>>(n, n);
      HDSA::Ptr<Teuchos::SerialDenseMatrix<int, RealT>> C = HDSA::makePtr<Teuchos::SerialDenseMatrix<int, RealT>>(n, n);
      for (int i = 0; i < n; i++)
      {
        for (int j = 0; j < n; j++)
        {
          (*B)(i, j) = A(i, j);
          (*C)(i, j) = M(i, j);
        }
      }
      HDSA::Ptr<Teuchos::SerialDenseVector<int, RealT>> S_rev = HDSA::makePtr<Teuchos::SerialDenseVector<int, RealT>>(n); // SYGV outputs eigenvalues if reverse order
      Teuchos::LAPACK<int, RealT> lapack;
      int itype = 1;
      char JOBZ = 'V';
      char UPLO = 'U';
      HDSA::Ptr<Teuchos::SerialDenseMatrix<int, RealT>> WORK = HDSA::makePtr<Teuchos::SerialDenseMatrix<int, RealT>>(3 * n, 3 * n);
      int lwork = (*WORK).stride();
      int info;
      // lapack.SYEV(JOBZ, UPLO, n, (*B).values(), n, (*S_rev).values(), (*WORK).values(), lwork, &info);
      lapack.SYGV(itype, JOBZ, UPLO, n, (*B).values(), n, (*C).values(), n, (*S_rev).values(), (*WORK).values(), lwork, &info);

      for (int j = 0; j < n; j++)
      {
        S.Set_Entry(j, 0, (*S_rev)(n - 1 - j));
        RealT sign = 1.0;
        if ((*B)(0, n - 1 - j) < 0.0)
        {
          sign = -1.0;
        }
        for (int i = 0; i < n; i++)
        {
          V.Set_Entry(i, j, sign * (*B)(i, n - 1 - j));
        }
      }
    }

  }

}

#endif
