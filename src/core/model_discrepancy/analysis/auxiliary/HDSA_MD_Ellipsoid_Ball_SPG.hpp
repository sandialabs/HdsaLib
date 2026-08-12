/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_MD_ELLIPSOID_BALL_SPG_HPP
#define HDSA_MD_ELLIPSOID_BALL_SPG_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "HDSA_Dense_Matrix.hpp"
#include "HDSA_Dense_Vector_Utils.hpp"
#include "HDSA_Linear_Algebra.hpp"
#include "HDSA_Ptr.hpp"
#include "HDSA_Stack_Trace.hpp"
#include "HDSA_Vector.hpp"

namespace HDSA {

template <class RealT> class MD_Ellipsoid_Ball_SPG {
public:
  struct Projection_Data {
    int r;
    int p;
    RealT radius;

    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> beta_bar;
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Q;
    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> d;

    Projection_Data()
        : r(0), p(0), radius(static_cast<RealT>(0)), beta_bar(HDSA::nullPtr), Q(HDSA::nullPtr), d(HDSA::nullPtr) {}

    bool Is_Initialized() const {
      return r > 0 && p > 0 && radius >= static_cast<RealT>(0) && beta_bar != HDSA::nullPtr && Q != HDSA::nullPtr &&
             d != HDSA::nullPtr;
    }
  };

  struct SPG_Options {
    int max_iter = 1000;
    RealT pg_tol = static_cast<RealT>(1e-8);
    RealT armijo_c = static_cast<RealT>(1e-4);
    RealT backtrack_factor = static_cast<RealT>(0.5);
    int max_backtracks = 60;
    int nonmonotone_window = 10;
    RealT alpha_min = static_cast<RealT>(1e-12);
    RealT alpha_max = static_cast<RealT>(1e12);
    bool verbosity = false;
  };

  struct SPG_Info {
    int iter = 0;
    RealT final_objective = static_cast<RealT>(0);
    RealT projected_gradient_norm = static_cast<RealT>(0);
    bool converged = false;
    int n_fg_eval = 0;
    std::vector<RealT> f_hist;
  };

private:
  using DV = HDSA::Dense_Vector_Utils<RealT>;

  static void Check_SPG_Options(const SPG_Options& options) {
    HDSA_TEST_FOR_EXCEPTION(options.max_iter < 0, std::logic_error,
                            "Error in HDSA::MD_Ellipsoid_Ball_SPG::Minimize: "
                            "max_iter must be nonnegative."
                                << std::endl);

    HDSA_TEST_FOR_EXCEPTION(options.pg_tol < static_cast<RealT>(0), std::logic_error,
                            "Error in HDSA::MD_Ellipsoid_Ball_SPG::Minimize: "
                            "pg_tol must be nonnegative."
                                << std::endl);

    HDSA_TEST_FOR_EXCEPTION(options.armijo_c <= static_cast<RealT>(0) || options.armijo_c >= static_cast<RealT>(1),
                            std::logic_error,
                            "Error in HDSA::MD_Ellipsoid_Ball_SPG::Minimize: "
                            "armijo_c must be in (0,1)."
                                << std::endl);

    HDSA_TEST_FOR_EXCEPTION(options.backtrack_factor <= static_cast<RealT>(0) ||
                                options.backtrack_factor >= static_cast<RealT>(1),
                            std::logic_error,
                            "Error in HDSA::MD_Ellipsoid_Ball_SPG::Minimize: "
                            "backtrack_factor must be in (0,1)."
                                << std::endl);

    HDSA_TEST_FOR_EXCEPTION(options.max_backtracks < 0, std::logic_error,
                            "Error in HDSA::MD_Ellipsoid_Ball_SPG::Minimize: "
                            "max_backtracks must be nonnegative."
                                << std::endl);

    HDSA_TEST_FOR_EXCEPTION(options.nonmonotone_window <= 0, std::logic_error,
                            "Error in HDSA::MD_Ellipsoid_Ball_SPG::Minimize: "
                            "nonmonotone_window must be positive."
                                << std::endl);

    HDSA_TEST_FOR_EXCEPTION(options.alpha_min <= static_cast<RealT>(0) || options.alpha_max < options.alpha_min,
                            std::logic_error,
                            "Error in HDSA::MD_Ellipsoid_Ball_SPG::Minimize: "
                            "alpha bounds are invalid."
                                << std::endl);
  }

public:
  static Projection_Data Prepare_Projection(const HDSA::Dense_Matrix<RealT>& beta_bar, const int p, const RealT radius,
                                            const HDSA::Dense_Matrix<RealT>& K) {
    const int r = DV::Length(beta_bar);

    HDSA_TEST_FOR_EXCEPTION(r <= 0, std::logic_error,
                            "Error in HDSA::MD_Ellipsoid_Ball_SPG::Prepare_Projection: "
                            "beta_bar must be nonempty."
                                << std::endl);

    HDSA_TEST_FOR_EXCEPTION(p <= 0, std::logic_error,
                            "Error in HDSA::MD_Ellipsoid_Ball_SPG::Prepare_Projection: "
                            "p must be a positive integer."
                                << std::endl);

    HDSA_TEST_FOR_EXCEPTION(radius < static_cast<RealT>(0), std::logic_error,
                            "Error in HDSA::MD_Ellipsoid_Ball_SPG::Prepare_Projection: "
                            "The ellipsoid constraint radius must be nonnegative."
                                << std::endl);

    HDSA_TEST_FOR_EXCEPTION(K.Number_of_Rows() != r || K.Number_of_Columns() != r, std::logic_error,
                            "Error in HDSA::MD_Ellipsoid_Ball_SPG::Prepare_Projection: "
                            "K must be r-by-r, where r = length(beta_bar)."
                                << std::endl);

    Projection_Data projection_data;

    projection_data.r = r;
    projection_data.p = p;
    projection_data.radius = radius;

    projection_data.beta_bar = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(r, 1);

    for (int i = 0; i < r; ++i) {
      projection_data.beta_bar->Set_Entry(i, 0, DV::Get_Column_Major(beta_bar, i));
    }

    projection_data.Q = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(r, r);

    projection_data.d = HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(r, 1);

    HDSA::Linear_Algebra::Symmetric_Eig_Decomposition<RealT>(K, *projection_data.Q, *projection_data.d);

    RealT max_abs_eval = static_cast<RealT>(0);

    for (int i = 0; i < r; ++i) {
      max_abs_eval = std::max(max_abs_eval, std::abs((*projection_data.d)(i, 0)));
    }

    const RealT eval_tol =
        static_cast<RealT>(100) * std::numeric_limits<RealT>::epsilon() * std::max(static_cast<RealT>(1), max_abs_eval);

    for (int i = 0; i < r; ++i) {
      const RealT di = (*projection_data.d)(i, 0);

      HDSA_TEST_FOR_EXCEPTION(di < -eval_tol, std::logic_error,
                              "Error in HDSA::MD_Ellipsoid_Ball_SPG::Prepare_Projection: "
                              "Ellipsoid matrix has a negative eigenvalue outside roundoff tolerance."
                                  << std::endl);

      if (di < static_cast<RealT>(0)) { projection_data.d->Set_Entry(i, 0, static_cast<RealT>(0)); }
    }

    return projection_data;
  }

  static HDSA::Ptr<HDSA::Dense_Matrix<RealT>> Project(const HDSA::Dense_Matrix<RealT>& y,
                                                      const Projection_Data& projection_data) {
    HDSA_TEST_FOR_EXCEPTION(!projection_data.Is_Initialized(), std::logic_error,
                            "Error in HDSA::MD_Ellipsoid_Ball_SPG::Project: "
                            "projection_data are not initialized."
                                << std::endl);

    const int r = projection_data.r;
    const int p = projection_data.p;
    const bool matrix_layout = y.Number_of_Rows() == r && y.Number_of_Columns() == p;

    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x =
        HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(y.Number_of_Rows(), y.Number_of_Columns());

    x->Zeros();

    auto get_coeff = [&](const HDSA::Dense_Matrix<RealT>& mat, const int row, const int col) -> RealT {
      if (matrix_layout) { return mat(row, col); }

      return DV::Get_Column_Major(mat, row + r * col);
    };

    auto set_coeff = [&](HDSA::Dense_Matrix<RealT>& mat, const int row, const int col, const RealT val) {
      if (matrix_layout) {
        mat.Set_Entry(row, col, val);
      } else {
        DV::Set_Column_Major(mat, row + r * col, val);
      }
    };

    RealT max_abs_eval = static_cast<RealT>(0);

    for (int i = 0; i < r; ++i) {
      max_abs_eval = std::max(max_abs_eval, std::abs((*projection_data.d)(i, 0)));
    }

    const RealT eval_tol =
        static_cast<RealT>(100) * std::numeric_limits<RealT>::epsilon() * std::max(static_cast<RealT>(1), max_abs_eval);

    const RealT radius_sq = projection_data.radius * projection_data.radius;

    const RealT feasibility_tol =
        static_cast<RealT>(100) * std::numeric_limits<RealT>::epsilon() * std::max(static_cast<RealT>(1), radius_sq);

    std::vector<RealT> y_minus_beta_bar(r);
    std::vector<RealT> y_hat(r);
    std::vector<RealT> x_hat(r);

    for (int col = 0; col < p; ++col) {
      for (int i = 0; i < r; ++i) {
        y_minus_beta_bar[i] = get_coeff(y, i, col) - (*projection_data.beta_bar)(i, 0);
      }

      for (int i = 0; i < r; ++i) {
        RealT val = static_cast<RealT>(0);

        for (int k = 0; k < r; ++k) {
          val += (*projection_data.Q)(k, i) * y_minus_beta_bar[k];
        }

        y_hat[i] = val;
      }

      RealT current_radius_sq = static_cast<RealT>(0);

      for (int i = 0; i < r; ++i) {
        const RealT di = (*projection_data.d)(i, 0);
        current_radius_sq += di * y_hat[i] * y_hat[i];
      }

      if (current_radius_sq <= radius_sq + feasibility_tol) {
        for (int i = 0; i < r; ++i) {
          x_hat[i] = y_hat[i];
        }
      } else if (projection_data.radius == static_cast<RealT>(0)) {
        for (int i = 0; i < r; ++i) {
          const RealT di = (*projection_data.d)(i, 0);
          x_hat[i] = di > eval_tol ? static_cast<RealT>(0) : y_hat[i];
        }
      } else {
        auto phi = [&](const RealT lambda) -> RealT {
          RealT val = -radius_sq;

          for (int i = 0; i < r; ++i) {
            const RealT di = (*projection_data.d)(i, 0);
            const RealT denom = static_cast<RealT>(1) + static_cast<RealT>(2) * lambda * di;

            val += di * y_hat[i] * y_hat[i] / (denom * denom);
          }

          return val;
        };

        RealT lambda_lo = static_cast<RealT>(0);
        RealT lambda_hi = static_cast<RealT>(1);

        int expansion_count = 0;

        while (phi(lambda_hi) > static_cast<RealT>(0) && expansion_count < 200) {
          lambda_hi *= static_cast<RealT>(2);
          ++expansion_count;
        }

        HDSA_TEST_FOR_EXCEPTION(expansion_count >= 200, std::logic_error,
                                "Error in HDSA::MD_Ellipsoid_Ball_SPG::Project: "
                                "Failed to bracket ellipsoid projection multiplier."
                                    << std::endl);

        for (int iter = 0; iter < 100; ++iter) {
          const RealT lambda_mid = static_cast<RealT>(0.5) * (lambda_lo + lambda_hi);

          if (phi(lambda_mid) > static_cast<RealT>(0)) {
            lambda_lo = lambda_mid;
          } else {
            lambda_hi = lambda_mid;
          }
        }

        const RealT lambda = lambda_hi;

        for (int i = 0; i < r; ++i) {
          const RealT di = (*projection_data.d)(i, 0);
          const RealT denom = static_cast<RealT>(1) + static_cast<RealT>(2) * lambda * di;

          x_hat[i] = y_hat[i] / denom;
        }
      }

      for (int row = 0; row < r; ++row) {
        RealT val = (*projection_data.beta_bar)(row, 0);

        for (int k = 0; k < r; ++k) {
          val += (*projection_data.Q)(row, k) * x_hat[k];
        }

        set_coeff(*x, row, col, val);
      }
    }

    return x;
  }

  static HDSA::Ptr<HDSA::Dense_Matrix<RealT>>
  Minimize(const std::function<RealT(const HDSA::Dense_Matrix<RealT>& x, HDSA::Dense_Matrix<RealT>& grad)>& objective,
           const HDSA::Dense_Matrix<RealT>& x0, const Projection_Data& projection_data, SPG_Info& info,
           const SPG_Options& options = SPG_Options()) {
    HDSA_TEST_FOR_EXCEPTION(!projection_data.Is_Initialized(), std::logic_error,
                            "Error in HDSA::MD_Ellipsoid_Ball_SPG::Minimize: "
                            "projection_data are not initialized."
                                << std::endl);

    HDSA_TEST_FOR_EXCEPTION(!objective, std::logic_error,
                            "Error in HDSA::MD_Ellipsoid_Ball_SPG::Minimize: "
                            "objective callback is empty."
                                << std::endl);

    Check_SPG_Options(options);

    const int expected_len = projection_data.r * projection_data.p;

    HDSA_TEST_FOR_EXCEPTION(DV::Length(x0) != expected_len, std::logic_error,
                            "Error in HDSA::MD_Ellipsoid_Ball_SPG::Minimize: "
                            "Initial point length is incompatible with projection data."
                                << std::endl);

    info = SPG_Info();

    HDSA::Dense_Matrix<RealT> x = *(Project(x0, projection_data));
    HDSA::Dense_Matrix<RealT> g = x.Clone(false);
    g.Zeros();

    RealT f = objective(x, g);
    info.n_fg_eval = 1;
    info.f_hist.push_back(f);

    HDSA_TEST_FOR_EXCEPTION(!std::isfinite(f), std::logic_error,
                            "Error in HDSA::MD_Ellipsoid_Ball_SPG::Minimize: "
                            "Objective returned a nonfinite initial value."
                                << std::endl);

    HDSA_TEST_FOR_EXCEPTION(g.Number_of_Rows() != x.Number_of_Rows() || g.Number_of_Columns() != x.Number_of_Columns(),
                            std::logic_error,
                            "Error in HDSA::MD_Ellipsoid_Ball_SPG::Minimize: "
                            "Objective gradient has incompatible dimensions."
                                << std::endl);

    RealT alpha = static_cast<RealT>(1);
    RealT pg_norm = std::numeric_limits<RealT>::infinity();

    int iter = 0;
    bool converged = false;

    for (iter = 0; iter < options.max_iter; ++iter) {
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x_pg = Project(x - g, projection_data);
      pg_norm = DV::Norm(x - *x_pg);
      const RealT x_norm = DV::Norm(x);

      if (pg_norm <= options.pg_tol * std::max(static_cast<RealT>(1), x_norm)) {
        converged = true;
        break;
      }

      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x_proj = Project(x - alpha * g, projection_data);
      HDSA::Dense_Matrix<RealT> dir = *x_proj - x;
      RealT gtd = DV::Dot(g, dir);

      if (gtd >= static_cast<RealT>(0)) {
        alpha = static_cast<RealT>(1);

        dir = *x_pg - x;
        gtd = DV::Dot(g, dir);

        if (gtd >= static_cast<RealT>(0)) { break; }
      }

      const int hist_len = static_cast<int>(info.f_hist.size());
      const int hist_start = std::max(0, hist_len - options.nonmonotone_window);

      RealT f_ref = info.f_hist[hist_start];

      for (int k = hist_start + 1; k < hist_len; ++k) {
        f_ref = std::max(f_ref, info.f_hist[k]);
      }

      bool accepted = false;
      RealT step = static_cast<RealT>(1);

      HDSA::Dense_Matrix<RealT> x_trial;
      HDSA::Dense_Matrix<RealT> g_trial = x.Clone(false);

      RealT f_trial = std::numeric_limits<RealT>::infinity();

      for (int bt = 0; bt < options.max_backtracks; ++bt) {
        x_trial = x + step * dir;
        f_trial = objective(x_trial, g_trial);
        ++info.n_fg_eval;

        HDSA_TEST_FOR_EXCEPTION(!std::isfinite(f_trial), std::logic_error,
                                "Error in HDSA::MD_Ellipsoid_Ball_SPG::Minimize: "
                                "Objective returned a nonfinite trial value."
                                    << std::endl);

        if (f_trial <= f_ref + options.armijo_c * step * gtd) {
          accepted = true;
          break;
        }

        step *= options.backtrack_factor;
      }

      if (!accepted) {
        x_trial = x + step * dir;
        f_trial = objective(x_trial, g_trial);
        ++info.n_fg_eval;

        HDSA_TEST_FOR_EXCEPTION(!std::isfinite(f_trial), std::logic_error,
                                "Error in HDSA::MD_Ellipsoid_Ball_SPG::Minimize: "
                                "Objective returned a nonfinite fallback trial value."
                                    << std::endl);
      }

      HDSA::Dense_Matrix<RealT> s = x_trial - x;
      HDSA::Dense_Matrix<RealT> y = g_trial - g;

      const RealT sy = DV::Dot(s, y);
      const RealT ss = DV::Dot(s, s);
      const RealT s_norm = std::sqrt(std::max(static_cast<RealT>(0), ss));
      const RealT y_norm = DV::Norm(y);

      x = x_trial;
      f = f_trial;
      g.Assign(g_trial);
      info.f_hist.push_back(f);

      if (sy > std::numeric_limits<RealT>::epsilon() * s_norm * std::max(static_cast<RealT>(1), y_norm)) {
        alpha = ss / sy;
        alpha = std::min(std::max(alpha, options.alpha_min), options.alpha_max);
      } else {
        alpha = static_cast<RealT>(1);
      }
    }

    if (!converged) {
      HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x_pg = Project(x - g, projection_data);
      pg_norm = DV::Norm(x - *x_pg);
      if (pg_norm <= options.pg_tol * std::max(static_cast<RealT>(1), DV::Norm(x))) { converged = true; }
    }

    info.iter = iter;
    info.final_objective = f;
    info.projected_gradient_norm = pg_norm;
    info.converged = converged;

    if (options.verbosity) {
      std::cout << "SPG completed in " << info.iter << " iterations." << std::endl;
      std::cout << "Final objective: " << info.final_objective << std::endl;
      std::cout << "Projected gradient norm: " << info.projected_gradient_norm << std::endl;
      std::cout << "Objective/gradient evaluations: " << info.n_fg_eval << std::endl;
      std::cout << "Converged: " << (info.converged ? "true" : "false") << std::endl;
    }

    HDSA::Ptr<HDSA::Dense_Matrix<RealT>> x_output =
        HDSA::makePtr<HDSA::Dense_Matrix<RealT>>(x.Number_of_Rows(), x.Number_of_Columns());
    x_output->Assign(x);
    return x_output;
  }

  static HDSA::Ptr<HDSA::Dense_Matrix<RealT>>
  Minimize(const std::function<RealT(const HDSA::Dense_Matrix<RealT>& x, HDSA::Dense_Matrix<RealT>& grad)>& objective,
           const HDSA::Dense_Matrix<RealT>& x0, const Projection_Data& projection_data,
           const SPG_Options& options = SPG_Options()) {
    SPG_Info info;
    return Minimize(objective, x0, projection_data, info, options);
  }
};

} // namespace HDSA

#endif