/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_INCOMPLETE_CHOL_FACTOR_HPP
#define HDSA_INCOMPLETE_CHOL_FACTOR_HPP

#include "HDSA_Vector.hpp"

namespace HDSA {

template <class RealT> class Incomplete_Chol_Factor {

private:

public:
  Incomplete_Chol_Factor() {} 

  virtual ~Incomplete_Chol_Factor() {}

  virtual void Apply(HDSA::Vector<RealT>& vec_out, const HDSA::Vector<RealT>& vec_in) const = 0 ; 

  virtual void Apply_Inverse(HDSA::Vector<RealT>& vec_out, const HDSA::Vector<RealT>& vec_in) const = 0 ;

  virtual void Apply_Inverse_Transpose(HDSA::Vector<RealT>& vec_out, const HDSA::Vector<RealT>& vec_in) const = 0 ;

};
} // namespace HDSA
#endif
