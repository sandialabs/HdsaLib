/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_ROL_VECTOR_HPP
#define HDSA_ROL_VECTOR_HPP

#include "ROL_Vector.hpp"
#include "ROL_StdVector.hpp"
#include "HDSA_Ptr.hpp"

namespace HDSA
{

  // Transforms 2 uniformely distributed variables into a normally distributed one
  // X, Y ~ U(0, 1) -> Z ~ N(0,1) -- Box-Muller approach
  template<class RealT>
  class Uniform2Normal : public ROL::Elementwise::BinaryFunction<RealT> {
  public:
    Uniform2Normal() {};

    RealT apply( const RealT &x, const RealT &y ) const {
      constexpr RealT two_pi = 2.0 * M_PI;
      constexpr RealT safe_lower_bound = std::numeric_limits<RealT>::min();
      return std::sqrt(-2.0 * log(safe_lower_bound+x)) * cos(two_pi * y);
    }  
  }; // class Uniform2Normal

  template <class RealT>
  class ROL_Vector : public HDSA::Vector<RealT>
  {

  public:
    ROL::Ptr<ROL::Vector<RealT>> rol_vec;
    static Uniform2Normal<RealT> u2n;

    ROL_Vector(ROL::Ptr<ROL::Vector<RealT>> &rol_vec_in) : rol_vec(rol_vec_in)
    {
    }

    ROL_Vector(ROL::Vector<RealT> &rol_vec_in) 
    {
      rol_vec = rol_vec_in.clone();
      rol_vec->set(rol_vec_in);
    }

    ROL_Vector(const ROL::Ptr<ROL::Vector<RealT>> &rol_vec_in) : rol_vec(rol_vec_in) {};

    virtual ~ROL_Vector()
    {
    }

    // Clone the vector
    HDSA::Ptr<HDSA::Vector<RealT>> Clone() const override
    {
      ROL::Ptr<ROL::Vector<RealT>> rol_vec_clone = rol_vec->clone();
      rol_vec_clone->zero(); // ROL clone() vector is not initialized
      return Teuchos::rcp(new HDSA::ROL_Vector<RealT>(rol_vec_clone));
    }

    // compute the Dot product of this and x
    RealT Dot(const HDSA::Vector<RealT> &x) const override
    {
      const HDSA::ROL_Vector<RealT> &ex = dynamic_cast<const HDSA::ROL_Vector<RealT> &>(x);
      RealT val = ex.rol_vec->dot(*rol_vec);
      return val;
    }

    // add alpha*x to this
    void Scaled_Plus(const RealT alpha, const HDSA::Vector<RealT> &x) override
    {
      const HDSA::ROL_Vector<RealT> &ex = dynamic_cast<const HDSA::ROL_Vector<RealT> &>(x);
      rol_vec->axpy(alpha, *ex.rol_vec);
    }

    // return vector Dimension
    int Dimension() const override
    {
      return rol_vec->dimension();
    }

    // Set this=val elementwise
    void Set_Scalar(const RealT val) override
    {
      rol_vec->setScalar(val);
    }

    void Randomize_Standard_Normal() override
    {
      auto rol_vec_tmp = rol_vec->clone();
      rol_vec->randomize();
      rol_vec_tmp->randomize();
      rol_vec->applyBinary(u2n,*rol_vec_tmp);
    }

    HDSA::Ptr<HDSA::Vector<RealT>> Get_Basis(int i) const override
    {
      return Teuchos::rcp(new HDSA::ROL_Vector<RealT>(rol_vec->basis(i)));
    };

    RealT Get_Entry(int i) const override
    {
      return rol_vec->dot(*rol_vec->basis(i));
    };

    void Set_Entry(int i, RealT val) override
    {
      rol_vec->setScalar(0.0);
      rol_vec->axpy(val, *rol_vec->basis(i));
    };

    void Write_to_File(const std::string &name) const override
    {
      try
      {
        ROL::Ptr<std::vector<RealT>> vec = dynamic_cast<ROL::StdVector<RealT> &>(*rol_vec).getVector();
        std::ofstream fout;
        fout.open(name);
        for (int i = 0; i < rol_vec->dimension(); i++)
        {
          fout << std::setprecision(16) << (*vec)[i] << "  ";
        }
        fout.close();
      }
      catch (...)
      {
        std::cout << "Write_to_File is currently not supported for this vector type" << std::endl;
      }
    }
  };

  template <>
  Uniform2Normal<double> ROL_Vector<double>::u2n = Uniform2Normal<double>();

}

#endif
