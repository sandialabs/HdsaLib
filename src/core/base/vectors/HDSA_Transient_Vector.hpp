/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis

 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_TRANSIENT_VECTOR_HPP
#define HDSA_TRANSIENT_VECTOR_HPP

namespace HDSA
{

    template <class RealT>
    class Transient_Vector : public HDSA::Vector<RealT>
    {

    private:
        int n_t_;
        std::vector<HDSA::Ptr<HDSA::Vector<RealT>>> vec_;

    public:
        Transient_Vector()
        {
        }

        Transient_Vector(int n_t, const HDSA::Ptr<const HDSA::Vector<RealT>> &spatial_vec)
        {
            n_t_ = n_t;
            vec_.resize(n_t);
            for (int k = 0; k < n_t; k++)
            {
                vec_[k] = spatial_vec->Clone();
            }
        }

        Transient_Vector(std::vector<HDSA::Ptr<HDSA::Vector<RealT>>> &trans_vec)
        {
            n_t_ = trans_vec.size();
            vec_.resize(n_t_);
            for (int k = 0; k < n_t_; k++)
            {
                vec_[k] = trans_vec[k];
            }
        }

        ~Transient_Vector()
        {
        }

        int Get_n_t(void) const
        {
            return n_t_;
        }

        //////////////////////////////////////////////////////////////////////////////////
        // Overloading pure virtual functions in HDSA::Vector base class
        //////////////////////////////////////////////////////////////////////////////////

        HDSA::Ptr<HDSA::Vector<RealT>> Clone() const override
        {
            HDSA::Ptr<HDSA::Vector<RealT>> vec = HDSA::makePtr<Transient_Vector<RealT>>(n_t_, vec_[0]);
            return vec;
        }

        // compute the Dot product of this and x
        RealT Dot(const HDSA::Vector<RealT> &x) const override
        {
            RealT val = 0.0;
            const Transient_Vector<RealT> x_trans = dynamic_cast<const Transient_Vector<RealT> &>(x);
            for (int k = 0; k < n_t_; k++)
            {
                val += Get_Vector_Const(k)->Dot(*x_trans.Get_Vector_Const(k));
            }
            return val;
        }

        // add alpha*x to this
        void Scaled_Plus(const RealT alpha, const HDSA::Vector<RealT> &x) override
        {
            const Transient_Vector<RealT> x_trans = dynamic_cast<const Transient_Vector<RealT> &>(x);
            for (int k = 0; k < n_t_; k++)
            {
                vec_[k]->Scaled_Plus(alpha, *x_trans.Get_Vector_Const(k));
            }
        }

        // return vector Dimension
        int Dimension() const override
        {
            return n_t_ * vec_[0]->Dimension();
        }

        // Set this=val elementwise
        void Set_Scalar(const RealT val) override
        {
            for (int k = 0; k < n_t_; k++)
            {
                vec_[k]->Set_Scalar(val);
            }
        }

        void Randomize_Standard_Normal() override
        {
            for (int k = 0; k < n_t_; k++)
            {
                vec_[k]->Randomize_Standard_Normal();
            }
        }

        void Write_to_File(const std::string &name) const override
        {
            int num_char = name.size();
            std::string name_tmp = name.substr(0, num_char - 4);
            for (int k = 0; k < n_t_; k++)
            {
                std::string name_k = name_tmp + "_time_" + std::to_string(k + 1) + ".txt";
                vec_[k]->Write_to_File(name_k);
            }
        }

        HDSA::Ptr<HDSA::Vector<RealT>> Generate_Std_Vector(int r) const override
        {
            HDSA::Ptr<HDSA::Vector<RealT>> vec = vec_[0]->Generate_Std_Vector(r);
            return vec;
        }

        //////////////////////////////////////////////////////////////////////////////////
        // Function specific to this class for convenience
        //////////////////////////////////////////////////////////////////////////////////

        HDSA::Ptr<HDSA::Vector<RealT>> operator[](int k) const
        {
            return vec_[k];
        }

        virtual HDSA::Ptr<const HDSA::Vector<RealT>> Get_Vector_Const(int k) const
        {
            return vec_[k];
        }
    };

}

#endif
