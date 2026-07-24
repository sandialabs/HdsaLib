/***********************************************************************
 HdsaLib - A library for Hyper-differential Sensitivity Analysis
 
 Questions? Contact Joseph Hart (joshart@sandia.gov)
************************************************************************/

#ifndef HDSA_RANDOM_NUMBER_GENERATOR_HPP
#define HDSA_RANDOM_NUMBER_GENERATOR_HPP

#include <algorithm>
#include <cstdlib>
#include <random>
#include "HDSA_Stack_Trace.hpp"
#include "HDSA_Comm.hpp"
#include "HDSA_Ptr.hpp"

namespace HDSA
{

  template <class RealT>
  class Random_Number_Generator
  {

  private:
    unsigned seed_;
    std::default_random_engine generator_;
    std::normal_distribution<RealT> distribution_;
    std::vector<RealT> random_numbers_;
    bool use_numbers_from_file_;
    int file_reading_index_;
    int num_random_numbers_;

  public:
    Random_Number_Generator(void)
    {
      use_numbers_from_file_ = false;
      seed_ = time(NULL);
      generator_.seed(seed_);
      distribution_ = std::normal_distribution<RealT>(0.0, 1.0);
    }

    Random_Number_Generator(HDSA::Ptr<const HDSA::Comm<int>> &comm, bool seed_on_time = false)
    {
      use_numbers_from_file_ = false;
      if (seed_on_time)
      {
        seed_ = time(NULL) + comm->getRank();
      }
      else
      {
        seed_ = 123 + comm->getRank();
      }
      generator_.seed(seed_);
      distribution_ = std::normal_distribution<RealT>(0.0, 1.0);
    }

    Random_Number_Generator(int seed)
    {
      use_numbers_from_file_ = false;
      seed_ = seed;
      generator_.seed(seed_);
      distribution_ = std::normal_distribution<RealT>(0.0, 1.0);
    }

    Random_Number_Generator(int num_random_numbers, std::string random_number_file)
    {
      use_numbers_from_file_ = true;
      file_reading_index_ = 0;
      num_random_numbers_ = num_random_numbers;

      random_numbers_ = std::vector<RealT>(num_random_numbers_);

      RealT val = 0.0;
      // read in data
      std::ifstream in(random_number_file);
      // read the elements in the file into a vector
      if (in)
      {
        for (int i = 0; i < num_random_numbers; i++)
        {
          in >> val;
          random_numbers_[i] = val;
        }
      }
      else
      {
          HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                                  "Error in HDSA::Random_Number_Generator: Cannot open random number file" << std::endl);
      }
    }

    virtual ~Random_Number_Generator()
    {
    }

    RealT Generate_Standard_Normal_Sample(void)
    {
      RealT val = 0.0;

      if (use_numbers_from_file_)
      {
        if (file_reading_index_ > num_random_numbers_)
        {
          HDSA_TEST_FOR_EXCEPTION(true, std::logic_error,
                                  "Error in HDSA::Random_Number_Generator: Requested more random numbers than was provided" << std::endl);
        }
        val = random_numbers_[file_reading_index_];
        file_reading_index_ += 1;
      }
      else
      {
        val = distribution_(generator_);
      }
      return val;
    }
  };

}

#endif
