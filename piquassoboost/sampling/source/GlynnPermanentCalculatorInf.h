#ifndef GlynnPermanentCalculatorInf_H
#define GlynnPermanentCalculatorInf_H

#include "matrix.h"
#include "matrix32.h"
#include "PicState.h"
#include <vector>
#include "PicVector.hpp"
#include <mpfr.h>

#ifndef CPYTHON
#include <tbb/tbb.h>
#endif


namespace pic {
#define IEEE_DBL_MANT_DIG 53
# define MPFR_LDBL_MANT_DIG   64
class FloatInf
{
public:
  FloatInf() : FloatInf(0.0) {}
  virtual ~FloatInf() { uninit(); }
  FloatInf(const double d) { init = 1; mpfr_init2(this->f, IEEE_DBL_MANT_DIG); mpfr_set_d(this->f, d, MPFR_RNDN); }
  FloatInf(const long double ld) { init = 1; mpfr_init2(this->f, MPFR_LDBL_MANT_DIG); mpfr_set_ld(this->f, ld, MPFR_RNDN); }
  FloatInf(const mpfr_t& f) { init = 1; mpfr_init2(this->f, mpfr_get_prec(f)); mpfr_set(this->f, f, MPFR_RNDN); }
  FloatInf(mpfr_prec_t prec) { init = 1; mpfr_init2(this->f, prec); mpfr_set(this->f, f, MPFR_RNDN); }
  FloatInf(const FloatInf& f) : FloatInf(f.f) {}
  FloatInf(FloatInf&& f) { memcpy(&this->f, &f.f, sizeof(mpfr_t)); f.init = 0; } 
  FloatInf& operator=(const FloatInf& f) {
    if (this != &f) {
      if (init) mpfr_set_prec(this->f, mpfr_get_prec(f.f));
      else init = 1, mpfr_init2(this->f, mpfr_get_prec(f.f));
      mpfr_set(this->f, f.f, MPFR_RNDN);
    }
    return *this;
  }
  FloatInf& operator=(FloatInf&& other) {
    if (init) uninit();
    else init = 1;
    memcpy(&this->f, &other.f, sizeof(mpfr_t));
    other.init = 0;
    return *this;
  }
  void uninit() { if (init) mpfr_clear(this->f); init = 0; }
  operator double() { return mpfr_get_d(this->f, MPFR_RNDN); }
  //https://github.com/BrianGladman/MPC/blob/master/src/fma.c
  static mpfr_prec_t
  bound_prec_addsub (const mpfr_t x, const mpfr_t y)
  {
    if (!mpfr_regular_p (x))
      return mpfr_get_prec (y);
    else if (!mpfr_regular_p (y))
      return mpfr_get_prec (x);
    else /* neither x nor y are NaN, Inf or zero */
      {
        mpfr_exp_t ex = mpfr_get_exp (x);
        mpfr_exp_t ey = mpfr_get_exp (y);
        mpfr_exp_t ulpx = ex - mpfr_get_prec (x);
        mpfr_exp_t ulpy = ey - mpfr_get_prec (y);
        return ((ex >= ey) ? ex : ey) + 1 - ((ulpx <= ulpy) ? ulpx : ulpy);
      }
  }
  static mpfr_prec_t
  bound_prec_mul(const mpfr_t x, const mpfr_t y) {
    if (!mpfr_regular_p (x))
      return mpfr_get_prec (y);
    else if (!mpfr_regular_p (y))
      return mpfr_get_prec (x);
    return mpfr_get_prec (x) + mpfr_get_prec (y);
  }
  FloatInf& operator+=(const double d) {
    mpfr_add_d(this->f, this->f, d, MPFR_RNDN);
    return *this;
  }
  FloatInf& operator-=(const double d) { //subtracting only values added before, no precision change
    mpfr_sub_d(this->f, this->f, d, MPFR_RNDN);
    return *this;
  }
  FloatInf& operator+=(const FloatInf& f) {
    mpfr_prec_round(this->f, bound_prec_addsub(this->f, f.f), MPFR_RNDN);
    mpfr_add(this->f, this->f, f.f, MPFR_RNDN);
    if (mpfr_regular_p(this->f)) {
        mpfr_prec_t minprec = mpfr_min_prec(this->f);
        if (minprec != mpfr_get_prec(this->f)) mpfr_prec_round(this->f, minprec, MPFR_RNDN);
    }
    return *this;
  }
  FloatInf& operator-=(const FloatInf& f) {
    mpfr_prec_round(this->f, bound_prec_addsub(this->f, f.f), MPFR_RNDN);
    mpfr_sub(this->f, this->f, f.f, MPFR_RNDN);
    if (mpfr_regular_p(this->f)) {
        mpfr_prec_t minprec = mpfr_min_prec(this->f);
        if (minprec != mpfr_get_prec(this->f)) mpfr_prec_round(this->f, minprec, MPFR_RNDN);
    }
    return *this;
  }
  /*FloatInf& operator*=(const double d) { //d=1 or -1, changing sign only, no precision change 
    mpfr_mul_d(this->f, this->f, d, MPFR_RNDN);
    return *this;
  }*/
  FloatInf& operator*=(const FloatInf& f) {
    mpfr_prec_round(this->f, bound_prec_mul(this->f, f.f), MPFR_RNDN);
    mpfr_mul(this->f, this->f, f.f, MPFR_RNDN);
    return *this;
  }
  FloatInf& operator/=(const FloatInf& f) { //divides only by power of 2, no precision change
    mpfr_div(this->f, this->f, f.f, MPFR_RNDN);
    return *this;
  }
  /*FloatInf operator+(const FloatInf& f) {
    FloatInf newf(bound_prec_addsub(this->f, f.f));
    mpfr_add(newf.f, this->f, f.f, MPFR_RNDN);
    return newf;
  }  
  FloatInf operator-(const FloatInf& f) {
    FloatInf newf(bound_prec_addsub(this->f, f.f));
    mpfr_sub(newf.f, this->f, f.f, MPFR_RNDN);
    return newf;
  }  
  FloatInf operator*(const FloatInf& f) {
    FloatInf newf(bound_prec_mul(this->f, f.f));
    mpfr_mul(newf.f, this->f, f.f, MPFR_RNDN);
    return newf;
  }*/
private:
  int init;
  mpfr_t f;
};

/// @brief Structure type representing 16 byte complex numbers
typedef Complex_base<FloatInf> ComplexInf;


/**
@brief Call to print the elements of a container
@param vec a container
@return Returns with the sum of the elements of the container
*/
template <typename Container>
void print_state( Container state );


/**
@brief Interface class representing a Glynn permanent calculator
*/
class GlynnPermanentCalculatorInf {

protected:
    /// Unitary describing a quantum circuit
    matrix mtx;

public:

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
GlynnPermanentCalculatorInf();



/**
@brief Call to calculate the permanent via Glynn formula scaling with n*2^n. (Does not use gray coding, but does the calculation is similar but scalable fashion)
@param mtx The effective scattering matrix of a boson sampling instance
@return Returns with the calculated permanent
*/
Complex16 calculate(matrix &mtx);


}; //GlynnPermanentCalculatorInf





// relieve Python extension from TBB functionalities
#ifndef CPYTHON


/**
@brief Class to calculate a partial permanent via Glynn's formula scaling with n*2^n. (Does not use gray coding, but does the calculation is similar but scalable fashion) 
*/
class GlynnPermanentCalculatorInfTask {

public:

    /// Unitary describing a quantum circuit
    matrix mtx;
    /// 2*mtx used in the recursive calls (The storing of thos matrix spare many repeating multiplications)
    matrix mtx2;
    /// thread local storage for partial permanents
    tbb::combinable<pic::ComplexInf> priv_addend;

public:

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
GlynnPermanentCalculatorInfTask();


/**
@brief Call to calculate the permanent via Glynn formula. scales with n*2^n
@param mtx The effective scattering matrix of a boson sampling instance
@return Returns with the calculated permanent
*/
Complex16 calculate(matrix &mtx);


/**
@brief Method to span parallel tasks via iterative function calls. (new task is spanned by altering one element in the vector of deltas)
@param colSum The sum of \f$ \delta_j a_{ij} \f$ in Eq. (S2) of arXiv:1606.05836
@param sign The current product \f$ \prod\delta_i $\f
@param index_min \f$ \delta_j a_{ij} $\f with \f$ 0<i<index_min $\f are kept contstant, while the signs of \f$ \delta_i \f$  with \f$ i>=idx_min $\f are changed.
*/
void IterateOverDeltas( pic::ComplexInf* colSum_data, int sign, int index_min );


}; // partial permanent_Task


#endif // CPYTHON




} // PIC

#endif
