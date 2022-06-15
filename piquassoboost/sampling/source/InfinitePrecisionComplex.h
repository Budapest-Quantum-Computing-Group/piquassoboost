#ifndef InfinitePrecisionComplex_H
#define InfinitePrecisionComplex_H

#include <mpfr.h>

namespace pic {
#define IEEE_DBL_MANT_DIG 53
# define MPFR_LDBL_MANT_DIG   64
class FloatInf
{
public:
  FloatInf() { init = 0; }
  ~FloatInf() { uninit(); }
  FloatInf(const double d) { init = 1; mpfr_init2(this->f, IEEE_DBL_MANT_DIG); mpfr_set_d(this->f, d, MPFR_RNDN); }
  FloatInf(const long double ld) { init = 1; mpfr_init2(this->f, MPFR_LDBL_MANT_DIG); mpfr_set_ld(this->f, ld, MPFR_RNDN); }
  FloatInf(const long long unsigned int uj) { init = 1; mpfr_init2(this->f, sizeof(uintmax_t)*8); mpfr_set_uj(this->f, uj, MPFR_RNDN); }
  FloatInf(const int64_t i) { init = 1; mpfr_init2(this->f, sizeof(intmax_t)*8); mpfr_set_sj(this->f, i, MPFR_RNDN); }
  FloatInf(const char c) { init = 1; mpfr_init2(this->f, sizeof(char)*8); mpfr_set_si(this->f, c, MPFR_RNDN); }
  FloatInf(const int oinit, const mpfr_t& f) {      
      init = oinit;
      if (init) { mpfr_init2(this->f, mpfr_get_prec(f)); mpfr_set(this->f, f, MPFR_RNDN); }
  }
  FloatInf(const int oinit, mpfr_prec_t prec) { init = oinit; mpfr_init2(this->f, prec); }
  FloatInf(const FloatInf& f) : FloatInf(f.init, f.f) {}
  FloatInf(FloatInf&& f) {
      init = f.init;
      if (init) { memcpy(&this->f, &f.f, sizeof(mpfr_t)); f.init = 0; }
  } 
  FloatInf& operator=(const FloatInf& f) {
    if (this != &f) {
      if (f.init) {
          if (init) mpfr_set_prec(this->f, mpfr_get_prec(f.f));
          else init = 1, mpfr_init2(this->f, mpfr_get_prec(f.f));
          mpfr_set(this->f, f.f, MPFR_RNDN);
      } else {
          if (init) uninit();
      }
    }
    return *this;
  }
  FloatInf& operator=(FloatInf&& other) {
    if (init) uninit();
    init = other.init;
    if (init) {
        memcpy(&this->f, &other.f, sizeof(mpfr_t));
        other.init = 0;
    }
    return *this;
  }
  void uninit() { if (init) mpfr_clear(this->f); init = 0; }
  operator double() { return this->toDouble(); } //dangerous as could cause problems with operator*, operator+
  double toDouble() { return mpfr_get_d(this->f, MPFR_RNDN); }
  //https://github.com/BrianGladman/MPC/blob/master/src/fma.c
  static mpfr_prec_t
  bound_prec_addsub (const mpfr_t& x, const mpfr_t& y)
  {
    if (!mpfr_regular_p (x))
      return mpfr_regular_p (y) ?  mpfr_min_prec (y) : mpfr_get_prec (y);
    else if (!mpfr_regular_p (y))
      return mpfr_min_prec (x);
    else /* neither x nor y are NaN, Inf or zero */
      {
        mpfr_exp_t ex = mpfr_get_exp (x);
        mpfr_exp_t ey = mpfr_get_exp (y);
        mpfr_exp_t ulpx = ex - mpfr_min_prec (x);
        mpfr_exp_t ulpy = ey - mpfr_min_prec (y);
        return ((ex >= ey) ? ex : ey) + 1 - ((ulpx <= ulpy) ? ulpx : ulpy);
      }
  }
  static mpfr_prec_t
  bound_prec_addsub (const mpfr_t& x, const double y)
  {
    if (!mpfr_regular_p (x))
      return IEEE_DBL_MANT_DIG;
    else if (!std::isfinite(y) || y == 0)
      return mpfr_min_prec (x);
    else /* neither x nor y are NaN, Inf or zero */
      {
        mpfr_exp_t ex = mpfr_get_exp (x);
        int exp; frexp(y, &exp);
        mpfr_exp_t ey = exp;
        mpfr_exp_t ulpx = ex - mpfr_min_prec (x);
        mpfr_exp_t ulpy = ey - IEEE_DBL_MANT_DIG;
        return ((ex >= ey) ? ex : ey) + 1 - ((ulpx <= ulpy) ? ulpx : ulpy);
      }
  }
  static mpfr_prec_t
  bound_prec_mul(const mpfr_t& x, const mpfr_t& y) {
    if (!mpfr_regular_p (x))
      return mpfr_regular_p (y) ?  mpfr_min_prec (y) : mpfr_get_prec (y);
    else if (!mpfr_regular_p (y))
      return mpfr_min_prec (x);
    return mpfr_min_prec (x) + mpfr_min_prec (y);
  }
  static mpfr_prec_t
  bound_prec_mul(const mpfr_t& x, const double y) {
    if (!mpfr_regular_p (x))
      return IEEE_DBL_MANT_DIG;
    else if (!std::isfinite(y) || y == 0)
      return mpfr_min_prec (x);
    return mpfr_min_prec (x) + IEEE_DBL_MANT_DIG;
  }
  FloatInf& operator+=(const double d) {
    mpfr_prec_round(this->f, bound_prec_addsub(this->f, d), MPFR_RNDN);
    mpfr_add_d(this->f, this->f, d, MPFR_RNDN);
    return *this;
  }
  FloatInf& operator-=(const double d) { //subtracting only values added before, no precision change
    mpfr_prec_round(this->f, bound_prec_addsub(this->f, d), MPFR_RNDN);
    mpfr_sub_d(this->f, this->f, d, MPFR_RNDN);
    return *this;
  }
  FloatInf& operator+=(const FloatInf& f) {
    mpfr_prec_round(this->f, bound_prec_addsub(this->f, f.f), MPFR_RNDN);
    mpfr_add(this->f, this->f, f.f, MPFR_RNDN);
    return *this;
  }
  FloatInf& operator-=(const FloatInf& f) {
    mpfr_prec_round(this->f, bound_prec_addsub(this->f, f.f), MPFR_RNDN);
    mpfr_sub(this->f, this->f, f.f, MPFR_RNDN);
    return *this;
  }
  FloatInf& operator*=(const FloatInf& f) {
    mpfr_prec_round(this->f, bound_prec_mul(this->f, f.f), MPFR_RNDN);
    mpfr_mul(this->f, this->f, f.f, MPFR_RNDN);
    return *this;
  }
  FloatInf& operator/=(const FloatInf& f) { //divides only by power of 2, no precision change
    mpfr_div(this->f, this->f, f.f, MPFR_RNDN);
    return *this;
  }
  FloatInf operator+(const FloatInf& f) const {
    FloatInf newf(1, bound_prec_addsub(this->f, f.f));
    mpfr_add(newf.f, this->f, f.f, MPFR_RNDN);
    return newf;
  }  
  FloatInf operator-(const FloatInf& f) const {
    FloatInf newf(1, bound_prec_addsub(this->f, f.f));
    mpfr_sub(newf.f, this->f, f.f, MPFR_RNDN);
    return newf;
  }  
  FloatInf operator*(const FloatInf& f) const {
    FloatInf newf(1, bound_prec_mul(this->f, f.f));
    mpfr_mul(newf.f, this->f, f.f, MPFR_RNDN);
    return newf;
  }
  FloatInf operator*(const double& f) const {
    FloatInf newf(1, bound_prec_mul(this->f, f));
    mpfr_mul_d(newf.f, this->f, f, MPFR_RNDN);
    return newf;
  }
  void print()
  {
    if (init) mpfr_out_str(stdout, 2, mpfr_min_prec(f), f, MPFR_RNDN); 
  }
private:
  int init;
  mpfr_t f;
};

#define REALPART(c) reinterpret_cast<FloatInf(&)[2]>(c)[0]
#define IMAGPART(c) reinterpret_cast<FloatInf(&)[2]>(c)[1]
#define REALPARTC(c) reinterpret_cast<const FloatInf(&)[2]>(c)[0]
#define IMAGPARTC(c) reinterpret_cast<const FloatInf(&)[2]>(c)[1]

class ComplexInf : public Complex_base<FloatInf>
{
public:
    ComplexInf() : ComplexInf(0.0, 0.0) {
    }
    ComplexInf(double real, double imag) {
        ::new (&REALPART(*this)) FloatInf(real);
        ::new (&IMAGPART(*this)) FloatInf(imag);
    }
    ComplexInf(const Complex16& f) : ComplexInf(f.real(), f.imag()) {}
    ComplexInf(const FloatInf& r, const FloatInf& i) {
        ::new (&REALPART(*this)) FloatInf(r);
        ::new (&IMAGPART(*this)) FloatInf(i);
    }
    ComplexInf(const ComplexInf& f) : ComplexInf(REALPARTC(f), IMAGPARTC(f)) {}
    ~ComplexInf() {
        REALPART(*this).~FloatInf();
        IMAGPART(*this).~FloatInf();
    }
    /*operator Complex16() {
        return Complex16(REALPART(*this).toDouble(), IMAGPART(*this).toDouble());
    }*/
    ComplexInf operator*(const double& f)
    {
        return ComplexInf(REALPART(*this) * f, IMAGPART(*this) * f);
    }
    ComplexInf operator*(const FloatInf& f)
    {
        return ComplexInf(REALPART(*this) * f, IMAGPART(*this) * f);
    }
    ComplexInf& operator+=(const ComplexInf& f)
    {
        REALPART(*this) += REALPARTC(f);
        IMAGPART(*this) += IMAGPARTC(f);
        return *this;
    }
    ComplexInf& operator+=(const Complex16& f)
    {
        REALPART(*this) += f.real();
        IMAGPART(*this) += f.imag();
        return *this;
    }
    ComplexInf& operator-=(const ComplexInf& f)
    {
        REALPART(*this) -= REALPARTC(f);
        IMAGPART(*this) -= IMAGPARTC(f);
        return *this;
    }
    ComplexInf& operator-=(const Complex16& f)
    {
        REALPART(*this) -= f.real();
        IMAGPART(*this) -= f.imag();
        return *this;
    }
    ComplexInf& operator*=(const double& f)
    {
        REALPART(*this) *= f;
        IMAGPART(*this) *= f;
        return *this;
    }
    ComplexInf& operator*=(const ComplexInf& f)
    {
        //REALPART(*this) = REALPARTC(*this) * REALPARTC(f) - IMAGPARTC(*this) * IMAGPARTC(f);
        //IMAGPART(*this) = REALPARTC(*this) * IMAGPARTC(f) - IMAGPARTC(*this) * REALPARTC(f);
        //return *this;
        /*FloatInf acbd = REALPART(*this) * REALPARTC(f);
        acbd -= IMAGPART(*this) * IMAGPARTC(f);
        FloatInf bcad = IMAGPART(*this) * REALPARTC(f);
        bcad += REALPART(*this) * IMAGPARTC(f);
        REALPART(*this) = acbd;
        IMAGPART(*this) = bcad;*/
        FloatInf ac = REALPART(*this) * REALPARTC(f);
        FloatInf bd = IMAGPART(*this) * IMAGPARTC(f);
        FloatInf p = REALPART(*this) + IMAGPART(*this);
        p *= REALPARTC(f) + IMAGPARTC(f); p -= ac + bd;
        ac -= bd;
        REALPART(*this) = ac;
        IMAGPART(*this) = p;
        return *this;
    }
    ComplexInf& operator/=(const FloatInf& f)
    {
        REALPART(*this) /= f;
        IMAGPART(*this) /= f;
        return *this;
    }
    FloatInf& real() { return REALPART(*this); }
    FloatInf& imag() { return IMAGPART(*this); }
};

} // PIC

#endif
