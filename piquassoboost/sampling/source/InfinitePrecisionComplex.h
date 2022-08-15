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
    if (this != &other) {
        if (init) uninit();
        init = other.init;
        if (init) {
            memcpy(&this->f, &other.f, sizeof(mpfr_t));
            other.init = 0;
        }
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
  bound_prec_div(const mpfr_t& x, const mpfr_t& y) {
    if (!mpfr_regular_p (x))
      return mpfr_regular_p (y) ?  mpfr_min_prec (y) : mpfr_get_prec (y);
    else if (!mpfr_regular_p (y))
      return mpfr_min_prec (x);
    return mpfr_min_prec (x) - mpfr_min_prec (y) + 1;
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
  FloatInf& operator/=(const FloatInf& f) { //divides only by power of 2, no precision change, or exact precision with no remainder
    mpfr_div(this->f, this->f, f.f, MPFR_RNDN);
    return *this;
  }
  FloatInf& operator%=(const FloatInf& f) {
    mpfr_prec_round(this->f, std::max(mpfr_min_prec(this->f), (mpfr_prec_t)mpfr_get_exp(f.f)), MPFR_RNDN);
    mpfr_remainder(this->f, this->f, f.f, MPFR_RNDN);
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
  FloatInf operator/(const FloatInf& f) const {
    FloatInf newf(1, bound_prec_div(this->f, f.f)); //exact precision only if no remainder, assumed
    mpfr_div(newf.f, this->f, f.f, MPFR_RNDN);
    return newf;
  }
  FloatInf operator%(const FloatInf& f) const { //remainder can be maximum f bits if 0 extended, so its exponent, assuming arguments are integers
    FloatInf newf(1, (mpfr_prec_t)mpfr_get_exp(f.f));
    mpfr_remainder(newf.f, this->f, f.f, MPFR_RNDN);
    return newf;
  }
  FloatInf operator-() const {
      FloatInf newf(1, this->f);
      mpfr_neg(newf.f, newf.f, MPFR_RNDN);
      return newf;
  }
  int operator==(const FloatInf& f) const { return mpfr_equal_p(this->f, f.f); }
  int operator!=(const FloatInf& f) const { return mpfr_lessgreater_p(this->f, f.f); }
  int operator>(const FloatInf& f) const { return mpfr_greater_p(this->f, f.f); }
  int operator>=(const FloatInf& f) const { return mpfr_greaterequal_p(this->f, f.f); }
  int operator<(const FloatInf& f) const { return mpfr_less_p(this->f, f.f); }
  int operator<=(const FloatInf& f) const { return mpfr_lessequal_p(this->f, f.f); }
  bool isZero() const { return mpfr_zero_p(this->f); }
  int getExponent() const { return mpfr_regular_p(this->f) ? mpfr_get_exp(this->f) : 0; }
  int getFractionBits() const {
      if (!mpfr_regular_p(this->f)) return 0; 
      return mpfr_min_prec(this->f) - mpfr_get_exp(this->f);
  }
  int getMantBits() const {
      if (!mpfr_regular_p(this->f)) return 0;
      return mpfr_min_prec(this->f);
  }
  int getSignBit() const { return mpfr_signbit(this->f); }
  void setExponent(int exp) { mpfr_set_exp(this->f, exp); }
  void print() const
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
    ComplexInf(ComplexInf&& f) : ComplexInf(std::move(REALPARTC(f)), std::move(IMAGPARTC(f))) {}
    ~ComplexInf() {
        REALPART(*this).~FloatInf();
        IMAGPART(*this).~FloatInf();
    }
    /*operator Complex16() {
        return Complex16(REALPART(*this).toDouble(), IMAGPART(*this).toDouble());
    }*/
    ComplexInf& operator=(ComplexInf&& f) {
        if (this != &f) {
            REALPART(*this) = std::move(REALPART(f));
            IMAGPART(*this) = std::move(IMAGPART(f));
        }
        return *this;
    }
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

class RationalInf
{
public:
    RationalInf() : RationalInf(0.0) {}
    RationalInf(double num) {
        ::new (&this->num) FloatInf(num);
        ::new (&this->denom) FloatInf(1.0);
        normalize_int();
    }
    RationalInf(double num, double denom) {
        ::new (&this->num) FloatInf(num);
        ::new (&this->denom) FloatInf(denom);
        normalize();
    }
    RationalInf(const FloatInf& num, const FloatInf& denom) {
        ::new (&this->num) FloatInf(num);
        ::new (&this->denom) FloatInf(denom);
    }
    RationalInf(FloatInf&& num, FloatInf&& denom) {
        ::new (&this->num) FloatInf(num);
        ::new (&this->denom) FloatInf(denom);
    }
    RationalInf(const RationalInf& f) {
        ::new (&this->num) FloatInf(f.num);
        ::new (&this->denom) FloatInf(f.denom);
    }
    RationalInf(RationalInf&& f) : RationalInf(std::move(f.num), std::move(f.denom)) {}
    ~RationalInf() {
        num.~FloatInf();
        denom.~FloatInf();
    }
    RationalInf& operator=(const RationalInf& f)
    {
        this->num = f.num;
        this->denom = f.denom;
        return *this;
    }
    RationalInf& operator=(RationalInf&& f) 
    {
        if (this != &f) {
            this->num = std::move(f.num);
            this->denom = std::move(f.denom);
        }
        return *this;
    }
    RationalInf& operator=(const double d) {
        this->num = d;
        this->denom = 1.0;
        normalize_int();
        return *this;
    }
    //https://www.boost.org/doc/libs/1_76_0/boost/rational.hpp
    RationalInf& operator+=(const RationalInf& f) {
        FloatInf g = compute_gcd(denom, f.denom);
        denom /= g;
        num *= f.denom / g; num += f.num * denom;
        g = compute_gcd(num, g);
        num /= g;
        denom *= f.denom/g;
        //num *= f.denom; num += denom * f.num;
        //denom *= f.denom;
        //normalize();
        return *this;
    }
    RationalInf& operator-=(const RationalInf& f) {
        FloatInf g = compute_gcd(denom, f.denom);
        denom /= g;
        num *= f.denom / g; num -= f.num * denom;
        g = compute_gcd(num, g);
        num /= g;
        denom *= f.denom/g;
        //num *= f.denom; num -= denom * f.num;
        //denom *= f.denom;
        //normalize();
        return *this;
    }
    RationalInf& operator*=(const RationalInf& f) {
        FloatInf gcd1 = compute_gcd(num, f.denom);
        FloatInf gcd2 = compute_gcd(f.num, denom);
        num /= gcd1; num *= f.num/gcd2;
        denom /= gcd2; denom *= f.denom/gcd1;
        //num *= f.num; denom *= f.denom; //normalize();
        return *this;
    }
    RationalInf& operator/=(const RationalInf &f) {
        FloatInf gcd1 = compute_gcd(num, f.num);
        FloatInf gcd2 = compute_gcd(f.denom, denom);
        num /= gcd1; num *= f.denom/gcd2;
        denom /= gcd2; denom *= f.num/gcd1;
        return *this;
    }
    RationalInf& operator/=(const double d) {
        return *this /= RationalInf(d);
    }
    RationalInf operator*(const RationalInf& f) const {
        FloatInf gcd1 = compute_gcd(num, f.denom);
        FloatInf gcd2 = compute_gcd(f.num, denom);
        FloatInf newnum = num / gcd1; newnum *= f.num/gcd2;
        FloatInf newdenom = denom / gcd2; newdenom *= f.denom/gcd1;
        return RationalInf(newnum, newdenom);
        //return RationalInf(num * f.num, denom * f.denom);
    }
    RationalInf operator+(const RationalInf& f) const {
        FloatInf g = compute_gcd(denom, f.denom);
        FloatInf denomnew = denom / g;
        FloatInf numnew = num * (f.denom / g) + f.num * denomnew;
        g = compute_gcd(numnew, g);
        return RationalInf(numnew / g, denomnew * (f.denom/g));
        //return RationalInf(num * f.denom + denom * f.num, denom * f.denom);
    }
    RationalInf operator-(const RationalInf& f) const {
        FloatInf g = compute_gcd(denom, f.denom);
        FloatInf denomnew = denom / g;
        FloatInf numnew = num * (f.denom / g) - f.num * denomnew;
        g = compute_gcd(numnew, g);
        return RationalInf(numnew / g, denomnew * (f.denom/g));
        //return RationalInf(num * f.denom - denom * f.num, denom * f.denom);
    }
    RationalInf operator*(const double d) const {
        return *this * RationalInf(d);
        //return RationalInf(num * d, denom);
    }
    RationalInf operator/(const double d) const {
        return *this / RationalInf(d);
        //if (this->num.isZero()) return *this;
        //return RationalInf(num, denom * d);
    }
    RationalInf operator/(const RationalInf &f) const {
        if (this->num.isZero()) return *this;
        FloatInf gcd1 = compute_gcd(num, f.num);
        FloatInf gcd2 = compute_gcd(f.denom, denom);
        FloatInf newnum = num / gcd1; newnum *= f.denom/gcd2;
        FloatInf newdenom = denom / gcd2; newdenom *= f.num/gcd1;
        return RationalInf(newnum, newdenom);        
        //return RationalInf(num * f.denom, denom * f.num);
    }
    RationalInf operator-() const {
        return RationalInf(-num, denom);
    }
    void normalize_int()
    {
        int numFrac = num.getFractionBits(), denomFrac = denom.getFractionBits();
        if (numFrac > 0 || denomFrac > 0) {
            int expInc = std::max(numFrac, denomFrac);
            num.setExponent(num.getExponent() + expInc);
            denom.setExponent(denom.getExponent() + expInc);
        }
    }
    void normalize()
    {
        if (num.isZero()) { denom = 1.0; return; }
        normalize_int();
        FloatInf f = compute_gcd(num, denom);
        //num.print(); printf(" "); denom.print(); printf(" "); f.print(); printf("\n");
        num /= f; denom /= f;
    }
    static FloatInf compute_gcd(FloatInf a, FloatInf b)
    {
        //https://en.wikipedia.org/wiki/Binary_GCD_algorithm
        if (a.getSignBit()) a = -a;
        if (b.getSignBit()) b = -b;
        if (a.isZero()) return b;
        else if (b.isZero()) return a;
        int i = a.getFractionBits(), j = b.getFractionBits();
        a.setExponent(a.getExponent() + i);
        b.setExponent(b.getExponent() + j);
        int k = std::max(i, j);
        while (true) {
            if (a > b) {
                std::swap(a, b);
                if (a.getMantBits() == 1 && a.getExponent() == 1) {
                    a.setExponent(a.getExponent() - k);
                    return a;
                }
            }
            b -= a;
            if (b.isZero()) {
                a.setExponent(a.getExponent() - k);
                return a;
            }
            b.setExponent(b.getExponent() + b.getFractionBits());
        }
        /*while (!b.isZero()) { //Euclidean GCD but modulo very slow O(n^2)
            a = a % b;
            std::swap(a, b);
        }
        return a;*/
    }
    int operator==(int z) const {
        if (z == 0) return this->num.isZero();
        return this->num == (double)z && this->denom == 1.0;
    }
    int operator!=(int z) const {
        if (z == 0) return !this->num.isZero();
        return this->num != (double)z || this->denom != 1.0;
    }
    operator double() const {
        assert (denom.getMantBits() == 1);
        FloatInf result = num;
        result.setExponent(result.getExponent() - denom.getExponent() + 1); 
        if (denom.getSignBit()) result = -result;
        if (denom.getMantBits() != 1) {
            num.print(); printf(" "); denom.print(); printf(" "); result.print(); printf("\n");
        }
        return result;
        /*FloatInf result = num / denom;
        assert (result * denom == num); //accuracy check
        if (!(result * denom == num)) {
            //assert result * denom == num; //need gcd check
            num.print(); printf(" "); denom.print(); printf(" "); result.print(); printf("\n");
        }
        return (double)result;*/
    }
private:
    FloatInf num;
    FloatInf denom;
};

#define RREALPART(c) reinterpret_cast<RationalInf(&)[2]>(c)[0]
#define RIMAGPART(c) reinterpret_cast<RationalInf(&)[2]>(c)[1]
#define RREALPARTC(c) reinterpret_cast<const RationalInf(&)[2]>(c)[0]
#define RIMAGPARTC(c) reinterpret_cast<const RationalInf(&)[2]>(c)[1]

class ComplexRationalInf : public Complex_base<RationalInf>
{
public:
    ComplexRationalInf() : ComplexRationalInf(0.0, 0.0) {}
    ComplexRationalInf(double real, double imag) {
        ::new (&RREALPART(*this)) RationalInf(real);
        ::new (&RIMAGPART(*this)) RationalInf(imag);
    }
    ComplexRationalInf(double real) : ComplexRationalInf(real, 0.0) {}
    ComplexRationalInf(RationalInf&& r, RationalInf&& i)
    {
        ::new (&RREALPART(*this)) RationalInf(r);
        ::new (&RIMAGPART(*this)) RationalInf(i);
    }
    ComplexRationalInf(const RationalInf& r, const RationalInf& i)
    {
        ::new (&RREALPART(*this)) RationalInf(r);
        ::new (&RIMAGPART(*this)) RationalInf(i);
    }
    ComplexRationalInf(const ComplexRationalInf& f) : ComplexRationalInf(RREALPARTC(f), RIMAGPARTC(f)) {}
    ComplexRationalInf(ComplexRationalInf&& f) : ComplexRationalInf(std::move(RREALPARTC(f)), std::move(RIMAGPARTC(f))) {}
    ~ComplexRationalInf() {
        RREALPART(*this).~RationalInf();
        RIMAGPART(*this).~RationalInf();
    }

    ComplexRationalInf& operator=(double d)
    {
        RREALPART(*this) = d;
        RIMAGPART(*this) = 0.0;
        return *this;
    }
    ComplexRationalInf& operator=(const ComplexRationalInf& f)
    {
        RREALPART(*this) = RREALPARTC(f);
        RIMAGPART(*this) = RIMAGPARTC(f);
        return *this;   
    }
    ComplexRationalInf& operator=(ComplexRationalInf&& f)
    {
        if (this != &f) {
            RREALPART(*this) = std::move(RREALPART(f));
            RIMAGPART(*this) = std::move(RIMAGPART(f));
        }
        return *this;   
    }
    ComplexRationalInf& operator+=(const ComplexRationalInf& f)
    {
        RREALPART(*this) += RREALPARTC(f);
        RIMAGPART(*this) += RIMAGPARTC(f);
        return *this;
    }
    ComplexRationalInf& operator-=(const ComplexRationalInf& f)
    {
        RREALPART(*this) -= RREALPARTC(f);
        RIMAGPART(*this) -= RIMAGPARTC(f);
        return *this;
    }
    ComplexRationalInf& operator*=(const ComplexRationalInf& f)
    {
        RationalInf ac = RREALPART(*this) * RREALPARTC(f);
        RationalInf bd = RIMAGPART(*this) * RIMAGPARTC(f);
        RationalInf p = RREALPART(*this) + RIMAGPART(*this);
        p *= RREALPARTC(f) + RIMAGPARTC(f); p -= ac + bd;
        ac -= bd;
        RREALPART(*this) = std::move(ac);
        RIMAGPART(*this) = std::move(p);
        return *this;
    }
    ComplexRationalInf& operator/=(const double d) {
        RREALPART(*this) /= d;
        RIMAGPART(*this) /= d;
        return *this;
    }
    ComplexRationalInf operator+(const ComplexRationalInf& f)
    {
        return ComplexRationalInf(RREALPART(*this) + RREALPARTC(f), RIMAGPART(*this) + RIMAGPARTC(f));
    }
    ComplexRationalInf operator-(const ComplexRationalInf& f)
    {
        return ComplexRationalInf(RREALPART(*this) - RREALPARTC(f), RIMAGPART(*this) - RIMAGPARTC(f));
    }
    ComplexRationalInf operator*(const ComplexRationalInf& f)
    {
        //ac-bd+((a+b)(c+d)-ac-bd)i
        RationalInf ac = RREALPART(*this) * RREALPARTC(f);
        RationalInf bd = RIMAGPART(*this) * RIMAGPARTC(f);
        RationalInf p = RREALPART(*this) + RIMAGPART(*this);
        p *= RREALPARTC(f) + RIMAGPARTC(f); p -= ac + bd;
        ac -= bd;
        return ComplexRationalInf(std::move(ac), std::move(p));
    }
    ComplexRationalInf operator*(double d) {
        return ComplexRationalInf(std::move(RREALPART(*this) * d), std::move(RIMAGPART(*this) * d));
    }
    ComplexRationalInf operator/(double d)
    {
        return ComplexRationalInf(std::move(RREALPART(*this) / d), std::move(RIMAGPART(*this) / d));
    }
    ComplexRationalInf operator/(const RationalInf& f)
    {
        return ComplexRationalInf(std::move(RREALPART(*this) / f), std::move(RIMAGPART(*this) / f));
    }
    ComplexRationalInf operator-() const {
        return ComplexRationalInf(std::move(-RREALPARTC(*this)), std::move(-RIMAGPARTC(*this)));
    }
    ComplexRationalInf conj() const
    {
        return ComplexRationalInf(RREALPARTC(*this), -RIMAGPARTC(*this));
    }
    ComplexRationalInf operator/(const ComplexRationalInf& f)
    {        
        return (*this * f.conj()) / (RREALPARTC(f) * RREALPARTC(f) + RIMAGPARTC(f) * RIMAGPARTC(f));
    }
    int operator==(int z) const {
        return RREALPARTC(*this) == z && RIMAGPARTC(*this) == 0;
    }
    int operator!=(int z) const {
        return RREALPARTC(*this) != z || RIMAGPARTC(*this) != 0;
    }
    void normalize() { RREALPART(*this).normalize(); RIMAGPART(*this).normalize(); }
    RationalInf& real() { return RREALPART(*this); }
    RationalInf& imag() { return RIMAGPART(*this); }
    void real(double d) {
        RREALPART(*this) = d; 
    }
    void imag(double d) {
        RIMAGPART(*this) = d; 
    }
};

} // PIC

#endif
