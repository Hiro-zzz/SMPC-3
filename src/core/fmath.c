/* SMPC3 :: fmath.c -- exp, log, sin, cos без libm.
 *
 * Схема у всех одна: свести аргумент к узкому отрезку точной арифметикой,
 * там посчитать ряд, и вернуть масштаб. Константы деления на ln2 и pi/2
 * разбиты на части с нулевыми хвостами, как в fdlibm: произведение k на
 * старшую часть точно, и приведение не теряет бит.
 */
#include "smpc3/fmath.h"

#include <emmintrin.h>
#include <string.h>

static double from_bits(uint64_t u) { double d; memcpy(&d, &u, sizeof d); return d; }
static uint64_t to_bits(double d)   { uint64_t u; memcpy(&u, &d, sizeof u); return u; }

/* 2^n для n в [-1022, 1023]: порядок собирается напрямую. */
static double pow2i(int n) { return from_bits((uint64_t)(n + 1023) << 52); }

/* Ближайшее целое, половины — от нуля. Только для |t| < 2^62. */
static double round_half_away(double t)
{
    return (double)(int64_t)(t + (t < 0.0 ? -0.5 : 0.5));
}

double smp_sqrt(double x)
{
    return _mm_cvtsd_f64(_mm_sqrt_sd(_mm_setzero_pd(), _mm_set_sd(x)));
}

/* ========================================================================== */
/*  exp                                                                       */
/* ========================================================================== */

static const double LN2_HI  = 6.93147180369123816490e-01;   /* 33 бита */
static const double LN2_LO  = 1.90821492927058770002e-10;
static const double INV_LN2 = 1.44269504088896338700e+00;

double smp_exp(double x)
{
    if (x != x) return x;
    if (x >  709.782712893384)  return from_bits(0x7FF0000000000000ull);
    if (x < -745.1332191019412) return 0.0;

    /* x = k ln2 + r, |r| <= ln2/2. k ln2_hi точно: у ln2_hi 33 значащих
     * бита, а |k| < 2^11. */
    const double k = round_half_away(x * INV_LN2);
    const double r = (x - k * LN2_HI) - k * LN2_LO;

    /* e^r рядом Тейлора до r^13: следующий член меньше 4e-18. */
    double p = 1.0 / 6227020800.0;                    /* 1/13! */
    p = p * r + 1.0 / 479001600.0;
    p = p * r + 1.0 / 39916800.0;
    p = p * r + 1.0 / 3628800.0;
    p = p * r + 1.0 / 362880.0;
    p = p * r + 1.0 / 40320.0;
    p = p * r + 1.0 / 5040.0;
    p = p * r + 1.0 / 720.0;
    p = p * r + 1.0 / 120.0;
    p = p * r + 1.0 / 24.0;
    p = p * r + 1.0 / 6.0;
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;

    /* Масштаб 2^k; на краях — в два шага, чтобы порядок не вышел за
     * пределы ни в одном из них. */
    const int ki = (int)k;
    if (ki > 1023)  return p * pow2i(1023) * pow2i(ki - 1023);
    if (ki < -1022) return p * pow2i(ki + 600) * pow2i(-600);
    return p * pow2i(ki);
}

/* ========================================================================== */
/*  log                                                                       */
/* ========================================================================== */

double smp_log(double x)
{
    if (x != x || x < 0.0) return from_bits(0x7FF8000000000000ull);
    if (x == 0.0) return -from_bits(0x7FF0000000000000ull);
    if (x == from_bits(0x7FF0000000000000ull)) return x;

    /* x = m 2^e, m в [sqrt(1/2), sqrt(2)). Субнормальные сначала
     * нормализуем умножением на 2^54. */
    int e = 0;
    if (x < 2.2250738585072014e-308) { x *= 18014398509481984.0; e = -54; }
    uint64_t u = to_bits(x);
    e += (int)((u >> 52) & 0x7FF) - 1023;
    u = (u & 0x000FFFFFFFFFFFFFull) | 0x3FF0000000000000ull;
    double m = from_bits(u);
    if (m > 1.4142135623730951) { m *= 0.5; e++; }

    /* Как в fdlibm: f = m - 1 точно (Штербенц), log(1+f) = f - f²/2 +
     * s (f²/2 + R), s = f/(2+f). Главный член f не проходит через деление,
     * и округление s задевает только малую поправку. R — минимаксный
     * многочлен fdlibm по s². */
    const double f    = m - 1.0;
    const double s    = f / (2.0 + f);
    const double z    = s * s;
    const double w    = z * z;
    const double t1   = w * (3.999999999940941908e-01 + w * (2.222219843214978396e-01 +
                        w * 1.531383769920937332e-01));
    const double t2   = z * (6.666666666666735130e-01 + w * (2.857142874366239149e-01 +
                        w * (1.818357216161805012e-01 + w * 1.479819860511658591e-01)));
    const double R    = t2 + t1;
    const double hfsq = 0.5 * f * f;
    const double k    = (double)e;

    return k * LN2_HI - ((hfsq - (s * (hfsq + R) + k * LN2_LO)) - f);
}

/* ========================================================================== */
/*  sin, cos                                                                  */
/* ========================================================================== */

/* pi/2 тремя частями по 33 бита (fdlibm, __ieee754_rem_pio2). */
static const double PIO2_1  = 1.57079632673412561417e+00;
static const double PIO2_2  = 6.07710050630396597660e-11;
static const double PIO2_3  = 2.02226624871116645580e-21;
static const double INV_PIO2 = 6.36619772367581382433e-01;

static double sin_poly(double r)
{
    const double r2 = r * r;
    double p = -1.0 / 121645100408832000.0;           /* -1/19! */
    p = p * r2 + 1.0 / 355687428096000.0;
    p = p * r2 - 1.0 / 1307674368000.0;
    p = p * r2 + 1.0 / 6227020800.0;
    p = p * r2 - 1.0 / 39916800.0;
    p = p * r2 + 1.0 / 362880.0;
    p = p * r2 - 1.0 / 5040.0;
    p = p * r2 + 1.0 / 120.0;
    p = p * r2 - 1.0 / 6.0;
    return r + r * r2 * p;
}

static double cos_poly(double r)
{
    const double r2 = r * r;
    double p = 1.0 / 2432902008176640000.0;           /* 1/20! */
    p = p * r2 - 1.0 / 6402373705728000.0;
    p = p * r2 + 1.0 / 20922789888000.0;
    p = p * r2 - 1.0 / 87178291200.0;
    p = p * r2 + 1.0 / 479001600.0;
    p = p * r2 - 1.0 / 3628800.0;
    p = p * r2 + 1.0 / 40320.0;
    p = p * r2 - 1.0 / 720.0;
    p = p * r2 + 1.0 / 24.0;
    p = p * r2 - 0.5;
    return 1.0 + r2 * p;
}

void smp_sincos(double x, double *s, double *c)
{
    /* NaN, inf и то, чего приведение уже не различает: у |x| > 2^62 в
     * double нет дробной части, а k не влез бы в int64. */
    if (x != x || x > 4.6e18 || x < -4.6e18) {
        *s = *c = from_bits(0x7FF8000000000000ull);
        return;
    }
    /* x = k pi/2 + r, |r| <= pi/4. k pi/2_1 и k pi/2_2 точны при |k| < 2^20. */
    const double k = round_half_away(x * INV_PIO2);
    const double r = ((x - k * PIO2_1) - k * PIO2_2) - k * PIO2_3;

    const double sr = sin_poly(r), cr = cos_poly(r);
    switch ((int64_t)k & 3) {
        case 0:  *s =  sr; *c =  cr; break;
        case 1:  *s =  cr; *c = -sr; break;
        case 2:  *s = -sr; *c = -cr; break;
        default: *s = -cr; *c =  sr; break;
    }
}
