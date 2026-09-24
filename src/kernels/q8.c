/* SMPC3 :: q8.c -- формат Q8_0: половинная точность, квантование, эталон.
 *
 * Блок — 34 байта: масштаб d в f16 и 32 веса q в i8, значение — d * q. Так
 * же лежат веса в GGUF, и квантование повторяет его эталонное: d = max|x| /
 * 127, q = round(x / d) с округлением от нуля.
 *
 * Распаковка точна: у d одиннадцать значащих бит, у q — восемь, и
 * произведение укладывается в 24 бита мантиссы f32 без округления. Поэтому
 * скалярная и векторная ветки распаковывают бит в бит одинаково, а
 * расходиться могут только в порядке сложения.
 */
#include "impl.h"

#include <string.h>

/* ========================================================================== */
/*  f16                                                                       */
/* ========================================================================== */

float smp_f16_to_f32(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t       e    = (h >> 10) & 0x1Fu;
    uint32_t       m    = h & 0x3FFu;
    uint32_t       x;

    if (e == 0x1Fu) {
        x = sign | 0x7F800000u | (m << 13);             /* inf, nan */
    } else if (e != 0) {
        x = sign | ((e + 112u) << 23) | (m << 13);      /* 127 - 15 = 112 */
    } else if (m == 0) {
        x = sign;
    } else {
        /* Субнормальное: m * 2^-24. Нормализуем, пока старший бит мантиссы
         * не встанет на место неявной единицы. */
        e = 113u;
        while (!(m & 0x400u)) { m <<= 1; e--; }
        x = sign | (e << 23) | ((m & 0x3FFu) << 13);
    }
    float f;
    memcpy(&f, &x, sizeof f);
    return f;
}

/* Округление к ближайшему, при равенстве — к чётному, как у аппаратного
 * vcvtps2ph по умолчанию. */
uint16_t smp_f32_to_f16(float f)
{
    uint32_t x;
    memcpy(&x, &f, sizeof x);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t ax   = x & 0x7FFFFFFFu;

    if (ax >= 0x7F800000u)                              /* inf, nan */
        return (uint16_t)(sign | 0x7C00u | (ax > 0x7F800000u ? 0x200u : 0u));
    if (ax >= 0x477FF000u)                              /* от 65520 — в inf */
        return (uint16_t)(sign | 0x7C00u);

    if (ax < 0x38800000u) {                             /* меньше 2^-14 */
        if (ax < 0x33000000u) return (uint16_t)sign;    /* меньше 2^-25 */
        /* Субнормальное: единица последнего разряда — 2^-24. */
        const uint32_t e     = ax >> 23;
        const uint32_t m     = (ax & 0x7FFFFFu) | 0x800000u;
        const uint32_t shift = 126u - e;                /* 14..24 */
        uint32_t       q     = m >> shift;
        const uint32_t rem   = m & ((1u << shift) - 1u);
        const uint32_t half  = 1u << (shift - 1u);
        if (rem > half || (rem == half && (q & 1u))) q++;
        return (uint16_t)(sign | q);
    }

    /* Нормальное: сменить смещение порядка и отрезать 13 бит мантиссы.
     * Перенос при округлении сам уходит в порядок — так и должно быть. */
    const uint32_t r   = ax - 0x38000000u;
    uint32_t       q   = r >> 13;
    const uint32_t rem = r & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (q & 1u))) q++;
    return (uint16_t)(sign | q);
}

/* ========================================================================== */
/*  Блоки                                                                     */
/* ========================================================================== */

static float block_scale(const uint8_t *blk)
{
    uint16_t h;
    memcpy(&h, blk, sizeof h);
    return smp_f16_to_f32(h);
}

void smp_q8_0_quantize(void *dst, const float *src, size_t n)
{
    uint8_t *out = (uint8_t *)dst;
    for (size_t b = 0; b < n / SMP_Q8_0_BLOCK; b++) {
        const float *x = src + b * SMP_Q8_0_BLOCK;
        uint8_t     *y = out + b * SMP_Q8_0_BYTES;

        float amax = 0.0f;
        for (uint32_t j = 0; j < SMP_Q8_0_BLOCK; j++) {
            const float a = x[j] < 0.0f ? -x[j] : x[j];
            if (a > amax) amax = a;
        }
        const float    d  = amax / 127.0f;
        const float    id = d != 0.0f ? 1.0f / d : 0.0f;
        const uint16_t h  = smp_f32_to_f16(d);
        memcpy(y, &h, sizeof h);

        for (uint32_t j = 0; j < SMP_Q8_0_BLOCK; j++) {
            /* От нуля, как roundf. В double: x*id не больше 127 по модулю,
             * и прибавка 0.5 там точна — во float она сама бы округлялась. */
            const double v = (double)(x[j] * id);
            long q = (long)(v + (v < 0.0 ? -0.5 : 0.5));
            if (q >  127) q =  127;
            if (q < -127) q = -127;
            y[2 + j] = (uint8_t)(int8_t)q;
        }
    }
}

void smp_q8_0_dequantize(float *dst, const void *src, size_t n)
{
    const uint8_t *in = (const uint8_t *)src;
    for (size_t b = 0; b < n / SMP_Q8_0_BLOCK; b++) {
        const uint8_t *blk = in + b * SMP_Q8_0_BYTES;
        const float    d   = block_scale(blk);
        for (uint32_t j = 0; j < SMP_Q8_0_BLOCK; j++)
            dst[b * SMP_Q8_0_BLOCK + j] = d * (float)(int8_t)blk[2 + j];
    }
}

float smp_q8_0_get(const void *src, size_t i)
{
    const uint8_t *blk = (const uint8_t *)src + i / SMP_Q8_0_BLOCK * SMP_Q8_0_BYTES;
    return block_scale(blk) * (float)(int8_t)blk[2 + i % SMP_Q8_0_BLOCK];
}

/* ========================================================================== */
/*  Эталон: C = A x W^T                                                       */
/* ========================================================================== */

/* Сумма — в double, по порядку: быстрой ветке есть с чем сверяться, и
 * расхождение с ней — это её округления, а не эталона. */
void smp_ks_gemm_q8(const SmpBuf *c, const SmpBuf *a, const SmpBuf *w)
{
    const SmpTensor *tc = c->t, *ta = a->t;
    const size_t M = ta->shape[0], K = ta->shape[1], N = w->t->shape[0];
    const size_t row = K / SMP_Q8_0_BLOCK * SMP_Q8_0_BYTES;
    const float *A = (const float *)a->p;
    float       *C = (float *)c->p;

    for (size_t n = 0; n < N; n++) {
        const uint8_t *wr = (const uint8_t *)w->p + n * row;
        for (size_t m = 0; m < M; m++) {
            double acc = 0.0;
            for (size_t k = 0; k < K; k++)
                acc += (double)A[m * ta->stride[0] + k * ta->stride[1]] *
                       (double)smp_q8_0_get(wr, k);
            C[m * tc->stride[0] + n * tc->stride[1]] = (float)acc;
        }
    }
}
