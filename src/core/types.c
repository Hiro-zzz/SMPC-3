/* SMPC3 :: types.c */
#include "smpc3/types.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

char *smp_num_fmt(double v, SmpDType dt, char *buf, size_t cap)
{
    if (dt == SMP_DT_I32 || dt == SMP_DT_U64) {
        snprintf(buf, cap, "%.0f", v);
        return buf;
    }
    if (v != v || v - v != 0.0) {             /* NaN и бесконечности */
        snprintf(buf, cap, "%g", v);
        return buf;
    }
    /* Кратчайшая запись, которая читается обратно в то же число своего
     * типа: 0.1 + 0.2 в f32 — это «0.3», а не «0.300000012», а 1234567 —
     * «1234567», а не «1.23457e+06». У f32 хватает девяти цифр. У f64 —
     * не больше пятнадцати, как у калькулятора: 0.1 + 0.2 в double —
     * «0.3», хвост последнего бита человеку не показывают. */
    const bool f32 = (dt != SMP_DT_F64);
    const int  max = f32 ? 9 : 15;

    /* Цифр меньше, чем в целой части, не берём: иначе 40 записалось бы как
     * «4e+01» — оно тоже читается обратно в 40. Для чисел длиннее max
     * экспонента честнее: 1e+20 короче двадцати одной цифры. */
    int    first = 1;
    double a     = v < 0.0 ? -v : v;
    while (a >= 10.0 && first <= max) { a /= 10.0; first++; }
    if (first > max) first = 1;

    for (int p = first; p <= max; p++) {
        snprintf(buf, cap, "%.*g", p, v);
        const double back = strtod(buf, NULL);
        if (f32 ? (float)back == (float)v : back == v) break;
    }
    return buf;
}

typedef struct { const char *name; uint32_t size; bool is_float; } SmpDTypeInfo;

static const SmpDTypeInfo smp__dt[SMP_DT__COUNT] = {
    /* INVALID */ { "<invalid>", 0, false },
    /* F32     */ { "f32",       4, true  },
    /* F64     */ { "f64",       8, true  },
    /* I32     */ { "i32",       4, false },
    /* U64     */ { "u64",       8, false },
    /* RAW_PTR */ { "raw_ptr",   8, false },
    /* Q8_0    */ { "q8_0",      0, false }
};

const char *smp_dtype_name(SmpDType dt)
{
    if ((unsigned)dt >= SMP_DT__COUNT) return "<corrupt>";
    return smp__dt[dt].name;
}

uint32_t smp_dtype_size(SmpDType dt)
{
    if ((unsigned)dt >= SMP_DT__COUNT) return 0;
    return smp__dt[dt].size;
}

bool smp_dtype_is_block(SmpDType dt)
{
    return dt == SMP_DT_Q8_0;
}

/* Блоки считаются целыми: хвост строки блочного типа всё равно занимает
 * блок, и проверке границ нужен именно занятый размер. */
uint64_t smp_dtype_bytes(SmpDType dt, uint64_t nelem)
{
    if (dt == SMP_DT_Q8_0)
        return (nelem + SMP_Q8_0_BLOCK - 1u) / SMP_Q8_0_BLOCK * SMP_Q8_0_BYTES;
    return nelem * smp_dtype_size(dt);
}

bool smp_dtype_is_float(SmpDType dt)
{
    if ((unsigned)dt >= SMP_DT__COUNT) return false;
    return smp__dt[dt].is_float;
}

SmpDType smp_dtype_parse(const char *s, size_t len)
{
    for (unsigned i = 1; i < SMP_DT__COUNT; i++) {
        const char *n = smp__dt[i].name;
        if (strlen(n) == len && memcmp(n, s, len) == 0) return (SmpDType)i;
    }
    return SMP_DT_INVALID;
}

uint32_t smp_dtype_lanes(SmpDType dt, uint32_t vec_bits)
{
    uint32_t sz = smp_dtype_size(dt);
    if (sz == 0) return 0;
    return vec_bits / (sz * 8u);
}

void smp_tensor_dense(SmpTensor *t, SmpDType dt, uint32_t rank, const uint32_t *shape)
{
    memset(t, 0, sizeof(*t));
    if (rank == 0 || rank > SMP_MAX_RANK) return;

    t->dtype = (uint8_t)dt;
    t->rank  = (uint8_t)rank;
    t->off   = 0;
    t->flags = SMP_TF_CONTIG;

    uint32_t n = 1;
    for (uint32_t i = 0; i < rank; i++) {
        t->shape[i] = shape[i];
        n *= shape[i];
    }
    t->nelem = n;

    /* row-major: последняя ось самая быстрая */
    uint32_t acc = 1;
    for (uint32_t i = rank; i-- > 0; ) {
        t->stride[i] = acc;
        acc *= shape[i];
    }
}

uint64_t smp_tensor_bytes(const SmpTensor *t)
{
    return smp_dtype_bytes((SmpDType)t->dtype, t->nelem);
}

bool smp_tensor_is_contiguous(const SmpTensor *t)
{
    uint32_t acc = 1;
    for (uint32_t i = t->rank; i-- > 0; ) {
        if (t->stride[i] != acc) return false;
        acc *= (uint32_t)t->shape[i];
    }
    return true;
}

bool smp_tensor_same_shape(const SmpTensor *a, const SmpTensor *b)
{
    if (a->rank != b->rank) return false;
    for (uint32_t i = 0; i < a->rank; i++)
        if (a->shape[i] != b->shape[i]) return false;
    return true;
}

char *smp_tensor_sig(const SmpTensor *t, char *buf, size_t cap)
{
    size_t n = 0;
    int w = snprintf(buf, cap, "%s:", smp_dtype_name((SmpDType)t->dtype));
    if (w > 0) n = (size_t)w;
    for (uint32_t i = 0; i < t->rank && n < cap; i++) {
        w = snprintf(buf + n, cap - n, "%s%u", i ? "," : "", (unsigned)t->shape[i]);
        if (w > 0) n += (size_t)w;
    }
    return buf;
}
