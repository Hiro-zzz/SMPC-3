/* SMPC3 :: scalar.c -- скалярный эталон вычислительных ядер.
 *
 * Здесь не оптимизируют: этот файл существует, чтобы у быстрых версий из Ф6
 * было с чем сверяться поэлементно. Читаемость важнее тактов.
 */
#include "impl.h"

#include <string.h>
#include <math.h>

/* ========================================================================== */
/*  Обход по шагам                                                            */
/* ========================================================================== */

/* Смещение элемента в ЭЛЕМЕНТАХ по логическому индексу. */
static size_t elem_index(const SmpTensor *t, const uint32_t *idx)
{
    size_t off = 0;
    for (uint32_t i = 0; i < t->rank; i++)
        off += (size_t)idx[i] * (size_t)t->stride[i];
    return off;
}

/* Плотный ли тензор — тогда можно идти линейно, без пересчёта индексов. */
static bool dense(const SmpTensor *t)
{
    uint32_t acc = 1;
    for (uint32_t i = t->rank; i-- > 0; ) {
        if (t->stride[i] != (uint16_t)acc) return false;
        acc *= (uint32_t)t->shape[i];
    }
    return true;
}

/* Увеличить многомерный индекс на единицу. false — обход закончен. */
static bool idx_next(const SmpTensor *t, uint32_t *idx)
{
    for (uint32_t i = t->rank; i-- > 0; ) {
        if (++idx[i] < t->shape[i]) return true;
        idx[i] = 0;
    }
    return false;
}

/* ========================================================================== */
/*  Чтение и запись элемента                                                  */
/* ========================================================================== */

static double load_at(const void *p, SmpDType dt, size_t i)
{
    switch (dt) {
        case SMP_DT_F32: return (double)((const float    *)p)[i];
        case SMP_DT_F64: return           ((const double  *)p)[i];
        case SMP_DT_I32: return (double)((const int32_t  *)p)[i];
        case SMP_DT_U64: return (double)((const uint64_t *)p)[i];
        default:         return 0.0;
    }
}

/* Преобразование double в целое вне диапазона типа — это UB, а не «как
 * получится»: стандарт не обещает ни обрезания, ни насыщения, а на x86
 * cvttsd2si молча выдаёт 0x80000000. Насыщаем явно, NaN отправляем в ноль. */
static double clamp_to(double v, double lo, double hi)
{
    if (v != v) return 0.0;          /* NaN */
    return v < lo ? lo : (v > hi ? hi : v);
}

static void store_at(void *p, SmpDType dt, size_t i, double v)
{
    switch (dt) {
        case SMP_DT_F32: ((float  *)p)[i] = (float)v; break;
        case SMP_DT_F64: ((double *)p)[i] = v;        break;

        case SMP_DT_I32:
            ((int32_t *)p)[i] =
                (int32_t)clamp_to(v, -2147483648.0, 2147483647.0);
            break;

        /* 2^64 в double не представимо точно; ближайшее снизу — 2^64-2048. */
        case SMP_DT_U64:
            ((uint64_t *)p)[i] =
                (uint64_t)clamp_to(v, 0.0, 18446744073709549568.0);
            break;

        default: break;
    }
}

/* ========================================================================== */
/*  Поэлементные                                                              */
/* ========================================================================== */

typedef double (*UnaryFn)(double, double);

static double f_relu (double x, double k) { SMP_UNUSED(k); return x > 0.0 ? x : 0.0; }
static double f_abs  (double x, double k) { SMP_UNUSED(k); return x < 0.0 ? -x : x; }
static double f_scale(double x, double k) { return x * k; }
static double f_id   (double x, double k) { SMP_UNUSED(k); return x; }

static void unary(const SmpBuf *dst, const SmpBuf *src, UnaryFn f, double k)
{
    const SmpTensor *td = dst->t, *ts = src->t;
    const SmpDType   dd = (SmpDType)td->dtype, ds = (SmpDType)ts->dtype;

    if (dense(td) && dense(ts) && td->nelem == ts->nelem) {
        for (uint32_t i = 0; i < td->nelem; i++)
            store_at(dst->p, dd, i, f(load_at(src->p, ds, i), k));
        return;
    }

    uint32_t idx[SMP_MAX_RANK] = { 0, 0, 0, 0 };
    do {
        const size_t oi = elem_index(ts, idx);
        const size_t oo = elem_index(td, idx);
        store_at(dst->p, dd, oo, f(load_at(src->p, ds, oi), k));
    } while (idx_next(ts, idx));
}

void smp_ks_relu(const SmpBuf *d, const SmpBuf *s)           { unary(d, s, f_relu,  0.0); }
void smp_ks_abs (const SmpBuf *d, const SmpBuf *s)           { unary(d, s, f_abs,   0.0); }
void smp_ks_scale(const SmpBuf *d, const SmpBuf *s, double k) { unary(d, s, f_scale, k);   }
void smp_ks_copy(const SmpBuf *d, const SmpBuf *s)           { unary(d, s, f_id,    0.0); }
void smp_ks_cast(const SmpBuf *d, const SmpBuf *s)           { unary(d, s, f_id,    0.0); }

void smp_ks_fill(const SmpBuf *dst, double v)
{
    const SmpTensor *t = dst->t;
    const SmpDType   dt = (SmpDType)t->dtype;

    if (dense(t)) {
        for (uint32_t i = 0; i < t->nelem; i++) store_at(dst->p, dt, i, v);
        return;
    }
    uint32_t idx[SMP_MAX_RANK] = { 0, 0, 0, 0 };
    do { store_at(dst->p, dt, elem_index(t, idx), v); } while (idx_next(t, idx));
}

void smp_ks_zero(const SmpBuf *dst)
{
    const SmpTensor *t = dst->t;
    if (dense(t)) {
        memset(dst->p, 0, (size_t)t->nelem * smp_dtype_size((SmpDType)t->dtype));
        return;
    }
    smp_ks_fill(dst, 0.0);
}

/* ========================================================================== */
/*  Бинарные                                                                  */
/* ========================================================================== */

static void binary(const SmpBuf *dst, const SmpBuf *a, const SmpBuf *b, bool mul)
{
    const SmpTensor *td = dst->t, *ta = a->t, *tb = b->t;
    const SmpDType   dd = (SmpDType)td->dtype;
    const SmpDType   da = (SmpDType)ta->dtype, db = (SmpDType)tb->dtype;

    if (dense(td) && dense(ta) && dense(tb)) {
        for (uint32_t i = 0; i < td->nelem; i++) {
            const double x = load_at(a->p, da, i), y = load_at(b->p, db, i);
            store_at(dst->p, dd, i, mul ? x * y : x + y);
        }
        return;
    }

    uint32_t idx[SMP_MAX_RANK] = { 0, 0, 0, 0 };
    do {
        const double x = load_at(a->p, da, elem_index(ta, idx));
        const double y = load_at(b->p, db, elem_index(tb, idx));
        store_at(dst->p, dd, elem_index(td, idx), mul ? x * y : x + y);
    } while (idx_next(td, idx));
}

void smp_ks_add(const SmpBuf *d, const SmpBuf *a, const SmpBuf *b) { binary(d, a, b, false); }
void smp_ks_mul(const SmpBuf *d, const SmpBuf *a, const SmpBuf *b) { binary(d, a, b, true);  }

/* ========================================================================== */
/*  Свёртки                                                                   */
/* ========================================================================== */

double smp_ks_reduce_add(const SmpBuf *src)
{
    const SmpTensor *t = src->t;
    const SmpDType   dt = (SmpDType)t->dtype;
    double acc = 0.0;

    if (dense(t)) {
        for (uint32_t i = 0; i < t->nelem; i++) acc += load_at(src->p, dt, i);
        return acc;
    }
    uint32_t idx[SMP_MAX_RANK] = { 0, 0, 0, 0 };
    do { acc += load_at(src->p, dt, elem_index(t, idx)); } while (idx_next(t, idx));
    return acc;
}

double smp_ks_reduce_max(const SmpBuf *src)
{
    const SmpTensor *t = src->t;
    const SmpDType   dt = (SmpDType)t->dtype;
    double best = -INFINITY;

    if (dense(t)) {
        for (uint32_t i = 0; i < t->nelem; i++) {
            const double v = load_at(src->p, dt, i);
            if (v > best) best = v;
        }
        return best;
    }
    uint32_t idx[SMP_MAX_RANK] = { 0, 0, 0, 0 };
    do {
        const double v = load_at(src->p, dt, elem_index(t, idx));
        if (v > best) best = v;
    } while (idx_next(t, idx));
    return best;
}

/* ========================================================================== */
/*  GEMM                                                                      */
/* ========================================================================== */

void smp_ks_gemm(const SmpBuf *c, const SmpBuf *a, const SmpBuf *b)
{
    const SmpTensor *tc = c->t, *ta = a->t, *tb = b->t;
    const SmpDType   dc = (SmpDType)tc->dtype;
    const SmpDType   da = (SmpDType)ta->dtype, db = (SmpDType)tb->dtype;

    const uint32_t M = ta->shape[0], K = ta->shape[1], N = tb->shape[1];

    /* Порядок i-k-j, а не i-j-k: даже в эталоне внутренний цикл идёт по
     * последней оси обоих операндов, иначе на 1024x1024 это невыносимо. */
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++)
            store_at(c->p, dc, (size_t)i * tc->stride[0] + (size_t)j * tc->stride[1], 0.0);

        for (uint32_t k = 0; k < K; k++) {
            const double av = load_at(a->p, da,
                (size_t)i * ta->stride[0] + (size_t)k * ta->stride[1]);
            if (av == 0.0) continue;

            for (uint32_t j = 0; j < N; j++) {
                const size_t oc = (size_t)i * tc->stride[0] + (size_t)j * tc->stride[1];
                const double bv = load_at(b->p, db,
                    (size_t)k * tb->stride[0] + (size_t)j * tb->stride[1]);
                store_at(c->p, dc, oc, load_at(c->p, dc, oc) + av * bv);
            }
        }
    }
}
