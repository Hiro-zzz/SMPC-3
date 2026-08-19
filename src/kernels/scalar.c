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
/*  Плотный путь с заранее известным типом                                    */
/* ========================================================================== */

/* Общий обход ниже разбирает dtype на КАЖДОМ элементе через load_at/store_at и
 * вдобавок зовёт операцию по указателю. На элемент это стоит дороже самой
 * арифметики, а компилятор через косвенный вызов не видит цикла и не
 * векторизует его.
 *
 * Мириться с этим было бы можно, будь файл только эталоном для сверки. Но
 * диспетчер отправляет сюда всё, что не f32 с плотной раскладкой: весь f64,
 * i32, u64 и любой срез с шагом. Замер сложения 1024x1024 показывает цену:
 * f64 — 1550 мкс, f32 через avx2 — 200 мкс, и это при вдвое большем объёме
 * данных.
 *
 * Поэтому там, где типы совпадают и тензоры плотные, тип разбирается ОДИН раз
 * до цикла, а дальше идёт обычный цикл по массиву. Общий путь никуда не делся
 * и обслуживает всё остальное — разные типы, шаги, транспозицию.
 *
 * Семантика обязана совпадать с общим путём поэлементно, поэтому насыщение
 * целых берётся из того же clamp_to с теми же границами, а не пишется заново.
 * Порядок обхода тот же, так что и свёртки дают бит в бит прежний результат.
 *
 * restrict здесь не ставится сознательно: эмиттер умеет писать результат
 * поверх входа (relu r4, r4), то есть dst и src законно совпадают. */

/* Плотны ли оба и одного ли типа. */
static bool dense_same(const SmpBuf *dst, const SmpBuf *src)
{
    return dst->t->dtype == src->t->dtype &&
           dst->t->nelem == src->t->nelem &&
           dense(dst->t) && dense(src->t);
}

#define KS_U_FLT(T, EXPR)                                                          do {                                                                               T *dp = (T *)dst->p; const T *sp = (const T *)src->p;                          for (uint32_t i = 0; i < n; i++) {                                                 const double x = (double)sp[i];                                                dp[i] = (T)(EXPR);                                                         }                                                                          } while (0)

#define KS_U_INT(T, LO, HI, EXPR)                                                  do {                                                                               T *dp = (T *)dst->p; const T *sp = (const T *)src->p;                          for (uint32_t i = 0; i < n; i++) {                                                 const double x = (double)sp[i];                                                dp[i] = (T)clamp_to((EXPR), LO, HI);                                       }                                                                          } while (0)

/* Разворачивается по всем типам и возвращает управление, если тип известен.
 * Для raw_ptr и мусора проваливается в общий путь. */
#define KS_UNARY_DENSE(EXPR)                                                       switch (dt) {                                                                      case SMP_DT_F32: KS_U_FLT(float,  EXPR); return;                               case SMP_DT_F64: KS_U_FLT(double, EXPR); return;                               case SMP_DT_I32: KS_U_INT(int32_t, -2147483648.0, 2147483647.0, EXPR);                           return;                                                       case SMP_DT_U64: KS_U_INT(uint64_t, 0.0, 18446744073709549568.0, EXPR);                          return;                                                       default: break;                                                            }

#define KS_B_FLT(T, EXPR)                                                          do {                                                                               T *dp = (T *)dst->p;                                                           const T *ap = (const T *)a->p, *bp = (const T *)b->p;                          for (uint32_t i = 0; i < n; i++) {                                                 const double x = (double)ap[i], y = (double)bp[i];                             dp[i] = (T)(EXPR);                                                         }                                                                          } while (0)

#define KS_B_INT(T, LO, HI, EXPR)                                                  do {                                                                               T *dp = (T *)dst->p;                                                           const T *ap = (const T *)a->p, *bp = (const T *)b->p;                          for (uint32_t i = 0; i < n; i++) {                                                 const double x = (double)ap[i], y = (double)bp[i];                             dp[i] = (T)clamp_to((EXPR), LO, HI);                                       }                                                                          } while (0)

#define KS_BINARY_DENSE(EXPR)                                                      switch (dt) {                                                                      case SMP_DT_F32: KS_B_FLT(float,  EXPR); return;                               case SMP_DT_F64: KS_B_FLT(double, EXPR); return;                               case SMP_DT_I32: KS_B_INT(int32_t, -2147483648.0, 2147483647.0, EXPR);                           return;                                                       case SMP_DT_U64: KS_B_INT(uint64_t, 0.0, 18446744073709549568.0, EXPR);                          return;                                                       default: break;                                                            }

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

void smp_ks_relu(const SmpBuf *dst, const SmpBuf *src)
{
    if (dense_same(dst, src)) {
        const uint32_t n = dst->t->nelem;
        const SmpDType dt = (SmpDType)dst->t->dtype;
        KS_UNARY_DENSE(x > 0.0 ? x : 0.0);
    }
    unary(dst, src, f_relu, 0.0);
}

void smp_ks_abs(const SmpBuf *dst, const SmpBuf *src)
{
    if (dense_same(dst, src)) {
        const uint32_t n = dst->t->nelem;
        const SmpDType dt = (SmpDType)dst->t->dtype;
        KS_UNARY_DENSE(x < 0.0 ? -x : x);
    }
    unary(dst, src, f_abs, 0.0);
}

void smp_ks_scale(const SmpBuf *dst, const SmpBuf *src, double k)
{
    if (dense_same(dst, src)) {
        const uint32_t n = dst->t->nelem;
        const SmpDType dt = (SmpDType)dst->t->dtype;
        KS_UNARY_DENSE(x * k);
    }
    unary(dst, src, f_scale, k);
}

void smp_ks_copy(const SmpBuf *dst, const SmpBuf *src)
{
    /* Плотная копия одного типа — это memcpy и есть. Совпадение указателей
     * законно (эмиттер пишет поверх входа), а memcpy на равных указателях —
     * UB; копировать в таком случае и нечего. */
    if (dense_same(dst, src)) {
        if (dst->p != src->p)
            memcpy(dst->p, src->p,
                   (size_t)dst->t->nelem * smp_dtype_size((SmpDType)dst->t->dtype));
        return;
    }
    unary(dst, src, f_id, 0.0);
}

/* Приведение по определению меняет тип, так что быстрый путь ему достаётся
 * только на вырожденном случае «тип тот же». Остальное — общий обход. */
void smp_ks_cast(const SmpBuf *dst, const SmpBuf *src)
{
    if (dense_same(dst, src)) { smp_ks_copy(dst, src); return; }
    unary(dst, src, f_id, 0.0);
}

void smp_ks_fill(const SmpBuf *dst, double v)
{
    const SmpTensor *t = dst->t;
    const SmpDType   dt = (SmpDType)t->dtype;

    /* Значение одно на весь тензор, поэтому и насыщение считается один раз:
     * store_at пересчитывал бы его на каждом элементе с тем же результатом. */
    if (dense(t)) {
        const uint32_t n = t->nelem;
        switch (dt) {
            case SMP_DT_F32: {
                float *dp = (float *)dst->p; const float fv = (float)v;
                for (uint32_t i = 0; i < n; i++) dp[i] = fv;
                return;
            }
            case SMP_DT_F64: {
                double *dp = (double *)dst->p;
                for (uint32_t i = 0; i < n; i++) dp[i] = v;
                return;
            }
            case SMP_DT_I32: {
                int32_t *dp = (int32_t *)dst->p;
                const int32_t iv =
                    (int32_t)clamp_to(v, -2147483648.0, 2147483647.0);
                for (uint32_t i = 0; i < n; i++) dp[i] = iv;
                return;
            }
            case SMP_DT_U64: {
                uint64_t *dp = (uint64_t *)dst->p;
                const uint64_t uv =
                    (uint64_t)clamp_to(v, 0.0, 18446744073709549568.0);
                for (uint32_t i = 0; i < n; i++) dp[i] = uv;
                return;
            }
            default: break;
        }
        for (uint32_t i = 0; i < n; i++) store_at(dst->p, dt, i, v);
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

/* Все три одного типа и плотные — условие быстрого пути для бинарных. */
static bool dense_same3(const SmpBuf *dst, const SmpBuf *a, const SmpBuf *b)
{
    return dense_same(dst, a) && dense_same(dst, b);
}

void smp_ks_add(const SmpBuf *dst, const SmpBuf *a, const SmpBuf *b)
{
    if (dense_same3(dst, a, b)) {
        const uint32_t n = dst->t->nelem;
        const SmpDType dt = (SmpDType)dst->t->dtype;
        KS_BINARY_DENSE(x + y);
    }
    binary(dst, a, b, false);
}

void smp_ks_mul(const SmpBuf *dst, const SmpBuf *a, const SmpBuf *b)
{
    if (dense_same3(dst, a, b)) {
        const uint32_t n = dst->t->nelem;
        const SmpDType dt = (SmpDType)dst->t->dtype;
        KS_BINARY_DENSE(x * y);
    }
    binary(dst, a, b, true);
}

/* ========================================================================== */
/*  Свёртки                                                                   */
/* ========================================================================== */

double smp_ks_reduce_add(const SmpBuf *src)
{
    const SmpTensor *t = src->t;
    const SmpDType   dt = (SmpDType)t->dtype;
    double acc = 0.0;

    /* Порядок обхода и тип накопителя те же, что в общем пути, поэтому
     * результат совпадает бит в бит — на этом стоит сверка векторных ядер. */
    if (dense(t)) {
        const uint32_t n = t->nelem;
        switch (dt) {
            case SMP_DT_F32: { const float    *sp = (const float    *)src->p;
                for (uint32_t i = 0; i < n; i++) acc += (double)sp[i]; return acc; }
            case SMP_DT_F64: { const double   *sp = (const double   *)src->p;
                for (uint32_t i = 0; i < n; i++) acc += sp[i];         return acc; }
            case SMP_DT_I32: { const int32_t  *sp = (const int32_t  *)src->p;
                for (uint32_t i = 0; i < n; i++) acc += (double)sp[i]; return acc; }
            case SMP_DT_U64: { const uint64_t *sp = (const uint64_t *)src->p;
                for (uint32_t i = 0; i < n; i++) acc += (double)sp[i]; return acc; }
            default: break;
        }
        for (uint32_t i = 0; i < n; i++) acc += load_at(src->p, dt, i);
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
        const uint32_t n = t->nelem;
        switch (dt) {
            case SMP_DT_F32: { const float    *sp = (const float    *)src->p;
                for (uint32_t i = 0; i < n; i++) { const double v = (double)sp[i];
                    if (v > best) best = v; } return best; }
            case SMP_DT_F64: { const double   *sp = (const double   *)src->p;
                for (uint32_t i = 0; i < n; i++) { const double v = sp[i];
                    if (v > best) best = v; } return best; }
            case SMP_DT_I32: { const int32_t  *sp = (const int32_t  *)src->p;
                for (uint32_t i = 0; i < n; i++) { const double v = (double)sp[i];
                    if (v > best) best = v; } return best; }
            case SMP_DT_U64: { const uint64_t *sp = (const uint64_t *)src->p;
                for (uint32_t i = 0; i < n; i++) { const double v = (double)sp[i];
                    if (v > best) best = v; } return best; }
            default: break;
        }
        for (uint32_t i = 0; i < n; i++) {
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

/* Тело тройного цикла с известным типом. Шаги остаются произвольными: срез и
 * транспозиция обязаны работать и здесь, поэтому индексы считаются, а не
 * инкрементируются. STORE задаёт запись элемента — у целых через clamp_to. */
#define KS_GEMM_LOOP(T, STORE)                                                     do {                                                                               T *cp = (T *)c->p;                                                             const T *ap = (const T *)a->p, *bp = (const T *)b->p;                          for (uint32_t i = 0; i < M; i++) {                                                 for (uint32_t j = 0; j < N; j++)                                                   cp[(size_t)i * sc0 + (size_t)j * sc1] = (T)0;                              for (uint32_t k = 0; k < K; k++) {                                                 const double av =                                                                  (double)ap[(size_t)i * sa0 + (size_t)k * sa1];                             if (av == 0.0) continue;                                                       for (uint32_t j = 0; j < N; j++) {                                                 const size_t oc = (size_t)i * sc0 + (size_t)j * sc1;                           const double bv =                                                                  (double)bp[(size_t)k * sb0 + (size_t)j * sb1];                             cp[oc] = STORE((double)cp[oc] + av * bv);                                  }                                                                          }                                                                          }                                                                              return;                                                                    } while (0)

#define KS_GST_F32(V) (float)(V)
#define KS_GST_F64(V) (double)(V)
#define KS_GST_I32(V) (int32_t)clamp_to((V), -2147483648.0, 2147483647.0)
#define KS_GST_U64(V) (uint64_t)clamp_to((V), 0.0, 18446744073709549568.0)

void smp_ks_gemm(const SmpBuf *c, const SmpBuf *a, const SmpBuf *b)
{
    const SmpTensor *tc = c->t, *ta = a->t, *tb = b->t;
    const SmpDType   dc = (SmpDType)tc->dtype;
    const SmpDType   da = (SmpDType)ta->dtype, db = (SmpDType)tb->dtype;

    const uint32_t M = ta->shape[0], K = ta->shape[1], N = tb->shape[1];

    /* Во внутреннем цикле два чтения и запись, и каждая разбирала dtype заново.
     * На f64:256,256 это 1.6 ГФЛОПС — при том, что арифметики здесь на порядок
     * меньше, чем работы по выяснению типа. Когда типы совпадают, разбираем их
     * один раз до циклов. */
    const uint32_t sc0 = tc->stride[0], sc1 = tc->stride[1];
    const uint32_t sa0 = ta->stride[0], sa1 = ta->stride[1];
    const uint32_t sb0 = tb->stride[0], sb1 = tb->stride[1];

    if (dc == da && dc == db) {
        switch (dc) {
            case SMP_DT_F32: KS_GEMM_LOOP(float,    KS_GST_F32);
            case SMP_DT_F64: KS_GEMM_LOOP(double,   KS_GST_F64);
            case SMP_DT_I32: KS_GEMM_LOOP(int32_t,  KS_GST_I32);
            case SMP_DT_U64: KS_GEMM_LOOP(uint64_t, KS_GST_U64);
            default: break;
        }
    }

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
