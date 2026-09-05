/* SMPC3 :: dispatch.c -- выбор ветки ядер.
 *
 * Векторная ветка берётся, только когда сходится всё сразу: процессор умеет
 * AVX2+FMA, тип f32, раскладка плотная. Любое несовпадение — скалярный
 * эталон, а не «примерно то же самое».
 */
#include "impl.h"
#include "smpc3/cpu.h"

#include <string.h>

static SmpKernelBackend g_backend = SMP_KB_AUTO;
static bool             g_ready   = false;
static bool             g_use_avx2 = false;

static void resolve(void)
{
    if (g_ready) return;
    const bool hw = smp_cpu_has(SMP_ISA_AVX2 | SMP_ISA_FMA);
    switch (g_backend) {
        case SMP_KB_SCALAR: g_use_avx2 = false;      break;
        case SMP_KB_AVX2:   g_use_avx2 = hw;         break;
        default:            g_use_avx2 = hw;         break;
    }
    g_ready = true;
}

bool smp_kernels_select(SmpKernelBackend b)
{
    if (b == SMP_KB_AVX2 && !smp_cpu_has(SMP_ISA_AVX2 | SMP_ISA_FMA)) return false;
    g_backend = b;
    g_ready   = false;
    resolve();
    return true;
}

const char *smp_kernels_name(void)
{
    resolve();
    return g_use_avx2 ? "avx2+fma" : "scalar";
}

/* ========================================================================== */
/*  Применимость векторной ветки                                              */
/* ========================================================================== */

static bool dense_f32(const SmpTensor *t)
{
    if (t->dtype != SMP_DT_F32) return false;
    uint32_t acc = 1;
    for (uint32_t i = t->rank; i-- > 0; ) {
        if (t->stride[i] != (uint16_t)acc) return false;
        acc *= (uint32_t)t->shape[i];
    }
    return true;
}

/* Для поэлементных нужно, чтобы совпадало ещё и число элементов: иначе это
 * не та операция, которую векторная ветка умеет. */
static bool vec_unary(const SmpBuf *d, const SmpBuf *s)
{
    resolve();
    return g_use_avx2 && dense_f32(d->t) && dense_f32(s->t) &&
           d->t->nelem == s->t->nelem;
}

static bool vec_binary(const SmpBuf *d, const SmpBuf *a, const SmpBuf *b)
{
    resolve();
    return g_use_avx2 && dense_f32(d->t) && dense_f32(a->t) && dense_f32(b->t) &&
           d->t->nelem == a->t->nelem && d->t->nelem == b->t->nelem;
}

/* GEMM терпимее: ведущий шаг может быть любым, важна лишь единичность шага по
 * последней оси — на этом стоит и упаковка B, и загрузки в микроядре. */
static bool rowmajor_f32(const SmpTensor *t)
{
    return t->dtype == SMP_DT_F32 && t->rank == 2 && t->stride[1] == 1;
}

/* ========================================================================== */
/*  Публичные ядра                                                            */
/* ========================================================================== */

void smp_k_relu(const SmpBuf *d, const SmpBuf *s)
{
    if (vec_unary(d, s)) smp_ka_relu((float *)d->p, (const float *)s->p, s->t->nelem);
    else                 smp_ks_relu(d, s);
}

void smp_k_abs(const SmpBuf *d, const SmpBuf *s)
{
    if (vec_unary(d, s)) smp_ka_abs((float *)d->p, (const float *)s->p, s->t->nelem);
    else                 smp_ks_abs(d, s);
}

void smp_k_scale(const SmpBuf *d, const SmpBuf *s, double k)
{
    if (vec_unary(d, s))
        smp_ka_scale((float *)d->p, (const float *)s->p, s->t->nelem, (float)k);
    else
        smp_ks_scale(d, s, k);
}

void smp_k_copy(const SmpBuf *d, const SmpBuf *s)
{
    /* Копия одного и того же типа с плотной раскладкой — это memcpy, и никакая
     * векторная арифметика тут не нужна. */
    if (d->t->dtype == s->t->dtype && dense_f32(d->t) && dense_f32(s->t) &&
        d->t->nelem == s->t->nelem) {
        memcpy(d->p, s->p, (size_t)d->t->nelem * sizeof(float));
        return;
    }
    smp_ks_copy(d, s);
}

void smp_k_cast(const SmpBuf *d, const SmpBuf *s) { smp_ks_cast(d, s); }
void smp_k_fill(const SmpBuf *d, double v)        { smp_ks_fill(d, v); }
void smp_k_zero(const SmpBuf *d)                  { smp_ks_zero(d); }

void smp_k_add(const SmpBuf *d, const SmpBuf *a, const SmpBuf *b)
{
    if (vec_binary(d, a, b))
        smp_ka_add((float *)d->p, (const float *)a->p, (const float *)b->p, d->t->nelem);
    else
        smp_ks_add(d, a, b);
}

void smp_k_mul(const SmpBuf *d, const SmpBuf *a, const SmpBuf *b)
{
    if (vec_binary(d, a, b))
        smp_ka_mul((float *)d->p, (const float *)a->p, (const float *)b->p, d->t->nelem);
    else
        smp_ks_mul(d, a, b);
}

void smp_k_fuse(const SmpBuf *dst, const SmpBuf *src,
                const SmpFuseStep *steps, uint32_t nsteps)
{
    resolve();

    /* Векторная ветка требует того же, что и обычные поэлементные: f32,
     * плотно, совпадающее число элементов — и это должно держаться для КАЖДОГО
     * операнда цепочки, иначе считает эталон. */
    if (g_use_avx2 && dense_f32(dst->t) && dense_f32(src->t) &&
        dst->t->nelem == src->t->nelem) {
        bool ok = true;
        for (uint32_t i = 0; i < nsteps && ok; i++)
            if (steps[i].op == SMP_FOP_ADD || steps[i].op == SMP_FOP_MUL)
                ok = dense_f32(steps[i].b.t) &&
                     steps[i].b.t->nelem == dst->t->nelem;
        if (ok) {
            smp_ka_fuse((float *)dst->p, (const float *)src->p,
                        dst->t->nelem, steps, nsteps);
            return;
        }
    }
    smp_ks_fuse(dst, src, steps, nsteps);
}

double smp_k_fuse_reduce(const SmpBuf *src, const SmpFuseStep *steps,
                         uint32_t nsteps, uint8_t red)
{
    resolve();

    if (g_use_avx2 && dense_f32(src->t)) {
        bool ok = true;
        for (uint32_t i = 0; i < nsteps && ok; i++)
            if (steps[i].op == SMP_FOP_ADD || steps[i].op == SMP_FOP_MUL)
                ok = dense_f32(steps[i].b.t) &&
                     steps[i].b.t->nelem == src->t->nelem;
        if (ok)
            return smp_ka_fuse_reduce((const float *)src->p, src->t->nelem,
                                      steps, nsteps, red);
    }
    return smp_ks_fuse_reduce(src, steps, nsteps, red);
}

double smp_k_reduce_add(const SmpBuf *s)
{
    resolve();
    if (g_use_avx2 && dense_f32(s->t))
        return smp_ka_reduce_add((const float *)s->p, s->t->nelem);
    return smp_ks_reduce_add(s);
}

double smp_k_reduce_max(const SmpBuf *s)
{
    resolve();
    if (g_use_avx2 && dense_f32(s->t))
        return smp_ka_reduce_max((const float *)s->p, s->t->nelem);
    return smp_ks_reduce_max(s);
}

/* Плотны ли все операнды цепочки и той же длины, что C. Проверка ровно та же,
 * что у обычного слияния: векторный эпилог адресует их линейным индексом. */
static bool ep_dense(const SmpBuf *c, const SmpFuseStep *st, uint32_t ns)
{
    for (uint32_t i = 0; i < ns; i++)
        if (st[i].op == SMP_FOP_ADD || st[i].op == SMP_FOP_MUL)
            if (!dense_f32(st[i].b.t) || st[i].b.t->nelem != c->t->nelem)
                return false;
    return true;
}

void smp_k_gemm_ep(const SmpBuf *c, const SmpBuf *a, const SmpBuf *b,
                   SmpKScratch *scratch, const SmpFuseStep *steps, uint32_t nsteps)
{
    resolve();
    /* Без рабочей памяти векторная ветка не запускается: молча подсунуть ей
     * общий буфер значило бы вернуть ровно ту гонку, ради устранения которой
     * scratch и появился. */
    if (g_use_avx2 && scratch && scratch->bytes &&
        rowmajor_f32(c->t) && rowmajor_f32(a->t) && rowmajor_f32(b->t) &&
        ep_dense(c, steps, nsteps)) {
        smp_ka_gemm_ep((float *)c->p, c->t->stride[0],
                       (const float *)a->p, a->t->stride[0],
                       (const float *)b->p, b->t->stride[0],
                       a->t->shape[0], b->t->shape[1], a->t->shape[1],
                       scratch->apack, scratch->bpack, steps, nsteps);
        return;
    }

    /* Эталон считает GEMM как считал, а цепочку докладывает отдельным
     * проходом. Результат тот же; экономии прохода по C здесь нет, и обещать
     * её было бы нечестно. */
    smp_ks_gemm(c, a, b);
    if (nsteps) smp_ks_fuse(c, c, steps, nsteps);
}

void smp_k_gemm(const SmpBuf *c, const SmpBuf *a, const SmpBuf *b,
                SmpKScratch *scratch)
{
    smp_k_gemm_ep(c, a, b, scratch, NULL, 0u);
}
