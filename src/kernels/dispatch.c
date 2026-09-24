/* SMPC3 :: dispatch.c -- выбор ветки ядер.
 *
 * Векторная ветка берётся, только когда сходится всё сразу: процессор умеет
 * AVX2+FMA, тип f32, раскладка плотная. Любое несовпадение — скалярный
 * эталон, а не «примерно то же самое».
 */
#include "impl.h"
#include "smpc3/cpu.h"
#include "smpc3/thread.h"

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
        if (t->stride[i] != acc) return false;
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

/* --- Параллельный GEMM ------------------------------------------------------ */

/* Ниже этого M*N*K раздача потокам дороже самой арифметики: 2^22 — это
 * около восьми миллионов операций, доли миллисекунды на одном ядре. */
static uint64_t g_par_min = (uint64_t)1 << 22;

void smp_k_gemm_par_min(uint64_t work) { g_par_min = work; }

typedef struct {
    float             *C;
    const float       *A, *B;
    size_t             ldc, lda, ldb, M, N, K;
    size_t             tile_m, tile_n;
    uint32_t           tiles_n, tiles;
    const SmpFuseStep *steps;
    uint32_t           nsteps;
    const SmpKScratch *sc;
    uint32_t           mxcsr;
    volatile int32_t   next;       /* следующая свободная плитка */
} ParGemm;

/* Поток t пакует в свой комплект и берёт плитки, пока они не кончатся. */
static void par_worker(void *ctx, uint32_t t)
{
    ParGemm *g = (ParGemm *)ctx;

    /* FTZ и прочие режимы — свойство потока, а не программы. Без этого
     * плитки на чужих потоках считались бы в режиме по умолчанию. */
    const uint32_t saved = smp_fpu_get_mxcsr();
    smp_fpu_set_mxcsr(g->mxcsr);

    float *ap = g->sc->apack + (size_t)t * g->sc->set_floats;
    float *bp = g->sc->bpack + (size_t)t * g->sc->set_floats;

    for (;;) {
        const int32_t i = smp_atomic_fetch_add(&g->next, 1);
        if (i < 0 || (uint32_t)i >= g->tiles) break;

        const size_t r0 = ((uint32_t)i / g->tiles_n) * g->tile_m;
        const size_t c0 = ((uint32_t)i % g->tiles_n) * g->tile_n;
        const size_t m  = SMP_MIN(g->tile_m, g->M - r0);
        const size_t n  = SMP_MIN(g->tile_n, g->N - c0);

        smp_ka_gemm_ep(g->C + r0 * g->ldc + c0, g->ldc,
                       g->A + r0 * g->lda, g->lda,
                       g->B + c0, g->ldb, m, n, g->K,
                       ap, bp, g->steps, g->nsteps, r0 * g->N + c0, g->N);
    }
    smp_fpu_set_mxcsr(saved);
}

/* Плитки: по строкам — блоком MC, как в самом ядре; по столбцам — так, чтобы
 * плиток было хотя бы вдвое больше потоков и хвост не простаивал. При M=1,
 * то есть умножении вектора на матрицу, делятся одни столбцы. */
static bool par_gemm(float *C, size_t ldc, const float *A, size_t lda,
                     const float *B, size_t ldb, size_t M, size_t N, size_t K,
                     const SmpKScratch *sc, const SmpFuseStep *steps, uint32_t nsteps)
{
    if (sc->sets < 2 || (uint64_t)M * N * K < g_par_min) return false;

    uint32_t mc, kc, nc;
    smp_k_gemm_block(&mc, &kc, &nc);

    const size_t   tile_m  = SMP_MIN((size_t)mc, M);
    const uint32_t tiles_m = (uint32_t)((M + tile_m - 1) / tile_m);
    const uint32_t want    = 2u * sc->sets;
    const uint32_t cols    = (want + tiles_m - 1) / tiles_m;

    size_t tile_n = (N + cols - 1) / cols;
    tile_n = SMP_ALIGN_UP(tile_n, 16u);           /* ширина микроядра, NR */
    if (tile_n > N) tile_n = N;

    ParGemm g;
    g.C = C; g.A = A; g.B = B;
    g.ldc = ldc; g.lda = lda; g.ldb = ldb;
    g.M = M; g.N = N; g.K = K;
    g.tile_m  = tile_m;
    g.tile_n  = tile_n;
    g.tiles_n = (uint32_t)((N + tile_n - 1) / tile_n);
    g.tiles   = tiles_m * g.tiles_n;
    g.steps   = steps;
    g.nsteps  = nsteps;
    g.sc      = sc;
    g.mxcsr   = smp_fpu_get_mxcsr();
    g.next    = 0;

    if (g.tiles < 2) return false;

    const uint32_t threads = SMP_MIN(sc->sets, g.tiles);
    smp_threads_run(par_worker, &g, threads);
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
        float       *C = (float *)c->p;
        const float *A = (const float *)a->p, *B = (const float *)b->p;
        const size_t M = a->t->shape[0], N = b->t->shape[1], K = a->t->shape[1];

        if (par_gemm(C, c->t->stride[0], A, a->t->stride[0], B, b->t->stride[0],
                     M, N, K, scratch, steps, nsteps))
            return;

        smp_ka_gemm_ep(C, c->t->stride[0], A, a->t->stride[0], B, b->t->stride[0],
                       M, N, K, scratch->apack, scratch->bpack, steps, nsteps, 0u, N);
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

/* --- C = A x B^T ------------------------------------------------------------ */

/* Порог ниже, чем у GEMM: здесь не арифметика, а проход по весам, и делить
 * его между ядрами выгодно раньше. 2^18 — матрица 512x512 на один вектор. */
static uint64_t g_q8_par_min = (uint64_t)1 << 18;

void smp_k_gemm_q8_par_min(uint64_t work) { g_q8_par_min = work; }

/* Строк B на одну порцию: достаточно, чтобы раздача не стоила больше
 * работы, и достаточно мало, чтобы хвост не простаивал. */
#define Q8_ROWS 64u

typedef struct {
    float            *C;
    const float      *A;
    const uint8_t    *W;
    size_t            ldc, lda, M, N, K;
    uint32_t          chunks;
    uint32_t          mxcsr;
    volatile int32_t  next;
} ParQ8;

static void q8_worker(void *ctx, uint32_t t)
{
    (void)t;
    ParQ8 *g = (ParQ8 *)ctx;
    const uint32_t saved = smp_fpu_get_mxcsr();
    smp_fpu_set_mxcsr(g->mxcsr);
    for (;;) {
        const int32_t i = smp_atomic_fetch_add(&g->next, 1);
        if (i < 0 || (uint32_t)i >= g->chunks) break;
        const size_t n0 = (size_t)i * Q8_ROWS;
        const size_t n1 = SMP_MIN(n0 + Q8_ROWS, g->N);
        smp_ka_gemm_q8(g->C, g->ldc, g->A, g->lda, g->W, g->M, g->K, n0, n1);
    }
    smp_fpu_set_mxcsr(saved);
}

void smp_k_mmul_t(const SmpBuf *c, const SmpBuf *a, const SmpBuf *b,
                  SmpKScratch *scratch)
{
    if (b->t->dtype != SMP_DT_Q8_0) {
        /* Транспонированный вид — перестановка шагов, данные на месте. */
        SmpTensor bt = *b->t;
        bt.shape[0]  = b->t->shape[1]; bt.shape[1]  = b->t->shape[0];
        bt.stride[0] = b->t->stride[1]; bt.stride[1] = b->t->stride[0];
        bt.flags = (uint16_t)((bt.flags & (uint16_t)~SMP_TF_CONTIG) | SMP_TF_TRANSPOSED);
        const SmpBuf bb = { b->p, &bt };
        smp_k_gemm(c, a, &bb, scratch);
        return;
    }

    resolve();
    if (!g_use_avx2 || !rowmajor_f32(c->t) || !rowmajor_f32(a->t)) {
        smp_ks_gemm_q8(c, a, b);
        return;
    }

    ParQ8 g;
    g.C   = (float *)c->p;
    g.A   = (const float *)a->p;
    g.W   = (const uint8_t *)b->p;
    g.ldc = c->t->stride[0];
    g.lda = a->t->stride[0];
    g.M   = a->t->shape[0];
    g.N   = b->t->shape[0];
    g.K   = a->t->shape[1];
    g.chunks = (uint32_t)((g.N + Q8_ROWS - 1) / Q8_ROWS);
    g.mxcsr  = smp_fpu_get_mxcsr();
    g.next   = 0;

    const uint32_t sets = scratch ? scratch->sets : 0u;
    if (sets >= 2 && g.chunks >= 2 && (uint64_t)g.M * g.N * g.K >= g_q8_par_min) {
        smp_threads_run(q8_worker, &g, SMP_MIN(sets, g.chunks));
        return;
    }
    smp_ka_gemm_q8(g.C, g.ldc, g.A, g.lda, g.W, g.M, g.K, 0, g.N);
}
