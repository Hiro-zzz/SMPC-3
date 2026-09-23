/* SMPC3 :: avx2.c -- ядра на AVX2 + FMA для f32.
 *
 * Файл собирается с -mavx2 -mfma, поэтому компилятор вправе разложить в AVX
 * ЛЮБУЮ функцию отсюда. Ни одну из них нельзя вызывать без проверки CPUID —
 * этим занимается dispatch.c и только он.
 *
 * Результаты намеренно не совпадают со скалярным эталоном бит в бит: FMA не
 * округляет промежуточное произведение, а свёртка идёт по нескольким
 * накопителям. Оба отличия улучшают точность, но меняют её; для тех, кому
 * это важно, есть [~precise].
 */
#include "impl.h"

#include "smpc3/cpu.h"
#include "smpc3/plat.h"

#include <immintrin.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* ========================================================================== */
/*  Поэлементные                                                              */
/* ========================================================================== */

/* Хвост (n % 8) добирается скалярно. Маскированные загрузки тут не нужны:
 * на длинах, ради которых всё затевалось, хвост — это шум. */
#define SMP_UNARY_BODY(OPV, OPS)                                              \
    size_t i = 0;                                                             \
    for (; i + 32 <= n; i += 32) {                                            \
        __m256 x0 = _mm256_loadu_ps(src + i +  0);                            \
        __m256 x1 = _mm256_loadu_ps(src + i +  8);                            \
        __m256 x2 = _mm256_loadu_ps(src + i + 16);                            \
        __m256 x3 = _mm256_loadu_ps(src + i + 24);                            \
        _mm256_storeu_ps(dst + i +  0, OPV(x0));                              \
        _mm256_storeu_ps(dst + i +  8, OPV(x1));                              \
        _mm256_storeu_ps(dst + i + 16, OPV(x2));                              \
        _mm256_storeu_ps(dst + i + 24, OPV(x3));                              \
    }                                                                         \
    for (; i + 8 <= n; i += 8)                                                \
        _mm256_storeu_ps(dst + i, OPV(_mm256_loadu_ps(src + i)));             \
    for (; i < n; i++) { const float x = src[i]; dst[i] = (OPS); }

void smp_ka_relu(float *dst, const float *src, size_t n)
{
    const __m256 zero = _mm256_setzero_ps();
#define RELU_V(v) _mm256_max_ps((v), zero)
    SMP_UNARY_BODY(RELU_V, x > 0.0f ? x : 0.0f)
#undef RELU_V
}

void smp_ka_abs(float *dst, const float *src, size_t n)
{
    /* Модуль — это снятие знакового бита, а не сравнение. */
    const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
#define ABS_V(v) _mm256_and_ps((v), mask)
    SMP_UNARY_BODY(ABS_V, x < 0.0f ? -x : x)
#undef ABS_V
}

void smp_ka_scale(float *dst, const float *src, size_t n, float k)
{
    const __m256 kv = _mm256_set1_ps(k);
#define SCALE_V(v) _mm256_mul_ps((v), kv)
    SMP_UNARY_BODY(SCALE_V, x * k)
#undef SCALE_V
}

/* ========================================================================== */
/*  Слитая цепочка                                                            */
/* ========================================================================== */

/* Разбор стадии сидит внутри цикла, но он приходится на восемь элементов
 * разом и на фоне обращения к памяти не виден. Широковещания констант вынесены
 * наружу: пересобирать их на каждом векторе незачем. */
void smp_ka_fuse(float *dst, const float *src, size_t n,
                 const SmpFuseStep *st, uint32_t ns)
{
    const __m256 zero = _mm256_setzero_ps();
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    __m256 kv[SMP_FUSE_MAX];
    for (uint32_t s = 0; s < ns; s++)
        kv[s] = _mm256_set1_ps((float)st[s].k);

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(src + i);
        for (uint32_t s = 0; s < ns; s++) {
            switch (st[s].op) {
                case SMP_FOP_RELU:  x = _mm256_max_ps(x, zero); break;
                case SMP_FOP_ABS:   x = _mm256_and_ps(x, absm); break;
                case SMP_FOP_SCALE: x = _mm256_mul_ps(x, kv[s]); break;
                case SMP_FOP_ADD:
                    x = _mm256_add_ps(x, _mm256_loadu_ps((const float *)st[s].b.p + i));
                    break;
                case SMP_FOP_MUL:
                    x = _mm256_mul_ps(x, _mm256_loadu_ps((const float *)st[s].b.p + i));
                    break;
                default: break;
            }
        }
        _mm256_storeu_ps(dst + i, x);
    }

    /* Хвост (n % 8) — скалярно, теми же формулами. */
    for (; i < n; i++) {
        float x = src[i];
        for (uint32_t s = 0; s < ns; s++) {
            const float y = (st[s].op == SMP_FOP_ADD || st[s].op == SMP_FOP_MUL)
                                ? ((const float *)st[s].b.p)[i] : 0.0f;
            switch (st[s].op) {
                case SMP_FOP_RELU:  x = x > 0.0f ? x : 0.0f; break;
                case SMP_FOP_ABS:   x = x < 0.0f ? -x : x;   break;
                case SMP_FOP_SCALE: x = x * (float)st[s].k;  break;
                case SMP_FOP_ADD:   x = x + y;               break;
                case SMP_FOP_MUL:   x = x * y;               break;
                default: break;
            }
        }
        dst[i] = x;
    }
}

#define SMP_BINARY_BODY(OPV, OPS)                                             \
    size_t i = 0;                                                             \
    for (; i + 16 <= n; i += 16) {                                            \
        __m256 x0 = _mm256_loadu_ps(a + i + 0), y0 = _mm256_loadu_ps(b + i + 0); \
        __m256 x1 = _mm256_loadu_ps(a + i + 8), y1 = _mm256_loadu_ps(b + i + 8); \
        _mm256_storeu_ps(dst + i + 0, OPV(x0, y0));                           \
        _mm256_storeu_ps(dst + i + 8, OPV(x1, y1));                           \
    }                                                                         \
    for (; i + 8 <= n; i += 8)                                                \
        _mm256_storeu_ps(dst + i,                                             \
            OPV(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));             \
    for (; i < n; i++) { const float x = a[i], y = b[i]; dst[i] = (OPS); }

void smp_ka_add(float *dst, const float *a, const float *b, size_t n)
{
    SMP_BINARY_BODY(_mm256_add_ps, x + y)
}

void smp_ka_mul(float *dst, const float *a, const float *b, size_t n)
{
    SMP_BINARY_BODY(_mm256_mul_ps, x * y)
}

/* ========================================================================== */
/*  Свёртки                                                                   */
/* ========================================================================== */

static float hsum256(__m256 v)
{
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

static float hmax256(__m256 v)
{
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_max_ps(lo, hi);
    s = _mm_max_ps(s, _mm_movehl_ps(s, s));
    s = _mm_max_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

double smp_ka_reduce_add(const float *src, size_t n)
{
    /* Четыре независимых накопителя: у сложения f32 задержка ~4 такта, одна
     * цепочка зависимостей загрузила бы конвейер на четверть. Побочный
     * эффект — попарное суммирование, то есть меньшая ошибка накопления. */
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        a0 = _mm256_add_ps(a0, _mm256_loadu_ps(src + i +  0));
        a1 = _mm256_add_ps(a1, _mm256_loadu_ps(src + i +  8));
        a2 = _mm256_add_ps(a2, _mm256_loadu_ps(src + i + 16));
        a3 = _mm256_add_ps(a3, _mm256_loadu_ps(src + i + 24));
    }
    for (; i + 8 <= n; i += 8)
        a0 = _mm256_add_ps(a0, _mm256_loadu_ps(src + i));

    double acc = (double)hsum256(_mm256_add_ps(_mm256_add_ps(a0, a1),
                                               _mm256_add_ps(a2, a3)));
    for (; i < n; i++) acc += (double)src[i];
    return acc;
}

double smp_ka_reduce_max(const float *src, size_t n)
{
    if (n == 0) return -INFINITY;

    __m256 m0 = _mm256_set1_ps(-INFINITY), m1 = m0, m2 = m0, m3 = m0;

    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        m0 = _mm256_max_ps(m0, _mm256_loadu_ps(src + i +  0));
        m1 = _mm256_max_ps(m1, _mm256_loadu_ps(src + i +  8));
        m2 = _mm256_max_ps(m2, _mm256_loadu_ps(src + i + 16));
        m3 = _mm256_max_ps(m3, _mm256_loadu_ps(src + i + 24));
    }
    for (; i + 8 <= n; i += 8)
        m0 = _mm256_max_ps(m0, _mm256_loadu_ps(src + i));

    double best = (double)hmax256(_mm256_max_ps(_mm256_max_ps(m0, m1),
                                                _mm256_max_ps(m2, m3)));
    for (; i < n; i++) if ((double)src[i] > best) best = (double)src[i];
    return best;
}

/* Свёртка поверх цепочки. Накопителей четыре — ровно как в обычной свёртке: у
 * сложения f32 задержка около четырёх тактов, и одна цепочка зависимостей
 * загрузила бы конвейер на четверть. Структура накопления та же, значит и
 * точность та же: сверять слитый вариант с обычным можно с прежним допуском. */
double smp_ka_fuse_reduce(const float *src, size_t n,
                          const SmpFuseStep *st, uint32_t ns, uint8_t red)
{
    const __m256 zero = _mm256_setzero_ps();
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    __m256 kv[SMP_FUSE_MAX];
    for (uint32_t s = 0; s < ns; s++)
        kv[s] = _mm256_set1_ps((float)st[s].k);

/* Прогоняет вектор через всю цепочку. OFF — смещение для бинарных операндов. */
#define FUSE_APPLY(V, OFF)                                                     \
    for (uint32_t s = 0; s < ns; s++) {                                        \
        switch (st[s].op) {                                                    \
            case SMP_FOP_RELU:  V = _mm256_max_ps(V, zero); break;             \
            case SMP_FOP_ABS:   V = _mm256_and_ps(V, absm); break;             \
            case SMP_FOP_SCALE: V = _mm256_mul_ps(V, kv[s]); break;            \
            case SMP_FOP_ADD:                                                  \
                V = _mm256_add_ps(V,                                           \
                    _mm256_loadu_ps((const float *)st[s].b.p + (OFF)));        \
                break;                                                         \
            case SMP_FOP_MUL:                                                  \
                V = _mm256_mul_ps(V,                                           \
                    _mm256_loadu_ps((const float *)st[s].b.p + (OFF)));        \
                break;                                                         \
            default: break;                                                    \
        }                                                                      \
    }

#define FUSE_RED_BODY(COMBINE)                                                 \
    for (; i + 32 <= n; i += 32) {                                             \
        __m256 x0 = _mm256_loadu_ps(src + i +  0); FUSE_APPLY(x0, i +  0)      \
        __m256 x1 = _mm256_loadu_ps(src + i +  8); FUSE_APPLY(x1, i +  8)      \
        __m256 x2 = _mm256_loadu_ps(src + i + 16); FUSE_APPLY(x2, i + 16)      \
        __m256 x3 = _mm256_loadu_ps(src + i + 24); FUSE_APPLY(x3, i + 24)      \
        a0 = COMBINE(a0, x0); a1 = COMBINE(a1, x1);                            \
        a2 = COMBINE(a2, x2); a3 = COMBINE(a3, x3);                            \
    }                                                                          \
    for (; i + 8 <= n; i += 8) {                                               \
        __m256 x = _mm256_loadu_ps(src + i); FUSE_APPLY(x, i)                  \
        a0 = COMBINE(a0, x);                                                   \
    }

    size_t i = 0;
    double acc;

    if (red == SMP_FRED_MAX) {
        __m256 a0 = _mm256_set1_ps(-INFINITY), a1 = a0, a2 = a0, a3 = a0;
        FUSE_RED_BODY(_mm256_max_ps)
        acc = (n >= 8) ? (double)hmax256(_mm256_max_ps(_mm256_max_ps(a0, a1),
                                                       _mm256_max_ps(a2, a3)))
                       : -INFINITY;
    } else {
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        FUSE_RED_BODY(_mm256_add_ps)
        acc = (double)hsum256(_mm256_add_ps(_mm256_add_ps(a0, a1),
                                            _mm256_add_ps(a2, a3)));
    }

#undef FUSE_RED_BODY
#undef FUSE_APPLY

    /* Хвост (n % 8) — скалярно, теми же формулами. */
    for (; i < n; i++) {
        float x = src[i];
        for (uint32_t s = 0; s < ns; s++) {
            const float y = (st[s].op == SMP_FOP_ADD || st[s].op == SMP_FOP_MUL)
                                ? ((const float *)st[s].b.p)[i] : 0.0f;
            switch (st[s].op) {
                case SMP_FOP_RELU:  x = x > 0.0f ? x : 0.0f; break;
                case SMP_FOP_ABS:   x = x < 0.0f ? -x : x;   break;
                case SMP_FOP_SCALE: x = x * (float)st[s].k;  break;
                case SMP_FOP_ADD:   x = x + y;               break;
                case SMP_FOP_MUL:   x = x * y;               break;
                default: break;
            }
        }
        if (red == SMP_FRED_MAX) { if ((double)x > acc) acc = (double)x; }
        else                     { acc += (double)x; }
    }
    return acc;
}

/* ========================================================================== */
/*  GEMM                                                                      */
/* ========================================================================== */

/* Микроядро 6x16: шесть строк C по шестнадцать столбцов — это 12 накопителей
 * ymm, плюс два регистра под B и один под широковещание A. Итого 15 из 16
 * доступных: последний оставлен компилятору под адресную арифметику.
 *
 * Упаковываются ОБА операнда. Первая версия паковала только B, и на 1024x1024
 * пропускная способность падала со 104 до 20 ГФЛОПС: панель A перечитывалась
 * на каждом блоке столбцов и не удерживалась в L2. Блокировка по строкам (MC)
 * плюс упаковка A убирают этот обрыв. */
#define MR 6u
#define NR 16u

/* Значения по умолчанию выбраны развёрткой на Raptor Lake и подтверждены
 * замером: соседние точки либо в пределах шума, либо хуже. «Больше» здесь не
 * значит «лучше» — MC=384 роняет N=1024 со 117 до 85 ГФЛОПС, KC=512 роняет
 * N=512 со 119 до 88. Разброс между прогонами одной сборки при этом доходит
 * до 13%, так что подбирать эти числа тоньше попросту нечем. */
#define MC_DEF 96u    /* кратно MR */
#define KC_DEF 256u
#define NC_DEF 256u   /* кратно NR */

/* Ниже этого ужимать бессмысленно: панель перестаёт окупать упаковку. */
#define MC_MIN 24u
#define KC_MIN 64u
#define NC_MIN 64u

static uint32_t g_mc = MC_DEF, g_kc = KC_DEF, g_nc = NC_DEF;
static bool     g_blk_ready = false;

static size_t panels_bytes(uint32_t mc, uint32_t kc, uint32_t nc)
{
    return ((size_t)mc * kc + (size_t)kc * nc) * sizeof(float);
}

void smp_k_gemm_block_for(uint32_t l2_bytes,
                          uint32_t *mc, uint32_t *kc, uint32_t *nc)
{
    uint32_t m = MC_DEF, k = KC_DEF, n = NC_DEF;

    /* Четверть L2 оставлена под строку C, стек и всё, что ядру тоже нужно. */
    if (l2_bytes) {
        const size_t budget = (size_t)l2_bytes * 3u / 4u;

        /* KC входит в обе панели, поэтому ужимается первым — он сокращает их
         * разом. Затем NC: панель B крупнее. MC трогается последним. */
        while (panels_bytes(m, k, n) > budget && k > KC_MIN) k -= 8u;
        while (panels_bytes(m, k, n) > budget && n > NC_MIN) n -= NR;
        while (panels_bytes(m, k, n) > budget && m > MC_MIN) m -= MR;
    }

    if (mc) *mc = m;
    if (kc) *kc = k;
    if (nc) *nc = n;
}

/* Разрешается один раз. Точка входа гарантированно одна: GEMM не запускается
 * без рабочей памяти, а её размер спрашивают через smp_k_scratch_bytes — и в
 * пуле это происходит на подъёме, до появления потоков. */
/* "MC,KC,NC": три положительных числа через запятую. Через strtoul, а не
 * sscanf: форматного ввода от libc больше не требует никто, и ядру ОС, где
 * этот код тоже собирается, не придётся заводить его ради одной строки. */
static bool parse_block(const char *s, unsigned v[3])
{
    for (int i = 0; i < 3; i++) {
        char *end;
        const unsigned long x = strtoul(s, &end, 10);
        if (end == s || x == 0) return false;
        v[i] = (unsigned)x;
        if (i < 2 && *end != ',') return false;
        s = end + 1;
    }
    return true;
}

static void blk_resolve(void)
{
    if (g_blk_ready) return;

    /* Подмена руками — чтобы воспроизвести чужой замер, не имея того же
     * процессора. Ровно та же роль, что у SMPC3_VEC_BITS. */
    const char *env = smp_plat_env("SMPC3_GEMM_BLOCK");
    unsigned    v[3];
    if (env && parse_block(env, v)) {
        const unsigned em = v[0], ek = v[1], en = v[2];
        g_mc = (uint32_t)(em - em % MR); if (g_mc < MR) g_mc = MR;
        g_kc = (uint32_t)ek;
        g_nc = (uint32_t)(en - en % NR); if (g_nc < NR) g_nc = NR;
    } else {
        smp_k_gemm_block_for(smp_cpu()->l2_bytes, &g_mc, &g_kc, &g_nc);
    }
    g_blk_ready = true;
}

void smp_k_gemm_block(uint32_t *mc, uint32_t *kc, uint32_t *nc)
{
    blk_resolve();
    if (mc) *mc = g_mc;
    if (kc) *kc = g_kc;
    if (nc) *nc = g_nc;
}

/* Размеры панелей известны на этапе компиляции, а сама память приходит
 * снаружи — по одному комплекту на инстанс VM.
 *   B: KC*NC*4 = 256 КиБ
 *   A: MC*KC*4 =  96 КиБ
 * Вместе укладываются в L2 современного ядра. */
/* Комплект выровнен на 64, чтобы B следующего комплекта тоже был. */
static size_t set_bytes(void)
{
    return SMP_ALIGN_UP(panels_bytes(g_mc, g_kc, g_nc), SMP_CACHELINE);
}

size_t smp_k_scratch_bytes_for(uint32_t sets)
{
    blk_resolve();
    return set_bytes() * (sets ? sets : 1u) + SMP_CACHELINE;
}

size_t smp_k_scratch_bytes(void)
{
    return smp_k_scratch_bytes_for(1);
}

void smp_k_scratch_bind(SmpKScratch *s, void *mem, size_t bytes)
{
    if (!s) return;
    if (!mem || bytes < smp_k_scratch_bytes()) {
        s->apack = NULL; s->bpack = NULL; s->bytes = 0;
        s->sets = 0;     s->set_floats = 0;
        return;
    }
    /* Обе панели выровнены на 64: микроядро читает B выровненными загрузками. */
    uint8_t *p = (uint8_t *)mem;
    p = (uint8_t *)SMP_ALIGN_UP((uintptr_t)p, SMP_CACHELINE);
    s->apack = (float *)p;
    s->bpack = (float *)(p + (size_t)g_mc * g_kc * sizeof(float));
    s->bytes = bytes;
    s->sets       = (uint32_t)((bytes - SMP_CACHELINE) / set_bytes());
    s->set_floats = set_bytes() / sizeof(float);
}

/* B[kc][nc] -> последовательность панелей шириной NR; внутри панели строки
 * идут подряд. Хвост по столбцам дополняется нулями. */
static void pack_b(float *dst, const float *B, size_t ldb, size_t kc, size_t nc)
{
    for (size_t j0 = 0; j0 < nc; j0 += NR) {
        const size_t w = (nc - j0 < NR) ? (nc - j0) : NR;
        for (size_t p = 0; p < kc; p++) {
            const float *s = B + p * ldb + j0;
            float       *d = dst + p * NR;
            for (size_t j = 0; j < w; j++)  d[j] = s[j];
            for (size_t j = w; j < NR; j++) d[j] = 0.0f;
        }
        dst += kc * NR;
    }
}

/* A[mc][kc] -> панели по MR строк: панель ir, шаг p -> MR подряд идущих
 * элементов. Строки за пределами mc обнуляются — именно поэтому микроядру не
 * нужен отдельный узкий путь для хвоста по строкам: лишние строки дают нули,
 * а наружу они просто не копируются. */
static void pack_a(float *dst, const float *A, size_t lda, size_t mc, size_t kc)
{
    for (size_t i0 = 0; i0 < mc; i0 += MR) {
        const size_t h = (mc - i0 < MR) ? (mc - i0) : MR;
        for (size_t p = 0; p < kc; p++) {
            float *d = dst + p * MR;
            for (size_t r = 0; r < h; r++)  d[r] = A[(i0 + r) * lda + p];
            for (size_t r = h; r < MR; r++) d[r] = 0.0f;
        }
        dst += kc * MR;
    }
}

/* ctile[MR][NR] = Ap[kc][MR] x Bp[kc][NR]. */
static void micro_6x16(size_t kc, const float *Ap, const float *Bp, float *ctile)
{
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();

    for (size_t p = 0; p < kc; p++) {
        const __m256 b0 = _mm256_load_ps(Bp);
        const __m256 b1 = _mm256_load_ps(Bp + 8);
        Bp += NR;

        __m256 av = _mm256_broadcast_ss(Ap + 0);
        c00 = _mm256_fmadd_ps(av, b0, c00);
        c01 = _mm256_fmadd_ps(av, b1, c01);

        av  = _mm256_broadcast_ss(Ap + 1);
        c10 = _mm256_fmadd_ps(av, b0, c10);
        c11 = _mm256_fmadd_ps(av, b1, c11);

        av  = _mm256_broadcast_ss(Ap + 2);
        c20 = _mm256_fmadd_ps(av, b0, c20);
        c21 = _mm256_fmadd_ps(av, b1, c21);

        av  = _mm256_broadcast_ss(Ap + 3);
        c30 = _mm256_fmadd_ps(av, b0, c30);
        c31 = _mm256_fmadd_ps(av, b1, c31);

        av  = _mm256_broadcast_ss(Ap + 4);
        c40 = _mm256_fmadd_ps(av, b0, c40);
        c41 = _mm256_fmadd_ps(av, b1, c41);

        av  = _mm256_broadcast_ss(Ap + 5);
        c50 = _mm256_fmadd_ps(av, b0, c50);
        c51 = _mm256_fmadd_ps(av, b1, c51);

        Ap += MR;
    }

    _mm256_store_ps(ctile +  0, c00); _mm256_store_ps(ctile +  8, c01);
    _mm256_store_ps(ctile + 16, c10); _mm256_store_ps(ctile + 24, c11);
    _mm256_store_ps(ctile + 32, c20); _mm256_store_ps(ctile + 40, c21);
    _mm256_store_ps(ctile + 48, c30); _mm256_store_ps(ctile + 56, c31);
    _mm256_store_ps(ctile + 64, c40); _mm256_store_ps(ctile + 72, c41);
    _mm256_store_ps(ctile + 80, c50); _mm256_store_ps(ctile + 88, c51);
}

/* ========================================================================== */
/*  Эпилог                                                                    */
/* ========================================================================== */

/* Цепочка по строке панели C.
 *
 * Стадии идут отдельными проходами по строке, а не все сразу на каждый
 * элемент: они поэлементные и независимые, так что порядок «стадия за
 * стадией» даёт то же самое. Зато разбор стадии перестаёт приходиться на
 * каждые восемь чисел — в тайле шириной NR=16 это стоило дороже самой
 * арифметики, и слияние съедало собственный выигрыш.
 *
 * Лишние проходы здесь ничего не стоят: строка не длиннее NC и только что
 * посчитана, то есть лежит в L1. Тем же и объясняется, почему стадии зовутся
 * готовыми ядрами вместо своей копии формул — второй реализации relu в файле
 * быть не должно.
 *
 * base — линейный индекс начала строки в плотном [M,N]: так адресуются
 * операнды @add и @mul, их плотность проверил диспетчер. */
static void ep_row(float *row, size_t n, const SmpFuseStep *st, uint32_t ns,
                   size_t base)
{
    for (uint32_t s = 0; s < ns; s++) {
        const float *b = (const float *)st[s].b.p + base;
        switch (st[s].op) {
            case SMP_FOP_RELU:  smp_ka_relu (row, row, n);                  break;
            case SMP_FOP_ABS:   smp_ka_abs  (row, row, n);                  break;
            case SMP_FOP_SCALE: smp_ka_scale(row, row, n, (float)st[s].k);  break;
            case SMP_FOP_ADD:   smp_ka_add  (row, row, b, n);               break;
            case SMP_FOP_MUL:   smp_ka_mul  (row, row, b, n);               break;
            default: break;
        }
    }
}

void smp_ka_gemm_ep(float *C, size_t ldc, const float *A, size_t lda,
                    const float *B, size_t ldb, size_t M, size_t N, size_t K,
                    float *apack, float *bpack,
                    const SmpFuseStep *steps, uint32_t nsteps,
                    size_t ep_base, size_t ep_ld)
{
    for (size_t i = 0; i < M; i++)
        memset(C + i * ldc, 0, N * sizeof(float));

    float SMP_ALIGNED(64) ctile[MR * NR];

    const size_t BMC = g_mc, BKC = g_kc, BNC = g_nc;

    for (size_t j0 = 0; j0 < N; j0 += BNC) {
        const size_t nc = (N - j0 < BNC) ? (N - j0) : BNC;

        for (size_t k0 = 0; k0 < K; k0 += BKC) {
            const size_t kc = (K - k0 < BKC) ? (K - k0) : BKC;

            /* Эпилог ложится ровно один раз — на последнем k-блоке. До него в
             * C лежит незаконченная сумма, и @relu над ней дал бы не «relu от
             * произведения», а срез промежуточного шага. */
            const bool ep = (nsteps != 0u) && (k0 + kc == K);

            pack_b(bpack, B + k0 * ldb + j0, ldb, kc, nc);

            for (size_t i0 = 0; i0 < M; i0 += BMC) {
                const size_t mc = (M - i0 < BMC) ? (M - i0) : BMC;

                pack_a(apack, A + i0 * lda + k0, lda, mc, kc);

                for (size_t ir = 0; ir < mc; ir += MR) {
                    const size_t mr = (mc - ir < MR) ? (mc - ir) : MR;
                    const float *Ap = apack + (ir / MR) * kc * MR;

                    for (size_t jr = 0; jr < nc; jr += NR) {
                        const size_t nr = (nc - jr < NR) ? (nc - jr) : NR;
                        const float *Bp = bpack + (jr / NR) * kc * NR;

                        micro_6x16(kc, Ap, Bp, ctile);

                        for (size_t r = 0; r < mr; r++) {
                            float *c = C + (i0 + ir + r) * ldc + j0 + jr;
                            for (size_t j = 0; j < nr; j++) c[j] += ctile[r * NR + j];
                        }
                    }

                    /* Строки панели готовы целиком — эпилог берёт их одним
                     * куском по nc, а не кусками по NR: так проход длиннее
                     * разбора стадии, а не короче. */
                    if (ep)
                        for (size_t r = 0; r < mr; r++)
                            ep_row(C + (i0 + ir + r) * ldc + j0, nc,
                                   steps, nsteps, ep_base + (i0 + ir + r) * ep_ld + j0);
                }
            }
        }
    }
}

void smp_ka_gemm(float *C, size_t ldc, const float *A, size_t lda,
                 const float *B, size_t ldb, size_t M, size_t N, size_t K,
                 float *apack, float *bpack)
{
    smp_ka_gemm_ep(C, ldc, A, lda, B, ldb, M, N, K, apack, bpack, NULL, 0u, 0u, N);
}
