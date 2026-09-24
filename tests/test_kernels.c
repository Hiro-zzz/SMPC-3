/* SMPC3 :: test_kernels.c -- проверки Ф6: векторные ядра против эталона.
 *
 * Смысл всего файла: быстрая ветка обязана давать ТОТ ЖЕ результат, что и
 * скалярная. Бит в бит совпадения не требуется — FMA не округляет
 * промежуточное произведение, а свёртка идёт по нескольким накопителям, —
 * поэтому сверка идёт по относительной погрешности.
 */
#include "smpc3/kernels.h"
#include "smpc3/cpu.h"
#include "smpc3/diag.h"

#include "harness.h"

#include <string.h>
#include <stdlib.h>
#include <math.h>

/* ========================================================================== */
/*  Данные                                                                    */
/* ========================================================================== */

#define MAXN (600u * 600u)

static float SMP_ALIGNED(64) g_a[MAXN];
static float SMP_ALIGNED(64) g_b[MAXN];
static float SMP_ALIGNED(64) g_ref[MAXN];
static float SMP_ALIGNED(64) g_got[MAXN];
static float SMP_ALIGNED(64) g_e[MAXN];   /* операнд @add/@mul в эпилоге */

/* Детерминированный генератор: тест обязан падать одинаково при каждом
 * запуске, иначе отлаживать его невозможно. */
static uint64_t g_rng = 0x243F6A8885A308D3ull;

/* Рабочая память ядер: с тех пор как она перестала быть глобальной, её обязан
 * предоставить вызывающий — иначе GEMM честно уходит в скалярный эталон. */
static SmpKScratch g_sc;
static void *g_scmem;

static float frand(void)
{
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    /* Диапазон [-2, 2): нужны и отрицательные, иначе relu не проверяется. */
    return (float)((double)(g_rng >> 40) / 2097152.0 - 2.0);
}

static void fill_rand(float *p, size_t n)
{
    for (size_t i = 0; i < n; i++) p[i] = frand();
}

static SmpTensor mk(uint32_t r0, uint32_t r1)
{
    SmpTensor t;
    memset(&t, 0, sizeof t);
    t.dtype = SMP_DT_F32;
    if (r1) {
        t.rank = 2; t.shape[0] = r0; t.shape[1] = r1;
        t.stride[0] = r1; t.stride[1] = 1;
        t.nelem = r0 * r1;
    } else {
        t.rank = 1; t.shape[0] = r0; t.stride[0] = 1;
        t.nelem = r0;
    }
    t.flags = SMP_TF_CONTIG;
    return t;
}

/* Относительная погрешность между эталоном и быстрой веткой. */
static double max_rel_err(const float *ref, const float *got, size_t n)
{
    double worst = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double r = ref[i], g = got[i];
        if (isnan(r) != isnan(g)) return INFINITY;
        const double scale = fabs(r) > 1.0 ? fabs(r) : 1.0;
        const double e = fabs(r - g) / scale;
        if (e > worst) worst = e;
    }
    return worst;
}

/* ========================================================================== */

static void test_available(void)
{
    SECTION("доступность ветки");

    const SmpCpu *c = smp_cpu();
    const bool hw = (c->isa & SMP_ISA_AVX2) && (c->isa & SMP_ISA_FMA);

    CHECK(smp_kernels_select(SMP_KB_SCALAR), "скалярная ветка обязана быть всегда");
    CHECK(strcmp(smp_kernels_name(), "scalar") == 0, "имя: %s", smp_kernels_name());

    CHECK(smp_kernels_select(SMP_KB_AVX2) == hw,
          "выбор avx2 вернул %d, а CPUID говорит %d",
          (int)smp_kernels_select(SMP_KB_AVX2), (int)hw);

    if (!hw) {
        fprintf(stderr, "  (на этом процессоре нет AVX2+FMA — сверять не с чем)\n");
        return;
    }
    CHECK(strcmp(smp_kernels_name(), "avx2+fma") == 0, "имя: %s", smp_kernels_name());

    smp_kernels_select(SMP_KB_AUTO);
    CHECK(strcmp(smp_kernels_name(), "avx2+fma") == 0,
          "автовыбор не поднял векторную ветку");
}

/* ========================================================================== */

typedef void (*UnaryOp)(const SmpBuf *, const SmpBuf *);

static void cmp_unary(UnaryOp op, const char *name, size_t n)
{
    SmpTensor t = mk((uint32_t)n, 0);
    SmpBuf src = { g_a, &t }, ref = { g_ref, &t }, got = { g_got, &t };

    smp_kernels_select(SMP_KB_SCALAR);
    op(&ref, &src);
    smp_kernels_select(SMP_KB_AVX2);
    op(&got, &src);

    const double e = max_rel_err(g_ref, g_got, n);
    CHECK(e == 0.0, "%s(n=%zu): расхождение %g (поэлементные обязаны совпадать точно)",
          name, n, e);
}

static void test_elementwise(void)
{
    SECTION("поэлементные");

    if (!smp_kernels_select(SMP_KB_AVX2)) return;

    fill_rand(g_a, MAXN > 4096 ? 4096 : MAXN);
    fill_rand(g_b, 4096);

    /* Длины подобраны так, чтобы задеть все хвосты: кратные 32, 8 и никаким. */
    static const size_t lens[] = { 0, 1, 7, 8, 9, 31, 32, 33, 255, 256, 1000, 4096 };

    for (size_t i = 0; i < SMP_ARRLEN(lens); i++) {
        const size_t n = lens[i];
        if (n == 0) continue;
        cmp_unary(smp_k_relu, "relu", n);
        cmp_unary(smp_k_abs,  "abs",  n);

        /* scale и бинарные — отдельно, у них своя сигнатура. */
        SmpTensor t = mk((uint32_t)n, 0);
        SmpBuf sa = { g_a, &t }, sb = { g_b, &t };
        SmpBuf ref = { g_ref, &t }, got = { g_got, &t };

        smp_kernels_select(SMP_KB_SCALAR); smp_k_scale(&ref, &sa, 0.375);
        smp_kernels_select(SMP_KB_AVX2);   smp_k_scale(&got, &sa, 0.375);
        CHECK(max_rel_err(g_ref, g_got, n) == 0.0, "scale(n=%zu)", n);

        smp_kernels_select(SMP_KB_SCALAR); smp_k_add(&ref, &sa, &sb);
        smp_kernels_select(SMP_KB_AVX2);   smp_k_add(&got, &sa, &sb);
        CHECK(max_rel_err(g_ref, g_got, n) == 0.0, "add(n=%zu)", n);

        smp_kernels_select(SMP_KB_SCALAR); smp_k_mul(&ref, &sa, &sb);
        smp_kernels_select(SMP_KB_AVX2);   smp_k_mul(&got, &sa, &sb);
        CHECK(max_rel_err(g_ref, g_got, n) == 0.0, "mul(n=%zu)", n);
    }
}

/* ========================================================================== */

static void test_reduce(void)
{
    SECTION("свёртки");

    if (!smp_kernels_select(SMP_KB_AVX2)) return;

    static const size_t lens[] = { 1, 7, 8, 33, 255, 256, 1000, 100000 };

    for (size_t i = 0; i < SMP_ARRLEN(lens); i++) {
        const size_t n = lens[i];
        fill_rand(g_a, n);

        SmpTensor t = mk((uint32_t)n, 0);
        SmpBuf s = { g_a, &t };

        smp_kernels_select(SMP_KB_SCALAR);
        const double r_add = smp_k_reduce_add(&s);
        const double r_max = smp_k_reduce_max(&s);

        smp_kernels_select(SMP_KB_AVX2);
        const double v_add = smp_k_reduce_add(&s);
        const double v_max = smp_k_reduce_max(&s);

        /* Максимум обязан совпасть точно: порядок сравнений на результат не
         * влияет. */
        CHECK(r_max == v_max, "reduce.max(n=%zu): %g против %g", n, r_max, v_max);

        /* Сумма — с допуском: порядок сложения разный. */
        const double scale = fabs(r_add) > 1.0 ? fabs(r_add) : 1.0;
        const double err   = fabs(r_add - v_add) / scale;
        CHECK(err < 1e-6, "reduce.add(n=%zu): %.9g против %.9g, отн. ошибка %g",
              n, r_add, v_add, err);
    }
}

/* ========================================================================== */

static void gemm_case(uint32_t M, uint32_t N, uint32_t K)
{
    if ((size_t)M * K > MAXN || (size_t)K * N > MAXN || (size_t)M * N > MAXN) return;

    fill_rand(g_a, (size_t)M * K);
    fill_rand(g_b, (size_t)K * N);

    SmpTensor ta = mk(M, K), tb = mk(K, N), tc = mk(M, N);
    SmpBuf a = { g_a, &ta }, b = { g_b, &tb };
    SmpBuf ref = { g_ref, &tc }, got = { g_got, &tc };

    memset(g_ref, 0xCD, (size_t)M * N * sizeof(float));
    memset(g_got, 0xCD, (size_t)M * N * sizeof(float));

    smp_kernels_select(SMP_KB_SCALAR); smp_k_gemm(&ref, &a, &b, &g_sc);
    smp_kernels_select(SMP_KB_AVX2);   smp_k_gemm(&got, &a, &b, &g_sc);

    const double e = max_rel_err(g_ref, g_got, (size_t)M * N);
    CHECK(e < 1e-4, "gemm %ux%ux%u: отн. ошибка %g", M, N, K, e);
}

/* Эпилог за GEMM: C = chain(A x B). Эталон собирается порознь — скалярный
 * GEMM плюс отдельный проход цепочки, ровно то, что делал неслитый путь.
 * Слияние меняет число проходов по памяти, а не арифметику, поэтому
 * расхождение здесь означало бы ошибку, а не иной порядок сложения. */
static void gemm_ep_case(uint32_t M, uint32_t N, uint32_t K,
                         const uint8_t *ops, uint32_t ns, const char *what)
{
    if ((size_t)M * K > MAXN || (size_t)K * N > MAXN || (size_t)M * N > MAXN) return;

    fill_rand(g_a, (size_t)M * K);
    fill_rand(g_b, (size_t)K * N);
    fill_rand(g_e, (size_t)M * N);

    SmpTensor ta = mk(M, K), tb = mk(K, N), tc = mk(M, N);
    SmpBuf a = { g_a, &ta }, b = { g_b, &tb };
    SmpBuf ref = { g_ref, &tc }, got = { g_got, &tc };
    const SmpBuf eb = { g_e, &tc };

    SmpFuseStep st[SMP_FUSE_MAX];
    memset(st, 0, sizeof st);
    for (uint32_t i = 0; i < ns; i++) {
        st[i].op = ops[i];
        st[i].k  = 0.5 + (double)i;
        st[i].b  = eb;
    }

    memset(g_ref, 0xCD, (size_t)M * N * sizeof(float));
    memset(g_got, 0xCD, (size_t)M * N * sizeof(float));

    smp_kernels_select(SMP_KB_SCALAR);
    smp_k_gemm(&ref, &a, &b, &g_sc);
    if (ns) smp_k_fuse(&ref, &ref, st, ns);

    smp_kernels_select(SMP_KB_AVX2);
    smp_k_gemm_ep(&got, &a, &b, &g_sc, st, ns);

    const double e = max_rel_err(g_ref, g_got, (size_t)M * N);
    CHECK(e < 1e-4, "эпилог %s на %ux%ux%u: отн. ошибка %g", what, M, N, K, e);
}

static void test_gemm_epilogue(void)
{
    SECTION("эпилог GEMM");

    if (!smp_kernels_select(SMP_KB_AVX2)) return;

    static const uint8_t relu[]    = { SMP_FOP_RELU };
    static const uint8_t relu_add[] = { SMP_FOP_RELU, SMP_FOP_ADD };
    static const uint8_t scale_abs[] = { SMP_FOP_SCALE, SMP_FOP_ABS };
    static const uint8_t triple[] = { SMP_FOP_MUL, SMP_FOP_RELU, SMP_FOP_SCALE };

    /* Те же хвосты, что у обычного GEMM: эпилог живёт в выгрузке тайла, и
     * ошибиться он может ровно там же, где ошибается она. */
    gemm_ep_case(6, 16, 8,  relu, 1, "relu");
    gemm_ep_case(1, 1, 1,   relu, 1, "relu");
    gemm_ep_case(7, 17, 13, relu, 1, "relu");
    gemm_ep_case(5, 16, 8,  relu_add, 2, "relu+add");
    gemm_ep_case(6, 15, 8,  relu_add, 2, "relu+add");
    gemm_ep_case(6, 17, 8,  relu_add, 2, "relu+add");
    gemm_ep_case(13, 100, 7, scale_abs, 2, "scale+abs");
    gemm_ep_case(23, 41, 37, triple, 3, "mul+relu+scale");

    /* Больше одного k-блока. Эпилог обязан лечь ровно один раз: применённый на
     * каждом, он дал бы scale в кубе — и это видно, в отличие от повторного
     * relu, который прошёл бы незамеченным. */
    gemm_ep_case(70, 260, 520, scale_abs, 2, "scale+abs, k-блоков много");
    gemm_ep_case(40, 40, 600,  triple, 3, "mul+relu+scale, k-блоков много");
    gemm_ep_case(64, 300, 300, relu_add, 2, "relu+add, блоков много");

    /* Пустая цепочка — это просто GEMM, и вести себя обязана как он. */
    gemm_ep_case(12, 32, 16, NULL, 0, "пустая цепочка");
}

/* ========================================================================== */

static void test_gemm(void)
{
    SECTION("GEMM");

    if (!smp_kernels_select(SMP_KB_AVX2)) return;

    /* Ровно по микроядру. */
    gemm_case(6, 16, 8);
    gemm_case(12, 32, 16);

    /* Хвосты по строкам: M не кратно MR=6. */
    gemm_case(1, 16, 8);
    gemm_case(5, 16, 8);
    gemm_case(7, 16, 8);
    gemm_case(13, 16, 8);

    /* Хвосты по столбцам: N не кратно NR=16. */
    gemm_case(6, 1, 8);
    gemm_case(6, 15, 8);
    gemm_case(6, 17, 8);
    gemm_case(6, 100, 8);

    /* Хвосты по K. */
    gemm_case(6, 16, 1);
    gemm_case(6, 16, 7);
    gemm_case(6, 16, 257);

    /* Оба измерения кривые сразу. */
    gemm_case(7, 17, 13);
    gemm_case(23, 41, 37);
    gemm_case(1, 1, 1);

    /* Больше блоков KC=256 и NC=256 — работает ли сама блокировка. */
    gemm_case(64, 300, 300);
    gemm_case(70, 260, 520);

    /* Прямоугольные и «плоские». */
    gemm_case(200, 3, 200);
    gemm_case(3, 200, 200);
    gemm_case(128, 128, 128);
}

/* ========================================================================== */

static void test_fallback(void)
{
    SECTION("отказ от векторной ветки");

    if (!smp_kernels_select(SMP_KB_AVX2)) return;

    /* Разреженный вид: векторная ветка обязана уступить эталону, а результат
     * остаться правильным. */
    enum { R = 8, C = 8 };
    fill_rand(g_a, R * C);

    SmpTensor col;
    memset(&col, 0, sizeof col);
    col.dtype = SMP_DT_F32; col.rank = 1;
    col.shape[0] = R; col.stride[0] = C;   /* столбец: шаг C элементов */
    col.nelem = R;

    SmpBuf s = { g_a, &col };

    smp_kernels_select(SMP_KB_SCALAR);
    const double r = smp_k_reduce_add(&s);
    smp_kernels_select(SMP_KB_AVX2);
    const double v = smp_k_reduce_add(&s);

    CHECK(r == v, "разреженная свёртка: %.9g против %.9g", r, v);

    /* Проверим вручную: сумма нулевого столбца. */
    double manual = 0.0;
    for (int i = 0; i < R; i++) manual += g_a[i * C];
    CHECK(fabs(manual - v) < 1e-6, "ручная сумма %.9g против %.9g", manual, v);

    /* f64 через векторную ветку не идёт, но обязан считаться верно. */
    {
        static double d[64];
        for (int i = 0; i < 64; i++) d[i] = (double)i;
        SmpTensor t;
        memset(&t, 0, sizeof t);
        t.dtype = SMP_DT_F64; t.rank = 1; t.shape[0] = 64; t.stride[0] = 1; t.nelem = 64;
        SmpBuf b = { d, &t };
        smp_kernels_select(SMP_KB_AVX2);
        CHECK(fabs(smp_k_reduce_add(&b) - 2016.0) < 1e-9,
              "f64-свёртка: %g", smp_k_reduce_add(&b));
    }

    /* GEMM с транспонированным (не row-major) операндом — тоже эталон. */
    {
        enum { S = 8 };
        fill_rand(g_a, S * S);
        fill_rand(g_b, S * S);

        SmpTensor ta = mk(S, S), tc = mk(S, S);
        SmpTensor tbT;                    /* B как транспонированный вид */
        memset(&tbT, 0, sizeof tbT);
        tbT.dtype = SMP_DT_F32; tbT.rank = 2;
        tbT.shape[0] = S; tbT.shape[1] = S;
        tbT.stride[0] = 1; tbT.stride[1] = S;   /* шаг по последней оси != 1 */
        tbT.nelem = S * S;

        SmpBuf a = { g_a, &ta }, b = { g_b, &tbT };
        SmpBuf ref = { g_ref, &tc }, got = { g_got, &tc };

        smp_kernels_select(SMP_KB_SCALAR); smp_k_gemm(&ref, &a, &b, &g_sc);
        smp_kernels_select(SMP_KB_AVX2);   smp_k_gemm(&got, &a, &b, &g_sc);
        CHECK(max_rel_err(g_ref, g_got, S * S) == 0.0,
              "gemm с транспонированным B ушёл в векторную ветку");
    }
}

/* ========================================================================== */

static void test_identity(void)
{
    SECTION("проверяемая вручную арифметика");

    if (!smp_kernels_select(SMP_KB_AVX2)) return;

    /* A x I = A для размеров, задевающих и микроядро, и хвосты. */
    static const uint32_t sizes[] = { 6, 7, 16, 17, 64, 100 };

    for (size_t s = 0; s < SMP_ARRLEN(sizes); s++) {
        const uint32_t n = sizes[s];
        fill_rand(g_a, (size_t)n * n);

        memset(g_b, 0, (size_t)n * n * sizeof(float));
        for (uint32_t i = 0; i < n; i++) g_b[i * n + i] = 1.0f;

        SmpTensor t = mk(n, n);
        SmpBuf a = { g_a, &t }, b = { g_b, &t }, c = { g_got, &t };
        smp_k_gemm(&c, &a, &b, &g_sc);

        bool same = true;
        for (size_t i = 0; i < (size_t)n * n; i++)
            if (g_got[i] != g_a[i]) same = false;
        CHECK(same, "A x I != A при n=%u", n);
    }

    /* Матрица из единиц в квадрате даёт n во всех элементах. */
    {
        const uint32_t n = 40;
        for (size_t i = 0; i < (size_t)n * n; i++) g_a[i] = 1.0f;
        SmpTensor t = mk(n, n);
        SmpBuf a = { g_a, &t }, c = { g_got, &t };
        smp_k_gemm(&c, &a, &a, &g_sc);
        bool ok = true;
        for (size_t i = 0; i < (size_t)n * n; i++) if (g_got[i] != (float)n) ok = false;
        CHECK(ok, "матрица единиц в квадрате: got[0]=%g, ждали %u", g_got[0], n);
    }
}


/* ========================================================================== */
/*  Блокировка GEMM под размер L2                                             */
/* ========================================================================== */

static size_t blk_panels(uint32_t mc, uint32_t kc, uint32_t nc)
{
    return ((size_t)mc * kc + (size_t)kc * nc) * sizeof(float);
}

static void test_gemm_block(void)
{
    SECTION("блокировка под L2");

    uint32_t mc = 0, kc = 0, nc = 0;

    /* Размер неизвестен — берутся значения по умолчанию, а не что попало. */
    smp_k_gemm_block_for(0, &mc, &kc, &nc);
    CHECK(mc == 96u && kc == 256u && nc == 256u,
          "без данных о L2 вышло MC=%u KC=%u NC=%u", mc, kc, nc);

    /* Там, где умолчания и так помещаются, их трогать незачем: на этой машине
     * поведение обязано остаться прежним до такта. */
    const uint32_t roomy[] = { 512u * 1024u, 1280u * 1024u, 2048u * 1024u };
    for (unsigned i = 0; i < 3; i++) {
        smp_k_gemm_block_for(roomy[i], &mc, &kc, &nc);
        CHECK(mc == 96u && kc == 256u && nc == 256u,
              "L2=%u КиБ изменил блокировку на MC=%u KC=%u NC=%u",
              roomy[i] / 1024u, mc, kc, nc);
    }

    /* А вот на 256 КиБ умолчания не влезают — 352 КиБ панелей против бюджета
     * в 192. Ради этого случая всё и делалось: иначе панели вытесняют друг
     * друга, и упаковка теряет смысл. */
    smp_k_gemm_block_for(256u * 1024u, &mc, &kc, &nc);
    CHECK(blk_panels(mc, kc, nc) <= (size_t)(256u * 1024u) * 3u / 4u,
          "на L2=256 КиБ панели остались %zu Б", blk_panels(mc, kc, nc));
    CHECK(kc < 256u, "KC не ужался: %u", kc);

    /* Инварианты микроядра: MC кратно MR=6, NC кратно NR=16, и ничего не
     * схлопнулось в ноль. */
    const uint32_t sizes[] = { 0u, 128u * 1024u, 256u * 1024u, 384u * 1024u,
                               512u * 1024u, 1024u * 1024u, 8192u * 1024u };
    size_t prev = 0;
    for (unsigned i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        smp_k_gemm_block_for(sizes[i], &mc, &kc, &nc);
        CHECK(mc > 0 && kc > 0 && nc > 0,
              "L2=%u дал нулевой блок", sizes[i]);
        CHECK(mc % 6u == 0u, "MC=%u не кратно MR", mc);
        CHECK(nc % 16u == 0u, "NC=%u не кратно NR", nc);

        /* Больше кэш — не меньше панели. Ноль стоит первым и в счёт не идёт. */
        if (i > 1) {
            const size_t cur = blk_panels(mc, kc, nc);
            CHECK(cur >= prev, "L2=%u КиБ дал панели меньше предыдущего шага",
                  sizes[i] / 1024u);
            prev = cur;
        } else if (i == 1) {
            prev = blk_panels(mc, kc, nc);
        }
    }

    /* Рабочая память обязана соответствовать выбранной блокировке: иначе
     * упаковка вылезет за буфер. */
    smp_k_gemm_block(&mc, &kc, &nc);
    CHECK(smp_k_scratch_bytes() >= blk_panels(mc, kc, nc),
          "рабочей памяти меньше панелей: %zu против %zu",
          smp_k_scratch_bytes(), blk_panels(mc, kc, nc));
}

/* Параллельный GEMM обязан давать ТО ЖЕ самое, бит в бит: каждая плитка
 * считается тем же циклом с тем же разбиением по K, а значит, и с тем же
 * порядком сложения. Сравнение поэтому точное, а не с допуском. */
static bool par_case(SmpKScratch *par, uint32_t M, uint32_t N, uint32_t K,
                     const SmpFuseStep *st, uint32_t ns)
{
    SmpTensor ta = mk(M, K), tb = mk(K, N), tc = mk(M, N);
    SmpBuf a = { g_a, &ta }, b = { g_b, &tb };
    SmpBuf ref = { g_ref, &tc }, got = { g_got, &tc };

    memset(g_ref, 0xCD, (size_t)M * N * sizeof(float));
    memset(g_got, 0xAB, (size_t)M * N * sizeof(float));
    smp_k_gemm_ep(&ref, &a, &b, &g_sc, st, ns);      /* один комплект: один поток */
    smp_k_gemm_ep(&got, &a, &b, par, st, ns);
    return memcmp(g_ref, g_got, (size_t)M * N * sizeof(float)) == 0;
}

static void test_gemm_parallel(void)
{
    SECTION("параллельный GEMM");

    const size_t bytes = smp_k_scratch_bytes_for(4);
    void *mem = malloc(bytes);
    if (!mem) { CHECK(0, "нет памяти под четыре комплекта"); return; }
    SmpKScratch par;
    smp_k_scratch_bind(&par, mem, bytes);
    CHECK(par.sets == 4, "комплектов %u, ждали 4", par.sets);

    smp_kernels_select(SMP_KB_AVX2);
    smp_k_gemm_par_min(0);          /* делить всё, даже мелкое */

    /* M=1 — вектор на матрицу: делятся одни столбцы. Остальные — хвосты по
     * M, N и K мимо MR, NR, MC и KC. */
    static const uint32_t shapes[][3] = {
        { 1, 600, 300 }, { 1, 17, 5 }, { 7, 33, 500 }, { 100, 257, 64 },
        { 293, 47, 129 }, { 300, 300, 300 }, { 96, 256, 256 },
    };
    for (size_t i = 0; i < SMP_ARRLEN(shapes); i++) {
        const uint32_t M = shapes[i][0], N = shapes[i][1], K = shapes[i][2];
        fill_rand(g_a, (size_t)M * K);
        fill_rand(g_b, (size_t)K * N);
        CHECK(par_case(&par, M, N, K, NULL, 0),
              "параллельный %ux%ux%u разошёлся с однопоточным", M, N, K);
    }

    /* Эпилог: операнд @add адресуется от угла плитки в полной матрице.
     * Сдвиг хоть на элемент — и результат разойдётся. */
    {
        const uint32_t M = 150, N = 200, K = 90;
        fill_rand(g_a, (size_t)M * K);
        fill_rand(g_b, (size_t)K * N);
        fill_rand(g_e, (size_t)M * N);
        SmpTensor te = mk(M, N);
        SmpFuseStep st[2];
        memset(st, 0, sizeof st);
        st[0].op = SMP_FOP_ADD;  st[0].b = (SmpBuf){ g_e, &te };
        st[1].op = SMP_FOP_RELU;
        CHECK(par_case(&par, M, N, K, st, 2),
              "параллельный эпилог @add -> @relu разошёлся с однопоточным");
    }

    /* FTZ переносится на исполнителей: 1e-20 * 1e-20 — денормаль, и с FTZ
     * она обязана обнулиться на всех потоках, а не только на вызывающем. */
    {
        const uint32_t M = 64, N = 64, K = 64;
        for (size_t i = 0; i < (size_t)M * K; i++) g_a[i] = 1e-20f;
        for (size_t i = 0; i < (size_t)K * N; i++) g_b[i] = 1e-20f;

        SmpTensor ta = mk(M, K), tb = mk(K, N), tc = mk(M, N);
        SmpBuf a = { g_a, &ta }, b = { g_b, &tb }, got = { g_got, &tc };
        smp_k_gemm(&got, &a, &b, &g_sc);
        const bool denormal = g_got[0] != 0.0f;

        const uint32_t saved = smp_fpu_get_mxcsr();
        smp_fpu_set_ftz(true);
        const bool same = par_case(&par, M, N, K, NULL, 0);
        const bool zero = g_got[(size_t)M * N - 1] == 0.0f;
        smp_fpu_set_mxcsr(saved);

        CHECK(denormal, "без FTZ результат не денормаль — проверка ни о чём");
        CHECK(same && zero, "FTZ не дошёл до потоков параллельного GEMM");
    }

    smp_k_gemm_par_min((uint64_t)1 << 22);
    free(mem);
}

/* ========================================================================== */

int main(void)
{
    smp_console_setup();
    fprintf(stderr, "SMPC3 tests :: Ф6\n\n");

    g_scmem = malloc(smp_k_scratch_bytes());
    if (!g_scmem) { fprintf(stderr, "нет памяти под ядра\n"); return 70; }
    smp_k_scratch_bind(&g_sc, g_scmem, smp_k_scratch_bytes());

    test_available();
    test_elementwise();
    test_reduce();
    test_gemm();
    test_gemm_epilogue();
    test_fallback();
    test_identity();
    test_gemm_block();
    test_gemm_parallel();

    smp_kernels_select(SMP_KB_AUTO);
    free(g_scmem);
    return REPORT();
}
