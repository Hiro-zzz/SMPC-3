/* SMPC3 :: main.c -- CLI. */
#include "smpc3/common.h"
#include "smpc3/arena.h"
#include "smpc3/cpu.h"
#include "smpc3/diag.h"
#include "smpc3/io.h"
#include "smpc3/lex.h"
#include "smpc3/parse.h"
#include "smpc3/sema.h"
#include "smpc3/emit.h"
#include "smpc3/kernels.h"
#include "smpc3/pool.h"
#include "smpc3/thread.h"
#include "smpc3/types.h"
#include "smpc3/vm.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* ========================================================================== */

static void cmd_help(void)
{
    printf(
        "SuperMegaPlexCalc3000 " SMPC3_VERSION_STR "\n"
        "\n"
        "  smpc3 info              профиль процессора и состояние арены\n"
        "  smpc3 codes             весь реестр диагностических кодов\n"
        "  smpc3 explain <CODE>    развернуть один код, например E0418\n"
        "  smpc3 demo              показать диагностику в работе\n"
        "  smpc3 lex <файл.smpc>   дамп потока токенов\n"
        "  smpc3 parse <файл.smpc> дамп дерева разбора\n"
        "  smpc3 check <файл.smpc> семантическая проверка (-v: таблицы)\n"
        "  smpc3 build <файл.smpc> собрать .s3b (-o путь, -S: показать байткод)\n"
        "  smpc3 dis <файл.s3b>    дизассемблировать модуль\n"
        "  smpc3 run <файл>        исполнить .smpc или .s3b\n"
        "                          -v статистика, -n N инстансов, -j N потоков\n"
        "\n"
        "  smpc3 ops               реестр операций\n"
        "  smpc3 attrs             реестр директив, режимов и атрибутов\n"
        "  smpc3 bench             замер GEMM: скаляр против avx2+fma\n");
}

/* ========================================================================== */

static void cmd_info(void)
{
    const SmpCpu *c = smp_cpu();

    printf("CPU         : %s\n", c->brand);
    printf("Логических  : %u\n", c->n_logical);
    printf("Макс. вектор: %s (%u бит)\n", smp_vec_name(c->max_vec_bits), c->max_vec_bits);
    printf("ISA         :");
    struct { uint32_t bit; const char *name; } f[] = {
        { SMP_ISA_SSE2, "sse2" }, { SMP_ISA_SSE42, "sse4.2" },
        { SMP_ISA_AVX, "avx" },   { SMP_ISA_AVX2, "avx2" },
        { SMP_ISA_FMA, "fma" },   { SMP_ISA_AVX512F, "avx512f" },
        { SMP_ISA_AVX512DQ, "avx512dq" }, { SMP_ISA_AVX512BW, "avx512bw" },
        { SMP_ISA_AVX512VL, "avx512vl" }
    };
    for (size_t i = 0; i < SMP_ARRLEN(f); i++)
        if (c->isa & f[i].bit) printf(" %s", f[i].name);
    printf("\n");

    printf("Ядра        : %s\n", smp_kernels_name());

    /* Кэши и выбранная под них блокировка GEMM: упаковка панелей окупается,
     * только пока обе держатся в L2. Это единственное место, где видно, что
     * рантайм про L2 действительно спросил и что он с этим сделал. */
    printf("Кэш L1d/L2/L3: %u / %u / %u КиБ\n",
           c->l1d_bytes / 1024u, c->l2_bytes / 1024u, c->l3_bytes / 1024u);
    {
        uint32_t mc = 0, kc = 0, nc = 0;
        smp_k_gemm_block(&mc, &kc, &nc);
        /* Целочисленно: вещественная арифметика взвела бы флаг неточности
         * в MXCSR, а строкой ниже печатается именно он. */
        /* Целочисленно: вещественная арифметика взвела бы флаг неточности
         * в MXCSR, а строкой ниже печатается именно он. */
        printf("Блоки GEMM  : MC=%u KC=%u NC=%u (панели %u КиБ)\n",
               mc, kc, nc, (mc * kc + kc * nc) * 4u / 1024u);
    }
    printf("MXCSR       : 0x%04X\n", smp_fpu_get_mxcsr());
    printf("Дескриптор  : %zu байт\n", sizeof(SmpTensor));
    printf("Токен       : %zu байт\n", sizeof(SmpToken));

    /* Проверка арены на живом сценарии из спецификации: две матрицы 1024x1024. */
    SmpArena a;
    if (smp_arena_init(&a, 64u << 20, 0, "main") != SMP_OK) {
        fprintf(stderr, "арена не поднялась\n");
        return;
    }

    uint16_t shape[2] = { 1024, 1024 };
    SmpTensor A, B;
    smp_tensor_dense(&A, SMP_DT_F32, 2, shape);
    smp_tensor_dense(&B, SMP_DT_F32, 2, shape);

    A.off = smp_arena_push_off(&a, (size_t)smp_tensor_bytes(&A), SMP_ALIGN_V512);
    B.off = smp_arena_push_off(&a, (size_t)smp_tensor_bytes(&B), SMP_ALIGN_V512);

    char buf[160], sig[64];
    printf("\n%s\n", smp_arena_report(&a, buf, sizeof buf));
    printf("A  <%s>  off=%u  align64=%s  contig=%s\n",
           smp_tensor_sig(&A, sig, sizeof sig), A.off,
           SMP_IS_ALIGNED(smp_arena_at(&a, A.off), 64) ? "да" : "НЕТ",
           smp_tensor_is_contiguous(&A) ? "да" : "нет");
    printf("B  <%s>  off=%u  align64=%s\n",
           smp_tensor_sig(&B, sig, sizeof sig), B.off,
           SMP_IS_ALIGNED(smp_arena_at(&a, B.off), 64) ? "да" : "НЕТ");

    smp_arena_release(&a);
}

/* ========================================================================== */

static void cmd_codes(void)
{
    for (unsigned i = 0; i < SMP_DIAG__COUNT; i++) {
        const SmpDiagInfo *inf = smp_diag_info((SmpDiagCode)i);
        const char *sev = inf->sev == SMP_SEV_FATAL ? "FATAL"
                        : inf->sev == SMP_SEV_WARN  ? "WARN " : "NOTE ";
        printf("%s  %s  %s\n", inf->text, sev, inf->title);
    }
    printf("\nвсего кодов: %u\n", (unsigned)SMP_DIAG__COUNT);
}

static int cmd_explain(const char *what)
{
    SmpDiagCode c = smp_diag_lookup(what);
    if (c == SMP_DIAG__COUNT) {
        fprintf(stderr, "код '%s' не зарегистрирован\n", what);
        return 1;
    }
    const SmpDiagInfo *inf = smp_diag_info(c);
    printf("%s\n", inf->text);
    printf("  ОШИБКА      : %s\n", inf->title);
    printf("  ИСПРАВЛЕНИЕ : %s\n", inf->fix);
    return 0;
}

/* ========================================================================== */

/* printf("%-Ns") считает БАЙТЫ, а имена видов токенов кириллические, поэтому
 * колонки в дампе разъезжаются. Дополняем по кодовым точкам. */
static void print_pad(const char *s, unsigned width)
{
    unsigned cps = 0;
    for (const char *q = s; *q; q++)
        if (((unsigned char)*q & 0xC0u) != 0x80u) cps++;
    fputs(s, stdout);
    for (unsigned i = cps; i < width; i++) fputc(' ', stdout);
}

static int cmd_lex(const char *path)
{
    SmpArena arena;
    if (smp_arena_init(&arena, 16u << 20, 1, "compile") != SMP_OK) {
        fprintf(stderr, "арена не поднялась\n");
        return 70;
    }

    SmpSource src;
    if (smp_source_load(&src, path, &arena) != SMP_OK) {
        fprintf(stderr, "не открывается: %s\n", path);
        smp_arena_release(&arena);
        return 66;
    }

    SmpDiagCtx D;
    smp_diag_init(&D, &src, stderr);

    SmpLexer lx;
    smp_lex_init(&lx, &src, &D);

    SmpToken *toks = NULL;
    uint32_t  n    = 0;
    smp_lex_all(&lx, &arena, &toks, &n);

    printf("%-5s ", "#");
    print_pad("поз.", 9);
    print_pad("вид", 22);
    printf("лексема\n");
    printf("----- -------- ---------------------- ----------------------------\n");

    char lex[128], pos[24];
    for (uint32_t i = 0; i < n; i++) {
        const SmpToken *t = &toks[i];
        snprintf(pos, sizeof pos, "%u:%u", t->span.line, t->span.col);
        printf("%-5u %-8s ", i, pos);
        print_pad(smp_tok_name(t->kind), 22);
        printf("%s\n", smp_tok_str(t, lex, sizeof lex));
    }

    char rep[160], sum[128];
    printf("\nтокенов: %u   %s\n", n, smp_arena_report(&arena, rep, sizeof rep));
    if (lx.n_errors)
        fprintf(stderr, "\nитог: %s\n", smp_diag_summary(&D, sum, sizeof sum));

    smp_arena_release(&arena);
    return lx.n_errors ? 65 : 0;
}

/* ========================================================================== */

static int cmd_parse(const char *path)
{
    SmpArena arena;
    if (smp_arena_init(&arena, 64u << 20, 1, "compile") != SMP_OK) {
        fprintf(stderr, "арена не поднялась\n");
        return 70;
    }

    SmpSource src;
    if (smp_source_load(&src, path, &arena) != SMP_OK) {
        fprintf(stderr, "не открывается: %s\n", path);
        smp_arena_release(&arena);
        return 66;
    }

    SmpDiagCtx D;
    smp_diag_init(&D, &src, stderr);

    SmpLexer lx;
    smp_lex_init(&lx, &src, &D);
    SmpToken *toks = NULL;
    uint32_t  ntok = 0;
    smp_lex_all(&lx, &arena, &toks, &ntok);

    /* Разбирать поток, в котором лексер уже нашёл мусор, бессмысленно:
     * посыплются мнимые синтаксические ошибки поверх настоящих. */
    if (lx.n_errors) {
        char sum[128];
        fprintf(stderr, "\nразбор не начат: %s\n", smp_diag_summary(&D, sum, sizeof sum));
        smp_arena_release(&arena);
        return 65;
    }

    SmpParser P;
    smp_parse_init(&P, &arena, &D, &src, toks, ntok);
    SmpAstProgram prog;
    smp_parse(&P, &prog);

    smp_ast_dump(stdout, &prog, false);

    char rep[160], sum[128];
    printf("\n%s\n", smp_arena_report(&arena, rep, sizeof rep));
    if (P.n_errors)
        fprintf(stderr, "\nитог: %s\n", smp_diag_summary(&D, sum, sizeof sum));

    smp_arena_release(&arena);
    return P.n_errors ? 65 : 0;
}

/* ========================================================================== */

/* Общий фронтенд: загрузка, лексер, парсер, семантика. Возвращает код выхода
 * или -1, если дошли до конца без фатальных ошибок. */
static int frontend(const char *path, SmpArena *arena, SmpDiagCtx *D,
                    SmpSource *src, SmpAstProgram *prog, SmpSemaResult *res,
                    SmpSema *sm)
{
    /* Контекст инициализируется ПЕРВЫМ. Если сделать это после загрузки, то
     * на неоткрывшемся файле вызывающий получит счётчики из мусора на стеке. */
    memset(src, 0, sizeof *src);
    src->path = path;
    smp_diag_init(D, src, stderr);

    if (smp_source_load(src, path, arena) != SMP_OK) {
        fprintf(stderr, "не открывается: %s\n", path);
        return 66;
    }
    smp_diag_init(D, src, stderr);   /* теперь с настоящим текстом */

    SmpLexer lx;
    smp_lex_init(&lx, src, D);
    SmpToken *toks = NULL;
    uint32_t  ntok = 0;
    smp_lex_all(&lx, arena, &toks, &ntok);
    if (lx.n_errors) return 65;

    SmpParser P;
    smp_parse_init(&P, arena, D, src, toks, ntok);
    smp_parse(&P, prog);
    if (P.n_errors) return 65;

    smp_sema_init(sm, arena, D, src);
    smp_sema_run(sm, prog, res);
    return sm->n_errors ? 65 : -1;
}

static int cmd_check(const char *path, bool dump)
{
    SmpArena arena;
    if (smp_arena_init(&arena, 64u << 20, 1, "compile") != SMP_OK) {
        fprintf(stderr, "арена не поднялась\n");
        return 70;
    }

    SmpSource     src;
    SmpAstProgram prog;
    SmpSemaResult res;
    SmpSema       sm;
    SmpDiagCtx    D;

    memset(&res, 0, sizeof res);
    memset(&sm,  0, sizeof sm);

    const int rc = frontend(path, &arena, &D, &src, &prog, &res, &sm);

    if (dump && res.ninfo) {
        smp_sema_dump(stdout, &res);
        char rep[160];
        printf("\n%s\n", smp_arena_report(&arena, rep, sizeof rep));
    }

    char sum[128];
    if (sm.n_errors || sm.n_warnings)
        fprintf(stderr, "\nитог: %s\n", smp_diag_summary(&D, sum, sizeof sum));
    else if (rc < 0)
        fprintf(stderr, "проверка пройдена: замечаний нет\n");

    smp_arena_release(&arena);
    return rc < 0 ? 0 : rc;
}

/* ========================================================================== */

/* Заменяет расширение на .s3b: kernel.smpc -> kernel.s3b */
static void default_output(const char *in, char *out, size_t cap)
{
    snprintf(out, cap, "%s", in);
    char *dot = strrchr(out, '.');
    char *sep = strrchr(out, '/');
    char *sep2 = strrchr(out, '\\');
    if (sep2 > sep) sep = sep2;
    if (dot && dot > sep) *dot = '\0';
    const size_t n = strlen(out);
    snprintf(out + n, cap - n, ".s3b");
}

static int cmd_build(const char *path, const char *outpath, bool show)
{
    SmpArena arena;
    if (smp_arena_init(&arena, 128u << 20, 1, "compile") != SMP_OK) {
        fprintf(stderr, "арена не поднялась\n");
        return 70;
    }

    SmpSource     src;
    SmpAstProgram prog;
    SmpSemaResult res;
    SmpSema       sm;
    SmpDiagCtx    D;
    memset(&res, 0, sizeof res);
    memset(&sm,  0, sizeof sm);

    const int rc = frontend(path, &arena, &D, &src, &prog, &res, &sm);
    if (rc >= 0) {
        char sum[128];
        fprintf(stderr, "\nсборка прервана: %s\n", smp_diag_summary(&D, sum, sizeof sum));
        smp_arena_release(&arena);
        return rc;
    }

    SmpEmitter em;
    smp_emit_init(&em, &arena, &D, &src);
    SmpModule mod;
    if (smp_emit(&em, &prog, &res, &mod) != SMP_OK) {
        smp_arena_release(&arena);
        return 70;
    }

    char buf[512];
    if (!outpath) { default_output(path, buf, sizeof buf); outpath = buf; }

    if (smp_s3b_write(&mod, outpath) != SMP_OK) {
        fprintf(stderr, "не записывается: %s\n", outpath);
        smp_arena_release(&arena);
        return 73;
    }

    if (show) { fputc('\n', stdout); smp_disasm(stdout, &mod, false); }

    char rep[160];
    fprintf(stderr, "собрано: %s (%u инстр., %u дескр., %u конст.)\n",
            outpath, mod.n_code, mod.n_tens, mod.n_consts);
    if (sm.n_warnings) {
        char sum[128];
        fprintf(stderr, "итог: %s\n", smp_diag_summary(&D, sum, sizeof sum));
    }
    fprintf(stderr, "%s\n", smp_arena_report(&arena, rep, sizeof rep));

    smp_arena_release(&arena);
    return 0;
}

static int cmd_dis(const char *path)
{
    SmpArena arena;
    if (smp_arena_init(&arena, 64u << 20, 1, "load") != SMP_OK) return 70;

    SmpSource  src = { path, "", 0 };
    SmpDiagCtx D;
    smp_diag_init(&D, &src, stderr);

    SmpModule mod;
    if (smp_s3b_read(&mod, path, &arena, &D) != SMP_OK) {
        smp_arena_release(&arena);
        return 65;
    }
    smp_disasm(stdout, &mod, false);
    smp_arena_release(&arena);
    return 0;
}

/* ========================================================================== */

/* Общий хвост для run: поднять VM, исполнить, показать что вышло. */
static int execute(SmpModule *mod, SmpDiagCtx *D, SmpArena *arena, bool verbose)
{
    SmpVM vm;
    if (smp_vm_init(&vm, mod, D) != SMP_OK) {
        fprintf(stderr, "не поднялись арены рантайма\n");
        return 70;
    }
    vm.out = stdout;

    vm.out = stdout;

    /* С этого момента фатальная диагностика печатает дамп регистров. */
    D->regdump      = smp_vm_regdump;
    D->regdump_user = &vm;

    const SmpStatus st = smp_vm_run(&vm);

    if (st == SMP_OK) {
        smp_vm_dump_tensors(stdout, &vm, 8);
        if (verbose) {
            char rep[160];
            fprintf(stderr, "\nисполнено инструкций: %llu   ядра: %s\n",
                    (unsigned long long)vm.n_executed, smp_vm_backend());
            for (uint32_t i = 0; i < vm.n_arenas; i++)
                fprintf(stderr, "%s\n", smp_arena_report(&vm.arenas[i], rep, sizeof rep));
            SMP_UNUSED(arena);
        }
    }

    smp_vm_release(&vm);
    D->regdump = NULL;
    return st == SMP_OK ? 0 : 70;
}

/* Пул инстансов: N независимых копий одного модуля, раскиданных по потокам. */
static int run_pool(SmpModule *mod, uint32_t n_inst, uint32_t n_threads)
{
    SmpVMPool pool;
    if (smp_vm_pool_init(&pool, mod, n_inst, n_threads) != SMP_OK) {
        fprintf(stderr, "пул не поднялся\n");
        return 70;
    }

    const SmpStatus st = smp_vm_pool_run(&pool);

    printf("инстансов : %u на %u потоках\n", pool.n_inst, pool.n_threads);
    printf("инструкций: %llu\n", (unsigned long long)pool.n_executed);
    printf("время     : %.4f c\n", pool.seconds);
    if (pool.seconds > 0.0)
        printf("темп      : %.1f инстансов/с\n", pool.n_inst / pool.seconds);

    /* Журналы упавших печатаются здесь и по очереди: во время исполнения
     * шестнадцать потоков перемешали бы диагностику в кашу. */
    if (pool.n_failed) {
        fprintf(stderr, "\nупало инстансов: %u из %u\n", pool.n_failed, pool.n_inst);
        smp_vm_pool_report(stderr, &pool);
    }

    smp_vm_pool_release(&pool);
    return st == SMP_OK ? 0 : 70;
}

static int cmd_run(const char *path, bool verbose,
                   uint32_t n_inst, uint32_t n_threads)
{
    SmpArena arena;
    if (smp_arena_init(&arena, 128u << 20, 1, "compile") != SMP_OK) return 70;

    SmpSource     src;
    SmpAstProgram prog;
    SmpSemaResult res;
    SmpSema       sm;
    SmpDiagCtx    D;
    memset(&res, 0, sizeof res);
    memset(&sm,  0, sizeof sm);

    const size_t n = strlen(path);
    SmpModule mod;

    if (n > 4 && strcmp(path + n - 4, ".s3b") == 0) {
        /* Готовый модуль: компилировать нечего. */
        smp_source_from_memory(&src, path, "", 0);
        smp_diag_init(&D, &src, stderr);
        if (smp_s3b_read(&mod, path, &arena, &D) != SMP_OK) {
            smp_arena_release(&arena);
            return 65;
        }
    } else {
        const int rc = frontend(path, &arena, &D, &src, &prog, &res, &sm);
        if (rc >= 0) {
            char sum[128];
            fprintf(stderr, "\nзапуск отменён: %s\n", smp_diag_summary(&D, sum, sizeof sum));
            smp_arena_release(&arena);
            return rc;
        }
        SmpEmitter em;
        smp_emit_init(&em, &arena, &D, &src);
        if (smp_emit(&em, &prog, &res, &mod) != SMP_OK) {
            smp_arena_release(&arena);
            return 70;
        }
    }

    const int rc = (n_inst > 1)
        ? run_pool(&mod, n_inst, n_threads)
        : execute(&mod, &D, &arena, verbose);
    smp_arena_release(&arena);
    return rc;
}

/* ========================================================================== */

/* Замер GEMM на обеих ветках. Матрицы берутся из арены — той же, что в
 * рантайме, с тем же выравниванием на 64. */
static void bench_gemm(SmpArena *a, uint32_t n, double min_sec)
{
    SmpTensor ta, tb, tc;
    uint16_t shape[2] = { (uint16_t)n, (uint16_t)n };
    smp_tensor_dense(&ta, SMP_DT_F32, 2, shape);
    tb = ta; tc = ta;

    const size_t bytes = (size_t)smp_tensor_bytes(&ta);
    float *A = (float *)smp_arena_push(a, bytes, SMP_ALIGN_V512);
    float *B = (float *)smp_arena_push(a, bytes, SMP_ALIGN_V512);
    float *C = (float *)smp_arena_push(a, bytes, SMP_ALIGN_V512);
    if (!A || !B || !C) { printf("  %4u  — не хватило арены\n", n); return; }

    uint64_t r = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < (size_t)n * n; i++) {
        r ^= r << 13; r ^= r >> 7; r ^= r << 17;
        A[i] = (float)((double)(r >> 40) / 8388608.0 - 0.125);
        B[i] = (float)((double)(r >> 20 & 0xFFFFFF) / 8388608.0 - 0.125);
    }

    SmpBuf bc = { C, &tc }, ba = { A, &ta }, bb = { B, &tb };

    /* Замеру рабочая память нужна такая же, как инстансу VM. */
    /* Рабочая память берётся отдельным блоком, а НЕ из той же арены, что
     * матрицы. При N=1024 матрицы занимают ровно по 4 МиБ, ложатся в арене с
     * шагом 4 МиБ и попадают в одни и те же множества L2. Панели упаковки,
     * положенные следом, конфликтуют с ними — пропускная способность падает со
     * 114 до 20 ГФЛОПС. Инстанс VM берёт эту память отдельно и по той же
     * причине; здесь повторяем его поведение, а не маскируем проблему. */
    void *smem = malloc(smp_k_scratch_bytes());
    SmpKScratch sc;
    smp_k_scratch_bind(&sc, smem, smp_k_scratch_bytes());
    if (!smem) { printf("  %4u  — не хватило памяти\n", n); return; }
    const double flops = 2.0 * (double)n * (double)n * (double)n;

    double best[2] = { 0.0, 0.0 };
    const SmpKernelBackend which[2] = { SMP_KB_SCALAR, SMP_KB_AVX2 };

    /* Берём ЛУЧШЕЕ из нескольких прогонов, а не единственный замер. Причина
     * не в приукрашивании: у гибридного процессора шесть P-ядер и восемь
     * E-ядер, и планировщик волен закинуть поток на E-ядро, где AVX2+FMA идёт
     * впятеро медленнее. Одиночный замер на 1024 давал то 20, то 100 ГФЛОПС —
     * такое число не значит ничего. Лучшее из серии отвечает на честный
     * вопрос: сколько ядро выдаёт, когда ему не мешают. */
    for (int k = 0; k < 2; k++) {
        if (!smp_kernels_select(which[k])) continue;
        smp_k_gemm(&bc, &ba, &bb, &sc);            /* прогрев кэшей */

        for (int pass = 0; pass < 3; pass++) {
            uint32_t iters = 0;
            const double t0 = smp_now_sec();
            double dt;
            do {
                smp_k_gemm(&bc, &ba, &bb, &sc);
                iters++;
                dt = smp_now_sec() - t0;
            } while (dt < min_sec);
            const double g = flops * iters / dt / 1e9;
            if (g > best[k]) best[k] = g;
        }
    }
    smp_kernels_select(SMP_KB_AUTO);
    free(smem);

    printf("  %4u   %9.3f   %9.3f   %6.1fx\n", n, best[0], best[1],
           best[0] > 0.0 ? best[1] / best[0] : 0.0);
}

/* Совокупная пропускная способность на всех ядрах: каждый поток считает свой
 * GEMM со своей рабочей памятью — ровно как инстанс в пуле. */
typedef struct {
    uint32_t n, iters;
    float  **A, **B, **C;
    SmpKScratch *sc;
} BenchCtx;

static void bench_worker(void *ctx, uint32_t idx)
{
    BenchCtx *b = (BenchCtx *)ctx;
    SmpTensor t;
    uint16_t shape[2] = { (uint16_t)b->n, (uint16_t)b->n };
    smp_tensor_dense(&t, SMP_DT_F32, 2, shape);

    SmpBuf bc = { b->C[idx], &t }, ba = { b->A[idx], &t }, bb = { b->B[idx], &t };
    for (uint32_t i = 0; i < b->iters; i++)
        smp_k_gemm(&bc, &ba, &bb, &b->sc[idx]);
}

static void bench_threads(uint32_t n, uint32_t threads, uint32_t iters)
{
    SmpTensor t;
    uint16_t shape[2] = { (uint16_t)n, (uint16_t)n };
    smp_tensor_dense(&t, SMP_DT_F32, 2, shape);
    const size_t bytes = (size_t)smp_tensor_bytes(&t);

    BenchCtx b;
    b.n = n; b.iters = iters;
    b.A  = (float **)calloc(threads, sizeof(float *));
    b.B  = (float **)calloc(threads, sizeof(float *));
    b.C  = (float **)calloc(threads, sizeof(float *));
    b.sc = (SmpKScratch *)calloc(threads, sizeof(SmpKScratch));
    void **mem = (void **)calloc(threads, sizeof(void *));
    if (!b.A || !b.B || !b.C || !b.sc || !mem) { printf("  нет памяти\n"); return; }

    for (uint32_t i = 0; i < threads; i++) {
        b.A[i] = (float *)malloc(bytes);
        b.B[i] = (float *)malloc(bytes);
        b.C[i] = (float *)malloc(bytes);
        mem[i] = malloc(smp_k_scratch_bytes());
        if (!b.A[i] || !b.B[i] || !b.C[i] || !mem[i]) { printf("  нет памяти\n"); return; }
        smp_k_scratch_bind(&b.sc[i], mem[i], smp_k_scratch_bytes());
        for (size_t e = 0; e < bytes / sizeof(float); e++) {
            b.A[i][e] = 1.0f / 64.0f;
            b.B[i][e] = 1.0f / 32.0f;
        }
    }

    smp_threads_run(bench_worker, &b, threads);        /* прогрев */
    const double t0 = smp_now_sec();
    smp_threads_run(bench_worker, &b, threads);
    const double dt = smp_now_sec() - t0;

    const double flops = 2.0 * (double)n * n * n * (double)iters * threads;
    printf("  %4u   %9.1f\n", threads, dt > 0.0 ? flops / dt / 1e9 : 0.0);

    for (uint32_t i = 0; i < threads; i++) {
        free(b.A[i]); free(b.B[i]); free(b.C[i]); free(mem[i]);
    }
    free(b.A); free(b.B); free(b.C); free(b.sc); free(mem);
}

static void cmd_bench(void)
{
    const SmpCpu *c = smp_cpu();
    printf("%s\n", c->brand);
    printf("ISA: %s   ветка по умолчанию: %s\n\n",
           smp_vec_name(c->max_vec_bits), smp_kernels_name());
    /* Закрепляемся за P-ядром, но НЕ за нулевым: Windows вешает на логический
     * процессор 0 обработку прерываний, и он стабильно даёт вдвое меньше
     * остальных (93 против 127 ГФЛОПС на N=256). Без закрепления вообще замер
     * каждый раз попадает на разное ядро — то на P, то на E, — и разброс
     * доходит до пятикратного. */
    const bool pinned = smp_thread_pin(2);
    printf("GEMM f32 на одном ядре, ГФЛОПС%s\n",
           pinned ? " (поток закреплён за P-ядром)" : " (закрепить не вышло)");
    printf("     N      скаляр    avx2+fma   ускорение\n");
    printf("  ----   ---------   ---------   ---------\n");

    SmpArena a;
    if (smp_arena_init(&a, 512u << 20, 0, "bench") != SMP_OK) {
        fprintf(stderr, "арена не поднялась\n");
        return;
    }
    static const uint32_t sizes[] = { 64, 128, 256, 512, 1024 };
    for (size_t i = 0; i < SMP_ARRLEN(sizes); i++) {
        smp_arena_reset(&a);
        bench_gemm(&a, sizes[i], sizes[i] >= 512 ? 0.15 : 0.05);
    }
    smp_arena_release(&a);
    smp_thread_unpin();

    /* И то же самое на всех ядрах сразу — то, ради чего появились инстансы.
     * Здесь закреплять ничего не нужно: работа раздаётся всем ядрам, и E-ядра
     * вносят свой вклад честно. */
    printf("\nGEMM f32 512, совокупно по потокам, ГФЛОПС\n");
    printf("  пот.       всего\n");
    printf("  ----   ---------\n");
    const uint32_t maxt = smp_threads_default();
    for (uint32_t th = 1; th <= maxt; th *= 2) bench_threads(512, th, 8);
}

/* ========================================================================== */

static const char g_demo_src[] =
    "// Инициализация матриц в выровненной арене\n"
    "[#arena:0]  *&A<f32:1024,1024>  ->  @alloc  =>  $r1;\n"
    "[#arena:0]  *&B<f32:1024,512>   ->  @alloc  =>  $r2;\n"
    "\n"
    "// Параллельный GEMM с запретом алиасинга указателей\n"
    "[#simd:v512]  $r1 -> @mmul($r2) => *&C<f32:1024,1024>  [!no-alias, ?strict];\n"
    "\n"
    "// Выборка строки, активация и редукция в скаляр\n"
    "[^raw]  *&C[0, ..] -> @relu -> @reduce.add => $f0  [~flush-to-zero];\n";

static void demo_regdump(SmpDiagCtx *dg, void *user, bool color)
{
    SMP_UNUSED(user);
    const char *dim = color ? "\x1b[2m" : "";
    const char *rst = color ? "\x1b[0m" : "";
    smp_diag_write(dg, "%s  --- ДАМП РЕГИСТРОВ ---------------------------------%s\n", dim, rst);
    smp_diag_write(dg, "   $r1 = tensor{ off=0x00000000  f32:1024,1024  stride=1024,1  flags=CONTIG|A64 }\n");
    smp_diag_write(dg, "   $r2 = tensor{ off=0x00400000  f32:1024,512   stride=512,1   flags=CONTIG|A64 }\n");
    smp_diag_write(dg, "   $f0 = 0.000000e+00\n");
    smp_diag_write(dg, "   MXCSR = 0x%04X   ISA = %s\n",
            smp_fpu_get_mxcsr(), smp_vec_name(smp_cpu()->max_vec_bits));
}

static void cmd_demo(void)
{
    SmpSource src = { "kernel.smpc", g_demo_src, sizeof(g_demo_src) - 1u };
    SmpDiagCtx D;
    smp_diag_init(&D, &src, stderr);
    D.regdump = demo_regdump;

    /* 1. Ошибка из спецификации: K не сходится. */
    smp_diag_emit(&D, &(SmpDiagMsg){
        .code    = SMP_E0419,
        .span    = (SmpSpan){ 6, 32, 12 },   /* @mmul($r2) */
        .details = smp_fmt(&D,
            "$r2 указывает на матрицу %ux%u, а $r1 — %ux%u.\n"
            "Размерности K не сходятся (%u != %u).",
            1024u, 512u, 1024u, 1024u, 1024u, 512u),
        .fix       = "Вызови @transpose($r2) или перепиши свой код на Scratch.",
        .dump_regs = true
    });

    /* 2. Предупреждение от runtime-детекта ISA. */
    const SmpCpu *c = smp_cpu();
    if (!(c->isa & SMP_ISA_AVX512F)) {
        smp_diag_emit(&D, &(SmpDiagMsg){
            .code    = SMP_W0512,
            .span    = (SmpSpan){ 6, 2, 11 },  /* #simd:v512 */
            .details = smp_fmt(&D,
                "Процессор «%s» предоставляет максимум %s.\n"
                "При активном ?strict это стало бы фатальной ошибкой E0502.",
                c->brand, smp_vec_name(c->max_vec_bits))
        });
    }

    /* 3. Ошибка на сыром указателе без шага. */
    smp_diag_emit(&D, &(SmpDiagMsg){
        .code    = SMP_E0418,
        .span    = (SmpSpan){ 9, 9, 10 },     /* *&C[0, ..] */
        .details = "Режим [^raw] отключает вывод шага, а срез [0, ..] его требует."
    });

    char sum[128];
    fprintf(stderr, "\nитог: %s\n", smp_diag_summary(&D, sum, sizeof sum));
}

/* ========================================================================== */

int main(int argc, char **argv)
{
    smp_console_setup();

    if (argc < 2)                        { cmd_help();  return 0; }
    if (strcmp(argv[1], "info")  == 0)   { cmd_info();  return 0; }
    if (strcmp(argv[1], "codes") == 0)   { cmd_codes(); return 0; }
    if (strcmp(argv[1], "demo")  == 0)   { cmd_demo();  return 0; }
    if (strcmp(argv[1], "lex")   == 0) {
        if (argc < 3) { fprintf(stderr, "нужен путь к .smpc\n"); return 1; }
        return cmd_lex(argv[2]);
    }
    if (strcmp(argv[1], "parse") == 0) {
        if (argc < 3) { fprintf(stderr, "нужен путь к .smpc\n"); return 1; }
        return cmd_parse(argv[2]);
    }
    if (strcmp(argv[1], "check") == 0) {
        if (argc < 3) { fprintf(stderr, "нужен путь к .smpc\n"); return 1; }
        return cmd_check(argv[2], argc > 3 && strcmp(argv[3], "-v") == 0);
    }
    if (strcmp(argv[1], "build") == 0) {
        if (argc < 3) { fprintf(stderr, "нужен путь к .smpc\n"); return 1; }
        const char *o = NULL;
        bool show = false;
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) o = argv[++i];
            else if (strcmp(argv[i], "-S") == 0) show = true;
        }
        return cmd_build(argv[2], o, show);
    }
    if (strcmp(argv[1], "run") == 0) {
        if (argc < 3) { fprintf(stderr, "нужен путь к .smpc или .s3b\n"); return 1; }
        bool     verbose = false;
        uint32_t n_inst = 1, n_threads = 0;
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "-v") == 0) verbose = true;
            else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
                n_inst = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "-j") == 0 && i + 1 < argc)
                n_threads = (uint32_t)strtoul(argv[++i], NULL, 10);
        }
        if (n_inst == 0) n_inst = 1;
        return cmd_run(argv[2], verbose, n_inst, n_threads);
    }
    if (strcmp(argv[1], "dis") == 0) {
        if (argc < 3) { fprintf(stderr, "нужен путь к .s3b\n"); return 1; }
        return cmd_dis(argv[2]);
    }
    if (strcmp(argv[1], "bench") == 0) { cmd_bench(); return 0; }
    if (strcmp(argv[1], "ops")   == 0) { smp_ops_list(stdout);   return 0; }
    if (strcmp(argv[1], "attrs") == 0) { smp_attrs_list(stdout); return 0; }
    if (strcmp(argv[1], "explain") == 0) {
        if (argc < 3) { fprintf(stderr, "нужен код, например: smpc3 explain E0418\n"); return 1; }
        return cmd_explain(argv[2]);
    }

    fprintf(stderr, "неизвестная команда: %s\n\n", argv[1]);
    cmd_help();
    return 1;
}
