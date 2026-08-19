/* SMPC3 :: test_pool.c -- проверки пула инстансов.
 *
 * Главная опасность здесь — не падение, а МОЛЧА неверный результат. Гонка за
 * общий буфер упаковки GEMM не роняет процесс: она портит числа у части
 * инстансов, и заметить это можно только сверив каждый.
 */
#include "smpc3/emit.h"
#include "smpc3/io.h"
#include "smpc3/lex.h"
#include "smpc3/parse.h"
#include "smpc3/pool.h"
#include "smpc3/sema.h"
#include "smpc3/thread.h"

#include "harness.h"

#include <string.h>
#include <stdlib.h>
#include <math.h>

/* ========================================================================== */

static SmpArena   g_arena;
static SmpDiagCtx g_diag;
static SmpSource  g_src;
static FILE      *g_sink;
static SmpModule  g_mod;

/* Компилирует исходник в модуль. */
static bool compile_str(const char *text)
{
    smp_arena_reset(&g_arena);
    rewind(g_sink);
    memset(&g_mod, 0, sizeof g_mod);

    smp_source_from_memory(&g_src, "t.smpc", text, strlen(text));
    smp_diag_init(&g_diag, &g_src, g_sink);

    SmpLexer lx;
    smp_lex_init(&lx, &g_src, &g_diag);
    SmpToken *toks = NULL;
    uint32_t  ntok = 0;
    smp_lex_all(&lx, &g_arena, &toks, &ntok);
    if (lx.n_errors) { CHECK(0, "лексер"); return false; }

    SmpParser P;
    smp_parse_init(&P, &g_arena, &g_diag, &g_src, toks, ntok);
    SmpAstProgram prog;
    smp_parse(&P, &prog);
    if (P.n_errors) { CHECK(0, "парсер"); return false; }

    SmpSema sm;
    smp_sema_init(&sm, &g_arena, &g_diag, &g_src);
    sm.host_vec_bits = 256;
    SmpSemaResult res;
    smp_sema_run(&sm, &prog, &res);
    if (sm.n_errors) { CHECK(0, "семантика"); return false; }

    SmpEmitter em;
    smp_emit_init(&em, &g_arena, &g_diag, &g_src);
    return smp_emit(&em, &prog, &res, &g_mod) == SMP_OK;
}

static bool near(double a, double b) { return fabs(a - b) < 1e-3; }

/* Значение именованного регистра в конкретном инстансе. */
static double pool_reg(const SmpVMPool *p, uint32_t inst, const char *name)
{
    const SmpVM *vm = &p->inst[inst];
    for (uint32_t i = 0; i < vm->n_regs && i < p->mod->n_reg_names; i++) {
        if (!p->mod->reg_names[i]) continue;
        if (strcmp(smp_module_str(p->mod, p->mod->reg_names[i]), name) != 0) continue;
        return vm->regs[i].s.f;
    }
    return NAN;
}

/* ========================================================================== */

static void test_basics(void)
{
    SECTION("подъём и изоляция");

    if (!compile_str("*&A<f32:16,16> -> @fill(3.0) => *&A;\n"
                     "*&A -> @reduce.add => $s;\n")) return;

    SmpVMPool p;
    CHECK(smp_vm_pool_init(&p, &g_mod, 8, 4) == SMP_OK, "пул не поднялся");
    CHECK(p.n_inst == 8, "инстансов %u", p.n_inst);
    CHECK(p.n_threads == 4, "потоков %u", p.n_threads);

    /* Арены инстансов не должны пересекаться ни одним байтом: на этом стоит
     * вся безопасность пула. */
    for (uint32_t i = 0; i < p.n_inst; i++) {
        for (uint32_t a = 0; a < p.inst[i].n_arenas; a++) {
            const uint8_t *bi = p.inst[i].arenas[a].base;
            const size_t   ci = p.inst[i].arenas[a].cap;
            CHECK(SMP_IS_ALIGNED(bi, 64), "арена #%u инстанса %u не выровнена", a, i);

            for (uint32_t j = i + 1; j < p.n_inst; j++) {
                for (uint32_t b = 0; b < p.inst[j].n_arenas; b++) {
                    const uint8_t *bj = p.inst[j].arenas[b].base;
                    const size_t   cj = p.inst[j].arenas[b].cap;
                    CHECK(bi + ci <= bj || bj + cj <= bi,
                          "арены инстансов %u и %u пересеклись", i, j);
                }
            }
        }
    }

    /* Рабочая память ядер тоже у каждого своя — иначе GEMM в двух потоках
     * затрёт панели. */
    for (uint32_t i = 0; i < p.n_inst; i++) {
        CHECK(p.inst[i].scratch_mem != NULL, "у инстанса %u нет рабочей памяти", i);
        for (uint32_t j = i + 1; j < p.n_inst; j++)
            CHECK(p.inst[i].scratch_mem != p.inst[j].scratch_mem,
                  "инстансы %u и %u делят рабочую память ядер", i, j);
    }

    CHECK(smp_vm_pool_run(&p) == SMP_OK, "прогон провалился");
    CHECK(p.n_failed == 0, "упало %u инстансов", p.n_failed);

    for (uint32_t i = 0; i < p.n_inst; i++)
        CHECK(near(pool_reg(&p, i, "s"), 768.0),
              "инстанс %u дал %g, ждали 768", i, pool_reg(&p, i, "s"));

    smp_vm_pool_release(&p);
    CHECK(p.inst == NULL && p.n_inst == 0, "release не вычистил пул");
}

/* ========================================================================== */

static void test_instance_id(void)
{
    SECTION("номер инстанса");

    if (!compile_str("*&ID<f32:4> -> @fill.instance => *&ID;\n"
                     "*&ID -> @reduce.add => $sum;\n"
                     "*&ID -> @reduce.max => $id;\n")) return;

    SmpVMPool p;
    CHECK(smp_vm_pool_init(&p, &g_mod, 16, 8) == SMP_OK, "пул");
    CHECK(smp_vm_pool_run(&p) == SMP_OK, "прогон");

    /* Каждый инстанс обязан увидеть СВОЙ номер, а не чужой и не общий. */
    for (uint32_t i = 0; i < p.n_inst; i++) {
        CHECK(near(pool_reg(&p, i, "id"), (double)i),
              "инстанс %u считает себя %g", i, pool_reg(&p, i, "id"));
        CHECK(near(pool_reg(&p, i, "sum"), (double)i * 4.0),
              "инстанс %u: сумма %g, ждали %g", i, pool_reg(&p, i, "sum"), i * 4.0);
    }

    smp_vm_pool_release(&p);
}

/* ========================================================================== */

/* Тот самый тест. GEMM пакует панели во временные буферы; пока они лежали в
 * глобалях, два потока молча портили друг другу числа. Здесь каждый инстанс
 * считает одно и то же, и любое расхождение — это гонка. */
static void test_gemm_race(void)
{
    SECTION("GEMM в потоках");

    /* Размер подобран так, чтобы задеть блокировку (KC=256, NC=256, MC=96):
     * на мелких матрицах упаковка укладывается в один блок и гонка может
     * не проявиться. */
    if (!compile_str("*&A<f32:200,300> -> @fill(1.5) => *&A;\n"
                     "*&B<f32:300,200> -> @fill(2.0) => *&B;\n"
                     "[#simd:v256] *&A -> @mmul(*&B) => *&C<f32:200,200> [!no-alias];\n"
                     "*&C -> @reduce.add => $s;\n")) return;

    /* Каждый элемент C = 300 * 1.5 * 2.0 = 900; всего 200*200 элементов. */
    const double want = 200.0 * 200.0 * 900.0;

    /* Сначала эталон: один инстанс, один поток. */
    double single = 0.0;
    {
        SmpVMPool p;
        CHECK(smp_vm_pool_init(&p, &g_mod, 1, 1) == SMP_OK, "пул 1x1");
        CHECK(smp_vm_pool_run(&p) == SMP_OK, "прогон 1x1");
        single = pool_reg(&p, 0, "s");
        CHECK(near(single, want), "один инстанс дал %g, ждали %g", single, want);
        smp_vm_pool_release(&p);
    }

    /* Теперь та же работа на всех потоках. Совпасть обязано ВСЁ и у ВСЕХ. */
    for (uint32_t threads = 2; threads <= 16; threads *= 2) {
        SmpVMPool p;
        CHECK(smp_vm_pool_init(&p, &g_mod, 32, threads) == SMP_OK,
              "пул 32x%u", threads);
        CHECK(smp_vm_pool_run(&p) == SMP_OK, "прогон на %u потоках", threads);

        uint32_t bad = 0;
        for (uint32_t i = 0; i < p.n_inst; i++)
            if (!near(pool_reg(&p, i, "s"), single)) bad++;

        CHECK(bad == 0, "на %u потоках разошлось %u инстансов из %u "
                        "(гонка за рабочую память ядер?)", threads, bad, p.n_inst);

        /* И содержимое самих тензоров, а не только свёртка. */
        const SmpTensor *desc = NULL;
        for (uint32_t i = 0; i < p.n_inst && i < 4; i++) {
            const float *C = (const float *)smp_vm_pool_tensor(&p, i, "C", &desc);
            CHECK(C != NULL, "нет тензора C у инстанса %u", i);
            if (!C || !desc) continue;
            uint32_t wrong = 0;
            for (uint32_t e = 0; e < desc->nelem; e++)
                if (fabs((double)C[e] - 900.0) > 0.05) wrong++;
            CHECK(wrong == 0, "у инстанса %u испорчено %u элементов C", i, wrong);
        }

        smp_vm_pool_release(&p);
    }
}

/* ========================================================================== */

static void test_failure(void)
{
    SECTION("падение инстанса");

    if (!compile_str("*&A<f32:8,8> -> @fill(1.0e38) => *&A;\n"
                     "*&A -> @scale(1.0e30) => *&B<f32:8,8>;\n"
                     "*&B -> @reduce.add => $s [?strict];\n")) return;

    SmpVMPool p;
    CHECK(smp_vm_pool_init(&p, &g_mod, 8, 4) == SMP_OK, "пул");
    CHECK(smp_vm_pool_run(&p) != SMP_OK, "пул не заметил падений");
    CHECK(p.n_failed == p.n_inst, "упало %u из %u", p.n_failed, p.n_inst);

    /* У каждого инстанса свой журнал: диагностика не перемешалась. */
    FILE *f = tmpfile();
    CHECK(f != NULL, "tmpfile");
    if (f) {
        smp_vm_pool_report(f, &p);
        rewind(f);
        static char buf[65536];
        const size_t n = fread(buf, 1, sizeof(buf) - 1u, f);
        buf[n] = '\0';
        fclose(f);

        CHECK(strstr(buf, "E0602") != NULL, "в отчёте нет E0602");
        CHECK(strstr(buf, "инстанс #0") != NULL, "нет заголовка инстанса");
        CHECK(strstr(buf, "инстанс #7") != NULL, "отчёт не дошёл до последнего");

        /* Каждый упавший обязан дать ровно один блок диагностики, а не
         * склеенную кашу от восьми потоков. */
        uint32_t n_codes = 0;
        for (const char *q = buf; (q = strstr(q, "E0602")) != NULL; q++) n_codes++;
        CHECK(n_codes == p.n_inst, "блоков диагностики %u, инстансов %u",
              n_codes, p.n_inst);
    }

    smp_vm_pool_release(&p);
}

/* ========================================================================== */

static void test_limits(void)
{
    SECTION("границы");

    if (!compile_str("*&A<f32:4,4> -> @fill(1.0) => *&A;\n"
                     "*&A -> @reduce.add => $s;\n")) return;

    /* Потоков больше, чем инстансов — лишние не поднимаются. */
    SmpVMPool p;
    CHECK(smp_vm_pool_init(&p, &g_mod, 3, 64) == SMP_OK, "пул 3x64");
    CHECK(p.n_threads == 3, "потоков %u, ждали 3", p.n_threads);
    CHECK(smp_vm_pool_run(&p) == SMP_OK, "прогон");
    smp_vm_pool_release(&p);

    /* Ноль потоков — берём по числу процессоров. */
    CHECK(smp_vm_pool_init(&p, &g_mod, 8, 0) == SMP_OK, "пул 8x0");
    CHECK(p.n_threads >= 1 && p.n_threads <= 8, "автовыбор дал %u", p.n_threads);
    smp_vm_pool_release(&p);

    /* Ноль инстансов — отказ, а не пустой пул. */
    CHECK(smp_vm_pool_init(&p, &g_mod, 0, 4) != SMP_OK, "пул из нуля инстансов принят");

    /* Повторный прогон одного и того же пула даёт тот же результат. */
    CHECK(smp_vm_pool_init(&p, &g_mod, 4, 4) == SMP_OK, "пул 4x4");
    CHECK(smp_vm_pool_run(&p) == SMP_OK, "первый прогон");
    const double first = pool_reg(&p, 0, "s");
    CHECK(smp_vm_pool_run(&p) == SMP_OK, "второй прогон");
    CHECK(near(pool_reg(&p, 0, "s"), first), "повторный прогон дал другое");
    smp_vm_pool_release(&p);
}

static void test_threads(void)
{
    SECTION("обвязка потоков");

    CHECK(smp_threads_default() >= 1, "потоков по умолчанию 0");

    volatile int32_t c = 0;
    CHECK(smp_atomic_fetch_add(&c, 1) == 0, "fetch_add вернул не прежнее значение");
    CHECK(c == 1, "счётчик = %d", (int)c);
    CHECK(smp_atomic_fetch_add(&c, 5) == 1, "второй fetch_add");
    CHECK(c == 6, "счётчик = %d", (int)c);
}

/* Заявленный потолок обязан быть достижимым. Раньше журналом инстанса был
 * tmpfile(), и пул упирался в лимит открытых файлов CRT: 509 вместо 1024.
 * Отказ выглядел как "пул не поднялся" — без кода и без объяснения. */
static void test_max_instances(void)
{
    SECTION("заявленный потолок инстансов");

    if (!compile_str("*&A<f32:4,4> -> @fill(2.0) => *&A;\n"
                     "*&A -> @reduce.add => $s;\n")) return;

    SmpVMPool p;
    const uint32_t n = SMP_POOL_MAX_INSTANCES;

    CHECK(smp_vm_pool_init(&p, &g_mod, n, 4) == SMP_OK,
          "пул из %u инстансов не поднялся", n);
    CHECK(p.n_inst == n, "инстансов %u, ждали %u", p.n_inst, n);
    CHECK(smp_vm_pool_run(&p) == SMP_OK, "прогон %u инстансов", n);

    /* Каждый считает своё и независимо: 16 элементов по 2.0 дают 32. */
    bool all = true;
    for (uint32_t i = 0; i < p.n_inst; i++)
        if (!near(pool_reg(&p, i, "s"), 32.0)) all = false;
    CHECK(all, "не все инстансы досчитали до 32");

    smp_vm_pool_release(&p);
}

/* Журнал конечен, и переполнение обязано быть заметным, а не тихим. */
static void test_log_truncation(void)
{
    SECTION("обрезка журнала");

    SmpLog l;
    char   buf[16];
    smp_log_bind(&l, buf, sizeof buf);

    smp_log_write(&l, "0123456789", 10);
    CHECK(l.len == 10 && !l.truncated, "10 из 16 не влезли");

    smp_log_write(&l, "abcdefghij", 10);
    CHECK(l.len == 16, "журнал вышел за буфер: len=%zu", l.len);
    CHECK(l.truncated, "переполнение не отмечено");
    CHECK(memcmp(buf, "0123456789abcdef", 16) == 0, "хвост записан не тот");

    smp_log_reset(&l);
    CHECK(l.len == 0 && !l.truncated, "сброс не очистил журнал");
}

/* ========================================================================== */

int main(void)
{
    smp_console_setup();
    fprintf(stderr, "SMPC3 tests :: пул инстансов\n\n");

    if (smp_arena_init(&g_arena, 64u << 20, 9, "pooltest") != SMP_OK) {
        fprintf(stderr, "арена не поднялась\n");
        return 70;
    }
    g_sink = tmpfile();
    if (!g_sink) { fprintf(stderr, "tmpfile недоступен\n"); return 70; }

    test_threads();
    test_basics();
    test_instance_id();
    test_gemm_race();
    test_failure();
    test_limits();
    test_max_instances();
    test_log_truncation();

    fclose(g_sink);
    smp_arena_release(&g_arena);
    return REPORT();
}
