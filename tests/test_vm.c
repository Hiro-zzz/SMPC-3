/* SMPC3 :: test_vm.c -- проверки Ф5: исполнение байткода. */
#include "smpc3/emit.h"
#include "smpc3/io.h"
#include "smpc3/lex.h"
#include "smpc3/parse.h"
#include "smpc3/sema.h"
#include "smpc3/vm.h"
#include "smpc3/cpu.h"

#include "harness.h"

#include <string.h>
#include <stdlib.h>
#include <math.h>

/* ========================================================================== */
/*  Обвязка                                                                   */
/* ========================================================================== */

static SmpArena   g_arena;
static SmpDiagCtx g_diag;
static SmpSource  g_src;
static FILE      *g_sink;
static FILE      *g_emit;    /* сюда пишет @emit */
static char       g_out[16384];
static char       g_emitted[4096];

/* VM держит регистровый файл внутри себя (14 КиБ) — на стеке теста ему тесно
 * не будет, но статический экземпляр удобнее переиспользовать. */
static SmpVM      g_vm;
static SmpModule  g_mod;

/* Компилирует и исполняет. Возвращает true, если дошли до halt. */
static bool run_str(const char *text)
{
    smp_vm_release(&g_vm);
    smp_arena_reset(&g_arena);
    rewind(g_sink);
    g_out[0] = '\0';

    smp_source_from_memory(&g_src, "t.smpc", text, strlen(text));
    smp_diag_init(&g_diag, &g_src, g_sink);

    SmpLexer lx;
    smp_lex_init(&lx, &g_src, &g_diag);
    SmpToken *toks = NULL;
    uint32_t  ntok = 0;
    smp_lex_all(&lx, &g_arena, &toks, &ntok);
    if (lx.n_errors) { CHECK(0, "лексер: %s", text); return false; }

    SmpParser P;
    smp_parse_init(&P, &g_arena, &g_diag, &g_src, toks, ntok);
    SmpAstProgram prog;
    smp_parse(&P, &prog);
    if (P.n_errors) { CHECK(0, "парсер: %s", text); return false; }

    SmpSema sm;
    smp_sema_init(&sm, &g_arena, &g_diag, &g_src);
    sm.host_vec_bits = 256;
    SmpSemaResult res;
    smp_sema_run(&sm, &prog, &res);
    if (sm.n_errors) { CHECK(0, "семантика: %s", text); return false; }

    SmpEmitter em;
    smp_emit_init(&em, &g_arena, &g_diag, &g_src);
    if (smp_emit(&em, &prog, &res, &g_mod) != SMP_OK) {
        CHECK(0, "эмиттер: %s", text);
        return false;
    }

    if (smp_vm_init(&g_vm, &g_mod, &g_diag) != SMP_OK) {
        CHECK(0, "VM не поднялась");
        return false;
    }
    if (g_emit) { rewind(g_emit); g_vm.out = g_emit; }
    g_diag.regdump      = smp_vm_regdump;
    g_diag.regdump_user = &g_vm;

    const SmpStatus st = smp_vm_run(&g_vm);

    rewind(g_sink);
    const size_t n = fread(g_out, 1, sizeof(g_out) - 1u, g_sink);
    g_out[n] = '\0';

    return st == SMP_OK;
}

/* Указатель на данные именованного тензора (полный вид, не срез). */
static const void *tensor_data(const char *name, const SmpTensor **out)
{
    for (uint32_t i = 0; i < g_mod.n_tens; i++) {
        const SmpTensor *t = &g_mod.tens[i];
        if (strcmp(smp_module_str(&g_mod, t->name_id), name) != 0) continue;
        /* Первый встреченный — полный вид: срезы добавляются позже. */
        const uint32_t a = smp_tf_arena(t->flags);
        if (a >= g_vm.n_arenas) return NULL;
        if (out) *out = t;
        return g_vm.arenas[a].base + t->off;
    }
    return NULL;
}

static float felem(const char *name, uint32_t i)
{
    const float *p = (const float *)tensor_data(name, NULL);
    return p ? p[i] : NAN;
}

/* Значение именованного скалярного регистра. */
static double regval(const char *name, bool *found)
{
    if (found) *found = false;
    for (uint32_t i = 0; i < g_vm.n_regs && i < g_mod.n_reg_names; i++) {
        if (!g_mod.reg_names[i]) continue;
        if (strcmp(smp_module_str(&g_mod, g_mod.reg_names[i]), name) != 0) continue;
        if (found) *found = true;
        return g_vm.regs[i].s.f;
    }
    return NAN;
}

static bool near(double a, double b) { return fabs(a - b) < 1e-4; }

/* Что программа напечатала через @emit. */
static const char *emitted(void)
{
    if (!g_emit) return "";
    fflush(g_emit);
    const long n = ftell(g_emit);
    rewind(g_emit);
    const size_t cap = sizeof(g_emitted) - 1u;
    const size_t want = (n > 0 && (size_t)n < cap) ? (size_t)n : 0u;
    const size_t got  = want ? fread(g_emitted, 1, want, g_emit) : 0u;
    g_emitted[got] = '\0';
    return g_emitted;
}

/* ========================================================================== */

static void test_emit(void)
{
    SECTION("вывод");

    /* Кодовые точки становятся UTF-8. 1055..1057 — кириллические П, Р, С. */
    CHECK(run_str("*&H<i32:3> -> @alloc => $h;\n"
                  "1055 => *&H[0];  1056 => *&H[1];  1057 => *&H[2];\n"
                  "*&H -> @emit.text => $n;\n"), "utf-8");
    CHECK(strcmp(emitted(), "ПРС") == 0, "напечатано '%s', ждали 'ПРС'", emitted());

    /* Счётчик считает ЭЛЕМЕНТЫ, а не байты: трём кодовым точкам здесь
     * соответствует шесть байт, и путать их нельзя. */
    CHECK(near(regval("n", NULL), 3.0), "@emit вернул %g, ждали 3", regval("n", NULL));

    /* ASCII и перевод строки. */
    CHECK(run_str("*&H<i32:2> -> @alloc => $h;\n"
                  "72 => *&H[0];  105 => *&H[1];\n"
                  "*&H -> @emit.line => $n;\n"), "emit.line");
    CHECK(strcmp(emitted(), "Hi\n") == 0, "напечатано '%s'", emitted());

    /* Числовые форматы. */
    CHECK(run_str("*&H<i32:3> -> @alloc => $h;\n"
                  "10 => *&H[0];  255 => *&H[1];  0 => *&H[2];\n"
                  "*&H -> @emit.dec => $n;\n"), "emit.dec");
    CHECK(strcmp(emitted(), "10 255 0\n") == 0, "dec: '%s'", emitted());

    CHECK(run_str("*&H<i32:2> -> @alloc => $h;\n"
                  "255 => *&H[0];  16 => *&H[1];\n"
                  "*&H -> @emit.hex => $n;\n"), "emit.hex");
    CHECK(strcmp(emitted(), "0xFF 0x10\n") == 0, "hex: '%s'", emitted());

    CHECK(run_str("*&H<i32:3> -> @alloc => $h;\n"
                  "5 => *&H[0];  0 => *&H[1];  8 => *&H[2];\n"
                  "*&H -> @emit.bits => $n;\n"), "emit.bits");
    CHECK(strcmp(emitted(), "101 0 1000\n") == 0, "bits: '%s'", emitted());

    CHECK(run_str("*&A<f32:2> -> @fill(1.5) => *&A;\n"
                  "*&A -> @emit.num => $n;\n"), "emit.num");
    CHECK(strcmp(emitted(), "1.5 1.5\n") == 0, "num: '%s'", emitted());

    /* Печатать можно и срез со шагом: обход идёт по шагам, а не по памяти. */
    CHECK(run_str("*&M<i32:2,2> -> @alloc => $m;\n"
                  "1 => *&M[0,0];  2 => *&M[0,1];\n"
                  "3 => *&M[1,0];  4 => *&M[1,1];\n"
                  "*&M[.., 0] -> @emit.dec => $n;\n"), "срез со шагом");
    CHECK(strcmp(emitted(), "1 3\n") == 0, "столбец: '%s'", emitted());

    /* Вывод — часть конвейера, а не тупик: счётчик идёт дальше по стадиям. */
    CHECK(run_str("*&H<i32:4> -> @fill(65) => *&H;\n"
                  "*&H -> @emit.text -> @cast.f32 => $n;\n"), "emit в конвейере");
    CHECK(strcmp(emitted(), "AAAA") == 0, "'%s'", emitted());
    CHECK(near(regval("n", NULL), 4.0), "счётчик %g", regval("n", NULL));

    /* Вещественное в текстовый формат отвергает семантика — проверка живёт
     * в test_sema.c, здесь ей делать нечего. */

    /* Недопустимая кодовая точка — фатально, а не «поставим вопросик». */
    CHECK(!run_str("*&H<i32:2> -> @alloc => $h;\n"
                   "65 => *&H[0];  1114112 => *&H[1];\n"
                   "*&H -> @emit.text => $n;\n"), "точка за пределами Unicode");
    CHECK(strstr(g_out, "E0605") != NULL, "нет E0605");

    CHECK(!run_str("*&H<i32:1> -> @alloc => $h;\n"
                   "55296 => *&H[0];\n"
                   "*&H -> @emit.text => $n;\n"), "суррогат D800");
    CHECK(strstr(g_out, "E0605") != NULL, "нет E0605 на суррогате");
    CHECK(strstr(g_out, "суррогат") != NULL, "в деталях не сказано, что это суррогат");
}

/* ========================================================================== */

static void test_dispatch(void)
{
    SECTION("диспетчеризация");

    /* Ради этого всё и затевалось: на clang/gcc должна собираться прямая
     * ветка, а не switch-запасной вариант. */
    CHECK(SMP_HAS_COMPUTED_GOTO == 1,
          "computed goto недоступен — VM собрана в режиме switch");

    CHECK(run_str("*&A<f32:4,4> -> @alloc => $a;"), "простейшая программа");
    CHECK(g_vm.n_executed >= 3, "исполнено %llu инструкций",
          (unsigned long long)g_vm.n_executed);
    CHECK(g_vm.pc == g_mod.n_code - 1u, "остановились не на halt: pc=%u из %u",
          g_vm.pc, g_mod.n_code);
    CHECK(!g_vm.trapped, "ловушка на корректной программе");
}

static void test_memory(void)
{
    SECTION("память и арены");

    CHECK(run_str("[#arena:0] *&A<f32:4,4> -> @alloc => $a;\n"
                  "[#arena:1] *&B<f32:4,4> -> @alloc => $b;\n"), "две арены");
    CHECK(g_vm.n_arenas == 2, "поднято арен: %u", g_vm.n_arenas);

    /* Базы всех арен выровнены на 64 — весь SIMD стоит на этом. */
    for (uint32_t i = 0; i < g_vm.n_arenas; i++)
        CHECK(SMP_IS_ALIGNED(g_vm.arenas[i].base, 64),
              "база арены #%u не выровнена", i);

    /* Арены не пересекаются в адресном пространстве. */
    if (g_vm.n_arenas == 2) {
        const uint8_t *b0 = g_vm.arenas[0].base, *b1 = g_vm.arenas[1].base;
        const size_t   c0 = g_vm.arenas[0].cap,  c1 = g_vm.arenas[1].cap;
        CHECK(b0 + c0 <= b1 || b1 + c1 <= b0, "арены пересеклись");
    }

    /* @alloc обязан занулять: в debug арена залита 0xCD. */
    CHECK(run_str("*&A<f32:8,8> -> @alloc => $a;"), "alloc");
    bool all_zero = true;
    for (uint32_t i = 0; i < 64; i++) if (felem("A", i) != 0.0f) all_zero = false;
    CHECK(all_zero, "@alloc не занулил область");

    /* Каждый тензор начинается с границы 64 байт. */
    CHECK(run_str("*&A<f32:3,3> -> @alloc => $a;\n"
                  "*&B<f32:3,3> -> @alloc => $b;\n"
                  "*&C<f32:3,3> -> @alloc => $c;\n"), "три тензора");
    for (uint32_t i = 0; i < g_mod.n_tens; i++)
        CHECK(g_mod.tens[i].off % 64u == 0,
              "тензор %u лежит по смещению %u", i, g_mod.tens[i].off);
}

static void test_arith(void)
{
    SECTION("арифметика");

    /* fill + reduce: 16 элементов по 2.5 = 40. */
    CHECK(run_str("*&A<f32:4,4> -> @fill(2.5) => *&A;\n"
                  "*&A -> @reduce.add => $s;\n"), "fill+reduce");
    bool ok = false;
    CHECK(near(regval("s", &ok), 40.0) && ok, "sum = %g, ждали 40", regval("s", NULL));

    /* GEMM с проверяемым результатом: [2,3]x[3,2], всё по 2 и 3. */
    CHECK(run_str("*&A<f32:2,3> -> @fill(2.0) => *&A;\n"
                  "*&B<f32:3,2> -> @fill(3.0) => *&B;\n"
                  "*&A -> @mmul(*&B) => *&C<f32:2,2> [!no-alias];\n"), "gemm");
    for (uint32_t i = 0; i < 4; i++)
        CHECK(near(felem("C", i), 18.0), "C[%u] = %g, ждали 18", i, felem("C", i));

    /* relu отсекает отрицательные. */
    CHECK(run_str("*&A<f32:2,2> -> @fill(-3.0) => *&A;\n"
                  "*&A -> @relu => *&R<f32:2,2>;\n"
                  "*&A -> @abs  => *&B<f32:2,2>;\n"), "relu/abs");
    CHECK(near(felem("R", 0), 0.0), "relu(-3) = %g", felem("R", 0));
    CHECK(near(felem("B", 0), 3.0), "abs(-3) = %g", felem("B", 0));

    /* Поэлементные бинарные. */
    CHECK(run_str("*&A<f32:2,2> -> @fill(3.0) => *&A;\n"
                  "*&B<f32:2,2> -> @fill(4.0) => *&B;\n"
                  "*&A -> @add(*&B) => *&S<f32:2,2>;\n"
                  "*&A -> @mul(*&B) => *&M<f32:2,2>;\n"
                  "*&A -> @scale(0.5) => *&H<f32:2,2>;\n"), "add/mul/scale");
    CHECK(near(felem("S", 0),  7.0), "3+4 = %g", felem("S", 0));
    CHECK(near(felem("M", 0), 12.0), "3*4 = %g", felem("M", 0));
    CHECK(near(felem("H", 0),  1.5), "3*0.5 = %g", felem("H", 0));

    /* reduce.max. */
    CHECK(run_str("*&A<f32:4,4> -> @fill(-1.0) => *&A;\n"
                  "*&A -> @reduce.max => $m;\n"), "reduce.max");
    CHECK(near(regval("m", NULL), -1.0), "max = %g", regval("m", NULL));

    /* Приведение типа. */
    CHECK(run_str("*&A<f32:2,2> -> @fill(2.7) => *&A;\n"
                  "*&A -> @cast.i32 => *&I<i32:2,2>;\n"), "cast");
    {
        const int32_t *p = (const int32_t *)tensor_data("I", NULL);
        CHECK(p && p[0] == 2, "(i32)2.7 = %d", p ? p[0] : -1);
    }
}

static void test_literals(void)
{
    SECTION("целые литералы");

    /* Пул констант хранит сырые восемь байт. Когда-то все потребители читали
     * их как double, и целый литерал молча превращался в ноль: биты числа 72,
     * прочитанные как double, дают 3.6e-322. Теперь тип едет в поле aux. */

    CHECK(run_str("*&A<f32:4,4> -> @fill(3) => *&A;\n"
                  "*&A -> @reduce.add => $s;\n"), "@fill целым");
    CHECK(near(regval("s", NULL), 48.0), "@fill(3) над 16 элементами дал %g, ждали 48",
          regval("s", NULL));

    CHECK(run_str("*&A<f32:2,2> -> @fill(5.0) => *&A;\n"
                  "*&A -> @scale(3) => *&B<f32:2,2>;\n"), "@scale целым");
    CHECK(near(felem("B", 0), 15.0), "5 * 3 = %g", felem("B", 0));

    /* Поэлементная запись: единственный способ задать разные значения. */
    CHECK(run_str("*&H<i32:3> -> @alloc => $h;\n"
                  "72 => *&H[0];\n"
                  "73 => *&H[1];\n"
                  "33 => *&H[2];\n"
                  "*&H -> @reduce.add => $s;\n"), "поэлементная запись");
    {
        const int32_t *p = (const int32_t *)tensor_data("H", NULL);
        CHECK(p && p[0] == 72 && p[1] == 73 && p[2] == 33,
              "записалось [%d, %d, %d], ждали [72, 73, 33]",
              p ? p[0] : -1, p ? p[1] : -1, p ? p[2] : -1);
    }
    CHECK(near(regval("s", NULL), 178.0), "сумма %g, ждали 178", regval("s", NULL));

    /* Отрицательные литералы. */
    CHECK(run_str("*&A<f32:2,2> -> @fill(-7) => *&A;\n"
                  "*&A -> @reduce.max => $m;\n"), "отрицательный литерал");
    CHECK(near(regval("m", NULL), -7.0), "max = %g, ждали -7", regval("m", NULL));

    /* Приведение скаляра обязано менять значение, а не только ярлык. */
    CHECK(run_str("*&A<f32:2,2> -> @fill(2.75) => *&A;\n"
                  "*&A -> @reduce.max => $x;\n"
                  "$x -> @cast.i32 => $i;\n"), "cast скаляра");
    CHECK(near(regval("x", NULL), 2.75), "исходный скаляр %g", regval("x", NULL));
    CHECK(near(regval("i", NULL), 2.0),
          "@cast.i32 над 2.75 дал %g, ждали 2", regval("i", NULL));

    /* Отдельный элемент читается через свёртку: она принимает и скаляр,
     * потому что свёртка одного элемента — это сам элемент. */
    CHECK(run_str("*&M<f32:2,2> -> @alloc => $m;\n"
                  "1.0 => *&M[0,0];  2.0 => *&M[0,1];\n"
                  "3.0 => *&M[1,0];  4.0 => *&M[1,1];\n"
                  "*&M[1,1] -> @reduce.add => $e;\n"
                  "*&M[0,1] -> @reduce.max => $f;\n"), "чтение элемента");
    CHECK(near(regval("e", NULL), 4.0), "M[1,1] = %g, ждали 4", regval("e", NULL));
    CHECK(near(regval("f", NULL), 2.0), "M[0,1] = %g, ждали 2", regval("f", NULL));

    /* Преобразование double в целое вне диапазона — UB в C. Обязано
     * насыщаться, а не выдавать 0x80000000 от cvttsd2si. */
    CHECK(run_str("*&A<f32:4> -> @fill(-5.0) => *&A;\n"
                  "*&A -> @cast.u64 => *&U<u64:4>;\n"
                  "*&U -> @reduce.add => $u;\n"), "отрицательное в u64");
    CHECK(near(regval("u", NULL), 0.0),
          "-5.0 в u64 дало %g, ждали 0", regval("u", NULL));

    CHECK(run_str("*&A<f32:4> -> @fill(1.0e30) => *&A;\n"
                  "*&A -> @cast.i32 => *&I<i32:4>;\n"
                  "*&I -> @reduce.max => $i;\n"), "переполнение i32");
    CHECK(near(regval("i", NULL), 2147483647.0),
          "1e30 в i32 дало %g, ждали 2147483647", regval("i", NULL));
}

static void test_views(void)
{
    SECTION("срезы и транспозиция");

    /* Столбец матрицы: шаг 4 элемента. Заполним строки разными значениями
     * через срезы и проверим сумму столбца. */
    CHECK(run_str("*&A<f32:4,4> -> @fill(2.0) => *&A;\n"
                  "*&A[.., 0] -> @reduce.add => $col;\n"
                  "*&A[0, ..] -> @reduce.add => $row;\n"), "срезы");
    CHECK(near(regval("col", NULL), 8.0), "сумма столбца = %g, ждали 8",
          regval("col", NULL));
    CHECK(near(regval("row", NULL), 8.0), "сумма строки = %g, ждали 8",
          regval("row", NULL));

    /* Транспозиция без копии: адрес тот же, шаги переставлены. */
    CHECK(run_str("*&A<f32:2,3> -> @fill(5.0) => *&A;\n"
                  "*&A -> @transpose -> @pack => *&T<f32:3,2>;\n"), "transpose+pack");
    {
        const SmpTensor *t = NULL;
        CHECK(tensor_data("T", &t) != NULL, "нет тензора T");
        CHECK(t && t->shape[0] == 3 && t->shape[1] == 2, "форма T");
        for (uint32_t i = 0; i < 6; i++)
            CHECK(near(felem("T", i), 5.0), "T[%u] = %g", i, felem("T", i));
    }

    /* Транспозиция реально переставляет элементы, а не только форму.
     * A = [[1,1,1],[9,9,9]] через два среза. */
    CHECK(run_str("*&A<f32:2,3> -> @fill(1.0) => *&A;\n"
                  "*&A[1, ..] -> @scale(9.0) => *&A[1, ..];\n"
                  "*&A -> @transpose -> @pack => *&T<f32:3,2>;\n"), "перестановка");
    CHECK(near(felem("A", 0), 1.0) && near(felem("A", 3), 9.0),
          "исходник: A[0]=%g A[3]=%g", felem("A", 0), felem("A", 3));
    CHECK(near(felem("T", 0), 1.0) && near(felem("T", 1), 9.0),
          "транспонированный: T[0]=%g T[1]=%g", felem("T", 0), felem("T", 1));
}

static void test_traps(void)
{
    SECTION("ловушки рантайма");

    /* Переполнение f32 под ?strict обязано остановить исполнение. */
    CHECK(!run_str("*&A<f32:8,8> -> @fill(1.0e38) => *&A;\n"
                   "*&A -> @scale(1.0e30) => *&B<f32:8,8>;\n"
                   "*&B -> @reduce.add => $s [?strict];\n"),
          "переполнение под ?strict прошло незамеченным");
    CHECK(strstr(g_out, "E0602") != NULL, "нет E0602");
    CHECK(strstr(g_out, "ДАМП РЕГИСТРОВ") != NULL, "нет дампа регистров");
    CHECK(strstr(g_out, "MXCSR") != NULL, "в дампе нет MXCSR");
    CHECK(g_vm.trapped, "флаг ловушки не выставлен");

    /* Без ?strict та же программа доходит до конца. */
    CHECK(run_str("*&A<f32:8,8> -> @fill(1.0e38) => *&A;\n"
                  "*&A -> @scale(1.0e30) => *&B<f32:8,8>;\n"
                  "*&B -> @reduce.add => $s;\n"),
          "без ?strict исполнение обязано продолжиться");
    CHECK(isinf(regval("s", NULL)), "ждали бесконечность, вышло %g", regval("s", NULL));
}

static void test_fp_modes(void)
{
    SECTION("режимы MXCSR");

    const uint32_t before = smp_fpu_get_mxcsr();

    /* MXCSR обязан остаться нетронутым: маски исключений FPU в нём же, и
     * запись нуля туда убила бы процесс на первой неточной операции. */
    {
        SmpVM fresh;
        memset(&fresh, 0, sizeof fresh);
        smp_vm_release(&fresh);
        CHECK(smp_fpu_get_mxcsr() == before,
              "release на непроинициализированной VM испортил MXCSR: 0x%04X -> 0x%04X",
              before, smp_fpu_get_mxcsr());
        CHECK((smp_fpu_get_mxcsr() & 0x1F80u) == 0x1F80u,
              "сняты маски исключений FPU: 0x%04X", smp_fpu_get_mxcsr());
    }

    CHECK(run_str("*&A<f32:4,4> -> @fill(1.0) => *&A;\n"
                  "*&A -> @relu => *&B<f32:4,4> [~flush-to-zero];\n"), "~ftz");

    /* Атрибут дошёл до исполнения: к концу прогона FTZ был включён. */
    CHECK(g_vm.cur_fp_flags == SMP_IF_FTZ,
          "~flush-to-zero не доехал до MXCSR: cur_fp_flags=0x%02X", g_vm.cur_fp_flags);

    /* VM восстанавливает MXCSR к СВОЕЙ точке входа, а не к произвольному
     * прошлому: сравнивать нужно именно с ней. */
    const uint32_t entry = g_vm.entry_mxcsr;
    smp_vm_release(&g_vm);
    const uint32_t after = smp_fpu_get_mxcsr();
    CHECK(after == entry, "MXCSR не восстановлен: вход 0x%04X, после 0x%04X",
          entry, after);
    CHECK((after & (1u << 15)) == 0, "FTZ остался включённым после релиза");
    CHECK((after & 0x1F80u) == 0x1F80u,
          "после релиза сняты маски исключений FPU: 0x%04X", after);
}

static void test_regdump(void)
{
    SECTION("дамп регистров");

    CHECK(!run_str("*&A<f32:4,4> -> @fill(1.0e38) => *&A;\n"
                   "*&A -> @scale(1.0e30) -> @reduce.add => $total [?strict];\n"),
          "ждали ловушку");

    /* Имена регистров обязаны быть человеческими: пользователь писал $total,
     * а не r0. */
    CHECK(g_mod.reg_names != NULL && g_mod.n_reg_names >= 1,
          "модуль не несёт имён регистров");
    bool named = false;
    for (uint32_t i = 0; i < g_mod.n_reg_names; i++)
        if (g_mod.reg_names[i] &&
            strcmp(smp_module_str(&g_mod, g_mod.reg_names[i]), "total") == 0)
            named = true;
    CHECK(named, "имя $total не попало в модуль");

    CHECK(strstr(g_out, "ядра = ") != NULL, "в дампе нет имени ветки ядер");
    CHECK(strstr(g_out, "опкод = redadd") != NULL, "в дампе нет опкода");
    CHECK(strstr(g_out, "t.smpc:2") != NULL, "ловушка не привязана к строке исходника");
}

static void test_roundtrip(void)
{
    SECTION("исполнение из .s3b");

    CHECK(run_str("*&A<f32:4,4> -> @fill(2.0) => *&A;\n"
                  "*&B<f32:4,4> -> @fill(3.0) => *&B;\n"
                  "*&A -> @mmul(*&B) => *&C<f32:4,4> [!no-alias];\n"
                  "*&C -> @reduce.add => $s;\n"), "исходная программа");
    const double direct = regval("s", NULL);
    CHECK(near(direct, 4.0 * 4.0 * (4.0 * 6.0)), "прямой прогон: %g", direct);

    /* Тот же модуль через файл обязан дать тот же результат. */
    const char *path = "build/test_vm_tmp.s3b";
    CHECK(smp_s3b_write(&g_mod, path) == SMP_OK, "запись .s3b");

    smp_vm_release(&g_vm);
    SmpArena a2;
    CHECK(smp_arena_init(&a2, 16u << 20, 8, "load") == SMP_OK, "арена загрузки");

    SmpModule m2;
    CHECK(smp_s3b_read(&m2, path, &a2, &g_diag) == SMP_OK, "чтение .s3b");

    SmpVM vm2;
    CHECK(smp_vm_init(&vm2, &m2, &g_diag) == SMP_OK, "VM из файла");
    CHECK(smp_vm_run(&vm2) == SMP_OK, "прогон из файла");

    double from_file = NAN;
    for (uint32_t i = 0; i < vm2.n_regs && i < m2.n_reg_names; i++)
        if (m2.reg_names[i] &&
            strcmp(smp_module_str(&m2, m2.reg_names[i]), "s") == 0)
            from_file = vm2.regs[i].s.f;

    CHECK(near(from_file, direct), "из файла %g, напрямую %g", from_file, direct);
    CHECK(vm2.n_executed == g_vm.n_executed || g_vm.n_executed == 0,
          "разное число инструкций");

    smp_vm_release(&vm2);
    smp_arena_release(&a2);
    remove(path);
}

/* ========================================================================== */
/*  Слияние поэлементных стадий                                               */
/* ========================================================================== */

/* Сколько инструкций помечено под слияние. */
static uint32_t count_fuse(void)
{
    uint32_t n = 0;
    for (uint32_t i = 0; i < g_mod.n_code; i++)
        if (g_mod.code[i].flags & SMP_IF_FUSE) n++;
    return n;
}

/* Прогоняет уже собранный модуль ещё раз, сняв все пометки слияния. Это и есть
 * эталон: без флага VM обязана считать то же самое, инструкция за инструкцией.
 * Флаг — утверждение компилятора об уже мёртвом буфере, а не про арифметику,
 * поэтому расхождение означало бы, что слияние меняет результат. */
static bool rerun_unfused(void)
{
    SmpInstr *code = (SmpInstr *)g_mod.code;
    for (uint32_t i = 0; i < g_mod.n_code; i++)
        code[i].flags &= (uint8_t)~SMP_IF_FUSE;

    smp_vm_release(&g_vm);
    if (smp_vm_init(&g_vm, &g_mod, &g_diag) != SMP_OK) return false;
    return smp_vm_run(&g_vm) == SMP_OK;
}

/* Сверяет тензор с его же значениями до пересчёта. */
static void check_same(const char *name, const double *want, uint32_t n)
{
    const SmpTensor *t = NULL;
    const void      *p = tensor_data(name, &t);
    if (!p || !t) { CHECK(0, "тензор %s не найден", name); return; }

    bool ok = true;
    for (uint32_t i = 0; i < n && i < t->nelem; i++) {
        const double got = (t->dtype == SMP_DT_F64)
                               ? ((const double *)p)[i]
                               : (double)((const float *)p)[i];
        if (!near(got, want[i])) ok = false;
    }
    CHECK(ok, "слияние изменило %s", name);
}

static void snapshot(const char *name, double *out, uint32_t n)
{
    const SmpTensor *t = NULL;
    const void      *p = tensor_data(name, &t);
    if (!p || !t) return;
    for (uint32_t i = 0; i < n && i < t->nelem; i++)
        out[i] = (t->dtype == SMP_DT_F64) ? ((const double *)p)[i]
                                          : (double)((const float *)p)[i];
}

static void test_fusion(void)
{
    SECTION("слияние поэлементных стадий");

    static double want[64];

    /* Три стадии подряд. Вход намеренно со знаком, чтобы relu действительно
     * срезал, а не проходил насквозь. */
    static const char *three =
        "[#arena:0] *&A<f32:64> -> @fill(-3.0) => *&A;\n"
        "[#arena:0] *&B<f32:64> -> @fill(5.0) => *&B;\n"
        "[#simd:v256] *&A -> @add(*&B) -> @relu -> @scale(2.0)"
        " => *&D<f32:64> [!no-alias];\n";
    if (run_str(three)) {
        CHECK(count_fuse() == 2, "ждали 2 пометки слияния, нашли %u", count_fuse());
        snapshot("D", want, 64);
        CHECK(near(want[0], 4.0), "(-3+5)*2 дало %g, ждали 4", want[0]);
        CHECK(rerun_unfused(), "прогон без слияния не дошёл до halt");
        check_same("D", want, 64);
    }

    /* Цепочка, где relu обязан обнулить: (-3 + 1) < 0. */
    static const char *clipped =
        "[#arena:0] *&A<f32:64> -> @fill(-3.0) => *&A;\n"
        "[#arena:0] *&B<f32:64> -> @fill(1.0) => *&B;\n"
        "[#simd:v256] *&A -> @add(*&B) -> @relu -> @scale(7.0)"
        " => *&D<f32:64> [!no-alias];\n";
    if (run_str(clipped)) {
        snapshot("D", want, 64);
        CHECK(near(want[0], 0.0), "relu не срезал: %g", want[0]);
        CHECK(rerun_unfused(), "прогон без слияния не дошёл до halt");
        check_same("D", want, 64);
    }

    /* f64 мимо векторной ветки — считает скалярное слитое ядро. */
    static const char *wide =
        "[#arena:0] *&A<f64:32> -> @fill(-2.0) => *&A;\n"
        "[#arena:0] *&B<f64:32> -> @fill(6.0) => *&B;\n"
        "*&A -> @add(*&B) -> @abs -> @scale(3.0) => *&D<f64:32> [!no-alias];\n";
    if (run_str(wide)) {
        snapshot("D", want, 32);
        CHECK(near(want[0], 12.0), "|(-2+6)|*3 дало %g, ждали 12", want[0]);
        CHECK(rerun_unfused(), "прогон без слияния не дошёл до halt");
        check_same("D", want, 32);
    }

    /* Цепочка длиннее SMP_FUSE_MAX обязана остаться правильной: лишние стадии
     * просто исполнятся отдельно. */
    static const char *toolong =
        "[#arena:0] *&A<f32:16> -> @fill(1.0) => *&A;\n"
        "*&A -> @scale(2.0) -> @scale(2.0) -> @scale(2.0) -> @scale(2.0)"
        " -> @scale(2.0) -> @scale(2.0) -> @scale(2.0) -> @scale(2.0)"
        " -> @scale(2.0) -> @scale(2.0) => *&D<f32:16> [!no-alias];\n";
    if (run_str(toolong)) {
        snapshot("D", want, 16);
        CHECK(near(want[0], 1024.0), "2^10 дало %g", want[0]);
        CHECK(rerun_unfused(), "прогон без слияния не дошёл до halt");
        check_same("D", want, 16);
    }

    /* Счётчик исполненного не должен врать: слияние экономит проходы по
     * памяти, а не инструкции программы. */
    if (run_str(three)) {
        const uint64_t fused = g_vm.n_executed;
        CHECK(rerun_unfused(), "прогон без слияния не дошёл до halt");
        CHECK(g_vm.n_executed == fused,
              "исполнено %llu против %llu без слияния",
              (unsigned long long)g_vm.n_executed, (unsigned long long)fused);
    }
    /* --- свёрточный хвост ---------------------------------------------------
     *
     * Свёртка закрывает цепочку, и промежуточный буфер не пишется вовсе.
     * Проверяем тем же способом: со снятыми флагами обязано выйти то же самое. */

    /* @relu -> @reduce.add: половина значений срезается в ноль. */
    static const char *red_add =
        "[#arena:0] *&A<f32:64> -> @fill(-4.0) => *&A;\n"
        "[#arena:0] *&B<f32:64> -> @fill(7.0) => *&B;\n"
        "[#simd:v256] *&A -> @add(*&B) -> @relu -> @reduce.add => $s;\n"
        "*&A -> @emit.num => $n;\n";
    if (run_str(red_add)) {
        bool ok = false;
        const double fused = regval("s", &ok);
        CHECK(ok, "регистр $s не найден");
        CHECK(count_fuse() == 2, "ждали 2 пометки слияния, нашли %u", count_fuse());
        CHECK(near(fused, 3.0 * 64.0), "(-4+7)*64 дало %g, ждали 192", fused);

        CHECK(rerun_unfused(), "прогон без слияния не дошёл до halt");
        const double plain = regval("s", &ok);
        CHECK(ok && near(plain, fused),
              "слияние изменило свёртку: %g против %g", fused, plain);
    }

    /* @abs -> @reduce.max поверх отрицательных: максимум обязан взяться из
     * модулей, а не из исходных значений. */
    static const char *red_max =
        "[#arena:0] *&A<f32:64> -> @fill(-9.0) => *&A;\n"
        "[#simd:v256] *&A -> @abs -> @reduce.max => $m;\n"
        "*&A -> @emit.num => $n;\n";
    if (run_str(red_max)) {
        bool ok = false;
        const double fused = regval("m", &ok);
        CHECK(ok, "регистр $m не найден");
        CHECK(count_fuse() == 1, "ждали 1 пометку слияния, нашли %u", count_fuse());
        CHECK(near(fused, 9.0), "max(|-9|) дало %g, ждали 9", fused);

        CHECK(rerun_unfused(), "прогон без слияния не дошёл до halt");
        const double plain = regval("m", &ok);
        CHECK(ok && near(plain, fused),
              "слияние изменило свёртку: %g против %g", fused, plain);
    }

    /* f64 идёт мимо векторной ветки — считает скалярное слитое ядро. */
    static const char *red_wide =
        "[#arena:0] *&A<f64:32> -> @fill(-1.5) => *&A;\n"
        "*&A -> @abs -> @scale(2.0) -> @reduce.add => $s;\n"
        "*&A -> @emit.num => $n;\n";
    if (run_str(red_wide)) {
        bool ok = false;
        const double fused = regval("s", &ok);
        CHECK(ok && near(fused, 3.0 * 32.0),
              "|-1.5|*2*32 дало %g, ждали 96", fused);
        CHECK(rerun_unfused(), "прогон без слияния не дошёл до halt");
        const double plain = regval("s", &ok);
        CHECK(ok && near(plain, fused),
              "слияние изменило свёртку f64: %g против %g", fused, plain);
    }

    /* Счётчик исполненного обязан учесть и саму свёртку. */
    if (run_str(red_max)) {
        const uint64_t fused = g_vm.n_executed;
        CHECK(rerun_unfused(), "прогон без слияния не дошёл до halt");
        CHECK(g_vm.n_executed == fused,
              "со свёрткой исполнено %llu против %llu",
              (unsigned long long)g_vm.n_executed, (unsigned long long)fused);
    }
}


/* ========================================================================== */

/* ========================================================================== */
/*  Эпилог GEMM                                                               */
/* ========================================================================== */

/* Цепочка за @mmul считается в цикле выгрузки тайла. Проверка та же, что у
 * поэлементного слияния: со снятыми флагами обязано выйти то же самое.
 * Замером это не проверить — разброс bench больше ожидаемого выигрыша. */
static void test_gemm_epilogue(void)
{
    SECTION("эпилог GEMM");

    static double want[64];

    /* Одна стадия за произведением: 8 * (1.5 * 2) = 24, вдвое меньше — 12. */
    static const char *scaled =
        "[#arena:0] *&A<f32:8,8> -> @fill(1.5) => *&A;\n"
        "[#arena:0] *&B<f32:8,8> -> @fill(2.0) => *&B;\n"
        "[#simd:v256] *&A -> @mmul(*&B) -> @scale(0.5)"
        " => *&C<f32:8,8> [!no-alias];\n";
    if (run_str(scaled)) {
        CHECK(count_fuse() == 1, "ждали 1 пометку слияния, нашли %u", count_fuse());
        snapshot("C", want, 64);
        CHECK(near(want[0], 12.0), "8*(1.5*2)/2 дало %g, ждали 12", want[0]);
        CHECK(rerun_unfused(), "прогон без слияния не дошёл до halt");
        check_same("C", want, 64);
    }

    /* relu обязан срезать: произведение отрицательное. */
    static const char *clipped =
        "[#arena:0] *&A<f32:8,8> -> @fill(-1.5) => *&A;\n"
        "[#arena:0] *&B<f32:8,8> -> @fill(2.0) => *&B;\n"
        "[#simd:v256] *&A -> @mmul(*&B) -> @relu"
        " => *&C<f32:8,8> [!no-alias];\n";
    if (run_str(clipped)) {
        snapshot("C", want, 64);
        CHECK(near(want[0], 0.0), "relu не срезал произведение: %g", want[0]);
        CHECK(rerun_unfused(), "прогон без слияния не дошёл до halt");
        check_same("C", want, 64);
    }

    /* Две стадии, вторая с тензорным операндом: 0 + 100. */
    static const char *biased =
        "[#arena:0] *&A<f32:8,8> -> @fill(-1.5) => *&A;\n"
        "[#arena:0] *&B<f32:8,8> -> @fill(2.0) => *&B;\n"
        "[#arena:0] *&E<f32:8,8> -> @fill(100.0) => *&E;\n"
        "[#simd:v256] *&A -> @mmul(*&B) -> @relu -> @add(*&E)"
        " => *&C<f32:8,8> [!no-alias];\n";
    if (run_str(biased)) {
        CHECK(count_fuse() == 2, "ждали 2 пометки слияния, нашли %u", count_fuse());
        snapshot("C", want, 64);
        CHECK(near(want[0], 100.0), "relu(-24)+100 дало %g, ждали 100", want[0]);
        CHECK(rerun_unfused(), "прогон без слияния не дошёл до halt");
        check_same("C", want, 64);
    }

    /* K = 600 при KC = 256 — три k-блока. Эпилог обязан лечь ровно один раз:
     * применённый на каждом, дал бы 600/8, а не 600/2. */
    static const char *kblocks =
        "[#arena:0] *&A<f32:40,600> -> @fill(1.0) => *&A;\n"
        "[#arena:0] *&B<f32:600,40> -> @fill(1.0) => *&B;\n"
        "[#simd:v256] *&A -> @mmul(*&B) -> @scale(0.5)"
        " => *&C<f32:40,40> [!no-alias];\n";
    if (run_str(kblocks)) {
        snapshot("C", want, 40);
        CHECK(near(want[0], 300.0), "600/2 дало %g, ждали 300", want[0]);
        CHECK(rerun_unfused(), "прогон без слияния не дошёл до halt");
        check_same("C", want, 40);
    }

    /* Свёртка за произведением цепочку не начинает: приёмника у GEMM тогда
     * нет. Отказ обязан остаться правильным, а не потерять результат. */
    static const char *reduced =
        "[#arena:0] *&A<f32:8,8> -> @fill(1.5) => *&A;\n"
        "[#arena:0] *&B<f32:8,8> -> @fill(2.0) => *&B;\n"
        "[#simd:v256] *&A -> @mmul(*&B) -> @relu -> @reduce.add => $s;\n";
    if (run_str(reduced)) {
        bool ok = false;
        const double fused = regval("s", &ok);
        CHECK(ok, "регистр $s не найден");
        CHECK(near(fused, 24.0 * 64.0), "сумма дала %g, ждали 1536", fused);
        CHECK(rerun_unfused(), "прогон без слияния не дошёл до halt");
        const double plain = regval("s", &ok);
        CHECK(ok && near(plain, fused),
              "слияние изменило свёртку за mmul: %g против %g", fused, plain);
    }

    /* Счётчик исполненного не должен врать и здесь. */
    if (run_str(biased)) {
        const uint64_t fused = g_vm.n_executed;
        CHECK(rerun_unfused(), "прогон без слияния не дошёл до halt");
        CHECK(g_vm.n_executed == fused,
              "исполнено %llu против %llu без слияния",
              (unsigned long long)g_vm.n_executed, (unsigned long long)fused);
    }
}

int main(void)
{
    smp_console_setup();
    fprintf(stderr, "SMPC3 tests :: Ф5\n\n");

    if (smp_arena_init(&g_arena, 64u << 20, 9, "vmtest") != SMP_OK) {
        fprintf(stderr, "арена не поднялась\n");
        return 70;
    }
    g_sink = tmpfile();
    g_emit = tmpfile();
    if (!g_emit) { fprintf(stderr, "tmpfile для вывода недоступен\n"); return 70; }
    if (!g_sink) { fprintf(stderr, "tmpfile недоступен\n"); return 70; }

    test_dispatch();
    test_memory();
    test_arith();
    test_literals();
    test_emit();
    test_views();
    test_traps();
    test_fp_modes();
    test_regdump();
    test_roundtrip();
    test_fusion();
    test_gemm_epilogue();

    smp_vm_release(&g_vm);
    fclose(g_sink);
    smp_arena_release(&g_arena);
    return REPORT();
}
