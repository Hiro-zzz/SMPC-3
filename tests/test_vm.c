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
/* Привязки тензоров к файлам для текущего прогона. Проставляются тестом до
 * run_str; по умолчанию их нет, и @load с @store честно падают. */
static SmpBind  g_binds[4];
static uint32_t g_n_binds;

/* Хранилище для текущего прогона; NULL — работают привязки. */
static const SmpStore *g_store;

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

    /* Развёртка [#repeat:N] идёт до семантики — как в настоящем конвейере.
     * Без этого тест с [#repeat] проверял бы не тот компилятор. */
    uint32_t n_exp = 0;
    smp_ast_expand(&prog, &g_arena, &g_diag, &n_exp);
    if (n_exp) { CHECK(0, "развёртка: %s", text); return false; }

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
    smp_vm_bind(&g_vm, g_binds, g_n_binds);
    smp_vm_store(&g_vm, g_store);
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
        if (out) *out = t;
        return smp_vm_tensor_data(&g_vm, t);
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
              "тензор %u лежит по смещению %llu", i, (unsigned long long)g_mod.tens[i].off);
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

    /* Приёмник со срезом по регистру. Раньше индекс-регистр приёмника
     * терялся: результат ложился в строку 0, а не в строку $t. Так пишется
     * строка KV-кэша по номеру позиции, поэтому проверены все три пути:
     * копия в конце, прямая запись последней стадией и скаляр. */
    CHECK(run_str("*&E<f32:4,8> -> @fill(1.0) => *&E;\n"
                  "2 => $t;\n"
                  "*&E[$t, ..] -> @scale(3.0) => *&E[$t, ..];\n"), "срез-приёмник, копия");
    CHECK(near(felem("E", 0), 1.0) && near(felem("E", 16), 3.0) && near(felem("E", 24), 1.0),
          "копия: E[0]=%g E[16]=%g E[24]=%g", felem("E", 0), felem("E", 16), felem("E", 24));

    CHECK(run_str("*&E<f32:4,8> -> @fill(1.0) => *&E;\n"
                  "*&X<f32:8> -> @fill(-5.0) => *&X;\n"
                  "3 => $t;\n"
                  "*&X -> @abs => *&E[$t, ..];\n"), "срез-приёмник, прямая запись");
    CHECK(near(felem("E", 0), 1.0) && near(felem("E", 24), 5.0) && near(felem("E", 31), 5.0),
          "прямая: E[0]=%g E[24]=%g E[31]=%g", felem("E", 0), felem("E", 24), felem("E", 31));

    CHECK(run_str("*&E<f32:4,8> -> @fill(1.0) => *&E;\n"
                  "1 => $t;\n"
                  "*&E[$t, ..] -> @reduce.add => $s;\n"
                  "$s => *&E[$t, 0];\n"), "срез-приёмник, скаляр");
    CHECK(near(felem("E", 0), 1.0) && near(felem("E", 8), 8.0) && near(felem("E", 9), 1.0),
          "скаляр: E[0]=%g E[8]=%g E[9]=%g", felem("E", 0), felem("E", 8), felem("E", 9));
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

/* ========================================================================== */
/*  Обмен с файлами                                                           */
/* ========================================================================== */

/* Ровно то, что делает Python через array.tofile: сырые байты, без заголовка. */
static bool write_f32(const char *path, const float *v, size_t n)
{
    FILE *f = fopen(path, "wb");
    if (!f) return false;
    const size_t put = fwrite(v, sizeof(float), n, f);
    return fclose(f) == 0 && put == n;
}

static bool read_f32(const char *path, float *v, size_t n)
{
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    const size_t got = fread(v, sizeof(float), n, f);
    fclose(f);
    return got == n;
}

static bool saw_code(const char *code) { return strstr(g_out, code) != NULL; }

static void test_fileio(void)
{
    SECTION("обмен с файлами");

    const char *in  = "build/test_io_in.bin";
    const char *out = "build/test_io_out.bin";

    float src[16], got[16];
    for (uint32_t i = 0; i < 16; i++) src[i] = (float)i - 8.0f;   /* со знаком */
    if (!write_f32(in, src, 16)) { CHECK(0, "не записался входной файл"); return; }

    /* --- туда и обратно --- */
    g_binds[0].name = "A"; g_binds[0].path = in;  g_binds[0].write = false;
    g_binds[1].name = "C"; g_binds[1].path = out; g_binds[1].write = true;
    g_n_binds = 2;

    const char *roundtrip =
        "[#arena:0] *&A<f32:16> -> @load  => *&A;\n"
        "[#arena:0] *&C<f32:16> -> @alloc => *&C;\n"
        "[#simd:v256] *&A -> @relu -> @scale(2.0) => *&C [!no-alias];\n"
        "*&C -> @store => $put;\n";
    if (run_str(roundtrip)) {
        CHECK(read_f32(out, got, 16), "выходной файл не прочитался");

        bool ok = true;
        for (uint32_t i = 0; i < 16; i++) {
            const float want = (src[i] > 0.0f ? src[i] : 0.0f) * 2.0f;
            if (!near(got[i], want)) ok = false;
        }
        CHECK(ok, "через файлы посчиталось не то");

        bool found = false;
        const double put = regval("put", &found);
        CHECK(found && near(put, 64.0),
              "@store вернул %g байт, ждали 64", put);
    }

    /* --- вход обязан долетать до арены, а не просто не падать --- */
    g_n_binds = 1;   /* только A на чтение */
    if (run_str("[#arena:0] *&A<f32:16> -> @load => *&A;\n"
                "*&A -> @reduce.add => $s;\n")) {
        double want = 0.0;
        for (uint32_t i = 0; i < 16; i++) want += (double)src[i];
        bool found = false;
        const double s = regval("s", &found);
        CHECK(found && near(s, want), "сумма прочитанного %g, ждали %g", s, want);
    }

    /* --- привязки нет --- */
    g_n_binds = 0;
    CHECK(!run_str("[#arena:0] *&A<f32:16> -> @load => *&A;\n"
                   "*&A -> @reduce.add => $s;\n"),
          "прогон без привязки обязан упасть");
    CHECK(saw_code("E0606"), "нет E0606 при отсутствующей привязке");

    /* --- размер не сошёлся ---
     * Заголовка у файла нет намеренно, поэтому единственная защита от
     * подсунутого не того файла — точное совпадение размера. */
    g_binds[0].name = "A"; g_binds[0].path = in; g_binds[0].write = false;
    g_n_binds = 1;
    CHECK(!run_str("[#arena:0] *&A<f32:9> -> @load => *&A;\n"
                   "*&A -> @reduce.add => $s;\n"),
          "файл на 64 байта в тензор на 36 обязан быть отказом");
    CHECK(saw_code("E0607"), "нет E0607 при несовпадении размера");

    /* --- направление привязки различается ---
     * Тот же файл, привязанный на запись, не годится для @load: иначе
     * опечатка в --in/--out молча читала бы не то. */
    g_binds[0].name = "A"; g_binds[0].path = in; g_binds[0].write = true;
    g_n_binds = 1;
    CHECK(!run_str("[#arena:0] *&A<f32:16> -> @load => *&A;\n"
                   "*&A -> @reduce.add => $s;\n"),
          "@load не должен брать привязку, объявленную на запись");

    remove(in);
    remove(out);
}

/* Хранилище в памяти: как рабочее пространство ядра, только на четыре
 * объекта. full=true — любое write отказывает, как у исчерпанной памяти. */
typedef struct {
    char      name[16];
    SmpTensor desc;
    uint8_t   data[256];
    uint64_t  bytes;
    bool      used;
} FakeObj;

static FakeObj g_objs[4];
static bool    g_store_full;

static FakeObj *fake_find_obj(const char *name)
{
    for (int i = 0; i < 4; i++)
        if (g_objs[i].used && strcmp(g_objs[i].name, name) == 0) return &g_objs[i];
    return NULL;
}

static bool fake_find(void *ctx, const char *name, SmpTensor *desc)
{
    (void)ctx;
    FakeObj *o = fake_find_obj(name);
    if (!o) return false;
    *desc = o->desc;
    return true;
}

static void fake_read(void *ctx, const char *name, void *dst, uint64_t bytes)
{
    (void)ctx;
    memcpy(dst, fake_find_obj(name)->data, (size_t)bytes);
}

static bool fake_write(void *ctx, const char *name, const SmpTensor *desc,
                       const void *src, uint64_t bytes)
{
    (void)ctx;
    if (g_store_full || bytes > sizeof g_objs[0].data) return false;
    FakeObj *o = fake_find_obj(name);
    for (int i = 0; !o && i < 4; i++)
        if (!g_objs[i].used) o = &g_objs[i];
    if (!o) return false;
    snprintf(o->name, sizeof o->name, "%s", name);
    o->desc  = *desc;
    o->bytes = bytes;
    o->used  = true;
    memcpy(o->data, src, (size_t)bytes);
    return true;
}

static void test_store(void)
{
    SECTION("обмен с хранилищем");

    static const SmpStore store = { NULL, fake_find, fake_read, fake_write };
    memset(g_objs, 0, sizeof g_objs);
    g_store_full = false;
    g_store      = &store;
    g_n_binds    = 0;       /* с хранилищем привязки не нужны вовсе */

    /* --- одна программа кладёт, другая достаёт ---
     * Ради этого хранилище и есть: объект переживает программу. */
    if (run_str("[#arena:0] *&V<f32:16> -> @fill(1.5) => *&V;\n"
                "*&V -> @store => $put;\n")) {
        bool found = false;
        const double put = regval("put", &found);
        CHECK(found && near(put, 64.0), "@store вернул %g байт, ждали 64", put);
        CHECK(fake_find_obj("V") != NULL, "объект V не появился в хранилище");
    }
    if (run_str("[#arena:0] *&V<f32:16> -> @load => *&V;\n"
                "*&V -> @reduce.add => $s;\n")) {
        bool found = false;
        const double sum = regval("s", &found);
        CHECK(found && near(sum, 24.0), "сумма прочитанного %g, ждали 24", sum);
    }

    /* --- объекта нет --- */
    CHECK(!run_str("[#arena:0] *&W<f32:4> -> @load => *&W;\n"
                   "*&W -> @reduce.add => $s;\n"),
          "@load несуществующего объекта обязан упасть");
    CHECK(saw_code("E0608"), "нет E0608 для отсутствующего объекта");
    CHECK(!saw_code("E0606"), "с хранилищем не должно быть речи о привязках");

    /* --- те же 64 байта, но другой тип ---
     * Файл этого не различил бы; хранилище помнит, что положили f32:16. */
    CHECK(!run_str("[#arena:0] *&V<f64:8> -> @load => *&V;\n"
                   "*&V -> @reduce.add => $s;\n"),
          "f64:8 вместо f32:16 обязан быть отказом");
    CHECK(saw_code("E0609"), "нет E0609 при несовпадении типа");
    CHECK(strstr(g_out, "f32:16") && strstr(g_out, "f64:8"),
          "в сообщении нет обеих форм");

    /* --- та же форма, другая раскладка осей --- */
    CHECK(!run_str("[#arena:0] *&V<f32:4,4> -> @load => *&V;\n"
                   "*&V -> @reduce.add => $s;\n"),
          "f32:4,4 вместо f32:16 обязан быть отказом");
    CHECK(saw_code("E0609"), "нет E0609 при несовпадении формы");

    /* --- хранилище отказало --- */
    g_store_full = true;
    CHECK(!run_str("[#arena:0] *&Z<f32:4> -> @fill(1.0) => *&Z;\n"
                   "*&Z -> @store => $put;\n"),
          "отказ хранилища обязан остановить программу");
    CHECK(saw_code("E0610"), "нет E0610 при отказе хранилища");

    g_store = NULL;
}

/* Хранилище, которое отдаёт объект по адресу. */
static uint32_t g_mapped;

static const void *fake_map(void *ctx, const char *name)
{
    (void)ctx;
    FakeObj *o = fake_find_obj(name);
    g_mapped++;
    return o ? o->data : NULL;
}

/* Объект W<f32:3,4> со значениями 0..11 прямо в хранилище. */
static void put_w(float base)
{
    FakeObj *o = &g_objs[0];
    memset(o, 0, sizeof *o);
    snprintf(o->name, sizeof o->name, "W");
    const uint32_t shape[2] = { 3, 4 };
    smp_tensor_dense(&o->desc, SMP_DT_F32, 2, shape);
    for (int i = 0; i < 12; i++) ((float *)o->data)[i] = base + (float)i;
    o->bytes = 48;
    o->used  = true;
}

/* Тензор, который программа только загружает, — вид: лежит там, где его
 * держит хранилище, и не копируется ни на одном прогоне. */
static void test_store_views(void)
{
    SECTION("виды: @load без копии");

    static const SmpStore store = { NULL, fake_find, fake_read, fake_write, fake_map };
    memset(g_objs, 0, sizeof g_objs);
    g_store_full = false;
    g_store      = &store;
    g_n_binds    = 0;
    put_w(0.0f);

    static const char prog[] =
        "[#arena:0] *&W<f32:3,4> -> @load => *&W;\n"
        "2 => $i;\n"
        "*&W[$i, ..] -> @reduce.add => $row;\n"
        "*&W -> @reduce.add => $all;\n";

    g_mapped = 0;
    if (run_str(prog)) {
        const SmpTensor *t = NULL;
        CHECK(tensor_data("W", &t) == g_objs[0].data, "W — не адрес объекта хранилища");
        CHECK(t && (t->flags & SMP_TF_EXTERN) && (t->flags & SMP_TF_READONLY),
              "у W нет флагов вида");
        CHECK(g_mod.view_bytes == 64, "видам отведено %llu байт, ждали 64",
              (unsigned long long)g_mod.view_bytes);
        CHECK(g_mod.arena_bytes[0] < 48, "W всё равно занял место в арене (%llu байт)",
              (unsigned long long)g_mod.arena_bytes[0]);
        CHECK(near(regval("row", NULL), 38.0), "строка 2 даёт %g, ждали 38",
              regval("row", NULL));
        CHECK(near(regval("all", NULL), 66.0), "сумма %g, ждали 66", regval("all", NULL));
        CHECK(g_mapped == 1, "адрес брали %u раз, ждали 1", g_mapped);

        /* Второй прогон той же VM видит объект, каким он стал: копии,
         * которая могла бы устареть, нет. */
        put_w(100.0f);
        CHECK(smp_vm_run(&g_vm) == SMP_OK, "второй прогон упал");
        CHECK(near(regval("all", NULL), 1266.0), "после замены объекта сумма %g, ждали 1266",
              regval("all", NULL));
        CHECK(g_mapped == 2, "на втором прогоне адрес не взяли заново");
    }

    /* --- в тензор пишут: остаётся в арене, объект хранилища цел --- */
    put_w(0.0f);
    if (run_str("[#arena:0] *&W<f32:3,4> -> @load => *&W;\n"
                "*&W -> @scale(2.0) => *&W;\n"
                "*&W -> @reduce.add => $all;\n")) {
        const SmpTensor *t = NULL;
        CHECK(tensor_data("W", &t) != g_objs[0].data, "W с записью стал видом");
        CHECK(t && !(t->flags & SMP_TF_EXTERN), "W с записью помечен видом");
        CHECK(near(regval("all", NULL), 132.0), "сумма %g, ждали 132", regval("all", NULL));
        CHECK(((const float *)g_objs[0].data)[11] == 11.0f, "запись дошла до хранилища");
    }
    /* @fill пишет в свой источник, хоть приёмник и другой. */
    if (run_str("[#arena:0] *&W<f32:3,4> -> @load => *&W;\n"
                "*&W -> @fill(1.0) => *&Z<f32:3,4>;\n"
                "*&Z -> @reduce.add => $z;\n")) {
        const SmpTensor *t = NULL;
        CHECK(tensor_data("W", &t) != g_objs[0].data, "источник @fill стал видом");
        CHECK(((const float *)g_objs[0].data)[0] == 0.0f, "@fill дошёл до хранилища");
    }

    /* --- положить вид обратно и читать дальше --- */
    if (run_str("[#arena:0] *&W<f32:3,4> -> @load => *&W;\n"
                "*&W -> @store => $n;\n"
                "*&W -> @reduce.add => $all;\n")) {
        CHECK(near(regval("all", NULL), 66.0), "после @store сумма %g, ждали 66",
              regval("all", NULL));
    } else {
        CHECK(0, "load -> store -> чтение упало: %s", g_out);
    }

    /* --- читать до @load: адреса нет --- */
    CHECK(!run_str("[#arena:0] *&W<f32:3,4> -> @reduce.add => $a;\n"
                   "*&W -> @load => *&W;\n"),
          "чтение вида до @load обязано упасть");
    CHECK(saw_code("E0611"), "нет E0611 при чтении вида до @load");

    /* --- то же без адресов: копия у VM, ответы те же --- */
    static const SmpStore copy = { NULL, fake_find, fake_read, fake_write, NULL };
    g_store = &copy;
    if (run_str(prog)) {
        const SmpTensor *t = NULL;
        const void *p = tensor_data("W", &t);
        CHECK(p && p != g_objs[0].data, "без map вид обязан быть копией");
        CHECK(near(regval("row", NULL), 38.0) && near(regval("all", NULL), 66.0),
              "копия даёт %g и %g, ждали 38 и 66", regval("row", NULL), regval("all", NULL));
    }

    g_store = NULL;
}

/* Весь путь весов: квантовать, умножить @mmul.t, взять строку по номеру
 * из регистра и распаковать. Ожидания точные: распаковка q8_0 точна, а
 * суммы здесь — степени двойки на одно и то же число. */
static void test_q8(void)
{
    SECTION("q8_0 в исполнении");

    CHECK(run_str("[#arena:1] *&F<f32:64,32> -> @fill(0.5) => *&F;\n"
                  "*&F -> @cast.q8_0 => *&W<q8_0:64,32>;\n"
                  "[#arena:0] *&x<f32:1,32> -> @fill(2.0) => *&x;\n"
                  "*&x -> @mmul.t(*&W) => *&y<f32:1,64>;\n"
                  "*&y -> @reduce.add => $s;\n"), "линейный слой на q8_0");
    {
        const float d = smp_f16_to_f32(smp_f32_to_f16(0.5f / 127.0f));
        const float y = d * 8128.0f;                   /* 32 * 2 * 127 */
        CHECK(felem("y", 0) == y && felem("y", 63) == y,
              "y[0]=%.9g y[63]=%.9g, ждали %.9g", felem("y", 0), felem("y", 63), y);
        CHECK(regval("s", NULL) == 64.0 * y, "сумма %.9g, ждали %.9g",
              regval("s", NULL), 64.0 * y);
    }

    /* Строки разные: 0, 1, 2, 3. Взять надо вторую — по регистру. */
    CHECK(run_str("*&F<f32:4,32> -> @alloc => $f;\n"
                  "[#repeat:4, #index:i] *&F[$i, ..] -> @fill($i) => *&F[$i, ..];\n"
                  "*&F -> @cast.q8_0 => *&W<q8_0:4,32>;\n"
                  "2 => $t;\n"
                  "*&W[$t, ..] -> @cast.f32 => *&r<f32:32>;\n"
                  "*&r -> @reduce.add => $s;\n"), "строка q8_0 по регистру");
    {
        const float v = smp_f16_to_f32(smp_f32_to_f16(2.0f / 127.0f)) * 127.0f;
        CHECK(felem("r", 0) == v && felem("r", 31) == v,
              "r[0]=%.9g r[31]=%.9g, ждали %.9g", felem("r", 0), felem("r", 31), v);
        CHECK(regval("s", NULL) == 32.0 * v, "сумма строки %.9g, ждали %.9g",
              regval("s", NULL), 32.0 * v);
    }
}

/* Как в модели: линейный слой в скретч, @reshape в головы, RoPE по позиции
 * из регистра, softmax по живой длине — и всё это через регистры и виды,
 * без копий. Числа подобраны так, чтобы проверять точно. */
/* Внимание по KV-кэшу, как в Qwen2: кэш лежит по позициям [T, группы,
 * голова], новая строка пишется по номеру позиции, группа голов запроса
 * читает свою голову K/V видом с шагом, и softmax смотрит только на первые
 * $len позиций. Эталон — те же формулы в double по содержимому тензоров. */
static void test_attention(void)
{
    SECTION("внимание по KV-кэшу");

    enum { T_ = 8, G = 2, H = 3, D = 4, LEN = 5 };
    CHECK(run_str("[#arena:1] *&Kc<f32:8,2,4> -> @alloc => $kc;\n"
                  "[#arena:1] *&Vc<f32:8,2,4> -> @alloc => $vc;\n"
                  "[#arena:0] *&one<f32:2,4> -> @fill(1.0) => *&one;\n"
                  "*&one[1, ..] -> @scale(-0.5) => *&one[1, ..];\n"
                  "[#repeat:6, #index:t] *&one -> @rope($t, 10.0) => *&Kc[$t, .., ..];\n"
                  "[#repeat:6, #index:t] *&one -> @rope($t, 3.0) -> @scale(0.7) => *&Vc[$t, .., ..];\n"
                  "*&q6<f32:6,4> -> @alloc => $q6;\n"
                  "[#repeat:6, #index:i] *&q6[$i, ..] -> @fill($i) => *&q6[$i, ..];\n"
                  "*&q6 -> @scale(0.3) -> @rope(5, 7.0) -> @reshape(2,3,4) => *&q<f32:2,3,4>;\n"
                  "*&o<f32:2,3,4> -> @alloc => $o;\n"
                  "5 => $len;\n"
                  "[#repeat:2, #index:g] *&q[$g, .., ..] -> @mmul.t(*&Kc[.., $g, ..])"
                  " -> @scale(0.5) -> @softmax($len) -> @mmul(*&Vc[.., $g, ..])"
                  " => *&o[$g, .., ..];\n"), "внимание");

    double worst = 0.0;
    for (int g = 0; g < G; g++)
        for (int h = 0; h < H; h++) {
            double s[T_], m = -INFINITY, sum = 0.0;
            for (int t = 0; t < LEN; t++) {
                s[t] = 0.0;
                for (int d = 0; d < D; d++)
                    s[t] += (double)felem("q", (uint32_t)((g * H + h) * D + d)) *
                            (double)felem("Kc", (uint32_t)((t * G + g) * D + d));
                s[t] *= 0.5;
                if (s[t] > m) m = s[t];
            }
            for (int t = 0; t < LEN; t++) sum += exp(s[t] - m);
            for (int d = 0; d < D; d++) {
                double o = 0.0;
                for (int t = 0; t < LEN; t++)
                    o += exp(s[t] - m) / sum * (double)felem("Vc", (uint32_t)((t * G + g) * D + d));
                const double e = fabs(o - (double)felem("o", (uint32_t)((g * H + h) * D + d)));
                if (e > worst) worst = e;
            }
        }
    CHECK(worst < 1e-5, "внимание разошлось с эталоном: %g", worst);

    /* Позиции 5 и дальше в кэше заняты, но маской отрезаны: их вклад — ноль,
     * иначе строка 5 (записана) изменила бы ответ. Проверяем, что она
     * действительно записана и отличается. */
    CHECK(felem("Kc", 5 * G * D) != 0.0f, "строка 5 кэша не записана — проверка маски ни о чём");
}

/* Живая длина: @mmul.t считает только первые n строк правого множителя,
 * остальные столбцы — нули; @mmul обрезает ось K. Внимание с ней даёт тот
 * же ответ, что с маской одной @softmax, но мёртвые позиции не считает. */
static void test_live_len(void)
{
    SECTION("живая длина в @mmul.t и @mmul");

    if (run_str("*&A<f32:2,8> -> @fill(1.0) => *&A;\n"
                "*&B<f32:6,8> -> @fill(2.0) => *&B;\n"
                "4 => $n;\n"
                "*&A -> @mmul.t(*&B, $n) => *&C<f32:2,6>;\n")) {
        bool ok = true;
        for (uint32_t m = 0; m < 2; m++)
            for (uint32_t j = 0; j < 6; j++)
                ok &= felem("C", m * 6 + j) == (j < 4 ? 16.0f : 0.0f);
        CHECK(ok, "@mmul.t с длиной 4: ждали 16 в первых четырёх столбцах и нули дальше");
    }
    if (run_str("*&A<f32:1,64> -> @fill(0.5) => *&A;\n"
                "*&P<f32:5,64> -> @fill(0.25) => *&P;\n"
                "*&P -> @cast.q8_0 => *&W<q8_0:5,64>;\n"
                "*&A -> @mmul.t(*&W, 3) => *&C<f32:1,5>;\n")) {
        const float w = felem("C", 0);
        CHECK(w != 0.0f && felem("C", 2) == w && felem("C", 3) == 0.0f && felem("C", 4) == 0.0f,
              "@mmul.t по q8_0 с длиной-литералом 3: %g %g %g", w, felem("C", 3), felem("C", 4));
    }
    if (run_str("*&A<f32:2,6> -> @fill(1.0) => *&A;\n"
                "*&B<f32:6,3> -> @fill(2.0) => *&B;\n"
                "4 => $k;\n"
                "*&A -> @mmul(*&B, $k) => *&C<f32:2,3>;\n")) {
        CHECK(felem("C", 0) == 8.0f && felem("C", 5) == 8.0f,
              "@mmul с осью K, обрезанной до 4: %g, ждали 8", felem("C", 0));
    }
    /* Длина за краями: ноль и меньше — ничего, больше оси — вся ось. */
    if (run_str("*&A<f32:1,8> -> @fill(1.0) => *&A;\n"
                "*&B<f32:3,8> -> @fill(1.0) => *&B;\n"
                "-2 => $lo;\n"
                "100 => $hi;\n"
                "*&A -> @mmul.t(*&B, $lo) => *&Z<f32:1,3>;\n"
                "*&A -> @mmul.t(*&B, $hi) => *&F<f32:1,3>;\n")) {
        CHECK(felem("Z", 0) == 0.0f && felem("Z", 2) == 0.0f, "длина -2 не дала нулей");
        CHECK(felem("F", 0) == 8.0f && felem("F", 2) == 8.0f, "длина 100 не дала всю ось");
    }

    /* Внимание: маска одной @softmax против живой длины во всех трёх стадиях. */
    static const char head[] =
        "[#arena:1] *&Kc<f32:8,4> -> @alloc => $kc;\n"
        "[#arena:1] *&Vc<f32:8,4> -> @alloc => $vc;\n"
        "*&one<f32:4> -> @fill(1.0) => *&one;\n"
        "[#repeat:8, #index:t] *&one -> @rope($t, 10.0) => *&Kc[$t, ..];\n"
        "[#repeat:8, #index:t] *&one -> @rope($t, 3.0) -> @scale(0.7) => *&Vc[$t, ..];\n"
        "*&one -> @scale(0.3) -> @rope(5, 7.0) -> @reshape(1, 4) => *&q<f32:1,4>;\n"
        "5 => $len;\n";
    char prog[1024];
    float masked[4] = { 0 };
    snprintf(prog, sizeof prog, "%s*&q -> @mmul.t(*&Kc) -> @scale(0.5) -> @softmax($len)"
             " -> @mmul(*&Vc) => *&o<f32:1,4>;\n", head);
    if (run_str(prog))
        for (uint32_t d = 0; d < 4; d++) masked[d] = felem("o", d);
    snprintf(prog, sizeof prog, "%s*&q -> @mmul.t(*&Kc, $len) -> @scale(0.5) -> @softmax($len)"
             " -> @mmul(*&Vc, $len) => *&o<f32:1,4>;\n", head);
    if (run_str(prog)) {
        double worst = 0.0;
        for (uint32_t d = 0; d < 4; d++) {
            const double e = fabs((double)masked[d] - (double)felem("o", d));
            if (e > worst) worst = e;
        }
        CHECK(masked[0] != 0.0f && worst < 1e-6,
              "внимание с живой длиной разошлось с маской: %g", worst);
    }
}

/* Логический тип именованного скалярного регистра. */
static SmpDType regtype(const char *name)
{
    for (uint32_t i = 0; i < g_vm.n_regs && i < g_mod.n_reg_names; i++) {
        if (!g_mod.reg_names[i]) continue;
        if (strcmp(smp_module_str(&g_mod, g_mod.reg_names[i]), name) != 0) continue;
        return (SmpDType)g_vm.regs[i].dtype;
    }
    return SMP_DT_INVALID;
}

static void test_calc(void)
{
    SECTION("калькулятор: числа");

    CHECK(run_str("17 -> @mul(23) => $p;\n"
                  "7 -> @div(2) => $q;\n"
                  "-7 -> @div(2) => $qn;\n"
                  "7.0 -> @div(2) => $f;\n"
                  "0.1 -> @add(0.2) => $t;\n"
                  "2 -> @sqrt => $r;\n"
                  "-12 -> @abs => $a;\n"
                  "$f -> @mul($f) -> @sub(0.25) => $sq;\n"
                  "[0.1] => *&h<f32:1>;\n"
                  "*&h[0] -> @add(0.2) => $t32;\n"), "арифметика чисел");
    CHECK(regval("p", NULL) == 391.0 && regtype("p") == SMP_DT_I32, "17*23 = %g", regval("p", NULL));
    CHECK(regval("q", NULL) == 3.0, "7/2 в i32 = %g, ждали 3", regval("q", NULL));
    CHECK(regval("qn", NULL) == -3.0, "-7/2 в i32 = %g, ждали -3 (к нулю)", regval("qn", NULL));
    /* Дробь без типизированного соседа — f64. */
    CHECK(regval("f", NULL) == 3.5 && regtype("f") == SMP_DT_F64, "7.0/2 = %g", regval("f", NULL));
    CHECK(regval("t", NULL) == 0.1 + 0.2 && regtype("t") == SMP_DT_F64,
          "0.1+0.2 в f64 = %.17g", regval("t", NULL));
    CHECK(regval("r", NULL) == sqrt(2.0) && regtype("r") == SMP_DT_F64,
          "корень из 2 = %.17g", regval("r", NULL));
    /* f32 обязан быть f32: сумма округлена в float, а не оставлена double. */
    CHECK(regval("t32", NULL) == (double)(0.1f + 0.2f) && regtype("t32") == SMP_DT_F32,
          "0.1+0.2 в f32 = %.17g", regval("t32", NULL));
    CHECK(regval("a", NULL) == 12.0 && regtype("a") == SMP_DT_I32, "|-12| = %g", regval("a", NULL));
    CHECK(regval("sq", NULL) == 12.0, "3.5^2 - 0.25 = %g", regval("sq", NULL));

    /* Литерал берёт тип соседа: 1 рядом с u64 из @argmax — это u64. */
    CHECK(run_str("*&v<f32:3> -> @alloc => *&v;\n"
                  "9.0 => *&v[1];\n"
                  "*&v -> @argmax -> @add(1) => $pos;\n"), "argmax + 1");
    CHECK(regval("pos", NULL) == 2.0 && regtype("pos") == SMP_DT_U64,
          "argmax+1 = %g", regval("pos", NULL));

    /* Элемент тензора — тоже число. */
    CHECK(run_str("[2.0, 5.0] => *&v<f32:2>;\n"
                  "*&v[1] -> @mul(*&v[0]) => $m;\n"), "элементы как числа");
    CHECK(regval("m", NULL) == 10.0, "5*2 = %g", regval("m", NULL));

    SECTION("калькулятор: ловушки");

    CHECK(!run_str("5 -> @div(0) => $x;\n"), "деление на ноль прошло");
    CHECK(strstr(g_out, "E0601") != NULL, "нет E0601");
    CHECK(!run_str("2147483647 -> @add(1) => $x;\n"), "переполнение i32 прошло");
    CHECK(strstr(g_out, "E0613") != NULL, "нет E0613");
    CHECK(!run_str("*&v<f32:2> -> @alloc => *&v;\n"
                   "*&v -> @argmax -> @sub(1) => $x;\n"), "u64 ниже нуля прошло");
    CHECK(strstr(g_out, "E0613") != NULL, "нет E0613 для u64");
    /* У дробных деление на ноль — IEEE, а не ошибка. */
    CHECK(run_str("1.0 -> @div(0) => $x;\n"), "1.0/0 обязано дать бесконечность");
    CHECK(isinf(regval("x", NULL)), "1.0/0 = %g", regval("x", NULL));

    SECTION("калькулятор: число из регистра в @fill и @scale");

    /* Раньше VM брала число из пула констант по индексу 0 — то есть первую
     * попавшуюся константу программы, а регистр не читала вовсе. */
    CHECK(run_str("*&v<f32:4> -> @fill(3.0) => *&v;\n"
                  "*&v -> @reduce.add => $s;\n"
                  "*&w<f32:4> -> @fill($s) => *&w;\n"
                  "*&w -> @scale($s) => *&x<f32:4>;\n"
                  "*&w -> @scale($s) -> @relu => *&y<f32:4>;\n"
                  "*&z<f32:4> -> @fill(*&v[0]) => *&z;\n"), "@fill($s), @scale($s)");
    CHECK(felem("w", 3) == 12.0f, "@fill($s): %g, ждали 12", felem("w", 3));
    CHECK(felem("x", 0) == 144.0f, "@scale($s): %g, ждали 144", felem("x", 0));
    CHECK(felem("y", 1) == 144.0f, "слитая @scale($s): %g, ждали 144", felem("y", 1));
    CHECK(felem("z", 2) == 3.0f, "@fill(*&v[0]): %g, ждали 3", felem("z", 2));

    SECTION("калькулятор: литералы тензоров");

    CHECK(run_str("[[1, 2, 3], [4, 5, 6]] => *&M<f32:2,3>;\n"
                  "*&T<i32:3,2> -> @alloc => *&T;\n"
                  "[7, 8, 9] => *&T[.., 1];\n"
                  "[1.5, 2.5] -> @reduce.add => $s;\n"
                  "[-1, 2] => $r;\n"
                  "$r -> @reduce.add => $rs;\n"
                  "1 => $i;\n"
                  "[10.0, 20, 30] => *&M[$i, ..];\n"), "литералы");
    {
        const float   *m = (const float *)tensor_data("M", NULL);
        const int32_t *t = (const int32_t *)tensor_data("T", NULL);
        CHECK(m && m[0] == 1.0f && m[2] == 3.0f && m[3] == 10.0f && m[5] == 30.0f,
              "M = [%g %g %g; %g %g %g]", m ? m[0] : -1, m ? m[1] : -1, m ? m[2] : -1,
              m ? m[3] : -1, m ? m[4] : -1, m ? m[5] : -1);
        CHECK(t && t[0] == 0 && t[1] == 7 && t[3] == 8 && t[5] == 9,
              "столбец T: %d %d %d %d", t ? t[0] : -1, t ? t[1] : -1, t ? t[3] : -1,
              t ? t[5] : -1);
    }
    CHECK(regval("s", NULL) == 4.0 && regtype("s") == SMP_DT_F64, "сумма литерала %g",
          regval("s", NULL));
    CHECK(regval("rs", NULL) == 1.0 && regtype("rs") == SMP_DT_I32, "литерал в регистре: %g",
          regval("rs", NULL));

    /* Строка f32:3 по индексу-регистру начинается с 12-го байта. Без #simd
     * это законно; с явным #simd:v256 — ловушка, как у статического среза. */
    CHECK(run_str("*&M<f32:2,3> -> @fill(1.0) => *&M;\n"
                  "1 => $i;\n"
                  "*&M[$i, ..] -> @reduce.add => $s;\n"), "невыровненная строка без #simd");
    CHECK(regval("s", NULL) == 3.0, "сумма строки %g", regval("s", NULL));
    CHECK(!run_str("*&M<f32:2,3> -> @fill(1.0) => *&M;\n"
                   "1 => $i;\n"
                   "[#simd:v256] *&M[$i, ..] -> @reduce.add => $s;\n"),
          "невыровненная строка под #simd:v256 прошла");
    CHECK(strstr(g_out, "E0402") != NULL, "нет E0402");

    SECTION("калькулятор: деление и корень тензоров");

    CHECK(run_str("[1.0, 4.0, 9.0] => *&a<f32:3>;\n"
                  "[2.0, 8.0, 3.0] => *&b<f32:3>;\n"
                  "*&a -> @div(*&b) => *&q<f32:3>;\n"
                  "*&a -> @sqrt => *&r<f32:3>;\n"), "div/sqrt");
    CHECK(felem("q", 0) == 0.5f && felem("q", 1) == 0.5f && felem("q", 2) == 3.0f,
          "a/b = %g %g %g", felem("q", 0), felem("q", 1), felem("q", 2));
    CHECK(felem("r", 0) == 1.0f && felem("r", 1) == 2.0f && felem("r", 2) == 3.0f,
          "sqrt(a) = %g %g %g", felem("r", 0), felem("r", 1), felem("r", 2));
}

static void test_nn_chain(void)
{
    SECTION("цепочка модели");

    CHECK(run_str("[#arena:1] *&W<f32:8,4> -> @fill(0.25) => *&W;\n"
                  "[#arena:0] *&x<f32:1,4> -> @fill(1.0) => *&x;\n"
                  "*&x -> @mmul.t(*&W) -> @reshape(2,4) => $h;\n"
                  "0 => $p;\n"
                  "$h -> @rope($p, 10000.0) -> @softmax($p) => *&S<f32:2,4>;\n"
                  "2 => $n;\n"
                  "$h -> @softmax($n) => *&T<f32:2,4>;\n"
                  "*&T -> @reshape(8) -> @argmax => $i;\n"), "цепочка");
    CHECK(felem("S", 0) == 0.0f && felem("S", 7) == 0.0f, "softmax(0) — не нули");
    CHECK(felem("T", 0) == 0.5f && felem("T", 1) == 0.5f && felem("T", 2) == 0.0f &&
          felem("T", 5) == 0.5f && felem("T", 7) == 0.0f,
          "softmax(2): T = %g %g %g .. %g %g", felem("T", 0), felem("T", 1),
          felem("T", 2), felem("T", 5), felem("T", 7));
    CHECK(regval("i", NULL) == 0.0, "argmax %g, ждали 0", regval("i", NULL));
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
    test_calc();
    test_emit();
    test_views();
    test_traps();
    test_fp_modes();
    test_regdump();
    test_roundtrip();
    test_fusion();
    test_gemm_epilogue();
    test_fileio();
    test_store();
    test_store_views();
    test_q8();
    test_nn_chain();
    test_attention();
    test_live_len();

    smp_vm_release(&g_vm);
    fclose(g_sink);
    smp_arena_release(&g_arena);
    return REPORT();
}
