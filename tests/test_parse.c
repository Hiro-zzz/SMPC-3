/* SMPC3 :: test_parse.c -- проверки Ф2: парсер и AST. */
#include "smpc3/io.h"
#include "smpc3/lex.h"
#include "smpc3/parse.h"

#include "harness.h"

#include <string.h>
#include <stdlib.h>

/* ========================================================================== */
/*  Обвязка                                                                   */
/* ========================================================================== */

static SmpArena   g_arena;
static SmpDiagCtx g_diag;
static SmpSource  g_src;
static FILE      *g_sink;

/* Полный проход лексер+парсер. Возвращает число ошибок парсера. */
static uint32_t parse_str(const char *text, SmpAstProgram *prog)
{
    smp_arena_reset(&g_arena);
    rewind(g_sink);

    smp_source_from_memory(&g_src, "t.smpc", text, strlen(text));
    smp_diag_init(&g_diag, &g_src, g_sink);

    SmpLexer lx;
    smp_lex_init(&lx, &g_src, &g_diag);
    SmpToken *toks = NULL;
    uint32_t  ntok = 0;
    smp_lex_all(&lx, &g_arena, &toks, &ntok);

    SmpParser P;
    smp_parse_init(&P, &g_arena, &g_diag, &g_src, toks, ntok);
    smp_parse(&P, prog);
    return P.n_errors;
}

/* Ожидаем: ровно одна претензия, ровно указанного кода. */
static void expect_error(const char *text, SmpDiagCode want, const char *label)
{
    SmpAstProgram prog;
    const uint32_t errs = parse_str(text, &prog);

    CHECK(errs == 1, "%s: ошибок %u, ждали 1", label, errs);

    rewind(g_sink);
    static char out[8192];
    const size_t n = fread(out, 1, sizeof(out) - 1u, g_sink);
    out[n] = '\0';

    const char *code = smp_diag_info(want)->text;
    CHECK(strstr(out, code) != NULL, "%s: в выводе нет %s", label, code);
}

static void expect_ok(const char *text, uint32_t nstmts, const char *label)
{
    SmpAstProgram prog;
    const uint32_t errs = parse_str(text, &prog);
    CHECK(errs == 0, "%s: %u ошибок на корректном коде", label, errs);
    CHECK(prog.nstmts == nstmts, "%s: инструкций %u, ждали %u",
          label, prog.nstmts, nstmts);
}

/* ========================================================================== */

static const char g_kernel[] =
    "// Инициализация матриц в выровненной арене\n"
    "[#arena:0]  *&A<f32:1024,1024>  ->  @alloc  =>  $r1;\n"
    "[#arena:0]  *&B<f32:1024,1024>  ->  @alloc  =>  $r2;\n"
    "\n"
    "[#simd:v256]  $r1 -> @mmul($r2) => *&C<f32:1024,1024>  [!no-alias, ?strict];\n"
    "\n"
    "[^raw]  *&C[0, ..] -> @relu -> @reduce.add => $f0  [~flush-to-zero];\n";

static void test_kernel(void)
{
    SECTION("пример из спецификации");

    SmpAstProgram prog;
    const uint32_t errs = parse_str(g_kernel, &prog);

    CHECK(errs == 0, "пример из спецификации дал %u ошибок", errs);
    CHECK(prog.nstmts == 4, "инструкций %u, ждали 4", prog.nstmts);
    if (prog.nstmts != 4) return;

    /* --- #0: объявление тензора --- */
    {
        const SmpAstStmt *s = &prog.stmts[0];
        CHECK(s->nprefix == 1, "#0 префиксов %u", s->nprefix);
        const SmpAstPrefix *ar = smp_stmt_directive(s, "arena");
        CHECK(ar != NULL, "#0 нет директивы #arena");
        CHECK(ar && ar->has_value && ar->value_is_int && ar->ival == 0,
              "#0 значение #arena разобрано неверно");

        CHECK(s->source.kind == SMP_OPD_TENSOR, "#0 источник не тензор");
        const SmpAstTensor *t = s->source.tensor;
        CHECK(t && smp_name_is(t->name, "A"), "#0 имя тензора");
        CHECK(t && t->has_type, "#0 нет объявления типа");
        CHECK(t && t->dtype == SMP_DT_F32, "#0 тип не f32");
        CHECK(t && t->rank == 2, "#0 ранг %u", t ? t->rank : 0);
        CHECK(t && t->dims[0] == 1024 && t->dims[1] == 1024, "#0 размерности");
        CHECK(t && !t->has_index, "#0 индекса быть не должно");

        CHECK(s->nstages == 1, "#0 стадий %u", s->nstages);
        CHECK(s->nstages && smp_name_is(s->stages[0].name, "alloc"), "#0 стадия @alloc");
        CHECK(s->nstages && s->stages[0].nargs == 0, "#0 у @alloc есть аргументы");

        CHECK(s->has_dest && s->dest.kind == SMP_OPD_REG, "#0 приёмник не регистр");
        CHECK(smp_name_is(s->dest.reg, "r1"), "#0 приёмник не $r1");
        CHECK(s->nsuffix == 0, "#0 суффиксов %u", s->nsuffix);
    }

    /* --- #2: GEMM с аргументом, объявлением приёмника и суффиксом --- */
    {
        const SmpAstStmt *s = &prog.stmts[2];
        const SmpAstPrefix *sd = smp_stmt_directive(s, "simd");
        CHECK(sd != NULL, "#2 нет директивы #simd");
        CHECK(sd && sd->has_value && !sd->value_is_int && smp_name_is(sd->sval, "v256"),
              "#2 значение #simd не v256");

        CHECK(s->source.kind == SMP_OPD_REG && smp_name_is(s->source.reg, "r1"),
              "#2 источник не $r1");

        CHECK(s->nstages == 1, "#2 стадий %u", s->nstages);
        CHECK(s->nstages && smp_name_is(s->stages[0].name, "mmul"), "#2 стадия @mmul");
        CHECK(s->nstages && s->stages[0].nargs == 1, "#2 аргументов %u",
              s->nstages ? s->stages[0].nargs : 0);
        CHECK(s->nstages && s->stages[0].nargs &&
              s->stages[0].args[0].kind == SMP_OPD_REG &&
              smp_name_is(s->stages[0].args[0].reg, "r2"), "#2 аргумент не $r2");

        CHECK(s->dest.kind == SMP_OPD_TENSOR, "#2 приёмник не тензор");
        CHECK(s->dest.tensor && s->dest.tensor->has_type, "#2 приёмник без типа");

        /* Суффиксная группа не должна была прилипнуть к тензору как индекс. */
        CHECK(s->dest.tensor && !s->dest.tensor->has_index,
              "#2 суффикс ошибочно разобран как индекс приёмника");

        CHECK(s->nsuffix == 2, "#2 суффиксов %u", s->nsuffix);
        CHECK(smp_stmt_suffix(s, SMP_SFX_ASSERT, "no-alias") != NULL, "#2 нет !no-alias");
        CHECK(smp_stmt_suffix(s, SMP_SFX_QUERY, "strict") != NULL, "#2 нет ?strict");
        CHECK(smp_stmt_suffix(s, SMP_SFX_ASSERT, "strict") == NULL,
              "#2 поиск игнорирует вид суффикса");
    }

    /* --- #3: режим, срез, цепочка стадий --- */
    {
        const SmpAstStmt *s = &prog.stmts[3];
        CHECK(smp_stmt_has_mode(s, "raw"), "#3 нет режима ^raw");
        CHECK(!smp_stmt_has_mode(s, "cooked"), "#3 нашёлся несуществующий режим");

        CHECK(s->source.kind == SMP_OPD_TENSOR, "#3 источник не тензор");
        const SmpAstTensor *t = s->source.tensor;
        CHECK(t && !t->has_type, "#3 у ссылки не должно быть объявления типа");
        CHECK(t && t->has_index, "#3 нет индекса");
        CHECK(t && t->nidx == 2, "#3 индексов %u", t ? t->nidx : 0);
        CHECK(t && t->idx[0].kind == SMP_IDX_INT && t->idx[0].ival == 0, "#3 idx[0]");
        CHECK(t && t->idx[1].kind == SMP_IDX_ALL, "#3 idx[1] не '..'");

        CHECK(s->nstages == 2, "#3 стадий %u", s->nstages);
        CHECK(s->nstages == 2 && smp_name_is(s->stages[0].name, "relu"), "#3 @relu");
        CHECK(s->nstages == 2 && smp_name_is(s->stages[1].name, "reduce.add"),
              "#3 @reduce.add как одно имя");

        CHECK(s->nsuffix == 1 && smp_stmt_suffix(s, SMP_SFX_FPMODE, "flush-to-zero"),
              "#3 нет ~flush-to-zero");
    }
}

/* ========================================================================== */

static void test_shapes(void)
{
    SECTION("корректные формы");

    expect_ok("$r1 => $r2;", 1, "минимальная инструкция");
    expect_ok("$r1 -> @relu => $r2;", 1, "одна стадия");
    expect_ok("$a => $b; $c => $d;", 2, "две инструкции");
    expect_ok(";;; $a => $b; ;;", 1, "лишние ';' не ошибка");
    expect_ok("", 0, "пустой файл");
    expect_ok("// только комментарий\n", 0, "файл из комментария");

    expect_ok("[#arena:0][#simd:v256] $a => $b;", 1, "две префиксные группы");
    expect_ok("[#arena:0, ^raw] $a => $b;", 1, "префиксы через запятую");
    expect_ok("$a => $b [!x][?y][~z];", 1, "три суффиксные группы");

    expect_ok("*&A<raw_ptr> => $r1;", 1, "тип без размерностей");
    expect_ok("*&A<f64:8,8,8,8> => $r1;", 1, "ранг 4");
    expect_ok("*&A[$i, .., 3] => $r1;", 1, "индекс регистром");
    expect_ok("$a -> @f(1, -2, 1.5, -0.5) => $b;", 1, "литералы, в т.ч. отрицательные");
    expect_ok("$a -> @f() => $b;", 1, "пустой список аргументов");

    /* Отрицательные литералы должны доезжать со знаком. */
    {
        SmpAstProgram prog;
        CHECK(parse_str("$a -> @f(-7, -1.5) => $b;", &prog) == 0, "разбор минусов");
        CHECK(prog.nstmts == 1 && prog.stmts[0].nstages == 1, "структура минусов");
        if (prog.nstmts == 1 && prog.stmts[0].nstages == 1) {
            const SmpAstStage *st = &prog.stmts[0].stages[0];
            CHECK(st->nargs == 2, "аргументов %u", st->nargs);
            CHECK(st->nargs == 2 && (int64_t)st->args[0].ival == -7,
                  "-7 -> %lld", (long long)(int64_t)st->args[0].ival);
            CHECK(st->nargs == 2 && st->args[1].fval == -1.5,
                  "-1.5 -> %g", st->args[1].fval);
        }
    }

    /* Конвейер максимальной длины. */
    {
        static char buf[1024];
        size_t n = 0;
        n += (size_t)snprintf(buf + n, sizeof(buf) - n, "$a");
        for (unsigned i = 0; i < SMP_MAX_STAGES; i++)
            n += (size_t)snprintf(buf + n, sizeof(buf) - n, " -> @s");
        snprintf(buf + n, sizeof(buf) - n, " => $b;");
        expect_ok(buf, 1, "конвейер ровно из SMP_MAX_STAGES стадий");
    }
}

/* ========================================================================== */

static void test_errors(void)
{
    SECTION("диагностика");

    expect_error("$a => $b",              SMP_E0201, "нет ';'");
    expect_error("$a -> => $b;",          SMP_E0202, "стрелка без стадии");
    expect_error("$a -> @f($b => $c;",    SMP_E0203, "не закрыта '('");
    expect_error("*&A<f32:4 => $b;",      SMP_E0203, "не закрыт '<'");
    expect_error("*&A[0 => $b;",          SMP_E0203, "не закрыт '['");
    expect_error("$a => $b [@nope];",     SMP_E0204, "мусор в суффиксе");
    expect_error("[@nope] $a => $b;",     SMP_E0204, "мусор в префиксе");
    expect_error("$a -> @f => ;",         SMP_E0209, "нет приёмника после '=>'");
    expect_error("$a;",                   SMP_E0205, "нет '=>' вовсе");
    expect_error("$a => $b [#simd:v256];", SMP_E0206, "префикс после тела");
    expect_error("[!no-alias] $a => $b;", SMP_E0206, "суффикс до тела");
    expect_error("$a => 42;",             SMP_E0208, "литерал как приёмник");
    expect_error("$a => 1.5;",            SMP_E0208, "веществ. литерал как приёмник");
    expect_error("[] $a => $b;",          SMP_E0210, "пустая группа");
    expect_error("=> $b;",                SMP_E0209, "нет источника");
    expect_error("*& => $b;",             SMP_E0209, "'*&' без имени");

    expect_error("*&A<f16:4> => $b;",     SMP_E0301, "несуществующий тип");
    expect_error("*&A<:4> => $b;",        SMP_E0301, "тип пропущен");
    expect_error("*&A<f32:1,2,3,4,5> => $b;", SMP_E0302, "ранг 5");
    expect_error("*&A<f32:0> => $b;",     SMP_E0304, "нулевая ось");
    expect_error("*&A<f32:4294967296> => $b;", SMP_E0304, "ось больше uint32_t");
    expect_error("*&A<f32:65536,65536> => $b;", SMP_E0304, "элементов больше uint32_t");
    expect_ok("*&A<f32:151936,896> => $b;", 1, "ось словаря больше 65535");
    expect_ok("*&A<f32:4294967295> => $b;", 1, "ось ровно на пределе");
    expect_error("*&A<f32:x> => $b;",     SMP_E0304, "ось не число");

    /* Пределы. */
    expect_error("$a -> @f(1,2,3,4,5,6,7,8,9) => $b;", SMP_E0207, "9 аргументов");
    expect_error("*&A[1,2,3,4,5] => $b;",              SMP_E0207, "5 индексов");
    {
        static char buf[1024];
        size_t n = 0;
        n += (size_t)snprintf(buf + n, sizeof(buf) - n, "$a");
        for (unsigned i = 0; i <= SMP_MAX_STAGES; i++)
            n += (size_t)snprintf(buf + n, sizeof(buf) - n, " -> @s");
        snprintf(buf + n, sizeof(buf) - n, " => $b;");
        expect_error(buf, SMP_E0207, "стадий больше предела");
    }
    {
        static char buf[512];
        size_t n = (size_t)snprintf(buf, sizeof buf, "[#a");
        for (unsigned i = 0; i < SMP_MAX_PREFIX; i++)
            n += (size_t)snprintf(buf + n, sizeof(buf) - n, ", #b");
        snprintf(buf + n, sizeof(buf) - n, "] $a => $b;");
        expect_error(buf, SMP_E0207, "префиксов больше предела");
    }
}

/* ========================================================================== */

static void test_recovery(void)
{
    SECTION("восстановление");

    SmpAstProgram prog;

    /* Одна битая инструкция посреди корректных: остальные обязаны уцелеть. */
    uint32_t errs = parse_str("$a => $b;  $c => ;  $d => $e;", &prog);
    CHECK(errs == 1, "ошибок %u, ждали 1", errs);
    CHECK(prog.nstmts == 2, "уцелевших инструкций %u, ждали 2", prog.nstmts);

    /* Каждая битая инструкция даёт РОВНО одну претензию, а не каскад. */
    errs = parse_str("$a => ;  $b => ;  $c => ;", &prog);
    CHECK(errs == 3, "ошибок %u, ждали ровно 3", errs);
    CHECK(prog.nstmts == 0, "уцелело %u инструкций, ждали 0", prog.nstmts);

    /* Инструкция с несколькими дефектами сразу — всё равно одна претензия. */
    errs = parse_str("[@x] *&A<f16:0,0,0,0,0> -> => 42 [#p]\n$ok => $ok2;", &prog);
    CHECK(errs == 1, "многократно битая инструкция дала %u претензий", errs);

    /* Без ';' в конце восстановление доходит до конца файла и не зацикливается. */
    errs = parse_str("$a => $b;  $c => ", &prog);
    CHECK(errs == 1, "хвост без ';' дал %u ошибок", errs);
    CHECK(prog.nstmts == 1, "уцелело %u, ждали 1", prog.nstmts);

    /* Забытая ';' не должна съедать следующую инструкцию: она разобрана
     * целиком, не хватает лишь терминатора. */
    errs = parse_str("$a => $b\n$c => $d;", &prog);
    CHECK(errs == 1, "забытая ';' дала %u ошибок, ждали 1", errs);
    CHECK(prog.nstmts == 2, "после забытой ';' уцелело %u, ждали 2", prog.nstmts);

    /* То же, когда следующая инструкция начинается с префиксной группы:
     * претензия обязана быть про ';', а не про «группу не на своём месте». */
    errs = parse_str("$a => $b\n[^raw] $c => $d;", &prog);
    CHECK(errs == 1, "';' перед префиксом дала %u ошибок", errs);
    CHECK(prog.nstmts == 2, "уцелело %u, ждали 2", prog.nstmts);
    {
        rewind(g_sink);
        static char out[8192];
        const size_t n = fread(out, 1, sizeof(out) - 1u, g_sink);
        out[n] = '\0';
        CHECK(strstr(out, "E0201") != NULL, "ждали E0201, а не E0206");
        CHECK(strstr(out, "E0206") == NULL, "E0206 не должен возникать");
        /* Каретка обязана стоять в конце первой строки, а не на второй. */
        CHECK(strstr(out, "t.smpc:1:9") != NULL, "каретка не в месте пропуска ';'");
    }

    /* А вот префикс на ТОЙ ЖЕ строке — действительно перепутанный порядок. */
    errs = parse_str("$a => $b [#simd:v256];", &prog);
    CHECK(errs == 1, "префикс на своей строке дал %u ошибок", errs);
    {
        rewind(g_sink);
        static char out[8192];
        const size_t n = fread(out, 1, sizeof(out) - 1u, g_sink);
        out[n] = '\0';
        CHECK(strstr(out, "E0206") != NULL, "здесь ждали именно E0206");
    }
}

/* ========================================================================== */

static void test_ambiguity(void)
{
    SECTION("разрешение '['");

    SmpAstProgram prog;

    /* Индекс приёмника и суффикс стоят подряд — их нельзя перепутать. */
    CHECK(parse_str("$a => *&C[0, ..] [!no-alias];", &prog) == 0, "индекс+суффикс");
    if (prog.nstmts == 1) {
        const SmpAstStmt *s = &prog.stmts[0];
        CHECK(s->dest.kind == SMP_OPD_TENSOR, "приёмник не тензор");
        CHECK(s->dest.tensor && s->dest.tensor->has_index &&
              s->dest.tensor->nidx == 2, "индекс приёмника потерян");
        CHECK(s->nsuffix == 1, "суффикс потерян: %u", s->nsuffix);
    }

    /* Тензор без индекса, сразу суффикс — индекс не должен появиться. */
    CHECK(parse_str("$a => *&C [?strict];", &prog) == 0, "тензор + суффикс");
    if (prog.nstmts == 1) {
        CHECK(!prog.stmts[0].dest.tensor->has_index,
              "суффикс ошибочно стал индексом");
        CHECK(prog.stmts[0].nsuffix == 1, "суффикс не разобран");
    }

    /* Индекс в источнике при наличии префикса. */
    CHECK(parse_str("[^raw] *&C[$i] -> @f => $b;", &prog) == 0, "префикс + индекс");
    if (prog.nstmts == 1) {
        const SmpAstTensor *t = prog.stmts[0].source.tensor;
        CHECK(t && t->has_index && t->nidx == 1 &&
              t->idx[0].kind == SMP_IDX_REG && smp_name_is(t->idx[0].reg, "i"),
              "индекс-регистр разобран неверно");
    }
}

/* ========================================================================== */

int main(void)
{
    smp_console_setup();
    fprintf(stderr, "SMPC3 tests :: Ф2\n\n");

    if (smp_arena_init(&g_arena, 8u << 20, 9, "parsetest") != SMP_OK) {
        fprintf(stderr, "арена не поднялась\n");
        return 70;
    }
    g_sink = tmpfile();
    if (!g_sink) { fprintf(stderr, "tmpfile недоступен\n"); return 70; }

    test_kernel();
    test_shapes();
    test_errors();
    test_recovery();
    test_ambiguity();

    fclose(g_sink);
    smp_arena_release(&g_arena);
    return REPORT();
}
