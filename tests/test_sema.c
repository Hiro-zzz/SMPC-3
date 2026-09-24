/* SMPC3 :: test_sema.c -- проверки Ф3: семантика. */
#include "smpc3/io.h"
#include "smpc3/lex.h"
#include "smpc3/parse.h"
#include "smpc3/sema.h"

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
static char       g_out[16384];

/* Прогон всего фронтенда. Возвращает число семантических ошибок;
 * лексические/синтаксические считаются провалом теста. */
static uint32_t sema_str(const char *text, SmpSemaResult *res, uint32_t *warns)
{
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
    CHECK(lx.n_errors == 0, "лексер сломался на семантическом тесте: %s", text);

    SmpParser P;
    smp_parse_init(&P, &g_arena, &g_diag, &g_src, toks, ntok);
    SmpAstProgram prog;
    smp_parse(&P, &prog);
    CHECK(P.n_errors == 0, "парсер сломался на семантическом тесте: %s", text);

    /* Развёртка [#repeat:N] идёт до семантики — ровно как в настоящем
     * конвейере. Прогонять её иначе значило бы проверять не тот компилятор,
     * который потом собирает модули. */
    uint32_t n_exp = 0;
    smp_ast_expand(&prog, &g_arena, &g_diag, &n_exp);

    SmpSema sm;
    smp_sema_init(&sm, &g_arena, &g_diag, &g_src);
    /* Железо фиксируем, чтобы тесты не зависели от машины. */
    sm.host_vec_bits = 256;
    memset(res, 0, sizeof *res);
    if (!n_exp) smp_sema_run(&sm, &prog, res);

    /* Забираем напечатанное для проверки кодов. */
    rewind(g_sink);
    const size_t n = fread(g_out, 1, sizeof(g_out) - 1u, g_sink);
    g_out[n] = '\0';

    if (warns) *warns = sm.n_warnings;
    return n_exp ? n_exp : sm.n_errors;
}

/* Сколько инструкций осталось после развёртки. */
static uint32_t expanded(const char *text)
{
    SmpSemaResult res;
    sema_str(text, &res, NULL);
    return res.ninfo;
}

static bool saw(SmpDiagCode c) { return strstr(g_out, smp_diag_info(c)->text) != NULL; }

static void expect_err(const char *text, SmpDiagCode want, const char *label)
{
    SmpSemaResult res;
    const uint32_t errs = sema_str(text, &res, NULL);
    CHECK(errs == 1, "%s: ошибок %u, ждали 1", label, errs);
    CHECK(saw(want), "%s: нет %s в выводе", label, smp_diag_info(want)->text);
}

static void expect_clean(const char *text, const char *label)
{
    SmpSemaResult res;
    uint32_t warns = 0;
    const uint32_t errs = sema_str(text, &res, &warns);
    CHECK(errs == 0, "%s: %u ошибок на корректном коде", label, errs);
}

/* Пролог: две матрицы и один регистр, чтобы не повторять в каждом тесте. */
#define P2 "*&A<f32:4,4> -> @alloc => $a;  *&B<f32:4,4> -> @alloc => $b;  "

/* ========================================================================== */

static void test_flow(void)
{
    SECTION("протаскивание типов");

    SmpSemaResult res;
    uint32_t warns = 0;

    const uint32_t errs = sema_str(
        "[#arena:0] *&A<f32:8,4> -> @alloc => $a;\n"
        "[#arena:0] *&B<f32:4,2> -> @alloc => $b;\n"
        "[#simd:v256] $a -> @mmul($b) => *&C<f32:8,2>;\n"
        "*&C -> @relu -> @reduce.add => $s;\n"
        "$s -> @cast.f64 => $t;\n", &res, &warns);   /* @scale требует тензор */

    CHECK(errs == 0, "корректная программа дала %u ошибок", errs);
    CHECK(res.ninfo == 5, "инструкций %u", res.ninfo);
    if (res.ninfo != 5) return;

    /* mmul: [8,4] x [4,2] -> [8,2] */
    CHECK(res.info[2].ok, "#2 не прошла проверку");
    CHECK(res.info[2].result.rank == 2, "#2 ранг %u", res.info[2].result.rank);
    CHECK(res.info[2].result.shape[0] == 8 && res.info[2].result.shape[1] == 2,
          "#2 форма %ux%u", res.info[2].result.shape[0], res.info[2].result.shape[1]);
    CHECK(res.info[2].result.dtype == SMP_DT_F32, "#2 тип");

    /* reduce схлопывает в скаляр */
    CHECK(res.info[3].result.is_scalar, "#3 результат не скаляр");
    CHECK(res.info[3].result.rank == 0, "#3 ранг %u", res.info[3].result.rank);

    /* Разложение по арене: 64-байтные границы и никаких наложений. */
    uint32_t ntensors = 0;
    for (uint32_t i = 0; i < res.nsyms; i++) {
        const SmpSym *s = &res.syms[i];
        if (s->kind != SMP_SYM_TENSOR) continue;
        ntensors++;
        CHECK(s->offset % 64u == 0, "тензор '%.*s' лежит по смещению %llu — не кратно 64",
              (int)s->name.len, s->name.p, (unsigned long long)s->offset);
        for (uint32_t j = 0; j < res.nsyms; j++) {
            if (j == i || res.syms[j].kind != SMP_SYM_TENSOR) continue;
            const SmpSym *o = &res.syms[j];
            if (o->arena_id != s->arena_id) continue;
            const bool overlap = s->offset < o->offset + o->bytes &&
                                 o->offset < s->offset + s->bytes;
            CHECK(!overlap, "тензоры '%.*s' и '%.*s' наложились",
                  (int)s->name.len, s->name.p, (int)o->name.len, o->name.p);
        }
    }
    CHECK(ntensors == 3, "тензоров %u, ждали 3", ntensors);
    CHECK(res.arena_bytes[0] > 0, "арена #0 пуста");
}

/* ========================================================================== */

static void test_ops(void)
{
    SECTION("операции");

    expect_clean(P2 "$a -> @mmul($b) => *&C<f32:4,4>;", "mmul совместимых");
    expect_clean(P2 "$a -> @transpose => *&T<f32:4,4>;", "transpose");
    expect_clean(P2 "$a -> @add($b) => *&S<f32:4,4>;", "поэлементное сложение");
    expect_clean(P2 "$a -> @relu -> @abs => *&R<f32:4,4>;", "цепочка поэлементных");
    expect_clean(P2 "$a -> @reduce.max => $m;", "reduce.max");
    expect_clean(P2 "$a -> @cast.f64 => *&D<f64:4,4>;", "cast меняет тип");
    expect_clean(P2 "$a -> @scale(0.5) => *&Q<f32:4,4>;", "scale скаляром");
    expect_clean("*&A<f32:4,4> -> @fill(1.0) => *&A;", "fill в себя без no-alias");

    /* transpose неквадратной: форма обязана перевернуться */
    {
        SmpSemaResult res;
        CHECK(sema_str("*&A<f32:8,3> -> @alloc => $a;\n"
                       "$a -> @transpose => *&T<f32:3,8>;\n", &res, NULL) == 0,
              "transpose [8,3] -> [3,8]");
    }

    expect_err(P2 "$a -> @nosuchop => $c;",     SMP_E0305, "неизвестная операция");
    expect_err(P2 "$a -> @mmul => $c;",         SMP_E0306, "mmul без аргумента");
    expect_err(P2 "$a -> @relu($b) => $c;",     SMP_E0306, "relu с аргументом");
    expect_err(P2 "$a -> @scale($b) => $c;",    SMP_E0309, "scale тензором");
    expect_err(P2 "$a -> @add(2.0) => $c;",     SMP_E0309, "add скаляром");
    expect_err(P2 "$a -> @reduce.add -> @relu => $c;", SMP_E0309, "relu над скаляром");

    /* Вывод — такая же свёртка: тензор на входе, скаляр на выходе. */
    expect_clean(P2 "$a -> @emit.num => $n;", "@emit.num над f32");
    expect_clean("*&H<i32:4> -> @alloc => $h;  *&H -> @emit.text => $n;",
                 "@emit.text над i32");
    expect_clean("*&H<i32:4> -> @alloc => $h;\n"
                 "*&H -> @emit.dec -> @cast.f32 => $n;", "@emit внутри конвейера");

    /* Текстовые форматы печатают коды, а коды — целые. Молча округлять за
     * пользователя язык не станет. */
    expect_err(P2 "$a -> @emit.text => $n;", SMP_E0303, "f32 в @emit.text");
    expect_err(P2 "$a -> @emit.hex => $n;",  SMP_E0303, "f32 в @emit.hex");
    expect_err(P2 "$a -> @emit.bits => $n;", SMP_E0303, "f32 в @emit.bits");
    expect_err("*&P<raw_ptr:4> -> @alloc => $p;  *&P -> @emit.num => $n;",
               SMP_E0303, "печать сырого указателя");

    /* Подсказка по опечатке обязана появиться. */
    {
        SmpSemaResult res;
        sema_str(P2 "$a -> @realu => $c;", &res, NULL);
        CHECK(strstr(g_out, "Ты имел в виду @relu") != NULL,
              "нет подсказки про @relu");
    }
    /* А на совсем чужом слове — не должна выдумываться. */
    {
        SmpSemaResult res;
        sema_str(P2 "$a -> @quantize => $c;", &res, NULL);
        CHECK(strstr(g_out, "Ты имел в виду") == NULL,
              "подсказка выдумана на непохожем имени");
    }
}

/* ========================================================================== */

static void test_shapes(void)
{
    SECTION("формы и K-совместимость");

    /* Флагманская проверка спецификации. */
    {
        SmpSemaResult res;
        const uint32_t errs = sema_str(
            "*&A<f32:1024,1024> -> @alloc => $a;\n"
            "*&B<f32:512,512>   -> @alloc => $b;\n"
            "$a -> @mmul($b) => *&C<f32:1024,512>;\n", &res, NULL);
        CHECK(errs == 1, "K-несовпадение дало %u ошибок", errs);
        CHECK(saw(SMP_E0419), "нет E0419");
        CHECK(strstr(g_out, "1024 != 512") != NULL, "в деталях нет самих размерностей");
        CHECK(strstr(g_out, "@transpose") != NULL, "в исправлении нет @transpose");
    }

    /* А совместимые прямоугольные — законны. */
    expect_clean("*&A<f32:1024,1024> -> @alloc => $a;\n"
                 "*&B<f32:1024,512>  -> @alloc => $b;\n"
                 "$a -> @mmul($b) => *&C<f32:1024,512>;", "прямоугольный mmul");

    expect_err(P2 "$a -> @mmul($b) => *&C<f32:4,8>;", SMP_E0303, "форма приёмника");
    expect_err("*&A<f32:4,4> -> @alloc => $a;\n"
               "*&B<f64:4,4> -> @alloc => $b;\n"
               "$a -> @mmul($b) => *&C<f32:4,4>;", SMP_E0303, "разные типы множителей");
    expect_err("*&A<f32:4,4,4> -> @alloc => $a;\n"
               "$a -> @transpose => $t;", SMP_E0309, "transpose ранга 3");
    expect_err(P2 "$a -> @reduce.add => *&S<f32:4,4>;", SMP_E0303, "скаляр в матрицу");
}

/* ========================================================================== */

static void test_names(void)
{
    SECTION("имена и объявления");

    expect_err("*&A<f32:4,4> -> @alloc => $a;\n"
               "*&A<f32:4,4> -> @alloc => $b;", SMP_E0308, "повторное объявление");
    expect_err("*&NOPE -> @relu => $a;",        SMP_E0307, "неизвестный тензор");
    expect_err("$nope -> @relu => $a;",         SMP_E0307, "неизвестный регистр");

    /* Подсказка по близкому имени символа. */
    {
        SmpSemaResult res;
        sema_str("*&Alpha<f32:4,4> -> @alloc => $a;\n*&Alpah -> @relu => $b;",
                 &res, NULL);
        CHECK(strstr(g_out, "Ты имел в виду *&Alpha") != NULL,
              "нет подсказки по имени тензора");
    }

    /* Алиасинг без обещания [!no-alias] — законен; отдельный тест на это
     * живёт в разделе про память. */
}

/* ========================================================================== */

static void test_memory(void)
{
    SECTION("память, выравнивание, алиасинг");

    /* Ложное [!no-alias]. */
    expect_err("*&A<f32:4,4> -> @alloc => $a;\n"
               "$a -> @relu => *&A [!no-alias];", SMP_E0420, "ложное no-alias");

    /* Без обещания то же самое законно. */
    expect_clean("*&A<f32:4,4> -> @alloc => $a;\n"
                 "$a -> @relu => *&A;", "запись в себя без обещания");

    /* Строка матрицы плотная — под ^raw проходит. */
    expect_clean("*&A<f32:8,8> -> @alloc => $a;\n"
                 "[^raw] *&A[0, ..] -> @reduce.add => $s;", "^raw над плотной строкой");

    /* Столбец имеет шаг — ^raq его не вынесет. */
    expect_err("*&A<f32:8,8> -> @alloc => $a;\n"
               "[^raw] *&A[.., 0] -> @reduce.add => $s;", SMP_E0418, "^raw над столбцом");

    /* Без ^raw столбец законен: шаг известен компилятору. */
    expect_clean("*&A<f32:8,8> -> @alloc => $a;\n"
                 "*&A[.., 0] -> @reduce.add => $s;", "столбец без ^raw");

    /* Выравнивание: строка длиной 1023xf32 = 4092 байта, не кратно 32. */
    expect_err("*&W<f32:1023,1023> -> @alloc => $w;\n"
               "[#simd:v256] *&W[1, ..] -> @relu => *&V<f32:1023>;",
               SMP_E0402, "срез сбил выравнивание под v256");

    /* Строка длиной 1024xf32 = 4096 байт — кратно 32 и 64. */
    expect_clean("*&W<f32:1024,1024> -> @alloc => $w;\n"
                 "[#simd:v256] *&W[1, ..] -> @relu => *&V<f32:1024>;",
                 "выровненный срез");

    /* Без SIMD выравнивание не требуется. */
    expect_clean("*&W<f32:1023,1023> -> @alloc => $w;\n"
                 "*&W[1, ..] -> @relu => *&V<f32:1023>;", "срез без #simd");

    /* Запись плотного результата в разреженный вид. */
    expect_err("*&A<f32:8,8> -> @alloc => $a;\n"
               "*&B<f32:8,8> -> @alloc => $b;\n"
               "*&B[.., 0] -> @relu => *&A[.., 0];", SMP_E0421, "запись в столбец");

    /* Индекс за границей. */
    expect_err("*&A<f32:4,4> -> @alloc => $a;\n"
               "*&A[9, ..] -> @relu => $r;", SMP_E0410, "индекс за границей");

    /* @pack чинит разреженность. */
    expect_clean("*&A<f32:8,8> -> @alloc => $a;\n"
                 "[^raw] *&A[.., 0] -> @pack -> @reduce.add => $s;",
                 "@pack перед ^raw");
}

/* ========================================================================== */

static void test_isa(void)
{
    SECTION("ISA и атрибуты");

    /* v512 на машине с v256: предупреждение, но не ошибка. */
    {
        SmpSemaResult res;
        uint32_t warns = 0;
        const uint32_t errs = sema_str(
            "*&A<f32:8,8> -> @alloc => $a;\n"
            "[#simd:v512] $a -> @relu => *&R<f32:8,8>;", &res, &warns);
        CHECK(errs == 0, "v512 без ?strict дал %u ошибок", errs);
        CHECK(saw(SMP_W0512), "нет W0512");
        CHECK(res.ninfo == 2 && res.info[1].req_vec_bits == 512,
              "запрошенная ширина не сохранена");
        CHECK(res.ninfo == 2 && res.info[1].vec_bits == 256,
              "фактическая ширина %u, ждали 256",
              res.ninfo == 2 ? res.info[1].vec_bits : 0);
    }

    /* Тот же код с ?strict — фатально. */
    expect_err("*&A<f32:8,8> -> @alloc => $a;\n"
               "[#simd:v512] $a -> @relu => *&R<f32:8,8> [?strict];",
               SMP_E0502, "v512 при ?strict");

    /* v256 запрашивать можно всегда. */
    expect_clean("*&A<f32:8,8> -> @alloc => $a;\n"
                 "[#simd:v256] $a -> @relu => *&R<f32:8,8> [?strict];", "v256 strict");

    expect_err("*&A<f32:4,4> -> @alloc => $a; [#simd:v1024] $a -> @relu => $r;",
               SMP_E0501, "несуществующая ширина");
    expect_err("*&A<f32:4,4> -> @alloc => $a; [#arena:99] $a -> @relu => $r;",
               SMP_E0311, "арена вне диапазона");
    expect_err("*&A<f32:4,4> -> @alloc => $a; [#nosuch:1] $a -> @relu => $r;",
               SMP_E0311, "неизвестная директива");
    expect_err("*&A<f32:4,4> -> @alloc => $a; [^nosuch] $a -> @relu => $r;",
               SMP_E0311, "неизвестный режим");
    expect_err("*&A<f32:4,4> -> @alloc => $a; $a -> @relu => $r [!nosuch];",
               SMP_E0311, "неизвестное утверждение");
    expect_err("*&A<f32:4,4> -> @alloc => $a; [#simd] $a -> @relu => $r;",
               SMP_E0311, "#simd без значения");
    expect_err("*&A<f32:4,4> -> @alloc => $a; [#arena:x] $a -> @relu => $r;",
               SMP_E0311, "#arena не числом");

    /* Подсказка по опечатке в атрибуте. */
    {
        SmpSemaResult res;
        sema_str("*&A<f32:4,4> -> @alloc => $a; $a -> @relu => $r [~flush-to-zerro];",
                 &res, NULL);
        CHECK(strstr(g_out, "~flush-to-zero.") != NULL, "нет подсказки по fp-режиму");
    }

    /* Разные арены не пересекаются по адресам. */
    {
        SmpSemaResult res;
        CHECK(sema_str("[#arena:0] *&A<f32:4,4> -> @alloc => $a;\n"
                       "[#arena:1] *&B<f32:4,4> -> @alloc => $b;\n", &res, NULL) == 0,
              "две арены");
        CHECK(res.max_arena_id == 1, "max_arena_id %u", res.max_arena_id);
        CHECK(res.arena_bytes[0] == 64 && res.arena_bytes[1] == 64,
              "размеры арен %llu и %llu",
              (unsigned long long)res.arena_bytes[0],
              (unsigned long long)res.arena_bytes[1]);
    }
}

/* ========================================================================== */

static void test_warnings(void)
{
    SECTION("предупреждения");

    SmpSemaResult res;
    uint32_t warns = 0;

    /* Записали и не прочитали. */
    sema_str("*&A<f32:4,4> -> @alloc => $unused;", &res, &warns);
    CHECK(warns == 1 && saw(SMP_W0301), "нет W0301 про мёртвую запись");

    /* Прочитали — претензий нет. Приёмник-тензор под W0301 не подпадает:
     * предупреждение про мёртвую запись касается только регистров. */
    sema_str("*&A<f32:4,4> -> @alloc => $a;  $a -> @relu => *&R<f32:4,4>;",
             &res, &warns);
    CHECK(warns == 0, "предупреждений %u, ждали 0", warns);

    /* Чтение тензора без записи. */
    sema_str("*&A<f32:4,4> -> @alloc => $a;\n"
             "*&B<f32:4,4> -> @alloc => $b;\n"
             "*&B -> @relu => *&A;", &res, &warns);
    CHECK(!saw(SMP_W0302), "W0302 сработало на инициализированном тензоре");

    sema_str("*&A<f32:4,4> -> @alloc => $a;\n"
             "*&U<f32:4,4> -> @alloc => $u2;\n"
             "*&U -> @relu => *&A;", &res, &warns);
    CHECK(!saw(SMP_W0302), "@alloc должен считаться инициализацией");

    /* Одна претензия на инструкцию, а не каскад. */
    {
        const uint32_t errs = sema_str(
            "*&A<f32:4,4> -> @alloc => $a;\n"
            "[#nosuch:1, ^alsonosuch] $a -> @nope($undefined) => *&A [!bad];",
            &res, &warns);
        CHECK(errs == 1, "инструкция с пятью дефектами дала %u претензий", errs);
    }
}

/* ========================================================================== */

static void test_registries(void)
{
    SECTION("реестры");

    for (unsigned i = 0; i < SMP_OP__COUNT; i++) {
        const SmpOpDef *d = smp_op_def((SmpOpKind)i);
        CHECK(d->name && d->name[0], "операция #%u без имени", i);
        CHECK(d->min_args <= d->max_args, "у @%s min>max", d->name);
        SmpName n = { d->name, (uint32_t)strlen(d->name) };
        CHECK(smp_op_lookup(n) == (SmpOpKind)i, "поиск @%s", d->name);
    }
    {
        SmpName bogus = { "nosuchop", 8 };
        CHECK(smp_op_lookup(bogus) == SMP_OP__UNKNOWN, "несуществующая операция найдена");
    }
}

/* ========================================================================== */


/* ========================================================================== */
/*  Развёртка [#repeat:N]                                                     */
/* ========================================================================== */

#define PM "*&M<f32:4,8> -> @alloc => $m;  "

static void test_repeat(void)
{
    SECTION("развёртка [#repeat:N]");

    /* --- что обязано проходить --- */
    expect_clean(PM "[#repeat:4, #index:i] *&M[$i, ..] -> @abs => *&M[$i, ..];",
                 "четыре строки подряд");
    expect_clean(PM "[#repeat:4, #index:i] *&M[$i, ..] -> @fill($i) => *&M[$i, ..];",
                 "индекс подставляется и как значение");
    expect_clean(PM "[#repeat:1] *&M -> @abs => *&M;",
                 "повтор без индекса");
    expect_clean(PM "*&R<f32:4> -> @alloc => $r;  "
                    "[#repeat:4, #index:i] *&M[$i, ..] -> @reduce.add => *&R[$i];",
                 "скаляр в отдельный элемент приёмника");

    /* --- копий ровно столько, сколько просили --- */
    CHECK(expanded(PM "[#repeat:4, #index:i] *&M[$i, ..] -> @abs => *&M[$i, ..];") == 5,
          "развёртка на 4 дала %u инструкций вместо 5",
          expanded(PM "[#repeat:4, #index:i] *&M[$i, ..] -> @abs => *&M[$i, ..];"));
    CHECK(expanded(PM "*&M -> @abs => *&M;") == 2,
          "без развёртки инструкций стало %u вместо 2",
          expanded(PM "*&M -> @abs => *&M;"));

    /* --- индекс не переживает свою инструкцию ---
     * Он существует только на время развёртки: снаружи это просто имя, в
     * которое никто ничего не писал. */
    expect_err(PM "[#repeat:4, #index:i] *&M[$i, ..] -> @abs => *&M[$i, ..];  "
                  "$i -> @emit.num => $n;",
               SMP_E0307, "индекс виден за пределами развёртки");

    /* --- ошибки самой развёртки --- */
    expect_err(PM "[#repeat:0, #index:i] *&M[$i, ..] -> @abs => *&M[$i, ..];",
               SMP_E0310, "ноль повторов");
    expect_err(PM "[#repeat:9999, #index:i] *&M[$i, ..] -> @abs => *&M[$i, ..];",
               SMP_E0310, "повторов больше потолка");
    expect_err(PM "[#index:i] *&M -> @abs => *&M;",
               SMP_E0310, "#index без #repeat");
    expect_err(PM "[#repeat:4, #index:i] *&M[$i, ..] -> @abs => *&BAD<f32:8>;",
               SMP_E0310, "объявление тензора внутри развёртки");
    expect_err(PM "2 => $i;  "
                  "[#repeat:4, #index:i] *&M[$i, ..] -> @abs => *&M[$i, ..];",
               SMP_E0310, "имя индекса занято обычным регистром");

    /* --- выход за границу оси ловится на компиляции --- *
     * Ровно одним сообщением: копий-нарушителей две (повторы 4 и 5), но
     * написана-то одна строка. Если тут когда-нибудь станет две ошибки,
     * значит латч в serr перестал работать. */
    expect_err(PM "[#repeat:6, #index:i] *&M[$i, ..] -> @abs => *&M[$i, ..];",
               SMP_E0410, "индекс за границей оси на пятом повторе");
}

/* ========================================================================== */
/*  @load и @store                                                            */
/* ========================================================================== */

static void test_fileio_rules(void)
{
    SECTION("@load и @store");

    expect_clean("*&A<f32:16> -> @load => $a;", "загрузка в объявленный тензор");
    expect_clean("*&A<f32:16> -> @alloc => $a;  *&A -> @store => $n;  "
                 "$n -> @emit.dec => $m;", "выгрузка именованного тензора");
    expect_clean("*&M<f32:4,8> -> @alloc => $m;  *&M[1, ..] -> @store => $n;  "
                 "$n -> @emit.dec => $k;", "строка идёт подряд — выгружается");
    expect_clean("*&A<f32:16> -> @load -> @relu => *&B<f32:16>;",
                 "@load первой стадией, дальше обычный конвейер");

    /* Столбец идёт с шагом: записать его одним куском нельзя. */
    expect_err("*&M<f32:4,8> -> @alloc => $m;  *&M[.., 1] -> @store => $n;",
               SMP_E0309, "срез с шагом");

    /* После вычисляющей стадии имени, по которому искать файл, уже нет.
     * Ловиться это обязано здесь: в рантайме человек увидел бы внутреннее
     * имя временного буфера вместо объяснения. */
    expect_err("*&A<f32:16> -> @alloc => $a;  *&A -> @relu -> @store => $n;",
               SMP_E0309, "@store после вычисляющей стадии");
    expect_err("*&A<f32:16> -> @alloc => $a;  *&A -> @relu -> @load => $b;",
               SMP_E0309, "@load после вычисляющей стадии");

    /* Скаляр не тензор: выгружать нечего. */
    expect_err("*&A<f32:16> -> @alloc => $a;  *&A -> @reduce.add => $s;  "
               "$s -> @store => $n;",
               SMP_E0309, "@store от скаляра");
}

/* ========================================================================== */

/* q8_0 — формат весов: объявить, загрузить, взять строкой, распаковать,
 * умножить @mmul.t. Всё прочее над ним — E0312. */
#define Q8 "*&W<q8_0:64,32> -> @alloc => $w;\n"

static void test_q8(void)
{
    SECTION("q8_0 и @mmul.t");

    expect_clean(Q8 "*&x<f32:2,32> -> @alloc => $x;\n"
                    "$x -> @mmul.t(*&W) => *&y<f32:2,64>;", "линейный слой на q8_0");
    expect_clean(Q8 "*&W[5, ..] -> @cast.f32 => *&r<f32:32>;", "строка весов в f32");
    expect_clean(Q8 "3 => $t;\n*&W[$t, ..] -> @cast.f32 => *&r<f32:32>;",
                 "строка по номеру из регистра");
    expect_clean("*&F<f32:64,32> -> @alloc => $f;\n"
                 "$f -> @cast.q8_0 => *&Q<q8_0:64,32>;", "квантование");
    expect_clean("*&x<f32:3,8> -> @alloc => $x;\n"
                 "*&W<f32:5,8> -> @alloc => $w;\n"
                 "$x -> @mmul.t($w) => *&y<f32:3,5>;", "@mmul.t на f32");

    expect_err("*&W<q8_0:64,30> -> @alloc => $w;", SMP_E0312, "ось не кратна 32");
    expect_err("*&W<q8_0> -> @alloc => $w;", SMP_E0312, "скаляр q8_0");
    expect_err(Q8 "*&W -> @relu => *&V<q8_0:64,32>;", SMP_E0312, "@relu над q8_0");
    expect_err(Q8 "*&W -> @reduce.add => $s;", SMP_E0312, "свёртка q8_0");
    expect_err(Q8 "*&W[1, 3] -> @cast.f32 => $s;", SMP_E0312, "элемент q8_0");
    expect_err(Q8 "*&W -> @cast.f64 => *&D<f64:64,32>;", SMP_E0312, "q8_0 в f64");
    expect_err(Q8 "*&x<f32:32,64> -> @alloc => $x;\n"
                  "*&W -> @transpose -> @cast.f32 => *&T<f32:32,64>;", SMP_E0312, "transpose q8_0");
    expect_err(Q8 "*&x<f32:2,64> -> @alloc => $x;\n"
                  "$x -> @mmul(*&W) => *&y<f32:2,32>;", SMP_E0312, "q8_0 в обычном @mmul");
    expect_err(Q8 "*&x<f64:2,32> -> @alloc => $x;\n"
                  "$x -> @mmul.t(*&W) => *&y<f64:2,64>;", SMP_E0303, "q8_0 на f64");
    expect_err(Q8 "*&x<f32:2,64> -> @alloc => $x;\n"
                  "$x -> @mmul.t(*&W) => *&y<f32:2,64>;", SMP_E0419, "@mmul.t: K не сходится");
    expect_err("*&F<f64:64,32> -> @alloc => $f;\n"
               "$f -> @cast.q8_0 => *&Q<q8_0:64,32>;", SMP_E0312, "квантовать f64");
    expect_err("*&F<f32:64,32> -> @alloc => $f;\n"
               "*&F -> @transpose -> @cast.q8_0 => *&Q<q8_0:32,64>;", SMP_E0312,
               "квантовать вид с шагом");
}

/* ========================================================================== */

/* Операции модели: всё по строкам, типы — f32/f64, параметры модели —
 * литералами, данные — регистрами. */
#define NN "*&X<f32:2,64> -> @alloc => $x;\n"

static void test_nn_ops(void)
{
    SECTION("операции модели");

    expect_clean(NN "5 => $p;\n3 => $n;\n"
                    "*&X -> @rmsnorm(0.000001) -> @silu => *&A<f32:2,64>;\n"
                    "*&X -> @rope($p, 1000000.0) => *&B<f32:2,64>;\n"
                    "*&X -> @softmax($n) => *&C<f32:2,64>;\n"
                    "*&X -> @softmax -> @sub(*&C) => *&D<f32:2,64>;\n"
                    "*&X -> @reshape(4,32) => *&H<f32:4,32>;\n"
                    "*&X -> @reshape(128) -> @argmax => $i;",
                 "rmsnorm, silu, rope, softmax, sub, reshape, argmax");

    expect_err("*&I<i32:2,8> -> @alloc => $i;\n*&I -> @silu => *&J<i32:2,8>;",
               SMP_E0303, "silu над i32");
    expect_err(NN "1 => $e;\n*&X -> @rmsnorm($e) => *&A<f32:2,64>;",
               SMP_E0309, "eps регистром");
    expect_err(NN "*&X -> @softmax(*&X) => *&A<f32:2,64>;", SMP_E0309, "длина тензором");
    expect_err(NN "*&X -> @rope(*&X, 10000.0) => *&A<f32:2,64>;", SMP_E0309, "позиция тензором");
    expect_err(NN "1 => $p;\n2 => $t;\n*&X -> @rope($p, $t) => *&A<f32:2,64>;",
               SMP_E0309, "основание регистром");
    expect_err("*&O<f32:2,7> -> @alloc => $o;\n*&O -> @rope(1, 10000.0) => *&A<f32:2,7>;",
               SMP_E0309, "rope по нечётной оси");
    expect_err(NN "*&X -> @reshape(3,40) => *&A<f32:3,40>;", SMP_E0309, "reshape: не то число элементов");
    expect_err(NN "*&X -> @reshape(0,128) => $a;", SMP_E0309, "reshape: нулевая ось");
    expect_err(NN "*&X[.., 3] -> @reshape(2,1) => $a;", SMP_E0309, "reshape вида с шагом");
    expect_err(NN "*&X -> @argmax => *&A<u64:1>;", SMP_E0303, "argmax в тензор");
}

/* ========================================================================== */

int main(void)
{
    smp_console_setup();
    fprintf(stderr, "SMPC3 tests :: Ф3\n\n");

    if (smp_arena_init(&g_arena, 32u << 20, 9, "sematest") != SMP_OK) {
        fprintf(stderr, "арена не поднялась\n");
        return 70;
    }
    g_sink = tmpfile();
    if (!g_sink) { fprintf(stderr, "tmpfile недоступен\n"); return 70; }

    test_flow();
    test_ops();
    test_shapes();
    test_names();
    test_memory();
    test_isa();
    test_warnings();
    test_registries();
    test_repeat();
    test_q8();
    test_nn_ops();
    test_fileio_rules();

    fclose(g_sink);
    smp_arena_release(&g_arena);
    return REPORT();
}
