/* SMPC3 :: test_lex.c -- проверки Ф1: лексер. */
#include "smpc3/io.h"
#include "smpc3/lex.h"

#include "harness.h"

#include <string.h>
#include <stdlib.h>

/* ========================================================================== */
/*  Обвязка                                                                   */
/* ========================================================================== */

static SmpArena   g_arena;
static SmpDiagCtx g_diag;
static SmpSource  g_src;
static FILE      *g_sink;   /* диагностика уезжает сюда, а не в консоль */

/* Токенизирует строку целиком. Возвращает число токенов (включая END),
 * *errs — сколько ошибок насчитал лексер. */
static uint32_t lex_str(const char *text, SmpToken **out, uint32_t *errs)
{
    smp_arena_reset(&g_arena);
    rewind(g_sink);

    smp_source_from_memory(&g_src, "t.smpc", text, strlen(text));
    smp_diag_init(&g_diag, &g_src, g_sink);

    SmpLexer lx;
    smp_lex_init(&lx, &g_src, &g_diag);

    uint32_t n = 0;
    smp_lex_all(&lx, &g_arena, out, &n);
    if (errs) *errs = lx.n_errors;
    return n;
}

/* Проверяет, что поток видов токенов совпадает с ожидаемым (без END). */
static void expect_kinds(const char *text, const SmpTokKind *want, uint32_t nwant)
{
    SmpToken *t = NULL;
    uint32_t  errs = 0;
    const uint32_t n = lex_str(text, &t, &errs);

    CHECK(errs == 0, "'%s': %u ошибок лексера", text, errs);
    CHECK(n == nwant + 1u, "'%s': токенов %u, ждали %u", text, n, nwant + 1u);
    if (n != nwant + 1u) return;

    for (uint32_t i = 0; i < nwant; i++)
        CHECK(t[i].kind == want[i], "'%s' токен %u: %s, ждали %s",
              text, i, smp_tok_name(t[i].kind), smp_tok_name(want[i]));

    CHECK(t[nwant].kind == SMP_TK_END, "'%s': поток не завершён END", text);
}

/* ========================================================================== */

static void test_punct(void)
{
    SECTION("пунктуация и стрелки");

    /* Двухсимвольные лексемы должны выигрывать у односимвольных. */
    {
        static const SmpTokKind w[] = { SMP_TK_ARROW, SMP_TK_FATARROW,
                                        SMP_TK_TENSOR, SMP_TK_DOTDOT };
        expect_kinds("-> => *& ..", w, 4);
    }
    {
        static const SmpTokKind w[] = { SMP_TK_MINUS, SMP_TK_STAR,
                                        SMP_TK_DOT, SMP_TK_RANGLE };
        expect_kinds("- * . >", w, 4);
    }
    {
        static const SmpTokKind w[] = {
            SMP_TK_LBRACKET, SMP_TK_RBRACKET, SMP_TK_LPAREN, SMP_TK_RPAREN,
            SMP_TK_LANGLE, SMP_TK_RANGLE, SMP_TK_COMMA, SMP_TK_SEMI, SMP_TK_COLON
        };
        expect_kinds("[]()<>,;:", w, 9);
    }

    /* Без пробелов лексер обязан разбирать так же. */
    {
        static const SmpTokKind w[] = { SMP_TK_ARROW, SMP_TK_FATARROW, SMP_TK_TENSOR };
        expect_kinds("->=>*&", w, 3);
    }
}

/* ========================================================================== */

static void test_sigils(void)
{
    SECTION("сигилы");

    SmpToken *t = NULL;
    uint32_t  errs = 0;

    lex_str("$r1 @mmul #simd ^raw !no-alias ?strict ~flush-to-zero", &t, &errs);
    CHECK(errs == 0, "чистая строка сигилов дала %u ошибок", errs);

    CHECK(t[0].kind == SMP_TK_REG       && smp_tok_is(&t[0], "r1"),   "$r1");
    CHECK(t[1].kind == SMP_TK_OP        && smp_tok_is(&t[1], "mmul"), "@mmul");
    CHECK(t[2].kind == SMP_TK_DIRECTIVE && smp_tok_is(&t[2], "simd"), "#simd");
    CHECK(t[3].kind == SMP_TK_MODE      && smp_tok_is(&t[3], "raw"),  "^raw");
    CHECK(t[4].kind == SMP_TK_ASSERT    && smp_tok_is(&t[4], "no-alias"), "!no-alias");
    CHECK(t[5].kind == SMP_TK_QUERY     && smp_tok_is(&t[5], "strict"), "?strict");
    CHECK(t[6].kind == SMP_TK_FPMODE    && smp_tok_is(&t[6], "flush-to-zero"), "~ftz");

    /* Тело не включает сигил, а span — включает. */
    CHECK(t[0].tlen == 2, "тело $r1 = %u байт", t[0].tlen);
    CHECK(t[0].span.len == 3, "span $r1 = %u байт", t[0].span.len);
    CHECK(t[0].span.col == 1, "колонка $r1 = %u", t[0].span.col);

    /* Точечные имена операций — один токен. */
    lex_str("@reduce.add @a.b.c", &t, &errs);
    CHECK(errs == 0, "точечные операции дали %u ошибок", errs);
    CHECK(t[0].kind == SMP_TK_OP && smp_tok_is(&t[0], "reduce.add"), "@reduce.add");
    CHECK(t[1].kind == SMP_TK_OP && smp_tok_is(&t[1], "a.b.c"), "@a.b.c");
    CHECK(t[2].kind == SMP_TK_END, "после @a.b.c ожидался END");

    /* Точка, за которой нет буквы, к имени не приклеивается. */
    lex_str("@relu.", &t, &errs);
    CHECK(t[0].kind == SMP_TK_OP && smp_tok_is(&t[0], "relu"), "@relu. -> тело 'relu'");
    CHECK(t[1].kind == SMP_TK_DOT, "@relu. -> отдельная точка");

    /* Кебаб только там, где разрешён: в @-именах дефис не часть имени. */
    lex_str("@no-alias", &t, &errs);
    CHECK(t[0].kind == SMP_TK_OP && smp_tok_is(&t[0], "no"), "@no-alias -> 'no'");
    CHECK(t[1].kind == SMP_TK_MINUS, "@no-alias -> MINUS");

    /* Висящий дефис не съедается. */
    lex_str("!no-", &t, &errs);
    CHECK(t[0].kind == SMP_TK_ASSERT && smp_tok_is(&t[0], "no"), "!no- -> тело 'no'");
    CHECK(t[1].kind == SMP_TK_MINUS, "!no- -> MINUS");

    /* Сигил без тела — E0104. */
    lex_str("$ ;", &t, &errs);
    CHECK(errs == 1, "'$ ;' дал %u ошибок, ждали 1", errs);
    CHECK(t[0].kind == SMP_TK_ERROR, "'$' должен дать ERROR");
    CHECK(t[1].kind == SMP_TK_SEMI, "после ошибки лексер обязан продолжить");
}

/* ========================================================================== */

static void test_numbers(void)
{
    SECTION("числа");

    SmpToken *t = NULL;
    uint32_t  errs = 0;

    lex_str("0 1024 0xDEAD 0b1011 1_000_000", &t, &errs);
    CHECK(errs == 0, "целые дали %u ошибок", errs);
    CHECK(t[0].kind == SMP_TK_INT && t[0].num.u == 0, "0");
    CHECK(t[1].kind == SMP_TK_INT && t[1].num.u == 1024, "1024 -> %llu",
          (unsigned long long)t[1].num.u);
    CHECK(t[2].kind == SMP_TK_INT && t[2].num.u == 0xDEAD, "0xDEAD -> %llu",
          (unsigned long long)t[2].num.u);
    CHECK(t[3].kind == SMP_TK_INT && t[3].num.u == 11, "0b1011 -> %llu",
          (unsigned long long)t[3].num.u);
    CHECK(t[4].kind == SMP_TK_INT && t[4].num.u == 1000000, "1_000_000 -> %llu",
          (unsigned long long)t[4].num.u);

    lex_str("1.5 1.5e-3 2E+4 0.0", &t, &errs);
    CHECK(errs == 0, "вещественные дали %u ошибок", errs);
    CHECK(t[0].kind == SMP_TK_FLOAT && t[0].num.f == 1.5, "1.5 -> %g", t[0].num.f);
    CHECK(t[1].kind == SMP_TK_FLOAT && t[1].num.f > 0.0014 && t[1].num.f < 0.0016,
          "1.5e-3 -> %g", t[1].num.f);
    CHECK(t[2].kind == SMP_TK_FLOAT && t[2].num.f == 20000.0, "2E+4 -> %g", t[2].num.f);
    CHECK(t[3].kind == SMP_TK_FLOAT && t[3].num.f == 0.0, "0.0 -> %g", t[3].num.f);

    /* Ключевой случай: '..' в индексе не должен съедаться как дробная часть. */
    lex_str("0..7", &t, &errs);
    CHECK(errs == 0, "'0..7' дал %u ошибок", errs);
    CHECK(t[0].kind == SMP_TK_INT && t[0].num.u == 0, "0..7 -> INT");
    CHECK(t[1].kind == SMP_TK_DOTDOT, "0..7 -> DOTDOT, а не FLOAT");
    CHECK(t[2].kind == SMP_TK_INT && t[2].num.u == 7, "0..7 -> INT");

    /* А одиночная точка — часть числа. */
    lex_str("0.7", &t, &errs);
    CHECK(t[0].kind == SMP_TK_FLOAT, "0.7 должно быть FLOAT");

    /* Битые литералы. */
    lex_str("1.", &t, &errs);
    CHECK(errs == 1 && t[0].kind == SMP_TK_ERROR, "'1.' должен быть E0103");

    lex_str("123abc", &t, &errs);
    CHECK(errs == 1 && t[0].kind == SMP_TK_ERROR, "'123abc' должен быть E0103");
    CHECK(t[1].kind == SMP_TK_END, "'123abc' обязан съесться целиком");

    lex_str("0x", &t, &errs);
    CHECK(errs == 1 && t[0].kind == SMP_TK_ERROR, "'0x' без цифр должен быть E0103");

    lex_str("0b2", &t, &errs);
    CHECK(errs == 1, "'0b2' должен быть E0103");
}

/* ========================================================================== */

static void test_trivia(void)
{
    SECTION("комментарии и позиции");

    SmpToken *t = NULL;
    uint32_t  errs = 0;

    lex_str("// весь этот текст игнорируется\n$r1", &t, &errs);
    CHECK(errs == 0, "строчный комментарий дал %u ошибок", errs);
    CHECK(t[0].kind == SMP_TK_REG, "после // ожидался $r1");
    CHECK(t[0].span.line == 2, "строка после комментария = %u", t[0].span.line);
    CHECK(t[0].span.col == 1, "колонка после комментария = %u", t[0].span.col);

    /* Кириллица внутри комментария законна: она не идентификатор. */
    lex_str("/* многострочный\n   комментарий */ $r1", &t, &errs);
    CHECK(errs == 0, "блочный комментарий дал %u ошибок", errs);
    CHECK(t[0].kind == SMP_TK_REG, "после /* */ ожидался $r1");
    CHECK(t[0].span.line == 2, "строка после блочного = %u", t[0].span.line);

    /* Незакрытый блок — E0102, и указывать он обязан на открытие. */
    lex_str("$r1 /* и всё", &t, &errs);
    CHECK(errs == 1, "незакрытый комментарий дал %u ошибок", errs);

    /* Позиции в многострочном тексте. */
    lex_str("$a\n  $b\n\n    $c", &t, &errs);
    CHECK(t[0].span.line == 1 && t[0].span.col == 1, "$a -> %u:%u",
          t[0].span.line, t[0].span.col);
    CHECK(t[1].span.line == 2 && t[1].span.col == 3, "$b -> %u:%u",
          t[1].span.line, t[1].span.col);
    CHECK(t[2].span.line == 4 && t[2].span.col == 5, "$c -> %u:%u",
          t[2].span.line, t[2].span.col);

    /* CRLF не должен сдвигать колонки. */
    lex_str("$a\r\n$b", &t, &errs);
    CHECK(errs == 0, "CRLF дал %u ошибок", errs);
    CHECK(t[1].span.line == 2 && t[1].span.col == 1, "после CRLF -> %u:%u",
          t[1].span.line, t[1].span.col);

    /* Пустой вход. */
    const uint32_t n = lex_str("", &t, &errs);
    CHECK(n == 1 && t[0].kind == SMP_TK_END, "пустой вход -> %u токенов", n);
    CHECK(errs == 0, "пустой вход дал %u ошибок", errs);

    /* Только комментарий. */
    CHECK(lex_str("// ничего\n", &t, &errs) == 1, "файл из одного комментария");
}

/* ========================================================================== */

static void test_garbage(void)
{
    SECTION("мусор и восстановление");

    SmpToken *t = NULL;
    uint32_t  errs = 0;

    /* Многобайтовая кодовая точка вне комментария — ОДНА ошибка, не три. */
    lex_str("$r1 щ $r2", &t, &errs);
    CHECK(errs == 1, "кириллица вне комментария дала %u ошибок, ждали 1", errs);
    CHECK(t[0].kind == SMP_TK_REG && t[1].kind == SMP_TK_REG,
          "лексер обязан восстановиться и добрать $r2");

    /* Целое кириллическое слово — тоже ОДНА ошибка, а не по одной на букву. */
    lex_str("$r1 фильтр $r2", &t, &errs);
    CHECK(errs == 1, "слово из 6 букв дало %u ошибок, ждали 1", errs);
    CHECK(t[0].kind == SMP_TK_REG && t[1].kind == SMP_TK_REG,
          "после кириллического слова ожидались оба регистра");

    /* Сигил с не-ASCII телом: ровно две претензии — пустое имя и сам мусор. */
    lex_str("@фильтр", &t, &errs);
    CHECK(errs == 2, "'@фильтр' дал %u ошибок, ждали 2", errs);

    /* Разные серии не сливаются между собой. */
    lex_str("аб $r1 вг", &t, &errs);
    CHECK(errs == 2, "две разделённые серии дали %u ошибок, ждали 2", errs);

    /* Несколько разных ошибок собираются за один проход. */
    lex_str("$r1 = & $r2 % $r3", &t, &errs);
    CHECK(errs == 3, "'= & %%' дали %u ошибок, ждали 3", errs);
    CHECK(t[0].kind == SMP_TK_REG && t[1].kind == SMP_TK_REG && t[2].kind == SMP_TK_REG,
          "все три регистра должны дойти до потока");

    /* END повторяется бесконечно, а не уезжает за конец буфера. */
    smp_source_from_memory(&g_src, "t.smpc", "$r1", 3);
    smp_diag_init(&g_diag, &g_src, g_sink);
    SmpLexer lx;
    smp_lex_init(&lx, &g_src, &g_diag);
    SmpToken tok;
    smp_lex_next(&lx, &tok);
    for (int i = 0; i < 5; i++) {
        smp_lex_next(&lx, &tok);
        CHECK(tok.kind == SMP_TK_END, "повторный END #%d = %s", i, smp_tok_name(tok.kind));
    }
}

/* ========================================================================== */

static const char g_kernel[] =
    "// Инициализация матриц в выровненной арене\n"
    "[#arena:0]  *&A<f32:1024,1024>  ->  @alloc  =>  $r1;\n"
    "[#simd:v256]  $r1 -> @mmul($r2) => *&C<f32:1024,1024>  [!no-alias, ?strict];\n"
    "[^raw]  *&C[0, ..] -> @relu -> @reduce.add => $f0  [~flush-to-zero];\n";

static void test_kernel(void)
{
    SECTION("пример из спецификации");

    SmpToken *t = NULL;
    uint32_t  errs = 0;
    const uint32_t n = lex_str(g_kernel, &t, &errs);

    CHECK(errs == 0, "пример из спецификации дал %u ошибок", errs);
    CHECK(n > 60, "токенов всего %u", n);

    /* Скобочные группы различимы по первому токену внутри — на этом держится
     * весь разбор в Ф2, поэтому проверяем инвариант прямо здесь. */
    uint32_t n_prefix = 0, n_suffix = 0, n_index = 0;
    for (uint32_t i = 0; i + 1 < n; i++) {
        if (t[i].kind != SMP_TK_LBRACKET) continue;
        switch (t[i + 1].kind) {
            case SMP_TK_DIRECTIVE:
            case SMP_TK_MODE:      n_prefix++; break;
            case SMP_TK_ASSERT:
            case SMP_TK_QUERY:
            case SMP_TK_FPMODE:    n_suffix++; break;
            case SMP_TK_INT:
            case SMP_TK_DOTDOT:
            case SMP_TK_REG:       n_index++;  break;
            default:
                CHECK(0, "группа '[' в %u:%u неклассифицируема: следом %s",
                      t[i].span.line, t[i].span.col, smp_tok_name(t[i + 1].kind));
                break;
        }
    }
    CHECK(n_prefix == 3, "префиксных групп %u, ждали 3", n_prefix);
    CHECK(n_suffix == 2, "суффиксных групп %u, ждали 2", n_suffix);
    CHECK(n_index  == 1, "индексных групп %u, ждали 1", n_index);

    /* Массив токенов непрерывен: на этом стоит smp_lex_all. */
    bool contiguous = true;
    for (uint32_t i = 1; i < n; i++)
        if ((const char *)&t[i] != (const char *)&t[i - 1] + sizeof(SmpToken))
            contiguous = false;
    CHECK(contiguous, "массив токенов не непрерывен");

    /* Смещения строго растут и указывают внутрь исходника. */
    bool ordered = true;
    for (uint32_t i = 1; i < n; i++)
        if (t[i].off < t[i - 1].off) ordered = false;
    CHECK(ordered, "смещения токенов не монотонны");
    CHECK(t[n - 1].off == g_src.len, "END не на конце: off=%u len=%zu",
          t[n - 1].off, g_src.len);
}

/* ========================================================================== */

static void test_registry(void)
{
    SECTION("реестр видов токенов");

    for (unsigned i = 0; i < SMP_TK__COUNT; i++) {
        const char *nm = smp_tok_name((SmpTokKind)i);
        CHECK(nm && nm[0], "вид #%u без имени", i);
    }
    CHECK(strcmp(smp_tok_name(SMP_TK_ARROW), "->") == 0, "имя ARROW");
    CHECK(strcmp(smp_tok_name((SmpTokKind)SMP_TK__COUNT), "<повреждён>") == 0,
          "выход за реестр не отловлен");
}

/* ========================================================================== */

int main(void)
{
    smp_console_setup();
    fprintf(stderr, "SMPC3 tests :: Ф1\n\n");

    if (smp_arena_init(&g_arena, 4u << 20, 9, "lextest") != SMP_OK) {
        fprintf(stderr, "арена не поднялась\n");
        return 70;
    }
    g_sink = tmpfile();
    if (!g_sink) { fprintf(stderr, "tmpfile недоступен\n"); return 70; }

    test_punct();
    test_sigils();
    test_numbers();
    test_trivia();
    test_garbage();
    test_kernel();
    test_registry();

    fclose(g_sink);
    smp_arena_release(&g_arena);
    return REPORT();
}
