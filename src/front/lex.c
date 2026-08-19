/* SMPC3 :: lex.c */
#include "smpc3/lex.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* ========================================================================== */
/*  Имена видов                                                               */
/* ========================================================================== */

#define SMP_TK_NAME(name, text) text,
static const char *const g_tok_name[SMP_TK__COUNT] = {
    SMP_TOKEN_KINDS(SMP_TK_NAME)
};
#undef SMP_TK_NAME

const char *smp_tok_name(SmpTokKind k)
{
    if ((unsigned)k >= (unsigned)SMP_TK__COUNT) return "<повреждён>";
    return g_tok_name[k];
}

bool smp_tok_is(const SmpToken *t, const char *s)
{
    const size_t n = strlen(s);
    return t->tlen == (uint32_t)n && memcmp(t->text, s, n) == 0;
}

char *smp_tok_str(const SmpToken *t, char *buf, size_t cap)
{
    switch (t->kind) {
        case SMP_TK_INT:
            snprintf(buf, cap, "%llu", (unsigned long long)t->num.u);
            break;
        case SMP_TK_FLOAT:
            snprintf(buf, cap, "%g", t->num.f);
            break;
        case SMP_TK_END:
            snprintf(buf, cap, "<конец файла>");
            break;
        default:
            snprintf(buf, cap, "%.*s", (int)t->tlen, t->text ? t->text : "");
            break;
    }
    return buf;
}

/* ========================================================================== */
/*  Классификация символов                                                    */
/* ========================================================================== */

static bool smp__is_digit(char c)   { return c >= '0' && c <= '9'; }
static bool smp__is_alpha(char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}
static bool smp__is_alnum(char c)   { return smp__is_alpha(c) || smp__is_digit(c); }
static bool smp__is_hex(char c)
{
    return smp__is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

/* ========================================================================== */
/*  Курсор                                                                    */
/* ========================================================================== */

static char smp__at(const SmpLexer *lx, ptrdiff_t k)
{
    const char *q = lx->p + k;
    return (q >= lx->src->text && q < lx->end) ? *q : '\0';
}
static char smp__cur(const SmpLexer *lx) { return smp__at(lx, 0); }
static bool smp__eof(const SmpLexer *lx) { return lx->p >= lx->end; }

static uint32_t smp__col(const SmpLexer *lx)
{
    return (uint32_t)(lx->p - lx->line_start) + 1u;
}

/* Продвижение на один байт с учётом перевода строки. */
static void smp__adv(SmpLexer *lx)
{
    if (smp__eof(lx)) return;
    if (*lx->p == '\n') {
        lx->line++;
        lx->line_start = lx->p + 1;
    }
    lx->p++;
}

static void smp__adv_n(SmpLexer *lx, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++) smp__adv(lx);
}

/* ========================================================================== */
/*  Инициализация                                                             */
/* ========================================================================== */

void smp_lex_init(SmpLexer *lx, const SmpSource *src, SmpDiagCtx *diag)
{
    memset(lx, 0, sizeof(*lx));
    lx->src        = src;
    lx->diag       = diag;
    lx->p          = src->text;
    lx->end        = src->text + src->len;
    lx->line_start = src->text;
    lx->line       = 1;
}

/* ========================================================================== */
/*  Ошибки                                                                    */
/* ========================================================================== */

static void smp__lex_err(SmpLexer *lx, SmpDiagCode code, SmpSpan sp,
                         const char *details)
{
    lx->n_errors++;
    if (!lx->diag) return;
    SmpDiagMsg m;
    memset(&m, 0, sizeof m);
    m.code    = code;
    m.span    = sp;
    m.details = details;
    smp_diag_emit(lx->diag, &m);
}

/* ========================================================================== */
/*  Пропуск незначащего                                                       */
/* ========================================================================== */

static void smp__skip_trivia(SmpLexer *lx)
{
    for (;;) {
        const char c = smp__cur(lx);

        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            smp__adv(lx);
            continue;
        }

        /* Строчный комментарий. */
        if (c == '/' && smp__at(lx, 1) == '/') {
            while (!smp__eof(lx) && smp__cur(lx) != '\n') smp__adv(lx);
            continue;
        }

        /* Блочный комментарий. Вложенности нет — как в C. */
        if (c == '/' && smp__at(lx, 1) == '*') {
            const SmpSpan open = { lx->line, smp__col(lx), 2 };
            smp__adv_n(lx, 2);
            bool closed = false;
            while (!smp__eof(lx)) {
                if (smp__cur(lx) == '*' && smp__at(lx, 1) == '/') {
                    smp__adv_n(lx, 2);
                    closed = true;
                    break;
                }
                smp__adv(lx);
            }
            if (!closed)
                smp__lex_err(lx, SMP_E0102, open,
                             "Комментарий открыт здесь и не закрыт до конца файла.");
            continue;
        }

        return;
    }
}

/* ========================================================================== */
/*  Сканеры лексем                                                            */
/* ========================================================================== */

/* Идентификатор. kebab=true разрешает дефис внутри (для !no-alias и
 * ~flush-to-zero), но не в конце: висящий дефис — это опечатка. */
static uint32_t smp__scan_ident(SmpLexer *lx, bool kebab)
{
    const char *start = lx->p;
    if (!smp__is_alpha(smp__cur(lx))) return 0;
    smp__adv(lx);

    for (;;) {
        const char c = smp__cur(lx);
        if (smp__is_alnum(c)) { smp__adv(lx); continue; }
        if (kebab && c == '-' && smp__is_alnum(smp__at(lx, 1))) {
            smp__adv(lx);
            continue;
        }
        break;
    }
    return (uint32_t)(lx->p - start);
}

/* Идентификатор с точками: reduce.add. */
static uint32_t smp__scan_dotted(SmpLexer *lx)
{
    const char *start = lx->p;
    if (smp__scan_ident(lx, false) == 0) return 0;

    while (smp__cur(lx) == '.' && smp__is_alpha(smp__at(lx, 1))) {
        smp__adv(lx);
        if (smp__scan_ident(lx, false) == 0) break;
    }
    return (uint32_t)(lx->p - start);
}

/* Число. Заполняет kind/num/numkind у токена. */
static void smp__scan_number(SmpLexer *lx, SmpToken *t)
{
    const char    *start = lx->p;
    const uint32_t line  = lx->line;
    const uint32_t col   = smp__col(lx);

    uint64_t uval  = 0;
    bool     isflt = false;
    bool     bad   = false;

    if (smp__cur(lx) == '0' && (smp__at(lx, 1) == 'x' || smp__at(lx, 1) == 'X')) {
        smp__adv_n(lx, 2);
        uint32_t nd = 0;
        while (smp__is_hex(smp__cur(lx)) || smp__cur(lx) == '_') {
            if (smp__cur(lx) != '_') {
                const char c = smp__cur(lx);
                const uint64_t d = (uint64_t)(smp__is_digit(c) ? c - '0'
                                   : (c | 0x20) - 'a' + 10);
                uval = uval * 16u + d;
                nd++;
            }
            smp__adv(lx);
        }
        if (nd == 0) bad = true;
    } else if (smp__cur(lx) == '0' && (smp__at(lx, 1) == 'b' || smp__at(lx, 1) == 'B')) {
        smp__adv_n(lx, 2);
        uint32_t nd = 0;
        while (smp__cur(lx) == '0' || smp__cur(lx) == '1' || smp__cur(lx) == '_') {
            if (smp__cur(lx) != '_') { uval = uval * 2u + (uint64_t)(smp__cur(lx) - '0'); nd++; }
            smp__adv(lx);
        }
        if (nd == 0) bad = true;
    } else {
        while (smp__is_digit(smp__cur(lx)) || smp__cur(lx) == '_') {
            if (smp__cur(lx) != '_') uval = uval * 10u + (uint64_t)(smp__cur(lx) - '0');
            smp__adv(lx);
        }

        /* Дробная часть. Одиночная точка перед второй точкой — это уже '..'
         * из индекса (*&C[0, ..]), а не десятичный разделитель. */
        if (smp__cur(lx) == '.' && smp__at(lx, 1) != '.') {
            if (!smp__is_digit(smp__at(lx, 1))) {
                /* "1." — незавершённый литерал */
                smp__adv(lx);
                bad = true;
            } else {
                isflt = true;
                smp__adv(lx);
                while (smp__is_digit(smp__cur(lx)) || smp__cur(lx) == '_') smp__adv(lx);
            }
        }

        /* Экспонента. */
        const char e = smp__cur(lx);
        if (e == 'e' || e == 'E') {
            const char s1 = smp__at(lx, 1);
            const char s2 = smp__at(lx, 2);
            if (smp__is_digit(s1) || ((s1 == '+' || s1 == '-') && smp__is_digit(s2))) {
                isflt = true;
                smp__adv(lx);
                if (smp__cur(lx) == '+' || smp__cur(lx) == '-') smp__adv(lx);
                while (smp__is_digit(smp__cur(lx))) smp__adv(lx);
            }
        }
    }

    /* Литерал, к которому вплотную прилип идентификатор, — всегда опечатка. */
    if (smp__is_alpha(smp__cur(lx))) {
        bad = true;
        while (smp__is_alnum(smp__cur(lx))) smp__adv(lx);
    }

    const uint32_t len = (uint32_t)(lx->p - start);
    t->span = (SmpSpan){ line, col, len };
    t->off  = (uint32_t)(start - lx->src->text);
    t->text = start;
    t->tlen = len;

    if (bad) {
        t->kind    = SMP_TK_ERROR;
        t->numkind = SMP_NUM_NONE;
        smp__lex_err(lx, SMP_E0103, t->span, NULL);
        return;
    }

    if (isflt) {
        /* strtod требует \0-терминированную строку, а исходник им не обязан
         * заканчиваться. Копируем лексему во временный буфер. */
        char tmp[64];
        const uint32_t n = len < sizeof(tmp) - 1u ? len : (uint32_t)sizeof(tmp) - 1u;
        memcpy(tmp, start, n);
        tmp[n] = '\0';
        t->kind    = SMP_TK_FLOAT;
        t->numkind = SMP_NUM_FLOAT;
        t->num.f   = strtod(tmp, NULL);
    } else {
        t->kind    = SMP_TK_INT;
        t->numkind = SMP_NUM_INT;
        t->num.u   = uval;
    }
}

/* Сигильная лексема: $r1, @mmul, #simd, ^raw, !no-alias, ?strict, ~ftz. */
static bool smp__scan_sigil(SmpLexer *lx, SmpToken *t, SmpTokKind kind,
                            bool dotted, bool kebab, const char *what)
{
    const char    *start = lx->p;
    const uint32_t line  = lx->line;
    const uint32_t col   = smp__col(lx);

    smp__adv(lx);                       /* сам сигил */
    const char *body = lx->p;

    const uint32_t blen = dotted ? smp__scan_dotted(lx) : smp__scan_ident(lx, kebab);

    const uint32_t len = (uint32_t)(lx->p - start);
    t->span = (SmpSpan){ line, col, len ? len : 1u };
    t->off  = (uint32_t)(start - lx->src->text);
    t->text = body;
    t->tlen = blen;

    if (blen == 0) {
        t->kind = SMP_TK_ERROR;
        smp__lex_err(lx, SMP_E0105, t->span,
                     lx->diag ? smp_fmt(lx->diag,
                         "После '%c' обязано идти %s.", *start, what) : NULL);
        return false;
    }

    t->kind = kind;
    return true;
}

/* ========================================================================== */
/*  Главный цикл                                                              */
/* ========================================================================== */

static void smp__punct(SmpLexer *lx, SmpToken *t, SmpTokKind kind, uint32_t n)
{
    const char *start = lx->p;
    t->span = (SmpSpan){ lx->line, smp__col(lx), n };
    t->off  = (uint32_t)(start - lx->src->text);
    t->text = start;
    t->tlen = n;
    t->kind = kind;
    smp__adv_n(lx, n);
}

void smp_lex_next(SmpLexer *lx, SmpToken *out)
{
    memset(out, 0, sizeof(*out));

    for (;;) {
        smp__skip_trivia(lx);

        if (smp__eof(lx)) {
            out->kind = SMP_TK_END;
            out->span = (SmpSpan){ lx->line, smp__col(lx), 0 };
            out->off  = (uint32_t)(lx->end - lx->src->text);
            out->text = lx->end;
            out->tlen = 0;
            return;
        }

        const char c  = smp__cur(lx);
        const char c1 = smp__at(lx, 1);

        /* --- двухсимвольные --- */
        if (c == '-' && c1 == '>') { smp__punct(lx, out, SMP_TK_ARROW,    2); return; }
        if (c == '=' && c1 == '>') { smp__punct(lx, out, SMP_TK_FATARROW, 2); return; }
        if (c == '*' && c1 == '&') { smp__punct(lx, out, SMP_TK_TENSOR,   2); return; }
        if (c == '.' && c1 == '.') { smp__punct(lx, out, SMP_TK_DOTDOT,   2); return; }

        /* --- односимвольные --- */
        switch (c) {
            case '[': smp__punct(lx, out, SMP_TK_LBRACKET, 1); return;
            case ']': smp__punct(lx, out, SMP_TK_RBRACKET, 1); return;
            case '(': smp__punct(lx, out, SMP_TK_LPAREN,   1); return;
            case ')': smp__punct(lx, out, SMP_TK_RPAREN,   1); return;
            case '<': smp__punct(lx, out, SMP_TK_LANGLE,   1); return;
            case '>': smp__punct(lx, out, SMP_TK_RANGLE,   1); return;
            case ',': smp__punct(lx, out, SMP_TK_COMMA,    1); return;
            case ';': smp__punct(lx, out, SMP_TK_SEMI,     1); return;
            case ':': smp__punct(lx, out, SMP_TK_COLON,    1); return;
            case '.': smp__punct(lx, out, SMP_TK_DOT,      1); return;
            case '*': smp__punct(lx, out, SMP_TK_STAR,     1); return;
            case '-': smp__punct(lx, out, SMP_TK_MINUS,    1); return;
            default: break;
        }

        /* --- сигилы --- */
        if (c == '$') { smp__scan_sigil(lx, out, SMP_TK_REG,       false, false, "имя регистра");    return; }
        if (c == '@') { smp__scan_sigil(lx, out, SMP_TK_OP,        true,  false, "имя операции");    return; }
        if (c == '#') { smp__scan_sigil(lx, out, SMP_TK_DIRECTIVE, false, false, "имя директивы");   return; }
        if (c == '^') { smp__scan_sigil(lx, out, SMP_TK_MODE,      false, false, "имя режима");      return; }
        if (c == '!') { smp__scan_sigil(lx, out, SMP_TK_ASSERT,    false, true,  "имя утверждения"); return; }
        if (c == '?') { smp__scan_sigil(lx, out, SMP_TK_QUERY,     false, false, "имя модификатора");return; }
        if (c == '~') { smp__scan_sigil(lx, out, SMP_TK_FPMODE,    false, true,  "имя fp-режима");   return; }

        /* --- числа --- */
        if (smp__is_digit(c)) { smp__scan_number(lx, out); return; }

        /* --- идентификаторы --- */
        if (smp__is_alpha(c)) {
            const char    *start = lx->p;
            const uint32_t line  = lx->line;
            const uint32_t col   = smp__col(lx);
            const uint32_t len   = smp__scan_ident(lx, false);
            out->kind = SMP_TK_IDENT;
            out->span = (SmpSpan){ line, col, len };
            out->off  = (uint32_t)(start - lx->src->text);
            out->text = start;
            out->tlen = len;
            return;
        }

        /* --- мусор --- */
        {
            const unsigned char uc    = (unsigned char)c;
            const char         *start = lx->p;
            const uint32_t      line  = lx->line;
            const uint32_t      col   = smp__col(lx);

            if (uc >= 0x80u) {
                /* Слово «фильтр» — это шесть кодовых точек и двенадцать байт.
                 * Ругаться на каждый байт (и даже на каждую точку) значит
                 * закопать одну настоящую претензию под стеной шума, поэтому
                 * непрерывная не-ASCII серия схлопывается в одну диагностику. */
                uint32_t cps = 0;
                while (!smp__eof(lx) && (unsigned char)smp__cur(lx) >= 0x80u) {
                    if (((unsigned char)smp__cur(lx) & 0xC0u) != 0x80u) cps++;
                    smp__adv(lx);
                }
                const SmpSpan sp = { line, col, (uint32_t)(lx->p - start) };
                smp__lex_err(lx, SMP_E0101, sp,
                    lx->diag ? smp_fmt(lx->diag,
                        "Здесь %u не-ASCII символ(ов), %u байт. "
                        "Идентификаторы языка состоят из ASCII; кириллица "
                        "допустима только в комментариях.",
                        cps, (unsigned)(lx->p - start)) : NULL);
            } else {
                const SmpSpan sp = { line, col, 1 };
                smp__lex_err(lx, SMP_E0101, sp,
                    lx->diag ? smp_fmt(lx->diag,
                        "Символ '%c' (0x%02X) не входит в алфавит языка.",
                        (uc >= 0x20u ? c : '?'), uc) : NULL);
                smp__adv(lx);
            }
            continue;
        }
    }
}

/* ========================================================================== */
/*  Пакетная токенизация                                                      */
/* ========================================================================== */

SmpStatus smp_lex_all(SmpLexer *lx, SmpArena *arena,
                      SmpToken **out_toks, uint32_t *out_n)
{
    SmpToken *first = NULL;
    uint32_t  n     = 0;

    for (;;) {
        SmpToken *slot = (SmpToken *)smp_arena_push_raw(arena, sizeof(SmpToken), 8);
        if (!slot) {
            if (lx->diag) {
                char rep[160];
                SmpDiagMsg m;
                memset(&m, 0, sizeof m);
                m.code    = SMP_E0401;
                m.details = smp_fmt(lx->diag,
                    "Поток токенов не поместился: %s\nУспели разобрать %u токенов.",
                    smp_arena_report(arena, rep, sizeof rep), n);
                smp_diag_emit(lx->diag, &m);
            }
            lx->n_errors++;
            return SMP_ERR_OOM;
        }
        if (!first) first = slot;

        smp_lex_next(lx, slot);
        n++;

        if (slot->kind == SMP_TK_END) break;
    }

    *out_toks = first;
    *out_n    = n;
    return lx->n_errors ? SMP_ERR_SYNTAX : SMP_OK;
}
