/* SMPC3 :: lex.h -- токенизация .smpc.
 *
 * Лексер не останавливается на первой ошибке: он печатает диагностику, делает
 * минимальное восстановление и идёт дальше, чтобы за один проход выдать все
 * претензии сразу. Решение о смерти принимает вызывающий по n_errors.
 *
 * Аллокаций нет: поток токенов пишется в арену, тела лексем остаются
 * указателями внутрь исходного текста и не копируются.
 */
#ifndef SMPC3_LEX_H
#define SMPC3_LEX_H

#include "smpc3/common.h"
#include "smpc3/arena.h"
#include "smpc3/diag.h"

/* --- Виды токенов --------------------------------------------------------- */
/*   X(суффикс, человекочитаемое имя)                                         */
#define SMP_TOKEN_KINDS(X)                            \
    X(END,       "<конец файла>")                     \
    X(ERROR,     "<ошибка>")                          \
    X(LBRACKET,  "[")                                 \
    X(RBRACKET,  "]")                                 \
    X(LPAREN,    "(")                                 \
    X(RPAREN,    ")")                                 \
    X(LANGLE,    "<")                                 \
    X(RANGLE,    ">")                                 \
    X(COMMA,     ",")                                 \
    X(SEMI,      ";")                                 \
    X(COLON,     ":")                                 \
    X(DOT,       ".")                                 \
    X(DOTDOT,    "..")                                \
    X(ARROW,     "->")                                \
    X(FATARROW,  "=>")                                \
    X(TENSOR,    "*&")                                \
    X(STAR,      "*")                                 \
    X(MINUS,     "-")                                 \
    X(REG,       "$регистр")                          \
    X(OP,        "@операция")                         \
    X(DIRECTIVE, "#директива")                        \
    X(MODE,      "^режим")                            \
    X(ASSERT,    "!утверждение")                      \
    X(QUERY,     "?модификатор")                      \
    X(FPMODE,    "~fp-режим")                         \
    X(IDENT,     "идентификатор")                     \
    X(INT,       "целый литерал")                     \
    X(FLOAT,     "вещественный литерал")

#define SMP_TK_ENUM(name, text) SMP_TK_##name,
typedef enum SmpTokKind {
    SMP_TOKEN_KINDS(SMP_TK_ENUM)
    SMP_TK__COUNT
} SmpTokKind;
#undef SMP_TK_ENUM

const char *smp_tok_name(SmpTokKind k);

/* --- Токен ---------------------------------------------------------------- */
typedef enum SmpNumKind {
    SMP_NUM_NONE = 0,
    SMP_NUM_INT,
    SMP_NUM_FLOAT
} SmpNumKind;

typedef struct SmpToken {
    SmpTokKind  kind;
    SmpSpan     span;      /* line/col/len — len в байтах всей лексемы        */
    uint32_t    off;       /* байтовое смещение начала лексемы в исходнике    */

    /* Тело лексемы БЕЗ сигила: для "$r1" это "r1", для "@reduce.add" —
     * "reduce.add". Указывает внутрь SmpSource.text, копий нет. */
    const char *text;
    uint32_t    tlen;

    uint8_t     numkind;   /* SmpNumKind                                      */
    uint8_t     _pad[3];
    union {
        uint64_t u;        /* SMP_NUM_INT                                     */
        double   f;        /* SMP_NUM_FLOAT                                   */
    } num;
} SmpToken;

SMP_STATIC_ASSERT(sizeof(SmpToken) % 8u == 0u, token_size_is_multiple_of_8);

/* Сравнение тела токена со строкой. */
bool smp_tok_is(const SmpToken *t, const char *s);

/* Печать одного токена в буфер (для дампа и сообщений). Возвращает buf. */
char *smp_tok_str(const SmpToken *t, char *buf, size_t cap);

/* --- Лексер --------------------------------------------------------------- */
typedef struct SmpLexer {
    const SmpSource *src;
    SmpDiagCtx      *diag;

    const char      *p;          /* текущая позиция                          */
    const char      *end;
    const char      *line_start;
    uint32_t         line;

    uint32_t         n_errors;
} SmpLexer;

void smp_lex_init(SmpLexer *lx, const SmpSource *src, SmpDiagCtx *diag);

/* Следующий токен. По исчерпании входа бесконечно отдаёт SMP_TK_END. */
void smp_lex_next(SmpLexer *lx, SmpToken *out);

/* Токенизирует весь вход в непрерывный массив внутри арены. Массив всегда
 * завершён токеном SMP_TK_END, он учтён в *out_n.
 *
 * ВАЖНО: пока идёт токенизация, из этой арены не должен выделять никто
 * другой — непрерывность массива держится на монотонности bump-указателя. */
SmpStatus smp_lex_all(SmpLexer *lx, SmpArena *arena,
                      SmpToken **out_toks, uint32_t *out_n);

#endif /* SMPC3_LEX_H */
