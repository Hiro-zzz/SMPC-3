/* SMPC3 :: parse.c */
#include "smpc3/parse.h"

#include <string.h>
#include <stdio.h>

/* ========================================================================== */
/*  Курсор по токенам                                                         */
/* ========================================================================== */

static const SmpToken *cur(const SmpParser *p)
{
    return &p->toks[p->pos < p->ntoks ? p->pos : p->ntoks - 1u];
}

static const SmpToken *peek(const SmpParser *p, uint32_t k)
{
    const uint32_t i = p->pos + k;
    return &p->toks[i < p->ntoks ? i : p->ntoks - 1u];
}

static bool at(const SmpParser *p, SmpTokKind k) { return cur(p)->kind == k; }
static bool at_end(const SmpParser *p)           { return cur(p)->kind == SMP_TK_END; }

static const SmpToken *advance(SmpParser *p)
{
    const SmpToken *t = cur(p);
    if (p->pos + 1u < p->ntoks) p->pos++;
    return t;
}

static bool accept(SmpParser *p, SmpTokKind k)
{
    if (!at(p, k)) return false;
    advance(p);
    return true;
}

static SmpName name_of(const SmpToken *t)
{
    SmpName n = { t->text, t->tlen };
    return n;
}

/* Склейка двух позиций в один диапазон: от начала a до конца b. */
static SmpSpan span_join(SmpSpan a, SmpSpan b)
{
    if (a.line != b.line) return a;          /* многострочное не подчёркиваем */
    const uint32_t end = b.col + b.len;
    a.len = end > a.col ? end - a.col : a.len;
    return a;
}

/* ========================================================================== */
/*  Ошибки                                                                    */
/* ========================================================================== */

/* Первая претензия внутри инструкции печатается, остальные глушатся: они
 * почти всегда следствие первой, а не отдельные дефекты. */
static void perr(SmpParser *p, SmpDiagCode code, SmpSpan sp,
                 const char *details, const char *fix)
{
    if (p->panic) return;
    p->panic = true;
    p->n_errors++;
    if (!p->diag) return;

    SmpDiagMsg m;
    memset(&m, 0, sizeof m);
    m.code    = code;
    m.span    = sp;
    m.details = details;
    m.fix     = fix;
    smp_diag_emit(p->diag, &m);
}

static const char *fmt(SmpParser *p, const char *f, ...) SMP_PRINTF(2, 3);
static const char *fmt(SmpParser *p, const char *f, ...)
{
    if (!p->diag) return NULL;
    static char scratch[SMP_FMT_SLOTLEN];
    va_list ap;
    va_start(ap, f);
    vsnprintf(scratch, sizeof scratch, f, ap);
    va_end(ap);
    return smp_fmt(p->diag, "%s", scratch);
}

/* Что тут вообще написано — для текста «ожидалось X, найдено Y». */
static const char *found_str(SmpParser *p)
{
    const SmpToken *t = cur(p);
    if (t->kind == SMP_TK_END) return "конец файла";

    char buf[96];
    smp_tok_str(t, buf, sizeof buf);
    const char *nm = smp_tok_name(t->kind);

    /* У пунктуации имя вида и есть сама лексема; «[ '['» выглядит глупо. */
    return strcmp(nm, buf) == 0 ? fmt(p, "'%s'", buf)
                                : fmt(p, "%s '%s'", nm, buf);
}

/* Паническая синхронизация: до ближайшей ';' включительно. */
static void sync_stmt(SmpParser *p)
{
    while (!at_end(p)) {
        if (accept(p, SMP_TK_SEMI)) return;
        advance(p);
    }
}

/* ========================================================================== */
/*  Классификация скобочных групп                                             */
/* ========================================================================== */

typedef enum { GRP_NONE = 0, GRP_PREFIX, GRP_SUFFIX, GRP_INDEX, GRP_UNKNOWN } GrpKind;

/* Решение принимается по одному токену за '[' — грамматика это гарантирует. */
static GrpKind group_kind(const SmpParser *p, uint32_t at_off)
{
    if (peek(p, at_off)->kind != SMP_TK_LBRACKET) return GRP_NONE;
    switch (peek(p, at_off + 1u)->kind) {
        case SMP_TK_DIRECTIVE:
        case SMP_TK_MODE:     return GRP_PREFIX;
        case SMP_TK_ASSERT:
        case SMP_TK_QUERY:
        case SMP_TK_FPMODE:   return GRP_SUFFIX;
        case SMP_TK_INT:
        case SMP_TK_DOTDOT:
        case SMP_TK_REG:      return GRP_INDEX;
        default:              return GRP_UNKNOWN;
    }
}

/* ========================================================================== */
/*  Тензорная ссылка                                                          */
/* ========================================================================== */

/* <f32:1024,1024> */
static bool parse_type(SmpParser *p, SmpAstTensor *t)
{
    const SmpToken *open = advance(p);        /* '<' */
    t->type_span = open->span;
    t->has_type  = true;

    if (!at(p, SMP_TK_IDENT)) {
        perr(p, SMP_E0301, cur(p)->span,
             fmt(p, "Здесь ожидалось имя типа, а найдено: %s.", found_str(p)), NULL);
        return false;
    }

    const SmpToken *dt = advance(p);
    t->dtype = smp_dtype_parse(dt->text, dt->tlen);
    if (t->dtype == SMP_DT_INVALID) {
        perr(p, SMP_E0301, dt->span,
             fmt(p, "Тип '%.*s' не существует.", (int)dt->tlen, dt->text), NULL);
        return false;
    }

    t->rank = 0;
    if (accept(p, SMP_TK_COLON)) {
        uint64_t nelem = 1;
        for (;;) {
            if (!at(p, SMP_TK_INT)) {
                perr(p, SMP_E0304, cur(p)->span,
                     fmt(p, "Размерность обязана быть целым литералом, а найдено: %s.",
                         found_str(p)), NULL);
                return false;
            }
            const SmpToken *d = advance(p);

            if (t->rank >= SMP_MAX_RANK) {
                perr(p, SMP_E0302, d->span,
                     fmt(p, "Уже объявлено %u осей, эта — лишняя.", SMP_MAX_RANK), NULL);
                return false;
            }
            if (d->num.u == 0 || d->num.u > SMP_DIM_MAX) {
                perr(p, SMP_E0304, d->span,
                     fmt(p, "Ось #%u имеет размер %llu; допустимо 1..%u.",
                         t->rank, (unsigned long long)d->num.u, SMP_DIM_MAX), NULL);
                return false;
            }
            /* Каждая ось влезает в uint32_t, но их произведение — число
             * элементов — обязано влезть туда же. */
            nelem *= d->num.u;
            if (nelem > SMP_NELEM_MAX) {
                perr(p, SMP_E0304, d->span,
                     fmt(p, "На оси #%u элементов уже больше %u.",
                         t->rank, SMP_NELEM_MAX), NULL);
                return false;
            }
            t->dims[t->rank++] = (uint32_t)d->num.u;

            if (!accept(p, SMP_TK_COMMA)) break;
        }
    }

    if (!at(p, SMP_TK_RANGLE)) {
        perr(p, SMP_E0203, cur(p)->span,
             fmt(p, "Объявление типа открыто '<' в %u:%u и не закрыто '>'; найдено: %s.",
                 open->span.line, open->span.col, found_str(p)), NULL);
        return false;
    }
    t->type_span = span_join(t->type_span, advance(p)->span);
    return true;
}

/* [0, ..] */
static bool parse_index(SmpParser *p, SmpAstTensor *t)
{
    const SmpToken *open = advance(p);        /* '[' */
    t->index_span = open->span;
    t->has_index  = true;
    t->nidx       = 0;

    for (;;) {
        if (t->nidx >= SMP_MAX_RANK) {
            perr(p, SMP_E0207, cur(p)->span,
                 fmt(p, "Индексов больше %u, а тензор не может быть глубже.",
                     SMP_MAX_RANK), NULL);
            return false;
        }

        SmpAstIndex *ix = &t->idx[t->nidx];
        const SmpToken *tk = cur(p);
        ix->span = tk->span;

        if (accept(p, SMP_TK_DOTDOT)) {
            ix->kind = SMP_IDX_ALL;
        } else if (at(p, SMP_TK_INT)) {
            ix->kind = SMP_IDX_INT;
            ix->ival = advance(p)->num.u;
        } else if (at(p, SMP_TK_REG)) {
            ix->kind = SMP_IDX_REG;
            ix->reg  = name_of(advance(p));
        } else {
            perr(p, SMP_E0209, tk->span,
                 fmt(p, "Индексом может быть целое, '..' или регистр; найдено: %s.",
                     found_str(p)), NULL);
            return false;
        }
        t->nidx++;

        if (!accept(p, SMP_TK_COMMA)) break;
    }

    if (!at(p, SMP_TK_RBRACKET)) {
        perr(p, SMP_E0203, cur(p)->span,
             fmt(p, "Индекс открыт '[' в %u:%u и не закрыт ']'; найдено: %s.",
                 open->span.line, open->span.col, found_str(p)), NULL);
        return false;
    }
    t->index_span = span_join(t->index_span, advance(p)->span);
    return true;
}

/* *&C<f32:1024,1024>[0, ..] */
static SmpAstTensor *parse_tensor(SmpParser *p)
{
    const SmpToken *sig = advance(p);         /* '*&' */

    SmpAstTensor *t = (SmpAstTensor *)smp_arena_push_raw(p->arena, sizeof *t, 8);
    if (!t) {
        perr(p, SMP_E0401, sig->span, "Арена компилятора исчерпана на разборе.", NULL);
        return NULL;
    }
    memset(t, 0, sizeof *t);
    t->span = sig->span;

    if (!at(p, SMP_TK_IDENT)) {
        perr(p, SMP_E0209, cur(p)->span,
             fmt(p, "После '*&' обязано идти имя тензора, а найдено: %s.",
                 found_str(p)), NULL);
        return NULL;
    }
    const SmpToken *nm = advance(p);
    t->name = name_of(nm);
    t->span = span_join(t->span, nm->span);

    if (at(p, SMP_TK_LANGLE)) {
        if (!parse_type(p, t)) return NULL;
        t->span = span_join(t->span, t->type_span);
    }

    /* Индексная группа — только если внутри действительно индекс. Суффикс
     * [!no-alias] к тензору не приклеивается, он принадлежит инструкции. */
    if (group_kind(p, 0) == GRP_INDEX) {
        if (!parse_index(p, t)) return NULL;
        t->span = span_join(t->span, t->index_span);
    }

    return t;
}

/* ========================================================================== */
/*  Литерал тензора                                                           */
/* ========================================================================== */

/* '[' с числом, минусом или ещё одной '[' следом — литерал, а не группа:
 * индекс стоит только за именем тензора, а префикс начинается с # или ^. */
static bool at_list(const SmpParser *p)
{
    if (!at(p, SMP_TK_LBRACKET)) return false;
    switch (peek(p, 1)->kind) {
        case SMP_TK_INT: case SMP_TK_FLOAT: case SMP_TK_MINUS: case SMP_TK_LBRACKET:
            return true;
        default:
            return false;
    }
}

/* Сборка литерала: числа подряд, а длина каждого уровня запоминается по
 * первой встреченной строке и сверяется со всеми остальными. */
typedef struct ListB {
    SmpAstNum vals[SMP_MAX_LIST];
    uint32_t  n;
    uint32_t  rank;                  /* 0 — чисел ещё не было            */
    uint32_t  dims[SMP_MAX_RANK];
    bool      dim_set[SMP_MAX_RANK];
    bool      any_float;
} ListB;

static bool list_level(SmpParser *p, ListB *b, uint32_t depth)
{
    const SmpToken *open = advance(p);        /* '[' */

    if (depth >= SMP_MAX_RANK) {
        perr(p, SMP_E0302, open->span,
             fmt(p, "Литерал вложен глубже %u уровней.", SMP_MAX_RANK), NULL);
        return false;
    }
    if (at(p, SMP_TK_RBRACKET)) {
        perr(p, SMP_E0211, span_join(open->span, cur(p)->span),
             "Пустая строка: нулевых осей не бывает.", NULL);
        return false;
    }

    uint32_t count = 0;
    for (;;) {
        const SmpToken *tk = cur(p);
        if (at(p, SMP_TK_LBRACKET)) {
            if (b->rank && depth + 1u >= b->rank) {
                perr(p, SMP_E0211, tk->span,
                     fmt(p, "Здесь ожидалось число, как в первой строке, а открыт "
                            "уровень %u.", depth + 2u), NULL);
                return false;
            }
            if (!list_level(p, b, depth + 1u)) return false;
        } else {
            const bool neg = accept(p, SMP_TK_MINUS);
            if (!at(p, SMP_TK_INT) && !at(p, SMP_TK_FLOAT)) {
                perr(p, SMP_E0209, cur(p)->span,
                     fmt(p, "В литерале тензора — только числа; найдено: %s.",
                         found_str(p)), NULL);
                return false;
            }
            if (!b->rank) b->rank = depth + 1u;
            else if (b->rank != depth + 1u) {
                perr(p, SMP_E0211, cur(p)->span,
                     fmt(p, "Число на уровне %u, а в первой строке числа на уровне %u.",
                         depth + 1u, b->rank), NULL);
                return false;
            }
            if (b->n >= SMP_MAX_LIST) {
                perr(p, SMP_E0207, cur(p)->span,
                     fmt(p, "В литерале больше %u чисел.", SMP_MAX_LIST),
                     "Большие таблицы кладут в рабочее пространство и берут @load.");
                return false;
            }
            const SmpToken *n = advance(p);
            SmpAstNum      *v = &b->vals[b->n++];
            v->is_float = (n->kind == SMP_TK_FLOAT);
            if (v->is_float) {
                v->fval = neg ? -n->num.f : n->num.f;
                b->any_float = true;
            } else {
                v->ival = neg ? (uint64_t)(-(int64_t)n->num.u) : n->num.u;
            }
        }
        count++;
        if (!accept(p, SMP_TK_COMMA)) break;
    }

    if (!at(p, SMP_TK_RBRACKET)) {
        perr(p, SMP_E0203, cur(p)->span,
             fmt(p, "Литерал открыт '[' в %u:%u и не закрыт ']'; найдено: %s.",
                 open->span.line, open->span.col, found_str(p)), NULL);
        return false;
    }
    const SmpToken *close = advance(p);

    if (!b->dim_set[depth]) {
        b->dim_set[depth] = true;
        b->dims[depth]    = count;
    } else if (b->dims[depth] != count) {
        perr(p, SMP_E0211, span_join(open->span, close->span),
             fmt(p, "В этой строке %u элемент(ов), а в первой строке того же уровня %u.",
                 count, b->dims[depth]), NULL);
        return false;
    }
    return true;
}

static bool parse_list(SmpParser *p, SmpAstOperand *o)
{
    static ListB b;                   /* 16 КиБ — в .bss, а не на стеке */
    memset(&b, 0, sizeof b);

    const SmpToken *open = cur(p);
    if (!list_level(p, &b, 0)) return false;

    SmpAstList *l = (SmpAstList *)smp_arena_push_raw(p->arena, sizeof *l, 8);
    SmpAstNum  *v = (SmpAstNum *)smp_arena_push_raw(p->arena, b.n * sizeof *v, 8);
    if (!l || !v) {
        perr(p, SMP_E0401, open->span, "Арена компилятора исчерпана на литерале.", NULL);
        return false;
    }
    memcpy(v, b.vals, b.n * sizeof *v);
    l->rank      = b.rank;
    l->n         = b.n;
    l->vals      = v;
    l->any_float = b.any_float;
    for (uint32_t i = 0; i < b.rank; i++) l->dims[i] = b.dims[i];

    o->kind = SMP_OPD_LIST;
    o->list = l;
    const SmpToken *last = &p->toks[p->pos ? p->pos - 1u : 0u];
    o->span = span_join(open->span, last->span);
    return true;
}

/* ========================================================================== */
/*  Операнды                                                                  */
/* ========================================================================== */

static bool parse_operand(SmpParser *p, SmpAstOperand *o)
{
    memset(o, 0, sizeof *o);
    const SmpToken *t = cur(p);
    o->span = t->span;

    if (at_list(p)) return parse_list(p, o);

    /* Отрицательный литерал: '-' приклеивается только к числу. */
    bool neg = false;
    if (at(p, SMP_TK_MINUS) &&
        (peek(p, 1)->kind == SMP_TK_INT || peek(p, 1)->kind == SMP_TK_FLOAT)) {
        neg = true;
        advance(p);
    }

    switch (cur(p)->kind) {
        case SMP_TK_REG:
            o->kind = SMP_OPD_REG;
            o->reg  = name_of(advance(p));
            return true;

        case SMP_TK_TENSOR: {
            SmpAstTensor *tn = parse_tensor(p);
            if (!tn) return false;
            o->kind   = SMP_OPD_TENSOR;
            o->tensor = tn;
            o->span   = tn->span;
            return true;
        }

        case SMP_TK_INT: {
            const SmpToken *n = advance(p);
            o->kind = SMP_OPD_INT;
            o->span = span_join(o->span, n->span);
            /* Отрицание целого хранится в дополнительном коде: знак разрешит
             * Ф3, когда станет известен тип приёмника. */
            o->ival = neg ? (uint64_t)(-(int64_t)n->num.u) : n->num.u;
            return true;
        }

        case SMP_TK_FLOAT: {
            const SmpToken *n = advance(p);
            o->kind = SMP_OPD_FLOAT;
            o->span = span_join(o->span, n->span);
            o->fval = neg ? -n->num.f : n->num.f;
            return true;
        }

        default:
            perr(p, SMP_E0209, cur(p)->span,
                 fmt(p, "Найдено: %s.", found_str(p)), NULL);
            return false;
    }
}

/* ========================================================================== */
/*  Стадия конвейера                                                          */
/* ========================================================================== */

static bool parse_stage(SmpParser *p, SmpAstStage *st)
{
    memset(st, 0, sizeof *st);

    if (!at(p, SMP_TK_OP)) {
        perr(p, SMP_E0202, cur(p)->span,
             fmt(p, "После '->' обязана идти стадия вида @имя, а найдено: %s.",
                 found_str(p)), NULL);
        return false;
    }

    const SmpToken *op = advance(p);
    st->name      = name_of(op);
    st->name_span = op->span;
    st->span      = op->span;

    if (!at(p, SMP_TK_LPAREN)) return true;

    const SmpToken *open = advance(p);        /* '(' */

    SmpAstOperand tmp[SMP_MAX_ARGS];
    uint32_t      n = 0;

    if (!at(p, SMP_TK_RPAREN)) {
        for (;;) {
            if (n >= SMP_MAX_ARGS) {
                perr(p, SMP_E0207, cur(p)->span,
                     fmt(p, "У стадии @%.*s больше %u аргументов.",
                         (int)st->name.len, st->name.p, SMP_MAX_ARGS), NULL);
                return false;
            }
            if (!parse_operand(p, &tmp[n])) return false;
            n++;
            if (!accept(p, SMP_TK_COMMA)) break;
        }
    }

    if (!at(p, SMP_TK_RPAREN)) {
        perr(p, SMP_E0203, cur(p)->span,
             fmt(p, "Список аргументов открыт '(' в %u:%u и не закрыт ')'; найдено: %s.",
                 open->span.line, open->span.col, found_str(p)), NULL);
        return false;
    }
    st->span = span_join(st->span, advance(p)->span);

    if (n) {
        st->args = (SmpAstOperand *)smp_arena_push_raw(p->arena, n * sizeof *st->args, 8);
        if (!st->args) {
            perr(p, SMP_E0401, st->span, "Арена компилятора исчерпана на аргументах.", NULL);
            return false;
        }
        memcpy(st->args, tmp, n * sizeof *st->args);
    }
    st->nargs = n;
    return true;
}

/* ========================================================================== */
/*  Префикс и суффикс                                                         */
/* ========================================================================== */

static bool parse_prefix_group(SmpParser *p, SmpAstPrefix *out, uint32_t *count)
{
    const SmpToken *open = advance(p);        /* '[' */

    for (;;) {
        if (*count >= SMP_MAX_PREFIX) {
            perr(p, SMP_E0207, cur(p)->span,
                 fmt(p, "Элементов префикса больше %u.", SMP_MAX_PREFIX), NULL);
            return false;
        }

        SmpAstPrefix *it = &out[*count];
        memset(it, 0, sizeof *it);
        const SmpToken *tk = cur(p);

        if (at(p, SMP_TK_DIRECTIVE)) {
            advance(p);
            it->kind = SMP_PFX_DIRECTIVE;
            it->name = name_of(tk);
            it->span = tk->span;

            if (accept(p, SMP_TK_COLON)) {
                const SmpToken *v = cur(p);
                if (at(p, SMP_TK_INT)) {
                    advance(p);
                    it->has_value    = true;
                    it->value_is_int = true;
                    it->ival         = v->num.u;
                } else if (at(p, SMP_TK_IDENT)) {
                    advance(p);
                    it->has_value    = true;
                    it->value_is_int = false;
                    it->sval         = name_of(v);
                } else {
                    perr(p, SMP_E0204, v->span,
                         fmt(p, "После '#%.*s:' ожидалось число или имя, а найдено: %s.",
                             (int)it->name.len, it->name.p, found_str(p)), NULL);
                    return false;
                }
                it->value_span = v->span;
                it->span       = span_join(it->span, v->span);
            }
        } else if (at(p, SMP_TK_MODE)) {
            advance(p);
            it->kind = SMP_PFX_MODE;
            it->name = name_of(tk);
            it->span = tk->span;
        } else {
            perr(p, SMP_E0204, tk->span,
                 fmt(p, "В префиксе допустимы только #директива и ^режим; найдено: %s.",
                     found_str(p)), NULL);
            return false;
        }

        (*count)++;
        if (!accept(p, SMP_TK_COMMA)) break;
    }

    if (!at(p, SMP_TK_RBRACKET)) {
        perr(p, SMP_E0203, cur(p)->span,
             fmt(p, "Префикс открыт '[' в %u:%u и не закрыт ']'; найдено: %s.",
                 open->span.line, open->span.col, found_str(p)), NULL);
        return false;
    }
    advance(p);
    return true;
}

static bool parse_suffix_group(SmpParser *p, SmpAstSuffix *out, uint32_t *count)
{
    const SmpToken *open = advance(p);        /* '[' */

    for (;;) {
        if (*count >= SMP_MAX_SUFFIX) {
            perr(p, SMP_E0207, cur(p)->span,
                 fmt(p, "Элементов суффикса больше %u.", SMP_MAX_SUFFIX), NULL);
            return false;
        }

        SmpAstSuffix *it = &out[*count];
        memset(it, 0, sizeof *it);
        const SmpToken *tk = cur(p);

        switch (tk->kind) {
            case SMP_TK_ASSERT: it->kind = SMP_SFX_ASSERT; break;
            case SMP_TK_QUERY:  it->kind = SMP_SFX_QUERY;  break;
            case SMP_TK_FPMODE: it->kind = SMP_SFX_FPMODE; break;
            default:
                perr(p, SMP_E0204, tk->span,
                     fmt(p, "В суффиксе допустимы только !утверждение, "
                            "?модификатор и ~fp-режим; найдено: %s.", found_str(p)), NULL);
                return false;
        }
        advance(p);
        it->name = name_of(tk);
        it->span = tk->span;

        (*count)++;
        if (!accept(p, SMP_TK_COMMA)) break;
    }

    if (!at(p, SMP_TK_RBRACKET)) {
        perr(p, SMP_E0203, cur(p)->span,
             fmt(p, "Суффикс открыт '[' в %u:%u и не закрыт ']'; найдено: %s.",
                 open->span.line, open->span.col, found_str(p)), NULL);
        return false;
    }
    advance(p);
    return true;
}

/* ========================================================================== */
/*  Инструкция                                                                */
/* ========================================================================== */

/* Перенос собранного на стеке в арену. Вынесено отдельно, потому что путь
 * «инструкция полна, но без ';'» тоже обязан её сохранить. */
static bool commit_stmt(SmpParser *p, SmpAstStmt *s,
                        const SmpAstPrefix *pfx, uint32_t npfx,
                        const SmpAstStage  *stg, uint32_t nstg,
                        const SmpAstSuffix *sfx, uint32_t nsfx)
{
    if (npfx) {
        s->prefix = (SmpAstPrefix *)smp_arena_push_raw(p->arena, npfx * sizeof *s->prefix, 8);
        if (!s->prefix) { perr(p, SMP_E0401, s->span, "Арена исчерпана на префиксе.", NULL); return false; }
        memcpy(s->prefix, pfx, npfx * sizeof *s->prefix);
    }
    if (nstg) {
        s->stages = (SmpAstStage *)smp_arena_push_raw(p->arena, nstg * sizeof *s->stages, 8);
        if (!s->stages) { perr(p, SMP_E0401, s->span, "Арена исчерпана на стадиях.", NULL); return false; }
        memcpy(s->stages, stg, nstg * sizeof *s->stages);
    }
    if (nsfx) {
        s->suffix = (SmpAstSuffix *)smp_arena_push_raw(p->arena, nsfx * sizeof *s->suffix, 8);
        if (!s->suffix) { perr(p, SMP_E0401, s->span, "Арена исчерпана на суффиксе.", NULL); return false; }
        memcpy(s->suffix, sfx, nsfx * sizeof *s->suffix);
    }
    s->nprefix = npfx;
    s->nstages = nstg;
    s->nsuffix = nsfx;
    return true;
}

static bool parse_stmt(SmpParser *p, SmpAstStmt *s)
{
    memset(s, 0, sizeof *s);
    s->span = cur(p)->span;

    /* --- пустая группа отлавливается до всего остального --- */
    if (at(p, SMP_TK_LBRACKET) && peek(p, 1)->kind == SMP_TK_RBRACKET) {
        perr(p, SMP_E0210, span_join(cur(p)->span, peek(p, 1)->span), NULL, NULL);
        return false;
    }

    /* --- префикс --- */
    SmpAstPrefix pfx[SMP_MAX_PREFIX];
    uint32_t     npfx = 0;
    while (group_kind(p, 0) == GRP_PREFIX)
        if (!parse_prefix_group(p, pfx, &npfx)) return false;

    if (group_kind(p, 0) == GRP_UNKNOWN && !at_list(p)) {
        perr(p, SMP_E0204, peek(p, 1)->span,
             fmt(p, "Группа '[' в %u:%u не опознана: следом идёт %s.",
                 cur(p)->span.line, cur(p)->span.col, found_str(p)), NULL);
        return false;
    }
    /* Зеркальный случай к E0206 ниже: суффикс управляет уже разобранным телом,
     * стоять до него он не может. */
    if (group_kind(p, 0) == GRP_SUFFIX) {
        perr(p, SMP_E0206, peek(p, 1)->span,
             "Это суффикс, а он описывает уже написанное тело инструкции.", NULL);
        return false;
    }

    /* --- источник --- */
    if (!parse_operand(p, &s->source)) return false;

    /* --- стадии --- */
    SmpAstStage stages[SMP_MAX_STAGES];
    uint32_t    nst = 0;
    while (at(p, SMP_TK_ARROW)) {
        advance(p);
        if (nst >= SMP_MAX_STAGES) {
            perr(p, SMP_E0207, cur(p)->span,
                 fmt(p, "Стадий в конвейере больше %u.", SMP_MAX_STAGES), NULL);
            return false;
        }
        if (!parse_stage(p, &stages[nst])) return false;
        nst++;
    }

    /* --- приёмник --- */
    if (!at(p, SMP_TK_FATARROW)) {
        perr(p, SMP_E0205, cur(p)->span,
             fmt(p, "Ожидалось '=>' и приёмник, а найдено: %s.", found_str(p)), NULL);
        return false;
    }
    s->arrow_span = advance(p)->span;

    if (!parse_operand(p, &s->dest)) return false;
    if (s->dest.kind == SMP_OPD_INT || s->dest.kind == SMP_OPD_FLOAT ||
        s->dest.kind == SMP_OPD_LIST) {
        perr(p, SMP_E0208, s->dest.span, NULL, NULL);
        return false;
    }
    s->has_dest = true;

    /* --- суффикс --- */
    SmpAstSuffix sfx[SMP_MAX_SUFFIX];
    uint32_t     nsfx = 0;
    for (;;) {
        const GrpKind g = group_kind(p, 0);
        if (g == GRP_SUFFIX) {
            if (!parse_suffix_group(p, sfx, &nsfx)) return false;
            continue;
        }
        if (g == GRP_PREFIX) {
            /* Префикс на другой строке почти наверняка означает не «группа не
             * там», а забытую ';' в конце текущей инструкции. Ругаться на
             * начало следующей строки — значит указать не на ту причину,
             * поэтому отдаём случай проверке ';' ниже. */
            if (cur(p)->span.line != s->arrow_span.line) break;

            perr(p, SMP_E0206, peek(p, 1)->span,
                 "Эта группа стоит после тела инструкции, а управляет её исполнением.",
                 NULL);
            return false;
        }
        if (g == GRP_INDEX || g == GRP_UNKNOWN) {
            perr(p, SMP_E0204, peek(p, 1)->span,
                 fmt(p, "После приёмника допустим только суффикс; найдено: %s.",
                     found_str(p)), NULL);
            return false;
        }
        break;
    }

    /* --- точка с запятой --- */
    if (!at(p, SMP_TK_SEMI)) {
        /* Каретка ставится в тот пробел, где ';' должна была быть, а не на
         * первый токен следующей инструкции. */
        const SmpToken *prev = &p->toks[p->pos ? p->pos - 1u : 0u];
        const SmpSpan   gap  = { prev->span.line, prev->span.col + prev->span.len, 1u };

        perr(p, SMP_E0201, gap,
             fmt(p, "Инструкция начата в %u:%u и разобрана целиком, "
                    "но вместо ';' дальше идёт %s.",
                 s->span.line, s->span.col, found_str(p)), NULL);

        /* Инструкция полна, не хватает лишь терминатора. Синхронизация по
         * следующей ';' проглотила бы соседнюю — корректную — инструкцию,
         * то есть наказала бы не того. Оставляем позицию как есть. */
        return commit_stmt(p, s, pfx, npfx, stages, nst, sfx, nsfx);
    }
    s->span = span_join(s->span, advance(p)->span);

    return commit_stmt(p, s, pfx, npfx, stages, nst, sfx, nsfx);
}

/* ========================================================================== */
/*  Программа                                                                 */
/* ========================================================================== */

typedef struct StmtLink {
    SmpAstStmt       stmt;
    struct StmtLink *next;
} StmtLink;

void smp_parse_init(SmpParser *p, SmpArena *arena, SmpDiagCtx *diag,
                    const SmpSource *src, const SmpToken *toks, uint32_t ntoks)
{
    memset(p, 0, sizeof *p);
    p->arena = arena;
    p->diag  = diag;
    p->src   = src;
    p->toks  = toks;
    p->ntoks = ntoks;
}

SmpStatus smp_parse(SmpParser *p, SmpAstProgram *prog)
{
    memset(prog, 0, sizeof *prog);
    prog->src = p->src;

    if (p->ntoks == 0) return SMP_ERR_INTERNAL;

    /* Инструкции копятся цепочкой: их подмассивы выделяются в ту же арену,
     * поэтому непрерывно расти сам список не может. Развернём его в массив
     * одним проходом в конце. */
    StmtLink *head = NULL, *tail = NULL;
    uint32_t  n = 0;

    while (!at_end(p)) {
        p->panic = false;

        /* Лишние ';' — не ошибка, просто пустая инструкция. */
        if (accept(p, SMP_TK_SEMI)) continue;

        StmtLink *link = (StmtLink *)smp_arena_push_raw(p->arena, sizeof *link, 8);
        if (!link) {
            perr(p, SMP_E0401, cur(p)->span, "Арена исчерпана на списке инструкций.", NULL);
            return SMP_ERR_OOM;
        }
        link->next = NULL;

        if (!parse_stmt(p, &link->stmt)) {
            sync_stmt(p);
            continue;
        }

        if (tail) tail->next = link; else head = link;
        tail = link;
        n++;
    }

    if (n) {
        prog->stmts = (SmpAstStmt *)smp_arena_push_raw(p->arena, n * sizeof *prog->stmts, 8);
        if (!prog->stmts) return SMP_ERR_OOM;
        uint32_t i = 0;
        for (StmtLink *l = head; l; l = l->next) prog->stmts[i++] = l->stmt;
    }
    prog->nstmts = n;

    return p->n_errors ? SMP_ERR_SYNTAX : SMP_OK;
}
