/* SMPC3 :: sema.c */
#include "smpc3/sema.h"
#include "smpc3/cpu.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* ========================================================================== */
/*  Реестр операций                                                           */
/* ========================================================================== */

#define SMP_OP_ROW(id, name, lo, hi, desc) { name, lo, hi, desc },
static const SmpOpDef g_ops[SMP_OP__COUNT] = { SMP_OPS(SMP_OP_ROW) };
#undef SMP_OP_ROW

static const SmpOpDef g_op_bogus = { "<нет>", 0, 0, "" };

const SmpOpDef *smp_op_def(SmpOpKind k)
{
    if ((unsigned)k >= (unsigned)SMP_OP__COUNT) return &g_op_bogus;
    return &g_ops[k];
}

SmpOpKind smp_op_lookup(SmpName n)
{
    for (unsigned i = 0; i < SMP_OP__COUNT; i++)
        if (smp_name_is(n, g_ops[i].name)) return (SmpOpKind)i;
    return SMP_OP__UNKNOWN;
}

void smp_ops_list(FILE *out)
{
    for (unsigned i = 0; i < SMP_OP__COUNT; i++) {
        const SmpOpDef *d = &g_ops[i];
        if (d->min_args == d->max_args)
            fprintf(out, "  @%-12s %u арг.   %s\n", d->name, d->min_args, d->desc);
        else
            fprintf(out, "  @%-12s %u-%u арг. %s\n", d->name, d->min_args,
                    d->max_args, d->desc);
    }
}

/* ========================================================================== */
/*  Реестр атрибутов                                                          */
/* ========================================================================== */

typedef enum { VAL_NONE = 0, VAL_INT, VAL_IDENT } AttrVal;

typedef struct { const char *name; AttrVal val; const char *desc; } AttrDef;

static const AttrDef g_directives[] = {
    { "arena",  VAL_INT,   "номер арены, 0..7" },
    { "simd",   VAL_IDENT, "ширина вектора: v128, v256, v512" },
    { "unroll", VAL_INT,   "коэффициент разворота цикла" },
    { "repeat", VAL_INT,   "развернуть инструкцию N раз, 1..4096" },
    { "index",  VAL_IDENT, "имя индекса повтора для #repeat" }
};
static const AttrDef g_modes[] = {
    { "raw",      VAL_NONE, "сырой указатель, вывод шага отключён" },
    { "readonly", VAL_NONE, "запись в операнды запрещена" }
};
static const AttrDef g_asserts[] = {
    { "no-alias",   VAL_NONE, "операнды не делят память" },
    { "aligned",    VAL_NONE, "база выровнена под текущую ширину вектора" },
    { "contiguous", VAL_NONE, "операнды плотные" }
};
static const AttrDef g_queries[] = {
    { "strict", VAL_NONE, "любое отступление от запрошенного — фатально" },
    { "fast",   VAL_NONE, "разрешены переассоциация и приближения" }
};
static const AttrDef g_fpmodes[] = {
    { "flush-to-zero", VAL_NONE, "FTZ+DAZ в MXCSR" },
    { "precise",       VAL_NONE, "без переассоциации" }
};

static const AttrDef *attr_find(const AttrDef *tab, size_t n, SmpName nm)
{
    for (size_t i = 0; i < n; i++)
        if (smp_name_is(nm, tab[i].name)) return &tab[i];
    return NULL;
}

static void attr_section(FILE *out, const char *title, const char *sigil,
                         const AttrDef *tab, size_t n)
{
    fprintf(out, "%s\n", title);
    for (size_t i = 0; i < n; i++)
        fprintf(out, "  %s%-14s %s\n", sigil, tab[i].name, tab[i].desc);
    fputc('\n', out);
}

void smp_attrs_list(FILE *out)
{
    attr_section(out, "Префикс — директивы:", "#", g_directives, SMP_ARRLEN(g_directives));
    attr_section(out, "Префикс — режимы:",    "^", g_modes,      SMP_ARRLEN(g_modes));
    attr_section(out, "Суффикс — утверждения:", "!", g_asserts,  SMP_ARRLEN(g_asserts));
    attr_section(out, "Суффикс — модификаторы:", "?", g_queries, SMP_ARRLEN(g_queries));
    attr_section(out, "Суффикс — fp-режимы:", "~", g_fpmodes,    SMP_ARRLEN(g_fpmodes));
}

/* ========================================================================== */
/*  «Ты имел в виду...»                                                       */
/* ========================================================================== */

/* Расстояние Левенштейна с потолком: длинные имена нам не встречаются. */
static uint32_t edit_dist(const char *a, uint32_t alen, const char *b, uint32_t blen)
{
    enum { CAP = 48 };
    if (alen >= CAP || blen >= CAP) return 99;

    uint32_t prev[CAP + 1], cur_[CAP + 1];
    for (uint32_t j = 0; j <= blen; j++) prev[j] = j;

    for (uint32_t i = 1; i <= alen; i++) {
        cur_[0] = i;
        for (uint32_t j = 1; j <= blen; j++) {
            const uint32_t cost = (a[i - 1] == b[j - 1]) ? 0u : 1u;
            uint32_t v = prev[j] + 1u;
            if (cur_[j - 1] + 1u < v) v = cur_[j - 1] + 1u;
            if (prev[j - 1] + cost < v) v = prev[j - 1] + cost;
            cur_[j] = v;
        }
        memcpy(prev, cur_, sizeof(uint32_t) * (blen + 1u));
    }
    return prev[blen];
}

/* Ближайшее имя из таблицы строк, если оно достаточно близко. */
static const char *suggest(SmpName got, const char *const *names, uint32_t n)
{
    const char *best = NULL;
    uint32_t    bd   = 99;
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t d = edit_dist(got.p, got.len, names[i], (uint32_t)strlen(names[i]));
        if (d < bd) { bd = d; best = names[i]; }
    }
    /* Порог пропорционален длине: для «mmul» правка в 3 символа — уже другое
     * слово, а для «flush-to-zerro» — очевидная опечатка. */
    const uint32_t limit = got.len <= 4u ? 1u : (got.len <= 8u ? 2u : 3u);
    return (bd <= limit) ? best : NULL;
}

static const char *suggest_attr(SmpName got, const AttrDef *tab, size_t n)
{
    const char *names[16];
    if (n > SMP_ARRLEN(names)) n = SMP_ARRLEN(names);
    for (size_t i = 0; i < n; i++) names[i] = tab[i].name;
    return suggest(got, names, (uint32_t)n);
}

/* ========================================================================== */
/*  Контекст прохода                                                          */
/* ========================================================================== */

typedef struct Ctx {
    SmpSema           *sm;
    SmpSemaResult     *res;
    const SmpAstStmt  *stmt;
    SmpStmtInfo       *info;
    bool               bad;      /* инструкция уже испорчена                 */

    /* Символы, прочитанные аргументами стадий. Собираются ровно один раз, при
     * фактическом разборе стадии: повторное чтение операнда задвоило бы и
     * счётчики, и предупреждения. */
    uint32_t           arg_syms[SMP_MAX_STAGES];
    uint32_t           n_arg_syms;

    uint32_t           cur_stage;   /* индекс разбираемой стадии */
} Ctx;

static const char *sfmt(Ctx *c, const char *f, ...) SMP_PRINTF(2, 3);

/* Дописать к деталям, каким по счёту повтором порождена эта инструкция.
 * Без этого N копий одной строки дают N неотличимых сообщений, и понять, на
 * каком именно индексе всё разъехалось, нечем. */
static const char *repeat_note(Ctx *c, const char *details)
{
    const SmpAstPrefix *r = smp_stmt_directive(c->stmt, "repeat");
    return sfmt(c, "%s\nЭто повтор #%u из %u: развёртка [#repeat] в %u:%u.",
                details ? details : "",
                c->stmt->repeat_idx, c->stmt->repeat_n,
                r ? r->span.line : c->stmt->span.line,
                r ? r->span.col  : c->stmt->span.col);
}

static void serr(Ctx *c, SmpDiagCode code, SmpSpan sp,
                 const char *details, const char *fix)
{
    if (c->bad) return;          /* одна претензия на инструкцию */
    c->bad = true;

    /* Копии одной развёртки — это одна написанная инструкция, и претензия к
     * ним одна. Латч стоит ДО счётчика намеренно: иначе опечатка внутри
     * [#repeat:1024] отчиталась бы тысячей ошибок об одной строке, и число в
     * итоговой сводке перестало бы что-либо значить. */
    if (c->stmt->from_repeat) {
        if (c->sm->repeat_told == c->stmt->repeat_of) return;
        c->sm->repeat_told = c->stmt->repeat_of;
    }

    c->sm->n_errors++;
    if (!c->sm->diag) return;
    if (c->stmt->from_repeat) details = repeat_note(c, details);

    SmpDiagMsg m;
    memset(&m, 0, sizeof m);
    m.code = code; m.span = sp; m.details = details; m.fix = fix;
    smp_diag_emit(c->sm->diag, &m);
}

static void swarn(Ctx *c, SmpDiagCode code, SmpSpan sp,
                  const char *details, const char *fix)
{
    if (c->stmt->from_repeat) {
        if (c->sm->repeat_warned == c->stmt->repeat_of) return;
        c->sm->repeat_warned = c->stmt->repeat_of;
    }

    c->sm->n_warnings++;
    if (!c->sm->diag) return;
    if (c->stmt->from_repeat) details = repeat_note(c, details);

    SmpDiagMsg m;
    memset(&m, 0, sizeof m);
    m.code = code; m.span = sp; m.details = details; m.fix = fix;
    smp_diag_emit(c->sm->diag, &m);
}

static const char *sfmt(Ctx *c, const char *f, ...) SMP_PRINTF(2, 3);
static const char *sfmt(Ctx *c, const char *f, ...)
{
    if (!c->sm->diag) return NULL;
    static char scratch[SMP_FMT_SLOTLEN];
    va_list ap;
    va_start(ap, f);
    vsnprintf(scratch, sizeof scratch, f, ap);
    va_end(ap);
    return smp_fmt(c->sm->diag, "%s", scratch);
}

/* ========================================================================== */
/*  Печать значений в сообщениях                                              */
/* ========================================================================== */

static char *val_sig(const SmpValue *v, char *buf, size_t cap)
{
    if (v->is_scalar) {
        snprintf(buf, cap, "скаляр %s", smp_dtype_name(v->dtype));
        return buf;
    }
    size_t n = 0;
    int    w = snprintf(buf, cap, "%s:", smp_dtype_name(v->dtype));
    if (w > 0) n = (size_t)w;
    for (uint32_t i = 0; i < v->rank && n < cap; i++) {
        w = snprintf(buf + n, cap - n, "%s%u", i ? "," : "", (unsigned)v->shape[i]);
        if (w > 0) n += (size_t)w;
    }
    return buf;
}

static bool val_contiguous(const SmpValue *v)
{
    uint32_t acc = 1;
    for (uint32_t i = v->rank; i-- > 0; ) {
        if (v->stride[i] != (uint16_t)acc) return false;
        acc *= (uint32_t)v->shape[i];
    }
    return true;
}

static void val_dense(SmpValue *v)
{
    uint32_t acc = 1;
    for (uint32_t i = v->rank; i-- > 0; ) {
        v->stride[i] = (uint16_t)acc;
        acc *= (uint32_t)v->shape[i];
    }
    v->flags |= SMP_TF_CONTIG;
}

static uint64_t val_nelem(const SmpValue *v)
{
    uint64_t n = 1;
    for (uint32_t i = 0; i < v->rank; i++) n *= v->shape[i];
    return v->rank ? n : 1u;
}

/* ========================================================================== */
/*  Таблица символов                                                          */
/* ========================================================================== */

static SmpSym *sym_find(SmpSemaResult *r, SmpSymKind k, SmpName n)
{
    for (uint32_t i = 0; i < r->nsyms; i++)
        if (r->syms[i].kind == k &&
            r->syms[i].name.len == n.len &&
            memcmp(r->syms[i].name.p, n.p, n.len) == 0)
            return &r->syms[i];
    return NULL;
}

static const char *suggest_sym(Ctx *c, SmpSymKind k, SmpName got)
{
    const char *names[64];
    uint32_t    n = 0;
    for (uint32_t i = 0; i < c->res->nsyms && n < SMP_ARRLEN(names); i++) {
        if (c->res->syms[i].kind != k) continue;
        /* Имена не \0-терминированы; кладём копию в кольцо диагностики. */
        names[n++] = sfmt(c, "%.*s", (int)c->res->syms[i].name.len,
                          c->res->syms[i].name.p);
    }
    return n ? suggest(got, names, n) : NULL;
}

static SmpSym *sym_add(Ctx *c, SmpSymKind k, SmpName n, SmpSpan sp)
{
    if (c->res->nsyms >= SMP_MAX_SYMS) {
        serr(c, SMP_E0207, sp,
             sfmt(c, "Символов больше %u.", SMP_MAX_SYMS), NULL);
        return NULL;
    }
    SmpSym *s = &c->res->syms[c->res->nsyms++];
    memset(s, 0, sizeof *s);
    s->kind      = k;
    s->name      = n;
    s->decl_span = sp;
    s->val.sym   = SMP_SYM_NONE;
    return s;
}

static uint32_t sym_index(const SmpSemaResult *r, const SmpSym *s)
{
    return s ? (uint32_t)(s - r->syms) : SMP_SYM_NONE;
}

/* Выделение места под тензор. Каждый тензор начинается с границы 64 байт —
 * на этом стоит вся SIMD-часть, обсуждать нечего. */
static bool sym_alloc(Ctx *c, SmpSym *s, SmpSpan sp)
{
    const uint64_t sz = val_nelem(&s->val) * smp_dtype_size(s->val.dtype);
    if (s->arena_id >= SMP_MAX_ARENAS) return false;

    uint64_t *cursor = &c->res->arena_bytes[s->arena_id];
    *cursor = SMP_ALIGN_UP(*cursor, SMP_CACHELINE);

    if (*cursor + sz > 0xFFFFFFFFull) {
        serr(c, SMP_E0401, sp,
             sfmt(c, "Арена #%u перевалила за 4 ГиБ: смещения хранятся в uint32_t.",
                  s->arena_id), NULL);
        return false;
    }

    s->offset       = (uint32_t)*cursor;
    s->bytes        = sz;
    s->val.byte_off = 0;
    s->val.flags   |= SMP_TF_ALIGN64 | SMP_TF_ALIGN32;
    *cursor        += sz;

    if (s->arena_id > c->res->max_arena_id) c->res->max_arena_id = s->arena_id;
    return true;
}

/* ========================================================================== */
/*  Префикс и суффикс                                                         */
/* ========================================================================== */

static void resolve_prefix(Ctx *c)
{
    SmpStmtInfo *in = c->info;
    in->arena_id     = 0;
    in->req_vec_bits = 0;

    for (uint32_t i = 0; i < c->stmt->nprefix; i++) {
        const SmpAstPrefix *p = &c->stmt->prefix[i];

        if (p->kind == SMP_PFX_MODE) {
            const AttrDef *d = attr_find(g_modes, SMP_ARRLEN(g_modes), p->name);
            if (!d) {
                const char *hint = suggest_attr(p->name, g_modes, SMP_ARRLEN(g_modes));
                serr(c, SMP_E0311, p->span,
                     sfmt(c, "Режим '^%.*s' не существует.%s%s%s",
                          (int)p->name.len, p->name.p,
                          hint ? " Ты имел в виду ^" : "", hint ? hint : "", hint ? "." : ""), NULL);
                continue;
            }
            if (smp_name_is(p->name, "raw")) in->raw = true;
            continue;
        }

        const AttrDef *d = attr_find(g_directives, SMP_ARRLEN(g_directives), p->name);
        if (!d) {
            const char *hint = suggest_attr(p->name, g_directives, SMP_ARRLEN(g_directives));
            serr(c, SMP_E0311, p->span,
                 sfmt(c, "Директива '#%.*s' не существует.%s%s%s",
                      (int)p->name.len, p->name.p,
                      hint ? " Ты имел в виду #" : "", hint ? hint : "", hint ? "." : ""), NULL);
            continue;
        }

        if (d->val != VAL_NONE && !p->has_value) {
            serr(c, SMP_E0311, p->span,
                 sfmt(c, "Директива #%s требует значение (%s).", d->name, d->desc), NULL);
            continue;
        }
        if (d->val == VAL_INT && !p->value_is_int) {
            serr(c, SMP_E0311, p->value_span,
                 sfmt(c, "Директива #%s требует число.", d->name), NULL);
            continue;
        }
        if (d->val == VAL_IDENT && p->value_is_int) {
            serr(c, SMP_E0311, p->value_span,
                 sfmt(c, "Директива #%s требует имя (%s).", d->name, d->desc), NULL);
            continue;
        }

        if (smp_name_is(p->name, "arena")) {
            if (p->ival >= SMP_MAX_ARENAS) {
                serr(c, SMP_E0311, p->value_span,
                     sfmt(c, "Арен всего %u, запрошена #%llu.",
                          SMP_MAX_ARENAS, (unsigned long long)p->ival), NULL);
                continue;
            }
            in->arena_id = (uint32_t)p->ival;
        } else if (smp_name_is(p->name, "simd")) {
            uint32_t bits = 0;
            if      (smp_name_is(p->sval, "v128")) bits = 128;
            else if (smp_name_is(p->sval, "v256")) bits = 256;
            else if (smp_name_is(p->sval, "v512")) bits = 512;
            else {
                serr(c, SMP_E0501, p->value_span,
                     sfmt(c, "Запрошено '%.*s'; процессор знает только v128, v256 и v512.",
                          (int)p->sval.len, p->sval.p), NULL);
                continue;
            }
            in->req_vec_bits = bits;
        }
    }
}

static void resolve_suffix(Ctx *c)
{
    for (uint32_t i = 0; i < c->stmt->nsuffix; i++) {
        const SmpAstSuffix *s = &c->stmt->suffix[i];
        const AttrDef *tab; size_t n; const char *sig;

        switch (s->kind) {
            case SMP_SFX_ASSERT: tab = g_asserts; n = SMP_ARRLEN(g_asserts); sig = "!"; break;
            case SMP_SFX_QUERY:  tab = g_queries; n = SMP_ARRLEN(g_queries); sig = "?"; break;
            default:             tab = g_fpmodes; n = SMP_ARRLEN(g_fpmodes); sig = "~"; break;
        }

        if (!attr_find(tab, n, s->name)) {
            const char *hint = suggest_attr(s->name, tab, n);
            serr(c, SMP_E0311, s->span,
                 sfmt(c, "Атрибут '%s%.*s' не существует.%s%s%s%s", sig,
                      (int)s->name.len, s->name.p,
                      hint ? " Ты имел в виду " : "", hint ? sig : "",
                      hint ? hint : "", hint ? "." : ""),
                 NULL);
            continue;
        }

        if (s->kind == SMP_SFX_ASSERT && smp_name_is(s->name, "no-alias")) c->info->no_alias = true;
        if (s->kind == SMP_SFX_QUERY  && smp_name_is(s->name, "strict"))   c->info->strict   = true;
        if (s->kind == SMP_SFX_FPMODE && smp_name_is(s->name, "flush-to-zero")) c->info->ftz = true;
    }
}

/* Сверка запрошенной ширины вектора с тем, что действительно есть в кремнии. */
static void check_isa(Ctx *c)
{
    SmpStmtInfo *in   = c->info;
    const uint32_t host = c->sm->host_vec_bits;

    in->vec_bits = in->req_vec_bits ? in->req_vec_bits : host;
    if (in->req_vec_bits == 0 || in->req_vec_bits <= host) {
        if (in->vec_bits > host) in->vec_bits = host;
        return;
    }

    const SmpAstPrefix *p = smp_stmt_directive(c->stmt, "simd");
    const SmpSpan sp = p ? p->span : c->stmt->span;
    const char *det = sfmt(c,
        "Запрошен %s, а процессор предоставляет максимум %s.\n"
        "Инструкция будет исполнена %u проходами по %s.",
        smp_vec_name(in->req_vec_bits), smp_vec_name(host),
        in->req_vec_bits / host, smp_vec_name(host));

    if (in->strict) {
        serr(c, SMP_E0502, sp,
             sfmt(c, "Запрошен %s, а процессор предоставляет максимум %s.\n"
                     "Активен ?strict, поэтому эмуляция запрещена.",
                  smp_vec_name(in->req_vec_bits), smp_vec_name(host)), NULL);
        return;
    }
    swarn(c, SMP_W0512, sp, det, NULL);
    in->vec_bits = host;
}

/* ========================================================================== */
/*  Операнды                                                                  */
/* ========================================================================== */

/* Применение среза: [0, ..] превращает [1024,1024] в [1024] со своим шагом. */
static bool apply_index(Ctx *c, const SmpAstTensor *t, SmpValue *v)
{
    if (t->nidx != v->rank) {
        serr(c, SMP_E0309, t->index_span,
             sfmt(c, "Тензор ранга %u, а индексов указано %u. "
                     "Частичная индексация здесь не поддерживается.",
                  v->rank, t->nidx), NULL);
        return false;
    }

    SmpValue out;
    memset(&out, 0, sizeof out);
    out.dtype    = v->dtype;
    out.sym      = v->sym;
    out.flags    = (uint16_t)(v->flags & (uint16_t)~SMP_TF_CONTIG);
    out.byte_off = v->byte_off;

    const uint32_t esz = smp_dtype_size(v->dtype);
    uint32_t rank = 0;

    for (uint32_t i = 0; i < t->nidx; i++) {
        const SmpAstIndex *ix = &t->idx[i];
        switch (ix->kind) {
            case SMP_IDX_ALL:
                out.shape[rank]  = v->shape[i];
                out.stride[rank] = v->stride[i];
                rank++;
                break;

            case SMP_IDX_INT:
                if (ix->ival >= v->shape[i]) {
                    serr(c, SMP_E0410, ix->span,
                         sfmt(c, "Ось #%u имеет размер %u, а запрошен элемент %llu.",
                              i, (unsigned)v->shape[i],
                              (unsigned long long)ix->ival), NULL);
                    return false;
                }
                out.byte_off += (uint32_t)ix->ival * v->stride[i] * esz;
                break;

            case SMP_IDX_REG: {
                SmpSym *r = sym_find(c->res, SMP_SYM_REG, ix->reg);
                if (!r) {
                    serr(c, SMP_E0307, ix->span,
                         sfmt(c, "Регистр $%.*s не определён.",
                              (int)ix->reg.len, ix->reg.p), NULL);
                    return false;
                }
                /* Индекс — такое же чтение регистра, как и любое другое.
                 * Пока это не считалось, `2 => $i;` рядом с *&M[$i, ..] давал
                 * ложный W0301: значение записано и «никем не прочитано». */
                r->n_reads++;

                /* Смещение станет известно только в рантайме — статические
                 * проверки выравнивания по этой оси отключаются. */
                out.flags |= SMP_TF_DYNOFF;
                break;
            }
        }
    }

    if (rank == 0) {                 /* все оси зафиксированы -> элемент */
        out.is_scalar = true;
        out.rank      = 0;
    } else {
        out.rank = rank;
        if (val_contiguous(&out)) out.flags |= SMP_TF_CONTIG;
    }

    *v = out;
    return true;
}

static bool resolve_source_tensor(Ctx *c, const SmpAstTensor *t, SmpValue *out)
{
    SmpSym *s = sym_find(c->res, SMP_SYM_TENSOR, t->name);

    if (t->has_type) {
        if (s) {
            serr(c, SMP_E0308, t->span,
                 sfmt(c, "Тензор '%.*s' уже объявлен в %u:%u.",
                      (int)t->name.len, t->name.p,
                      s->decl_span.line, s->decl_span.col), NULL);
            return false;
        }
        s = sym_add(c, SMP_SYM_TENSOR, t->name, t->span);
        if (!s) return false;

        s->arena_id      = c->info->arena_id;
        s->val.dtype     = t->dtype;
        s->val.rank      = t->rank;
        for (uint32_t i = 0; i < t->rank; i++) s->val.shape[i] = t->dims[i];
        val_dense(&s->val);
        s->val.sym       = sym_index(c->res, s);
        s->val.is_scalar = (t->rank == 0);

        if (!sym_alloc(c, s, t->span)) return false;
    } else {
        if (!s) {
            const char *hint = suggest_sym(c, SMP_SYM_TENSOR, t->name);
            serr(c, SMP_E0307, t->span,
                 sfmt(c, "Тензор '%.*s' нигде не объявлен.%s%s%s",
                      (int)t->name.len, t->name.p,
                      hint ? " Ты имел в виду *&" : "", hint ? hint : "", hint ? "." : ""), NULL);
            return false;
        }
    }

    *out = s->val;
    out->sym = sym_index(c->res, s);

    if (t->has_index && !apply_index(c, t, out)) return false;
    return true;
}

/* Операнд в позиции чтения: источник или аргумент стадии. */
static bool read_operand(Ctx *c, const SmpAstOperand *o, SmpValue *out)
{
    memset(out, 0, sizeof *out);
    out->sym = SMP_SYM_NONE;

    switch (o->kind) {
        case SMP_OPD_REG: {
            SmpSym *r = sym_find(c->res, SMP_SYM_REG, o->reg);
            if (!r) {
                const char *hint = suggest_sym(c, SMP_SYM_REG, o->reg);
                serr(c, SMP_E0307, o->span,
                     sfmt(c, "Регистр '$%.*s' читается раньше, чем в него что-то записали.%s%s%s",
                          (int)o->reg.len, o->reg.p,
                          hint ? " Ты имел в виду $" : "", hint ? hint : "", hint ? "." : ""), NULL);
                return false;
            }
            r->n_reads++;
            *out = r->val;
            /* Регистр, хранящий тензор, читается как этот тензор. */
            if (out->sym != SMP_SYM_NONE) c->res->syms[out->sym].n_reads++;
            return true;
        }

        case SMP_OPD_TENSOR: {
            if (!resolve_source_tensor(c, o->tensor, out)) return false;
            if (out->sym != SMP_SYM_NONE) {
                SmpSym *s = &c->res->syms[out->sym];
                s->n_reads++;
                if (!s->initialized && !o->tensor->has_type)
                    swarn(c, SMP_W0302, o->span,
                          sfmt(c, "Тензор '%.*s' объявлен в %u:%u, но записи в него не было.",
                               (int)s->name.len, s->name.p,
                               s->decl_span.line, s->decl_span.col), NULL);
            }
            return true;
        }

        case SMP_OPD_INT:
            out->is_scalar = true;
            out->dtype     = SMP_DT_I32;
            return true;

        case SMP_OPD_FLOAT:
            out->is_scalar = true;
            out->dtype     = SMP_DT_F32;
            return true;

        default:
            serr(c, SMP_E0209, o->span, NULL, NULL);
            return false;
    }
}

/* ========================================================================== */
/*  Стадии                                                                    */
/* ========================================================================== */

/* Определена ниже, среди проверок памяти; нужна уже здесь. */
static void check_align(Ctx *c, const SmpValue *v, SmpSpan sp, const char *what);

static bool require_tensor(Ctx *c, const SmpValue *v, SmpSpan sp, const char *op)
{
    if (!v->is_scalar) return true;
    serr(c, SMP_E0309, sp,
         sfmt(c, "@%s работает с тензором, а на входе скаляр.", op), NULL);
    return false;
}

static bool require_rank(Ctx *c, const SmpValue *v, uint32_t rank, SmpSpan sp,
                         const char *op)
{
    if (!v->is_scalar && v->rank == rank) return true;
    char sig[80];
    serr(c, SMP_E0309, sp,
         sfmt(c, "@%s требует ранг %u, а на входе %s (ранг %u).",
              op, rank, val_sig(v, sig, sizeof sig), v->rank), NULL);
    return false;
}

static bool apply_stage(Ctx *c, const SmpAstStage *st, SmpOpKind k, SmpValue *v)
{
    const SmpOpDef *def = smp_op_def(k);

    if (st->nargs < def->min_args || st->nargs > def->max_args) {
        serr(c, SMP_E0306, st->span,
             sfmt(c, "@%s принимает %u аргумент(ов), а передано %u.",
                  def->name, def->min_args, st->nargs), NULL);
        return false;
    }

    SmpValue a0;
    memset(&a0, 0, sizeof a0);
    a0.sym = SMP_SYM_NONE;

    if (st->nargs >= 1) {
        if (!read_operand(c, &st->args[0], &a0)) return false;
        if (c->cur_stage < SMP_MAX_STAGES) c->info->arg_val[c->cur_stage] = a0;
        if (a0.sym != SMP_SYM_NONE && c->n_arg_syms < SMP_ARRLEN(c->arg_syms))
            c->arg_syms[c->n_arg_syms++] = a0.sym;
        check_align(c, &a0, st->args[0].span, "Аргумент");
        if (c->bad) return false;
    }

    switch (k) {
        case SMP_OP_ALLOC:
            if (!require_tensor(c, v, st->span, "alloc")) return false;
            if (v->sym == SMP_SYM_NONE) {
                serr(c, SMP_E0309, st->span,
                     "@alloc применим только к именованному тензору.", NULL);
                return false;
            }
            c->res->syms[v->sym].initialized = true;
            return true;

        case SMP_OP_FILL_INST:
            /* Единственное, чем инстансы пула отличаются друг от друга внутри
             * языка. Без этого N копий одной программы посчитали бы N раз одно
             * и то же, и пул был бы пригоден только для замеров. */
            if (!require_tensor(c, v, st->span, "fill.instance")) return false;
            if (v->sym != SMP_SYM_NONE) c->res->syms[v->sym].initialized = true;
            return true;

        case SMP_OP_FILL:
            if (!require_tensor(c, v, st->span, "fill")) return false;
            if (!a0.is_scalar) {
                serr(c, SMP_E0309, st->args[0].span,
                     "@fill заполняет тензор скаляром, а получил тензор.", NULL);
                return false;
            }
            if (v->sym != SMP_SYM_NONE) c->res->syms[v->sym].initialized = true;
            return true;

        case SMP_OP_MMUL: {
            if (!require_rank(c, v, 2, st->span, "mmul")) return false;
            if (a0.is_scalar || a0.rank != 2) {
                char sig[80];
                serr(c, SMP_E0309, st->args[0].span,
                     sfmt(c, "Второй множитель обязан быть матрицей, а это %s.",
                          val_sig(&a0, sig, sizeof sig)), NULL);
                return false;
            }
            if (v->dtype != a0.dtype) {
                serr(c, SMP_E0303, st->args[0].span,
                     sfmt(c, "Множители имеют разные типы: %s и %s.",
                          smp_dtype_name(v->dtype), smp_dtype_name(a0.dtype)), NULL);
                return false;
            }

            const uint32_t M = v->shape[0], K1 = v->shape[1];
            const uint32_t K2 = a0.shape[0], N = a0.shape[1];
            if (K1 != K2) {
                char s1[80], s2[80];
                serr(c, SMP_E0419, st->span,
                     sfmt(c, "Левый множитель %s, правый %s.\n"
                             "Размерности K не сходятся (%u != %u).",
                          val_sig(v, s1, sizeof s1), val_sig(&a0, s2, sizeof s2),
                          K1, K2),
                     sfmt(c, "Вызови @transpose над правым множителем "
                             "(получится [%u,%u]) или перепиши свой код на Scratch.",
                          N, K2));
                return false;
            }

            SmpValue out;
            memset(&out, 0, sizeof out);
            out.dtype    = v->dtype;
            out.rank     = 2;
            out.shape[0] = (uint16_t)M;
            out.shape[1] = (uint16_t)N;
            out.sym      = SMP_SYM_NONE;
            val_dense(&out);
            *v = out;
            return true;
        }

        case SMP_OP_TRANSPOSE: {
            if (!require_rank(c, v, 2, st->span, "transpose")) return false;
            const uint16_t s0 = v->shape[0], s1 = v->shape[1];
            const uint16_t t0 = v->stride[0], t1 = v->stride[1];
            v->shape[0] = s1; v->shape[1] = s0;
            v->stride[0] = t1; v->stride[1] = t0;
            v->flags = (uint16_t)((v->flags & (uint16_t)~SMP_TF_CONTIG) | SMP_TF_TRANSPOSED);
            if (val_contiguous(v)) v->flags |= SMP_TF_CONTIG;
            return true;
        }

        case SMP_OP_PACK:
            if (!require_tensor(c, v, st->span, "pack")) return false;
            v->sym   = SMP_SYM_NONE;
            v->flags = (uint16_t)(v->flags & (uint16_t)~SMP_TF_TRANSPOSED);
            v->byte_off = 0;
            val_dense(v);
            return true;

        case SMP_OP_RELU:
        case SMP_OP_ABS:
            if (!require_tensor(c, v, st->span, def->name)) return false;
            if (v->dtype == SMP_DT_RAW_PTR) {
                serr(c, SMP_E0303, st->span,
                     sfmt(c, "@%s не определён для raw_ptr.", def->name), NULL);
                return false;
            }
            return true;

        case SMP_OP_SCALE:
            if (!require_tensor(c, v, st->span, "scale")) return false;
            if (!a0.is_scalar) {
                serr(c, SMP_E0309, st->args[0].span,
                     "@scale умножает на скаляр, а получил тензор.", NULL);
                return false;
            }
            return true;

        case SMP_OP_ADD:
        case SMP_OP_MUL: {
            if (!require_tensor(c, v, st->span, def->name)) return false;
            if (a0.is_scalar) {
                serr(c, SMP_E0309, st->args[0].span,
                     sfmt(c, "@%s поэлементный: оба операнда обязаны быть тензорами. "
                             "Для скаляра есть @scale.", def->name), NULL);
                return false;
            }
            if (v->dtype != a0.dtype) {
                serr(c, SMP_E0303, st->args[0].span,
                     sfmt(c, "Операнды имеют разные типы: %s и %s.",
                          smp_dtype_name(v->dtype), smp_dtype_name(a0.dtype)), NULL);
                return false;
            }
            if (v->rank != a0.rank ||
                memcmp(v->shape, a0.shape, sizeof(uint16_t) * v->rank) != 0) {
                char s1[80], s2[80];
                serr(c, SMP_E0309, st->args[0].span,
                     sfmt(c, "Формы не совпадают: %s и %s.\n"
                             "Broadcast в этом языке отсутствует намеренно.",
                          val_sig(v, s1, sizeof s1), val_sig(&a0, s2, sizeof s2)), NULL);
                return false;
            }
            return true;
        }

        case SMP_OP_REDUCE_ADD:
        case SMP_OP_REDUCE_MAX:
            /* Скаляр на входе допустим намеренно: свёртка одного элемента —
             * это сам элемент. Без этого послабления полностью
             * проиндексированный срез вида *&M[2,3] некуда деть: он уже
             * скаляр, а положить его в регистр значением больше нечем. */
            v->is_scalar = true;
            v->rank      = 0;
            v->sym       = SMP_SYM_NONE;
            v->flags     = SMP_TF_CONTIG;
            v->byte_off  = 0;
            return true;

        case SMP_OP_EMIT_TEXT:
        case SMP_OP_EMIT_LINE:
        case SMP_OP_EMIT_DEC:
        case SMP_OP_EMIT_HEX:
        case SMP_OP_EMIT_BITS:
        case SMP_OP_EMIT_NUM: {
            /* Вывод — такая же свёртка, как @reduce: съедает тензор, отдаёт
             * скаляр. Отдельной «процедуры без результата» в языке нет, и
             * заводить её ради печати незачем: число выведенных элементов —
             * вполне себе значение, а конвейер остаётся однородным. */
            if (k != SMP_OP_EMIT_NUM && smp_dtype_is_float(v->dtype)) {
                serr(c, SMP_E0303, st->span,
                     sfmt(c, "@%s печатает целые, а на входе %s. "
                             "Приведи явно через @cast.i32 либо возьми @emit.num.",
                          def->name, smp_dtype_name(v->dtype)), NULL);
                return false;
            }
            if (v->dtype == SMP_DT_RAW_PTR) {
                serr(c, SMP_E0303, st->span,
                     "Печатать сырой указатель бессмысленно.", NULL);
                return false;
            }
            v->is_scalar = true;
            v->rank      = 0;
            v->dtype     = SMP_DT_U64;      /* сколько элементов выведено */
            v->sym       = SMP_SYM_NONE;
            v->flags     = SMP_TF_CONTIG;
            v->byte_off  = 0;
            return true;
        }

        case SMP_OP_CAST_F32:
        case SMP_OP_CAST_F64:
        case SMP_OP_CAST_I32:
        case SMP_OP_CAST_U64: {
            const SmpDType to = k == SMP_OP_CAST_F32 ? SMP_DT_F32
                              : k == SMP_OP_CAST_F64 ? SMP_DT_F64
                              : k == SMP_OP_CAST_I32 ? SMP_DT_I32 : SMP_DT_U64;
            v->dtype = to;
            if (!v->is_scalar) {
                v->sym      = SMP_SYM_NONE;   /* приведение создаёт копию */
                v->byte_off = 0;
                val_dense(v);
            }
            return true;
        }

        default:
            return false;
    }
}

/* ========================================================================== */
/*  Проверки памяти                                                           */
/* ========================================================================== */

static void check_align(Ctx *c, const SmpValue *v, SmpSpan sp, const char *what)
{
    /* Требовать выравнивания можно только там, где ширина запрошена ЯВНО.
     * Без #simd компилятор сам выбирает загрузки и вправе взять невыровненные,
     * так что придираться к смещению среза не за что. */
    const uint32_t bits = c->info->req_vec_bits;
    if (bits < 256 || v->is_scalar) return;
    if (v->flags & SMP_TF_DYNOFF)   return;   /* смещение из регистра */

    const uint32_t req = bits / 8u;           /* 32 для v256, 64 для v512 */
    /* База тензора выровнена на 64 при выделении, поэтому весь вопрос —
     * в смещении среза. */
    if (v->byte_off % req == 0) return;

    serr(c, SMP_E0402, sp,
         sfmt(c, "%s начинается со смещения %u байт от базы тензора, "
                 "а %s требует кратности %u.\n"
                 "База выровнена на 64, срез сбил выравнивание.",
              what, v->byte_off, smp_vec_name(bits), req),
         sfmt(c, "Материализуй срез через @pack либо возьми ось, "
                 "длина которой кратна %u элементам.",
              req / (smp_dtype_size(v->dtype) ? smp_dtype_size(v->dtype) : 1u)));
}

/* Проверка ^raw. Вызывается перед каждой стадией, КРОМЕ @pack: именно @pack
 * умеет ходить по шагам и превращать разреженный вид в плотный, и он же стоит
 * в тексте исправления к E0421. Запрещать ему работать под ^raw значило бы
 * советовать средство и тут же его отбирать. */
static void check_raw(Ctx *c, const SmpValue *v, SmpSpan sp)
{
    if (!c->info->raw || v->is_scalar) return;
    if (v->flags & SMP_TF_CONTIG)    return;

    serr(c, SMP_E0418, sp,
         sfmt(c, "Режим ^raw отдаёт голый указатель, а этот срез имеет шаг "
                 "%u элементов при длине оси %u — плотным он не является.",
              (unsigned)v->stride[0], (unsigned)v->shape[0]), NULL);
}

/* ========================================================================== */
/*  Инструкция целиком                                                        */
/* ========================================================================== */

static void check_stmt(Ctx *c)
{
    const SmpAstStmt *s  = c->stmt;
    SmpStmtInfo      *in = c->info;

    resolve_prefix(c);
    resolve_suffix(c);
    check_isa(c);
    if (c->bad) return;

    /* --- источник --- */
    SmpValue v;
    if (!read_operand(c, &s->source, &v)) return;
    const uint32_t src_sym = v.sym;
    in->src_val = v;

    check_align(c, &v, s->source.span, "Источник");
    if (c->bad) return;

    /* --- конвейер --- */
    for (uint32_t i = 0; i < s->nstages; i++) {
        const SmpAstStage *st = &s->stages[i];
        const SmpOpKind    k  = smp_op_lookup(st->name);
        c->cur_stage = i;

        if (k == SMP_OP__UNKNOWN) {
            const char *names[SMP_OP__COUNT];
            for (unsigned j = 0; j < SMP_OP__COUNT; j++) names[j] = g_ops[j].name;
            const char *hint = suggest(st->name, names, SMP_OP__COUNT);
            serr(c, SMP_E0305, st->name_span,
                 sfmt(c, "Операции '@%.*s' не существует.%s%s%s",
                      (int)st->name.len, st->name.p,
                      hint ? " Ты имел в виду @" : "", hint ? hint : "", hint ? "." : ""), NULL);
            return;
        }
        in->ops[i] = k;

        if (k != SMP_OP_PACK) {
            check_raw(c, &v, i == 0 ? s->source.span : st->span);
            if (c->bad) return;
        }

        if (!apply_stage(c, st, k, &v)) return;
        in->stage_out[i] = v;
    }

    /* Конвейер без стадий: значение уходит в приёмник как есть. */
    if (s->nstages == 0) {
        check_raw(c, &v, s->source.span);
        if (c->bad) return;
    }

    /* --- приёмник --- */
    if (!s->has_dest) return;

    uint32_t dest_sym = SMP_SYM_NONE;

    if (s->dest.kind == SMP_OPD_REG) {
        SmpSym *r = sym_find(c->res, SMP_SYM_REG, s->dest.reg);
        if (!r) {
            r = sym_add(c, SMP_SYM_REG, s->dest.reg, s->dest.span);
            if (!r) return;
        }
        r->val        = v;
        r->n_writes++;
        r->last_write = s->dest.span;
        dest_sym      = v.sym;
        in->dest_val  = v;
    } else {
        SmpValue dv;
        if (!resolve_source_tensor(c, s->dest.tensor, &dv)) return;
        dest_sym            = dv.sym;
        in->dest_val        = dv;
        in->dest_is_tensor  = true;

        if (v.is_scalar != dv.is_scalar || v.dtype != dv.dtype ||
            v.rank != dv.rank ||
            (v.rank && memcmp(v.shape, dv.shape, sizeof(uint16_t) * v.rank) != 0)) {
            char s1[80], s2[80];
            serr(c, SMP_E0303, s->dest.span,
                 sfmt(c, "Конвейер даёт %s, а приёмник объявлен как %s.",
                      val_sig(&v, s1, sizeof s1), val_sig(&dv, s2, sizeof s2)), NULL);
            return;
        }

        /* Писать плотный результат в разреженный вид нельзя без упаковки. */
        if (!(dv.flags & SMP_TF_CONTIG) && dv.rank > 0) {
            serr(c, SMP_E0421, s->dest.span,
                 sfmt(c, "Приёмник — срез с шагом %u элементов, а результат плотный.",
                      (unsigned)dv.stride[0]), NULL);
            return;
        }

        check_align(c, &dv, s->dest.span, "Приёмник");
        if (c->bad) return;

        if (dest_sym != SMP_SYM_NONE) {
            SmpSym *ds = &c->res->syms[dest_sym];
            ds->initialized = true;
            ds->n_writes++;
            ds->last_write  = s->dest.span;
        }
    }

    /* --- честность [!no-alias] --- */
    if (in->no_alias && dest_sym != SMP_SYM_NONE) {
        uint32_t clash = SMP_SYM_NONE;
        if (src_sym == dest_sym) clash = src_sym;
        for (uint32_t i = 0; i < c->n_arg_syms && clash == SMP_SYM_NONE; i++)
            if (c->arg_syms[i] == dest_sym) clash = c->arg_syms[i];

        if (clash != SMP_SYM_NONE) {
            const SmpSym *cs = &c->res->syms[clash];
            const SmpAstSuffix *na = smp_stmt_suffix(s, SMP_SFX_ASSERT, "no-alias");
            serr(c, SMP_E0420, na ? na->span : s->span,
                 sfmt(c, "Тензор '%.*s' стоит и слева, и справа от '=>'. "
                         "Обещание [!no-alias] заведомо ложно.",
                      (int)cs->name.len, cs->name.p), NULL);
            return;
        }
    }

    in->result   = v;
    in->dest_sym = dest_sym;
    in->ok       = !c->bad;
}

/* ========================================================================== */
/*  Точка входа                                                               */
/* ========================================================================== */

void smp_sema_init(SmpSema *sm, SmpArena *arena, SmpDiagCtx *diag,
                   const SmpSource *src)
{
    memset(sm, 0, sizeof *sm);
    sm->arena         = arena;
    sm->diag          = diag;
    sm->src           = src;
    sm->host_vec_bits = smp_cpu()->max_vec_bits;
    sm->repeat_told   = SMP_REPEAT_NONE;
    sm->repeat_warned = SMP_REPEAT_NONE;

    /* SMPC3_VEC_BITS выдаёт компилятору другую ширину вектора, чем есть в
     * кремнии. Нужно в двух местах: снимки диагностики должны совпадать на
     * любой машине, и чужой отчёт об ошибке должно быть можно воспроизвести
     * у себя, не имея того же процессора. На кодогенерацию не влияет —
     * ветку ядер по-прежнему выбирает CPUID. */
    const char *env = getenv("SMPC3_VEC_BITS");
    if (env) {
        const unsigned long v = strtoul(env, NULL, 10);
        if (v == 128u || v == 256u || v == 512u) sm->host_vec_bits = (uint32_t)v;
    }
}

SmpStatus smp_sema_run(SmpSema *sm, const SmpAstProgram *prog, SmpSemaResult *res)
{
    memset(res, 0, sizeof *res);

    res->syms = (SmpSym *)smp_arena_push_raw(sm->arena, SMP_MAX_SYMS * sizeof(SmpSym), 8);
    if (!res->syms) return SMP_ERR_OOM;

    if (prog->nstmts) {
        res->info = (SmpStmtInfo *)smp_arena_push_raw(
            sm->arena, prog->nstmts * sizeof(SmpStmtInfo), 8);
        if (!res->info) return SMP_ERR_OOM;
        memset(res->info, 0, prog->nstmts * sizeof(SmpStmtInfo));
    }
    res->ninfo = prog->nstmts;

    for (uint32_t i = 0; i < prog->nstmts; i++) {
        Ctx c;
        memset(&c, 0, sizeof c);
        c.sm   = sm;
        c.res  = res;
        c.stmt = &prog->stmts[i];
        c.info = &res->info[i];
        c.info->dest_sym = SMP_SYM_NONE;
        check_stmt(&c);
    }

    /* Записали и не прочитали — почти всегда опечатка в имени регистра. */
    for (uint32_t i = 0; i < res->nsyms; i++) {
        const SmpSym *s = &res->syms[i];
        if (s->kind != SMP_SYM_REG || s->n_writes == 0 || s->n_reads > 0) continue;

        sm->n_warnings++;
        if (!sm->diag) continue;
        SmpDiagMsg m;
        memset(&m, 0, sizeof m);
        m.code    = SMP_W0301;
        m.span    = s->last_write;
        m.details = smp_fmt(sm->diag,
            "В регистр $%.*s записано значение, но ни одна инструкция его не читает.",
            (int)s->name.len, s->name.p);
        smp_diag_emit(sm->diag, &m);
    }

    return sm->n_errors ? SMP_ERR_TYPE : SMP_OK;
}

/* ========================================================================== */
/*  Печать                                                                    */
/* ========================================================================== */

void smp_sema_dump(FILE *out, const SmpSemaResult *res)
{
    fprintf(out, "СИМВОЛЫ\n");
    for (uint32_t i = 0; i < res->nsyms; i++) {
        const SmpSym *s = &res->syms[i];
        char sig[80];
        if (s->kind == SMP_SYM_TENSOR) {
            fprintf(out, "  *&%-8.*s арена#%u  off=0x%08X  %10llu Б  <%s>  %s  чт=%u зп=%u\n",
                    (int)s->name.len, s->name.p, s->arena_id, s->offset,
                    (unsigned long long)s->bytes, val_sig(&s->val, sig, sizeof sig),
                    s->initialized ? "инициализирован" : "ПУСТ",
                    s->n_reads, s->n_writes);
        } else {
            fprintf(out, "  $%-9.*s %s  чт=%u зп=%u\n",
                    (int)s->name.len, s->name.p,
                    val_sig(&s->val, sig, sizeof sig), s->n_reads, s->n_writes);
        }
    }

    fprintf(out, "\nИНСТРУКЦИИ\n");
    for (uint32_t i = 0; i < res->ninfo; i++) {
        const SmpStmtInfo *in = &res->info[i];
        char sig[80];
        fprintf(out, "  #%-3u арена#%u  %s", i, in->arena_id, smp_vec_name(in->vec_bits));
        if (in->req_vec_bits && in->req_vec_bits != in->vec_bits)
            fprintf(out, " (просили %s)", smp_vec_name(in->req_vec_bits));
        if (in->no_alias) fprintf(out, " !no-alias");
        if (in->strict)   fprintf(out, " ?strict");
        if (in->ftz)      fprintf(out, " ~ftz");
        if (in->raw)      fprintf(out, " ^raw");
        fprintf(out, "  ->  %s\n",
                in->ok ? val_sig(&in->result, sig, sizeof sig) : "<не прошла проверку>");
    }

    fprintf(out, "\nАРЕНЫ\n");
    for (uint32_t i = 0; i <= res->max_arena_id && i < SMP_MAX_ARENAS; i++)
        fprintf(out, "  #%u  %llu байт (%.2f МиБ)\n", i,
                (unsigned long long)res->arena_bytes[i],
                (double)res->arena_bytes[i] / (1024.0 * 1024.0));
}
