/* SMPC3 :: ast.c -- запросы к дереву и его печать. */
#include "smpc3/ast.h"

#include <string.h>
#include <stdio.h>

/* ========================================================================== */

bool smp_name_is(SmpName n, const char *s)
{
    const size_t len = strlen(s);
    return n.len == (uint32_t)len && n.p && memcmp(n.p, s, len) == 0;
}

const SmpAstPrefix *smp_stmt_directive(const SmpAstStmt *s, const char *name)
{
    for (uint32_t i = 0; i < s->nprefix; i++)
        if (s->prefix[i].kind == SMP_PFX_DIRECTIVE &&
            smp_name_is(s->prefix[i].name, name))
            return &s->prefix[i];
    return NULL;
}

const SmpAstPrefix *smp_block_directive(const SmpAstBlock *b, const char *name)
{
    for (uint32_t i = 0; i < b->nprefix; i++)
        if (b->prefix[i].kind == SMP_PFX_DIRECTIVE &&
            smp_name_is(b->prefix[i].name, name))
            return &b->prefix[i];
    return NULL;
}

bool smp_stmt_has_mode(const SmpAstStmt *s, const char *name)
{
    for (uint32_t i = 0; i < s->nprefix; i++)
        if (s->prefix[i].kind == SMP_PFX_MODE &&
            smp_name_is(s->prefix[i].name, name))
            return true;
    return false;
}

const SmpAstSuffix *smp_stmt_suffix(const SmpAstStmt *s, SmpSuffixKind k,
                                    const char *name)
{
    for (uint32_t i = 0; i < s->nsuffix; i++)
        if (s->suffix[i].kind == k && smp_name_is(s->suffix[i].name, name))
            return &s->suffix[i];
    return NULL;
}

/* ========================================================================== */
/*  Печать                                                                    */
/* ========================================================================== */

#define AC_RESET "\x1b[0m"
#define AC_DIM   "\x1b[2m"
#define AC_KEY   "\x1b[96m"
#define AC_NAME  "\x1b[93m"
#define AC_NUM   "\x1b[95m"

typedef struct {
    FILE *o;
    bool  color;
} Pr;

static const char *pc(const Pr *pr, const char *seq) { return pr->color ? seq : ""; }

static void pr_indent(const Pr *pr, uint32_t d)
{
    for (uint32_t i = 0; i < d; i++) fputs("  ", pr->o);
}

static void pr_name(const Pr *pr, SmpName n)
{
    fprintf(pr->o, "%.*s", (int)n.len, n.p ? n.p : "");
}

static void pr_tensor(const Pr *pr, const SmpAstTensor *t, uint32_t d)
{
    pr_indent(pr, d);
    fprintf(pr->o, "%sтензор%s ", pc(pr, AC_KEY), pc(pr, AC_RESET));
    fprintf(pr->o, "%s", pc(pr, AC_NAME));
    pr_name(pr, t->name);
    fprintf(pr->o, "%s", pc(pr, AC_RESET));

    if (t->has_type) {
        fprintf(pr->o, " <%s", smp_dtype_name(t->dtype));
        for (uint32_t i = 0; i < t->rank; i++)
            fprintf(pr->o, "%s%u", i ? "," : ":", (unsigned)t->dims[i]);
        fputc('>', pr->o);
        fprintf(pr->o, " %s(объявление)%s", pc(pr, AC_DIM), pc(pr, AC_RESET));
    }

    if (t->has_index) {
        fputs(" [", pr->o);
        for (uint32_t i = 0; i < t->nidx; i++) {
            if (i) fputs(", ", pr->o);
            switch (t->idx[i].kind) {
                case SMP_IDX_ALL: fputs("..", pr->o); break;
                case SMP_IDX_INT: fprintf(pr->o, "%llu",
                                          (unsigned long long)t->idx[i].ival); break;
                case SMP_IDX_REG: fputc('$', pr->o); pr_name(pr, t->idx[i].reg); break;
            }
        }
        fputc(']', pr->o);
    }
    fprintf(pr->o, " %s@%u:%u%s\n", pc(pr, AC_DIM),
            t->span.line, t->span.col, pc(pr, AC_RESET));
}

static void pr_operand(const Pr *pr, const SmpAstOperand *o, const char *label, uint32_t d)
{
    switch (o->kind) {
        case SMP_OPD_REG:
            pr_indent(pr, d);
            fprintf(pr->o, "%s%s%s регистр %s$", pc(pr, AC_KEY), label,
                    pc(pr, AC_RESET), pc(pr, AC_NAME));
            pr_name(pr, o->reg);
            fprintf(pr->o, "%s %s@%u:%u%s\n", pc(pr, AC_RESET), pc(pr, AC_DIM),
                    o->span.line, o->span.col, pc(pr, AC_RESET));
            break;

        case SMP_OPD_TENSOR:
            pr_indent(pr, d);
            fprintf(pr->o, "%s%s%s\n", pc(pr, AC_KEY), label, pc(pr, AC_RESET));
            pr_tensor(pr, o->tensor, d + 1u);
            break;

        case SMP_OPD_INT:
            pr_indent(pr, d);
            fprintf(pr->o, "%s%s%s целое %s%lld%s\n", pc(pr, AC_KEY), label,
                    pc(pr, AC_RESET), pc(pr, AC_NUM),
                    (long long)o->ival, pc(pr, AC_RESET));
            break;

        case SMP_OPD_FLOAT:
            pr_indent(pr, d);
            fprintf(pr->o, "%s%s%s веществ. %s%g%s\n", pc(pr, AC_KEY), label,
                    pc(pr, AC_RESET), pc(pr, AC_NUM), o->fval, pc(pr, AC_RESET));
            break;

        case SMP_OPD_LIST: {
            const SmpAstList *l = o->list;
            pr_indent(pr, d);
            fprintf(pr->o, "%s%s%s литерал %s", pc(pr, AC_KEY), label,
                    pc(pr, AC_RESET), pc(pr, AC_NUM));
            for (uint32_t i = 0; i < l->rank; i++)
                fprintf(pr->o, "%s%u", i ? "x" : "", l->dims[i]);
            fprintf(pr->o, " [");
            for (uint32_t i = 0; i < l->n && i < 8u; i++) {
                if (l->vals[i].is_float) fprintf(pr->o, "%s%g", i ? ", " : "", l->vals[i].fval);
                else fprintf(pr->o, "%s%lld", i ? ", " : "", (long long)l->vals[i].ival);
            }
            fprintf(pr->o, "%s]%s %s@%u:%u%s\n", l->n > 8u ? ", ..." : "",
                    pc(pr, AC_RESET), pc(pr, AC_DIM), o->span.line, o->span.col,
                    pc(pr, AC_RESET));
            break;
        }

        default:
            pr_indent(pr, d);
            fprintf(pr->o, "%s%s%s <пусто>\n", pc(pr, AC_KEY), label, pc(pr, AC_RESET));
            break;
    }
}

static void pr_prefix(const Pr *pr, const SmpAstPrefix *pf, const char *label)
{
    pr_indent(pr, 1);
    if (pf->kind == SMP_PFX_DIRECTIVE) {
        fprintf(pr->o, "%s  #", label);
        pr_name(pr, pf->name);
        if (pf->has_value) {
            fputc(':', pr->o);
            if (pf->value_is_int) fprintf(pr->o, "%llu", (unsigned long long)pf->ival);
            else                  pr_name(pr, pf->sval);
        }
    } else {
        fprintf(pr->o, "%s  ^", label);
        pr_name(pr, pf->name);
    }
    fputc('\n', pr->o);
}

static void pr_stmt(const Pr *pr, const SmpAstStmt *s, uint32_t idx)
{
    fprintf(pr->o, "%sинструкция #%u%s %s(строка %u)%s\n",
            pc(pr, AC_KEY), idx, pc(pr, AC_RESET),
            pc(pr, AC_DIM), s->span.line, pc(pr, AC_RESET));

    /* Префикс блока печатается у каждой его инструкции: дерево плоское, и
     * без этой строки не видно, что инструкция повторяется вместе с соседями. */
    if (s->block) {
        pr_indent(pr, 1);
        fprintf(pr->o, "в блоке  %s'{' в %u:%u%s\n", pc(pr, AC_DIM),
                s->block->open.line, s->block->open.col, pc(pr, AC_RESET));
        for (uint32_t i = 0; i < s->block->nprefix; i++)
            pr_prefix(pr, &s->block->prefix[i], "  блок");
    }

    for (uint32_t i = 0; i < s->nprefix; i++)
        pr_prefix(pr, &s->prefix[i], "префикс");

    pr_operand(pr, &s->source, "источник", 1);

    for (uint32_t i = 0; i < s->nstages; i++) {
        const SmpAstStage *st = &s->stages[i];
        pr_indent(pr, 1);
        fprintf(pr->o, "стадия %u  %s@", i, pc(pr, AC_NAME));
        pr_name(pr, st->name);
        fprintf(pr->o, "%s", pc(pr, AC_RESET));
        if (st->nargs) fprintf(pr->o, "  (%u арг.)", st->nargs);
        fputc('\n', pr->o);
        for (uint32_t a = 0; a < st->nargs; a++)
            pr_operand(pr, &st->args[a], "аргумент", 2);
    }

    if (s->has_dest) pr_operand(pr, &s->dest, "приёмник", 1);

    for (uint32_t i = 0; i < s->nsuffix; i++) {
        const SmpAstSuffix *sf = &s->suffix[i];
        const char sig = sf->kind == SMP_SFX_ASSERT ? '!'
                       : sf->kind == SMP_SFX_QUERY  ? '?' : '~';
        pr_indent(pr, 1);
        fprintf(pr->o, "суффикс  %c", sig);
        pr_name(pr, sf->name);
        fputc('\n', pr->o);
    }
}

/* ========================================================================== */
/*  Развёртка [#repeat:N] и блоков [#repeat:N] { … }                          */
/* ========================================================================== */

static bool name_eq(SmpName a, SmpName b)
{
    return a.len == b.len && a.p && b.p && memcmp(a.p, b.p, a.len) == 0;
}

static void rep_err(SmpDiagCtx *d, uint32_t *nerr, SmpSpan sp,
                    const char *details, const char *fix)
{
    (*nerr)++;
    if (!d) return;

    SmpDiagMsg m;
    memset(&m, 0, sizeof m);
    m.code    = SMP_E0310;
    m.span    = sp;
    m.details = details;
    m.fix     = fix;
    smp_diag_emit(d, &m);
}

/* Что подставить на этом повторе: имя индекса и его номер. Подстановок не
 * больше двух — индекс блока и индекс своей [#repeat] инструкции. */
typedef struct Subst {
    SmpName  name;
    uint64_t v;
} Subst;

static bool subst_find(const Subst *sb, uint32_t nsb, SmpName reg, uint64_t *v)
{
    for (uint32_t k = 0; k < nsb; k++)
        if (name_eq(reg, sb[k].name)) { *v = sb[k].v; return true; }
    return false;
}

/* Копия ссылки на тензор с подставленными индексами. Копия нужна даже там,
 * где подставлять нечего: узлы разных повторов не должны делить память,
 * иначе правка одного тихо меняла бы остальные. */
static SmpAstTensor *subst_tensor(const SmpAstTensor *t, const Subst *sb,
                                  uint32_t nsb, SmpArena *ar)
{
    SmpAstTensor *n = (SmpAstTensor *)smp_arena_push_raw(ar, sizeof *n, 8);
    if (!n) return NULL;
    *n = *t;

    for (uint32_t i = 0; i < n->nidx; i++) {
        uint64_t v;
        if (n->idx[i].kind == SMP_IDX_REG && subst_find(sb, nsb, n->idx[i].reg, &v)) {
            n->idx[i].kind = SMP_IDX_INT;
            n->idx[i].ival = v;
        }
    }
    return n;
}

/* Операнд на месте: регистр-индекс становится литералом, тензор копируется.
 *
 * Индекс подставляется и там, где он стоит просто операндом, а не только в
 * квадратных скобках. Иначе половина упоминаний одного имени была бы числом,
 * а половина осталась бы регистром, и объяснить это правило было бы нечем. */
static bool subst_operand(SmpAstOperand *o, const Subst *sb, uint32_t nsb,
                          SmpArena *ar)
{
    uint64_t v;
    if (o->kind == SMP_OPD_REG && subst_find(sb, nsb, o->reg, &v)) {
        o->kind = SMP_OPD_INT;
        o->ival = v;
        return true;
    }
    if (o->kind == SMP_OPD_TENSOR && o->tensor) {
        SmpAstTensor *t = subst_tensor(o->tensor, sb, nsb, ar);
        if (!t) return false;
        o->tensor = t;
    }
    return true;
}

/* Одна копия инструкции с подставленными номерами повтора. Префикс и суффикс
 * не копируются: развёртка их не трогает, а делить неизменяемое безопасно. */
static bool clone_stmt(SmpAstStmt *dst, const SmpAstStmt *src, const Subst *sb,
                       uint32_t nsb, SmpArena *ar)
{
    *dst = *src;

    if (!subst_operand(&dst->source, sb, nsb, ar)) return false;
    if (dst->has_dest && !subst_operand(&dst->dest, sb, nsb, ar)) return false;

    if (dst->nstages) {
        SmpAstStage *st = (SmpAstStage *)smp_arena_push_raw(
            ar, dst->nstages * sizeof *st, 8);
        if (!st) return false;
        memcpy(st, src->stages, dst->nstages * sizeof *st);

        for (uint32_t i = 0; i < dst->nstages; i++) {
            if (!st[i].nargs) continue;
            SmpAstOperand *ag = (SmpAstOperand *)smp_arena_push_raw(
                ar, st[i].nargs * sizeof *ag, 8);
            if (!ag) return false;
            memcpy(ag, src->stages[i].args, st[i].nargs * sizeof *ag);
            for (uint32_t j = 0; j < st[i].nargs; j++)
                if (!subst_operand(&ag[j], sb, nsb, ar)) return false;
            st[i].args = ag;
        }
        dst->stages = st;
    }
    return true;
}

/* Объявляет ли инструкция тензор. Внутри развёртки это всегда ошибка: N копий
 * объявили бы одно имя N раз, и человек получил бы N сообщений о повторном
 * объявлении, все указывающие на одну и ту же строку. Лучше сказать прямо. */
static const SmpAstTensor *declares_tensor(const SmpAstStmt *s)
{
    if (s->source.kind == SMP_OPD_TENSOR && s->source.tensor &&
        s->source.tensor->has_type) return s->source.tensor;
    if (s->has_dest && s->dest.kind == SMP_OPD_TENSOR && s->dest.tensor &&
        s->dest.tensor->has_type) return s->dest.tensor;

    for (uint32_t i = 0; i < s->nstages; i++)
        for (uint32_t j = 0; j < s->stages[i].nargs; j++) {
            const SmpAstOperand *a = &s->stages[i].args[j];
            if (a->kind == SMP_OPD_TENSOR && a->tensor && a->tensor->has_type)
                return a->tensor;
        }
    return NULL;
}

/* Занято ли имя индекса обычным регистром где-то ещё в программе. Развёртка
 * подставляет литерал по имени, так что совпадение молча затенило бы чужой
 * регистр внутри этой инструкции — ровно тот случай, после которого говорят
 * «оно почему-то считает не то». */
static bool name_taken_by_reg(const SmpAstProgram *prog, SmpName ix)
{
    for (uint32_t i = 0; i < prog->nstmts; i++) {
        const SmpAstStmt *s = &prog->stmts[i];
        if (s->has_dest && s->dest.kind == SMP_OPD_REG && name_eq(s->dest.reg, ix))
            return true;
    }
    return false;
}

/* Пара #repeat / #index — одни правила для инструкции и для блока. Число
 * повторов или 0, если претензия уже высказана. */
static uint32_t check_repeat(const SmpAstProgram *prog, const SmpAstPrefix *r,
                             const SmpAstPrefix *x, SmpDiagCtx *d, uint32_t *nerr)
{
    if (!r->has_value || !r->value_is_int) {
        rep_err(d, nerr, r->span,
                "Директива #repeat требует число повторов, известное на компиляции.",
                "Запиши #repeat:N, где N — целое от 1 до 4096.");
        return 0;
    }
    if (r->ival == 0 || r->ival > SMP_MAX_REPEAT) {
        rep_err(d, nerr, r->value_span,
                smp_fmt(d, "Запрошено %llu повторов, допустимо от 1 до %u.",
                        (unsigned long long)r->ival, SMP_MAX_REPEAT),
                "Это развёртка, а не цикл: каждый повтор — свои инструкции в модуле.");
        return 0;
    }
    if (x && (!x->has_value || x->value_is_int)) {
        rep_err(d, nerr, x->span,
                "Директива #index требует имя регистра, записанное без сигила.",
                "Запиши #index:i, а в теле пиши $i.");
        return 0;
    }
    if (x && name_taken_by_reg(prog, x->sval)) {
        rep_err(d, nerr, x->value_span,
                smp_fmt(d, "Имя '$%.*s' уже принадлежит обычному регистру.",
                        (int)x->sval.len, x->sval.p),
                "Возьми индексу другое имя: внутри развёртки он затенил бы регистр.");
        return 0;
    }
    return (uint32_t)r->ival;
}

/* Префикс блока: только #repeat и #index. Всё остальное управляет
 * исполнением одной инструкции, и у блока целиком смысла не имеет —
 * какой арене принадлежит «группа»? Число повторов или 0. */
static uint32_t check_block(const SmpAstProgram *prog, const SmpAstBlock *b,
                            SmpDiagCtx *d, uint32_t *nerr)
{
    for (uint32_t i = 0; i < b->nprefix; i++) {
        const SmpAstPrefix *pf = &b->prefix[i];
        if (pf->kind == SMP_PFX_DIRECTIVE &&
            (smp_name_is(pf->name, "repeat") || smp_name_is(pf->name, "index")))
            continue;
        rep_err(d, nerr, pf->span,
                smp_fmt(d, "У блока бывают только #repeat и #index, а здесь '%c%.*s'.",
                        pf->kind == SMP_PFX_DIRECTIVE ? '#' : '^',
                        (int)pf->name.len, pf->name.p),
                "Это свойство одной инструкции: поставь его ей, внутри блока.");
        return 0;
    }

    const SmpAstPrefix *r = smp_block_directive(b, "repeat");
    if (!r) {
        rep_err(d, nerr, b->open,
                "Блок без #repeat: сколько раз его повторять, не сказано.",
                "Запиши [#repeat:N] { … }.");
        return 0;
    }
    return check_repeat(prog, r, smp_block_directive(b, "index"), d, nerr);
}

/* Своя развёртка одной инструкции: 1 — её нет, 0 — претензия высказана. */
static uint32_t own_repeat(const SmpAstProgram *prog, const SmpAstStmt *s,
                           SmpDiagCtx *d, uint32_t *nerr)
{
    const SmpAstPrefix *r = smp_stmt_directive(s, "repeat");
    const SmpAstPrefix *x = smp_stmt_directive(s, "index");

    if (!r) {
        if (!x) return 1;
        rep_err(d, nerr, x->span,
                "Директива #index задана без #repeat: нумеровать нечего.",
                "Добавь #repeat:N или убери #index.");
        return 0;
    }
    return check_repeat(prog, r, x, d, nerr);
}

/* Конец цепочки инструкций одного блока: парсер кладёт их подряд. */
static uint32_t run_end(const SmpAstProgram *prog, uint32_t i)
{
    const SmpAstBlock *b = prog->stmts[i].block;
    uint32_t j = i + 1u;
    if (!b) return j;
    while (j < prog->nstmts && prog->stmts[j].block == b) j++;
    return j;
}

/* Сколько копий даёт цепочка [i, end): одна инструкция или весь блок. 0 —
 * претензия высказана. */
static uint64_t count_run(const SmpAstProgram *prog, uint32_t i, uint32_t end,
                          SmpDiagCtx *d, uint32_t *nerr)
{
    const SmpAstBlock *b = prog->stmts[i].block;
    if (!b) {
        const SmpAstStmt *s = &prog->stmts[i];
        const uint32_t    n = own_repeat(prog, s, d, nerr);
        if (n > 1u) {
            const SmpAstTensor *decl = declares_tensor(s);
            if (decl) {
                rep_err(d, nerr, decl->span,
                        smp_fmt(d, "Тензор '%.*s' объявлен внутри развёртки с числом "
                                   "повторов %u:\nодно имя объявлялось бы заново "
                                   "на каждом из них.",
                                (int)decl->name.len, decl->name.p, n),
                        "Объяви тензор отдельной инструкцией до [#repeat].");
                return 0;
            }
        }
        return n;
    }

    const uint32_t n = check_block(prog, b, d, nerr);
    if (!n) return 0;
    const SmpAstPrefix *bx = smp_block_directive(b, "index");

    uint64_t per = 0;
    bool     ok  = true;
    for (uint32_t k = i; k < end; k++) {
        const SmpAstStmt *s = &prog->stmts[k];
        const uint32_t    m = own_repeat(prog, s, d, nerr);
        if (!m) { ok = false; continue; }

        const SmpAstPrefix *x = smp_stmt_directive(s, "index");
        if (bx && x && name_eq(bx->sval, x->sval)) {
            rep_err(d, nerr, x->value_span,
                    smp_fmt(d, "Имя '$%.*s' уже занято индексом блока из %u:%u.",
                            (int)x->sval.len, x->sval.p,
                            bx->span.line, bx->span.col),
                    "Возьми индексу инструкции другое имя: блок и она считают разное.");
            ok = false;
            continue;
        }

        const SmpAstTensor *decl = declares_tensor(s);
        if (decl && (n > 1u || m > 1u)) {
            rep_err(d, nerr, decl->span,
                    smp_fmt(d, "Тензор '%.*s' объявлен внутри %s с числом повторов "
                               "%u:\nодно имя объявлялось бы заново на каждом из них.",
                            (int)decl->name.len, decl->name.p,
                            n > 1u ? "блока" : "развёртки", n > 1u ? n : m),
                    "Объяви тензор до блока.");
            ok = false;
            continue;
        }
        per += m;
    }
    if (!ok) return 0;

    const uint64_t total = (uint64_t)n * per;
    if (total > SMP_MAX_EXPANDED) {
        const SmpAstPrefix *r = smp_block_directive(b, "repeat");
        rep_err(d, nerr, r->value_span,
                smp_fmt(d, "Блок развернулся бы в %llu инструкций, а программа "
                           "вмещает %u.", (unsigned long long)total, SMP_MAX_EXPANDED),
                "Это развёртка, а не цикл: каждый повтор — свои инструкции в модуле.");
        return 0;
    }
    return total;
}

SmpStatus smp_ast_expand(SmpAstProgram *prog, SmpArena *arena,
                         SmpDiagCtx *diag, uint32_t *n_errors)
{
    uint32_t nerr  = 0;
    uint64_t total = 0;
    bool     any   = false;

    prog->nwritten = prog->nstmts;

    /* Сколько инструкций получится, считаем заранее: массив обязан остаться
     * непрерывным, а дописывать в bump-арену посреди чужих выделений нельзя. */
    for (uint32_t i = 0; i < prog->nstmts;) {
        const uint32_t end = run_end(prog, i);
        const uint64_t n   = count_run(prog, i, end, diag, &nerr);
        const SmpAstStmt *s = &prog->stmts[i];
        if (s->block || smp_stmt_directive(s, "repeat")) any = true;
        total += n;
        i = end;
    }

    if (n_errors) *n_errors = nerr;
    if (nerr) return SMP_ERR_SYNTAX;
    if (!any) return SMP_OK;           /* развёртывать нечего — дерево как было */
    if (total > SMP_MAX_EXPANDED) {
        /* Каждая развёртка по отдельности в пределе, а вместе — нет. Указать
         * на одну виноватую нельзя, поэтому — на начало программы. */
        rep_err(diag, &nerr, prog->stmts[0].span,
                smp_fmt(diag, "После развёртки инструкций %llu, а программа вмещает %u.",
                        (unsigned long long)total, SMP_MAX_EXPANDED),
                "Это развёртка, а не цикл: каждый повтор — свои инструкции в модуле.");
        if (n_errors) *n_errors = nerr;
        return SMP_ERR_SYNTAX;
    }

    SmpAstStmt *out = (SmpAstStmt *)smp_arena_push_raw(
        arena, (size_t)total * sizeof *out, 8);
    if (!out) return SMP_ERR_OOM;

    uint32_t w = 0;
    for (uint32_t i = 0; i < prog->nstmts;) {
        const uint32_t     end = run_end(prog, i);
        const SmpAstBlock *b   = prog->stmts[i].block;
        const SmpAstPrefix *br = b ? smp_block_directive(b, "repeat") : NULL;
        const SmpAstPrefix *bx = b ? smp_block_directive(b, "index")  : NULL;
        const uint32_t     bn  = br ? (uint32_t)br->ival : 1u;

        if (!b && !smp_stmt_directive(&prog->stmts[i], "repeat")) {
            out[w++] = prog->stmts[i];
            i = end;
            continue;
        }

        /* Блок — N раз вся группа подряд; в каждом повторе блока своя
         * [#repeat] инструкции разворачивается заново. */
        for (uint32_t k = 0; k < bn; k++)
            for (uint32_t t = i; t < end; t++) {
                const SmpAstStmt   *s  = &prog->stmts[t];
                const SmpAstPrefix *r  = smp_stmt_directive(s, "repeat");
                const SmpAstPrefix *x  = smp_stmt_directive(s, "index");
                const uint32_t      m  = r ? (uint32_t)r->ival : 1u;

                for (uint32_t j = 0; j < m; j++) {
                    Subst    sb[2];
                    uint32_t nsb = 0;
                    if (bx) { sb[nsb].name = bx->sval; sb[nsb].v = k; nsb++; }
                    if (x)  { sb[nsb].name = x->sval;  sb[nsb].v = j; nsb++; }

                    if (!clone_stmt(&out[w], s, sb, nsb, arena)) return SMP_ERR_OOM;
                    out[w].from_repeat = true;
                    out[w].repeat_of   = t;
                    out[w].repeat_idx  = j;
                    out[w].repeat_n    = r ? m : 0u;
                    out[w].block_idx   = k;
                    out[w].block_n     = b ? bn : 0u;
                    w++;
                }
            }
        i = end;
    }

    prog->stmts  = out;
    prog->nstmts = w;
    return SMP_OK;
}

void smp_ast_dump(FILE *out, const SmpAstProgram *prog, bool color)
{
    Pr pr = { out, color };
    for (uint32_t i = 0; i < prog->nstmts; i++) {
        if (i) fputc('\n', out);
        pr_stmt(&pr, &prog->stmts[i], i);
    }
    fprintf(out, "\n%sинструкций: %u%s\n",
            pc(&pr, AC_DIM), prog->nstmts, pc(&pr, AC_RESET));
}
