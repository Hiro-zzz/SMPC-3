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

        default:
            pr_indent(pr, d);
            fprintf(pr->o, "%s%s%s <пусто>\n", pc(pr, AC_KEY), label, pc(pr, AC_RESET));
            break;
    }
}

static void pr_stmt(const Pr *pr, const SmpAstStmt *s, uint32_t idx)
{
    fprintf(pr->o, "%sинструкция #%u%s %s(строка %u)%s\n",
            pc(pr, AC_KEY), idx, pc(pr, AC_RESET),
            pc(pr, AC_DIM), s->span.line, pc(pr, AC_RESET));

    for (uint32_t i = 0; i < s->nprefix; i++) {
        const SmpAstPrefix *pf = &s->prefix[i];
        pr_indent(pr, 1);
        if (pf->kind == SMP_PFX_DIRECTIVE) {
            fprintf(pr->o, "префикс  #");
            pr_name(pr, pf->name);
            if (pf->has_value) {
                fputc(':', pr->o);
                if (pf->value_is_int) fprintf(pr->o, "%llu", (unsigned long long)pf->ival);
                else                  pr_name(pr, pf->sval);
            }
        } else {
            fprintf(pr->o, "префикс  ^");
            pr_name(pr, pf->name);
        }
        fputc('\n', pr->o);
    }

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
