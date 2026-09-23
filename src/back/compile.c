/* SMPC3 :: compile.c -- от текста до модуля. */
#include "smpc3/compile.h"
#include "smpc3/lex.h"
#include "smpc3/parse.h"

#include <string.h>

bool smp_frontend(const SmpSource *src, SmpArena *arena, SmpDiagCtx *d,
                  SmpAstProgram *prog, SmpSemaResult *res, SmpSema *sm)
{
    SmpLexer lx;
    smp_lex_init(&lx, src, d);
    SmpToken *toks = NULL;
    uint32_t  ntok = 0;
    smp_lex_all(&lx, arena, &toks, &ntok);
    if (lx.n_errors) return false;

    SmpParser p;
    smp_parse_init(&p, arena, d, src, toks, ntok);
    smp_parse(&p, prog);
    if (p.n_errors) return false;

    /* Развёртка [#repeat:N] идёт до семантики: каждая копия обязана
     * проверяться со своим индексом, иначе смысл развёртки теряется. */
    uint32_t n_exp = 0;
    smp_ast_expand(prog, arena, d, &n_exp);
    if (n_exp) return false;

    smp_sema_init(sm, arena, d, src);
    smp_sema_run(sm, prog, res);
    return sm->n_errors == 0;
}

SmpStatus smp_compile(const SmpSource *src, SmpArena *arena, SmpDiagCtx *d,
                      SmpModule *mod)
{
    /* Промежуточные структуры — в той же арене, а не на стеке: у ядра стек
     * маленький, а таблицы семантики нет. Переживают вызов они всё равно
     * только как часть арены. */
    SmpAstProgram *prog = (SmpAstProgram *)smp_arena_push_raw(arena, sizeof *prog, 16);
    SmpSemaResult *res  = (SmpSemaResult *)smp_arena_push_raw(arena, sizeof *res, 16);
    SmpSema       *sm   = (SmpSema       *)smp_arena_push_raw(arena, sizeof *sm, 16);
    if (!prog || !res || !sm) return SMP_ERR_OOM;
    memset(prog, 0, sizeof *prog);
    memset(res,  0, sizeof *res);
    memset(sm,   0, sizeof *sm);

    if (!smp_frontend(src, arena, d, prog, res, sm)) return SMP_ERR_SYNTAX;

    SmpEmitter em;
    smp_emit_init(&em, arena, d, src);
    return smp_emit(&em, prog, res, mod);
}
