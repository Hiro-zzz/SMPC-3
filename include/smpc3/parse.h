/* SMPC3 :: parse.h -- разбор потока токенов в AST.
 *
 * Парсер рекурсивно-нисходящий и не возвращается: скобочные группы
 * различаются по одному токену вперёд (см. lex.h), поэтому решение
 * принимается сразу и навсегда.
 *
 * Восстановление — паническое, по границе ';'. На одну инструкцию печатается
 * не больше одной претензии: каскад из десяти сообщений об одной опечатке
 * это не диагностика, а наказание не по адресу.
 */
#ifndef SMPC3_PARSE_H
#define SMPC3_PARSE_H

#include "smpc3/ast.h"
#include "smpc3/arena.h"
#include "smpc3/diag.h"
#include "smpc3/lex.h"

typedef struct SmpParser {
    const SmpSource *src;
    SmpDiagCtx      *diag;
    SmpArena        *arena;

    const SmpToken  *toks;
    uint32_t         ntoks;
    uint32_t         pos;

    uint32_t         n_errors;
    bool             panic;    /* внутри уже испорченной инструкции */

    /* Открытый блок [#repeat:N] { … }: инструкции до его '}' получают на
     * него указатель. Вложенных блоков нет, поэтому хватает одного. */
    SmpAstBlock     *block;
    uint32_t         block_stmts;  /* инструкций в нём на сейчас           */
    bool             opened;       /* последний parse_stmt открыл блок     */

    /* '{', проглоченные восстановлением или отвергнутые: их '}' закрывают
     * не блок, а то, что уже отвергнуто, и второй претензии не стоят. */
    uint32_t         skip_braces;
} SmpParser;

void smp_parse_init(SmpParser *p, SmpArena *arena, SmpDiagCtx *diag,
                    const SmpSource *src, const SmpToken *toks, uint32_t ntoks);

/* Разбирает весь поток. Даже при ошибках заполняет prog корректными
 * инструкциями — испорченные просто пропускаются. */
SmpStatus smp_parse(SmpParser *p, SmpAstProgram *prog);

#endif /* SMPC3_PARSE_H */
