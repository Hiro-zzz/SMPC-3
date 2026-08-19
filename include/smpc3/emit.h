/* SMPC3 :: emit.h -- AST + результаты семантики -> байткод, и запись .s3b. */
#ifndef SMPC3_EMIT_H
#define SMPC3_EMIT_H

#include "smpc3/arena.h"
#include "smpc3/ast.h"
#include "smpc3/bytecode.h"
#include "smpc3/diag.h"
#include "smpc3/sema.h"

typedef struct SmpEmitter {
    SmpArena        *arena;
    SmpDiagCtx      *diag;
    const SmpSource *src;
    uint32_t         n_errors;
} SmpEmitter;

void      smp_emit_init(SmpEmitter *e, SmpArena *arena, SmpDiagCtx *diag,
                        const SmpSource *src);

/* Собирает модуль. Всё содержимое модуля лежит в арене эмиттера и живёт,
 * пока живёт она. */
SmpStatus smp_emit(SmpEmitter *e, const SmpAstProgram *prog,
                   const SmpSemaResult *sema, SmpModule *out);

/* --- Контейнер .s3b -------------------------------------------------------- */

/* Сериализует модуль в файл. */
SmpStatus smp_s3b_write(const SmpModule *m, const char *path);

/* Читает файл в арену и раскладывает по указателям. Проверяет magic, версию,
 * порядок байтов, контрольную сумму и границы всех секций: доверять файлу,
 * который открыл кто угодно, нельзя. */
SmpStatus smp_s3b_read(SmpModule *m, const char *path, SmpArena *arena,
                       SmpDiagCtx *diag);

/* --- Дизассемблер ---------------------------------------------------------- */
void smp_disasm(FILE *out, const SmpModule *m, bool color);

#endif /* SMPC3_EMIT_H */
