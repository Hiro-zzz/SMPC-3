/* SMPC3 :: compile.h -- от текста до модуля одним вызовом.
 *
 * Шесть фаз — лексер, парсер, развёртка [#repeat], семантика, эмиттер — раньше
 * склеивал только CLI. Теперь это библиотека: ей пользуются и smpc3.exe, и
 * ядро NablaOS, где программу приносит человек или модель, а не файл.
 *
 * Источник уже загружен, диагностический контекст уже на него настроен
 * (smp_diag_init). Всё, что создаётся по дороге, лежит в арене и живёт, пока
 * живёт она. Диагностика печатается по ходу, как обычно. */
#ifndef SMPC3_COMPILE_H
#define SMPC3_COMPILE_H

#include "smpc3/arena.h"
#include "smpc3/ast.h"
#include "smpc3/diag.h"
#include "smpc3/emit.h"
#include "smpc3/sema.h"

/* Лексер, парсер, развёртка, семантика. Останавливается на первой фазе, где
 * нашлись ошибки: следующая фаза на битом входе нашла бы только эхо первой.
 * true — ошибок нет, prog и res готовы к эмиттеру. */
bool      smp_frontend(const SmpSource *src, SmpArena *arena, SmpDiagCtx *d,
                       SmpAstProgram *prog, SmpSemaResult *res, SmpSema *sm);

/* Весь путь до модуля.
 *   SMP_OK          модуль собран;
 *   SMP_ERR_SYNTAX  фронтенд нашёл ошибки, и они уже напечатаны;
 *   другое          отказал эмиттер или не хватило арены. */
SmpStatus smp_compile(const SmpSource *src, SmpArena *arena, SmpDiagCtx *d,
                      SmpModule *mod);

#endif /* SMPC3_COMPILE_H */
