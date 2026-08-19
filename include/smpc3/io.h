/* SMPC3 :: io.h -- загрузка исходников. Файл целиком читается в арену:
 * лексер и парсер держат указатели внутрь этого буфера и ничего не копируют,
 * поэтому буфер обязан жить до конца компиляции. */
#ifndef SMPC3_IO_H
#define SMPC3_IO_H

#include "smpc3/common.h"
#include "smpc3/arena.h"
#include "smpc3/diag.h"

/* Читает файл в арену и заполняет src. Текст завершается '\0' (сверх len) —
 * это упрощает отладочную печать, но код обязан опираться на len, а не на
 * терминатор: в файле могут быть нулевые байты. */
SmpStatus smp_source_load(SmpSource *src, const char *path, SmpArena *arena);

/* Оборачивает уже имеющуюся строку в памяти. */
void      smp_source_from_memory(SmpSource *src, const char *path,
                                 const char *text, size_t len);

#endif /* SMPC3_IO_H */
