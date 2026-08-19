/* SMPC3 :: io.c */
#include "smpc3/io.h"

#include <stdio.h>
#include <string.h>

void smp_source_from_memory(SmpSource *src, const char *path,
                            const char *text, size_t len)
{
    src->path = path;
    src->text = text;
    src->len  = len;
}

SmpStatus smp_source_load(SmpSource *src, const char *path, SmpArena *arena)
{
    memset(src, 0, sizeof(*src));
    src->path = path;

    FILE *f = fopen(path, "rb");
    if (!f) return SMP_ERR_IO;

    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return SMP_ERR_IO; }
    const long sz = ftell(f);
    if (sz < 0)                     { fclose(f); return SMP_ERR_IO; }
    rewind(f);

    char *buf = (char *)smp_arena_push_raw(arena, (size_t)sz + 1u, 8);
    if (!buf) { fclose(f); return SMP_ERR_OOM; }

    const size_t got = (size_t)sz ? fread(buf, 1, (size_t)sz, f) : 0u;
    fclose(f);
    if (got != (size_t)sz) return SMP_ERR_IO;

    buf[got] = '\0';

    /* UTF-8 BOM: редакторы Windows ставят его молча, а лексер обязан видеть
     * первым байтом код, а не метку кодировки. */
    size_t off = 0;
    if (got >= 3 && (unsigned char)buf[0] == 0xEF
                 && (unsigned char)buf[1] == 0xBB
                 && (unsigned char)buf[2] == 0xBF) {
        off = 3;
    }

    src->text = buf + off;
    src->len  = got - off;
    return SMP_OK;
}
