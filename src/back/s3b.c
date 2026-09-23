/* SMPC3 :: s3b.c -- сериализация модуля и его чтение обратно.
 *
 * Читатель считает файл враждебным: смещения секций, размеры и контрольная
 * сумма проверяются до того, как по ним хоть раз пройдёт указатель. Файл на
 * диске мог править кто угодно, а рантайм тут без страховки. */
#include "smpc3/emit.h"
#include "smpc3/cpu.h"
#include "smpc3/plat.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* ========================================================================== */
/*  Опкоды                                                                    */
/* ========================================================================== */

#define SMP_BC_ROW(id, mn, fmt, desc) { mn, fmt, desc },
static const SmpOpcodeDef g_bc[SMP_BC__COUNT] = { SMP_BC_OPS(SMP_BC_ROW) };
#undef SMP_BC_ROW

static const SmpOpcodeDef g_bc_bogus = { "<?>", SMP_FMT_NONE, "неизвестный опкод" };

const SmpOpcodeDef *smp_opcode_def(SmpOpcode op)
{
    if ((unsigned)op >= (unsigned)SMP_BC__COUNT) return &g_bc_bogus;
    return &g_bc[op];
}

const char *smp_module_str(const SmpModule *m, uint32_t off)
{
    if (!m->strs || off >= m->strs_size) return "<нет>";
    return m->strs + off;
}

SmpDbgLine smp_module_dbg(const SmpModule *m, uint32_t instr)
{
    SmpDbgLine z = { 0, 0, 0 };
    if (!m->dbg) return z;
    for (uint32_t i = 0; i < m->n_dbg; i++)
        if (m->dbg[i].instr == instr) return m->dbg[i];
    return z;
}

/* ========================================================================== */
/*  Контрольная сумма                                                         */
/* ========================================================================== */

static uint32_t fnv1a(const void *p, size_t n)
{
    const uint8_t *b = (const uint8_t *)p;
    uint32_t h = 0x811C9DC5u;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= 0x01000193u;
    }
    return h;
}

/* ========================================================================== */
/*  Запись                                                                    */
/* ========================================================================== */

#define S3B_N_SECTIONS 7u

typedef struct { const char *tag; const void *data; uint32_t size, count; } SecSrc;

SmpStatus smp_s3b_write(const SmpModule *m, const char *path)
{
    SecSrc sec[S3B_N_SECTIONS] = {
        { "AREN", m->arena_bytes, (uint32_t)(m->n_arenas * sizeof(uint64_t)),   m->n_arenas  },
        { "TENS", m->tens,        (uint32_t)(m->n_tens   * sizeof(SmpTensor)),  m->n_tens    },
        { "CONS", m->consts,      (uint32_t)(m->n_consts * sizeof(SmpConst)),   m->n_consts  },
        { "CODE", m->code,        (uint32_t)(m->n_code   * sizeof(SmpInstr)),   m->n_code    },
        { "STRS", m->strs,        m->strs_size,                                 m->strs_size },
        { "DBGL", m->dbg,         (uint32_t)(m->n_dbg    * sizeof(SmpDbgLine)), m->n_dbg     },
        { "RNAM", m->reg_names,   (uint32_t)(m->n_reg_names * sizeof(uint32_t)), m->n_reg_names }
    };

    const uint32_t hdr_size = (uint32_t)sizeof(SmpS3bHeader);
    const uint32_t tab_size = (uint32_t)(S3B_N_SECTIONS * sizeof(SmpS3bSection));

    SmpS3bSection tab[S3B_N_SECTIONS];
    uint32_t off = hdr_size + tab_size;

    for (uint32_t i = 0; i < S3B_N_SECTIONS; i++) {
        off = (uint32_t)SMP_ALIGN_UP(off, 8u);
        memcpy(tab[i].tag, sec[i].tag, 4);
        tab[i].offset = off;
        tab[i].size   = sec[i].size;
        tab[i].count  = sec[i].count;
        off += sec[i].size;
    }
    const uint32_t total = off;

    /* Собираем образ целиком: писать по кускам и потом досчитывать сумму —
     * лишний повод разъехаться. */
    uint8_t *img = (uint8_t *)smp_plat_pages(total);
    if (!img) return SMP_ERR_OOM;

    SmpS3bHeader *h = (SmpS3bHeader *)img;
    h->magic[0]  = SMP_S3B_MAGIC0; h->magic[1] = SMP_S3B_MAGIC1;
    h->magic[2]  = SMP_S3B_MAGIC2; h->magic[3] = SMP_S3B_MAGIC3;
    h->ver_major = SMP_S3B_VER_MAJOR;
    h->ver_minor = SMP_S3B_VER_MINOR;
    h->flags     = SMP_S3B_LITTLE_ENDIAN | (m->n_dbg ? SMP_S3B_HAS_DEBUG : 0u);
    h->n_sections= S3B_N_SECTIONS;
    h->total_size= total;
    h->checksum  = 0;

    memcpy(img + hdr_size, tab, tab_size);
    for (uint32_t i = 0; i < S3B_N_SECTIONS; i++)
        if (sec[i].size && sec[i].data)
            memcpy(img + tab[i].offset, sec[i].data, sec[i].size);

    h->checksum = fnv1a(img + hdr_size, total - hdr_size);

    FILE *f = fopen(path, "wb");
    if (!f) { smp_plat_pages_free(img, total); return SMP_ERR_IO; }
    const size_t wrote = fwrite(img, 1, total, f);
    fclose(f);
    smp_plat_pages_free(img, total);

    return wrote == total ? SMP_OK : SMP_ERR_IO;
}

/* ========================================================================== */
/*  Чтение                                                                    */
/* ========================================================================== */

static void s3b_err(SmpDiagCtx *d, const char *details)
{
    if (!d) return;
    SmpDiagMsg m;
    memset(&m, 0, sizeof m);
    m.code    = SMP_E0604;
    m.details = details;
    smp_diag_emit(d, &m);
}

SmpStatus smp_s3b_read(SmpModule *m, const char *path, SmpArena *arena,
                       SmpDiagCtx *diag)
{
    memset(m, 0, sizeof *m);

    FILE *f = fopen(path, "rb");
    if (!f) return SMP_ERR_IO;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return SMP_ERR_IO; }
    const long sz = ftell(f);
    rewind(f);
    if (sz < (long)sizeof(SmpS3bHeader)) {
        fclose(f);
        s3b_err(diag, "Файл короче заголовка: это не .s3b.");
        return SMP_ERR_IO;
    }

    uint8_t *img = (uint8_t *)smp_arena_push_raw(arena, (size_t)sz, 64);
    if (!img) { fclose(f); return SMP_ERR_OOM; }
    const size_t got = fread(img, 1, (size_t)sz, f);
    fclose(f);
    if (got != (size_t)sz) return SMP_ERR_IO;

    const SmpS3bHeader *h = (const SmpS3bHeader *)img;

    if (h->magic[0] != SMP_S3B_MAGIC0 || h->magic[1] != SMP_S3B_MAGIC1 ||
        h->magic[2] != SMP_S3B_MAGIC2 || h->magic[3] != SMP_S3B_MAGIC3) {
        static char b[160];
        snprintf(b, sizeof b, "Ожидалась сигнатура 53 4D 50 33 ('SMP3'), "
                              "а в файле %02X %02X %02X %02X.",
                 h->magic[0], h->magic[1], h->magic[2], h->magic[3]);
        s3b_err(diag, b);
        return SMP_ERR_IO;
    }
    if (h->ver_major != SMP_S3B_VER_MAJOR || h->ver_minor != SMP_S3B_VER_MINOR) {
        static char b[160];
        snprintf(b, sizeof b, "Файл версии %u.%u, а рантайм понимает %u.%u.",
                 h->ver_major, h->ver_minor, SMP_S3B_VER_MAJOR, SMP_S3B_VER_MINOR);
        s3b_err(diag, b);
        return SMP_ERR_IO;
    }
    if (!(h->flags & SMP_S3B_LITTLE_ENDIAN)) {
        s3b_err(diag, "Файл собран под порядок байтов big-endian.");
        return SMP_ERR_IO;
    }
    if (h->total_size != (uint32_t)sz) {
        static char b[160];
        snprintf(b, sizeof b, "В заголовке заявлено %u байт, а файл занимает %ld.",
                 h->total_size, sz);
        s3b_err(diag, b);
        return SMP_ERR_IO;
    }

    const uint32_t hdr_size = (uint32_t)sizeof(SmpS3bHeader);
    {
        const uint32_t want = h->checksum;
        const uint32_t have = fnv1a(img + hdr_size, (size_t)sz - hdr_size);
        if (want != have) {
            static char b[160];
            snprintf(b, sizeof b, "Контрольная сумма не сошлась: в файле %08X, "
                                  "посчитано %08X. Содержимое повреждено.", want, have);
            s3b_err(diag, b);
            return SMP_ERR_IO;
        }
    }

    if (h->n_sections == 0 || h->n_sections > 64u ||
        (uint64_t)hdr_size + (uint64_t)h->n_sections * sizeof(SmpS3bSection) > (uint64_t)sz) {
        s3b_err(diag, "Таблица секций не помещается в файл.");
        return SMP_ERR_IO;
    }

    const SmpS3bSection *tab = (const SmpS3bSection *)(img + hdr_size);

    for (uint32_t i = 0; i < h->n_sections; i++) {
        const SmpS3bSection *s = &tab[i];

        /* Каждая секция обязана лежать внутри файла целиком. Без этой проверки
         * подправленное смещение превращается в чтение чужой памяти. */
        if ((uint64_t)s->offset + (uint64_t)s->size > (uint64_t)sz) {
            static char b[200];
            snprintf(b, sizeof b, "Секция '%.4s' заявлена на смещении %u размером %u, "
                                  "а файл всего %ld байт.", (const char *)s->tag,
                     s->offset, s->size, sz);
            s3b_err(diag, b);
            return SMP_ERR_IO;
        }

        const void *p = img + s->offset;

        if (memcmp(s->tag, "AREN", 4) == 0) {
            if (s->count * sizeof(uint64_t) != s->size) goto bad_count;
            m->arena_bytes = (const uint64_t *)p; m->n_arenas = s->count;
        } else if (memcmp(s->tag, "TENS", 4) == 0) {
            if (s->count * sizeof(SmpTensor) != s->size) goto bad_count;
            m->tens = (const SmpTensor *)p; m->n_tens = s->count;
        } else if (memcmp(s->tag, "CONS", 4) == 0) {
            if (s->count * sizeof(SmpConst) != s->size) goto bad_count;
            m->consts = (const SmpConst *)p; m->n_consts = s->count;
        } else if (memcmp(s->tag, "CODE", 4) == 0) {
            if (s->count * sizeof(SmpInstr) != s->size) goto bad_count;
            m->code = (const SmpInstr *)p; m->n_code = s->count;
        } else if (memcmp(s->tag, "STRS", 4) == 0) {
            m->strs = (const char *)p; m->strs_size = s->size;
        } else if (memcmp(s->tag, "DBGL", 4) == 0) {
            if (s->count * sizeof(SmpDbgLine) != s->size) goto bad_count;
            m->dbg = (const SmpDbgLine *)p; m->n_dbg = s->count;
        } else if (memcmp(s->tag, "RNAM", 4) == 0) {
            if (s->count * sizeof(uint32_t) != s->size) goto bad_count;
            m->reg_names = (const uint32_t *)p; m->n_reg_names = s->count;
        }
        continue;

    bad_count:
        {
            static char b[200];
            snprintf(b, sizeof b, "Секция '%.4s': %u элементов не дают %u байт.",
                     (const char *)s->tag, s->count, s->size);
            s3b_err(diag, b);
            return SMP_ERR_IO;
        }
    }

    if (!m->code || m->n_code == 0) {
        s3b_err(diag, "В модуле нет секции CODE.");
        return SMP_ERR_IO;
    }

    /* Опкоды и ссылки в пулы проверяются здесь, один раз, а не на горячем
     * пути диспетчеризации. */
    uint32_t max_reg = 0;
    for (uint32_t i = 0; i < m->n_code; i++) {
        const SmpInstr *in = &m->code[i];
        if (in->op >= SMP_BC__COUNT) {
            static char b[160];
            snprintf(b, sizeof b, "Инструкция #%u содержит опкод 0x%02X, "
                                  "а их всего %u.", i, in->op, SMP_BC__COUNT);
            s3b_err(diag, b);
            return SMP_ERR_IO;
        }
        const SmpOpFmt fmt = g_bc[in->op].fmt;
        const bool uses_tens = (fmt == SMP_FMT_D_T || fmt == SMP_FMT_T_A);
        const bool uses_const = (fmt == SMP_FMT_D_K || fmt == SMP_FMT_D_A_K ||
                                 fmt == SMP_FMT_D_A_B_K);
        if (uses_tens && in->k >= m->n_tens) {
            static char b[160];
            snprintf(b, sizeof b, "Инструкция #%u ссылается на дескриптор %u, "
                                  "а их %u.", i, in->k, m->n_tens);
            s3b_err(diag, b);
            return SMP_ERR_IO;
        }
        if (uses_const && in->k >= m->n_consts) {
            static char b[160];
            snprintf(b, sizeof b, "Инструкция #%u ссылается на константу %u, "
                                  "а их %u.", i, in->k, m->n_consts);
            s3b_err(diag, b);
            return SMP_ERR_IO;
        }
        if (in->d > max_reg) max_reg = in->d;
        if (in->a > max_reg) max_reg = in->a;
        if (in->b > max_reg) max_reg = in->b;
    }
    m->n_regs = max_reg + 1u;

    if (m->code[m->n_code - 1u].op != SMP_BC_HALT) {
        s3b_err(diag, "Последняя инструкция модуля не halt: исполнение уедет за код.");
        return SMP_ERR_IO;
    }

    /* Строковая таблица обязана быть \0-терминирована, иначе печать имени
     * уедет за секцию. */
    if (m->strs && m->strs_size && m->strs[m->strs_size - 1u] != '\0') {
        s3b_err(diag, "Строковая таблица не заканчивается нулевым байтом.");
        return SMP_ERR_IO;
    }

    return SMP_OK;
}

/* ========================================================================== */
/*  Дизассемблер                                                              */
/* ========================================================================== */

#define D_RESET "\x1b[0m"
#define D_DIM   "\x1b[2m"
#define D_OP    "\x1b[96m"
#define D_REG   "\x1b[93m"
#define D_NUM   "\x1b[95m"

static const char *dc(bool color, const char *s) { return color ? s : ""; }

static void print_tensor(FILE *o, const SmpModule *m, uint32_t idx, bool color)
{
    if (idx >= m->n_tens) { fprintf(o, "T%u <вне таблицы>", idx); return; }
    const SmpTensor *t = &m->tens[idx];

    fprintf(o, "%sT%u%s ", dc(color, D_NUM), idx, dc(color, D_RESET));
    fprintf(o, "%s<%s", smp_module_str(m, t->name_id), smp_dtype_name((SmpDType)t->dtype));
    for (uint32_t i = 0; i < t->rank; i++)
        fprintf(o, "%s%u", i ? "," : ":", (unsigned)t->shape[i]);
    fprintf(o, "> a%u:0x%X", smp_tf_arena(t->flags), t->off);
}

void smp_disasm(FILE *out, const SmpModule *m, bool color)
{
    fprintf(out, "%s; модуль: %u инструкций, %u дескрипторов, %u констант, "
                 "%u регистров%s\n",
            dc(color, D_DIM), m->n_code, m->n_tens, m->n_consts, m->n_regs,
            dc(color, D_RESET));

    for (uint32_t i = 0; i < m->n_arenas; i++)
        fprintf(out, "%s; арена #%u: %llu байт%s\n", dc(color, D_DIM), i,
                (unsigned long long)m->arena_bytes[i], dc(color, D_RESET));
    fputc('\n', out);

    uint32_t last_line = 0;

    for (uint32_t i = 0; i < m->n_code; i++) {
        const SmpInstr     *in  = &m->code[i];
        const SmpOpcodeDef *def = smp_opcode_def((SmpOpcode)in->op);
        const SmpDbgLine    dl  = smp_module_dbg(m, i);

        if (dl.line && dl.line != last_line) {
            fprintf(out, "%s; строка %u%s\n", dc(color, D_DIM), dl.line, dc(color, D_RESET));
            last_line = dl.line;
        }

        fprintf(out, "%s%04u%s  ", dc(color, D_DIM), i, dc(color, D_RESET));

        /* Флаги: ширина вектора и режимы этой инструкции. */
        char fl[32];
        size_t fn = 0;
        fn += (size_t)snprintf(fl + fn, sizeof(fl) - fn, "%s",
                               smp_vec_name(smp_vec_bits(in->flags & SMP_IF_VEC_MASK)));
        if (in->flags & SMP_IF_FTZ)     fn += (size_t)snprintf(fl + fn, sizeof(fl) - fn, ",ftz");
        if (in->flags & SMP_IF_STRICT)  fn += (size_t)snprintf(fl + fn, sizeof(fl) - fn, ",str");
        if (in->flags & SMP_IF_NOALIAS) fn += (size_t)snprintf(fl + fn, sizeof(fl) - fn, ",na");
        if (in->flags & SMP_IF_RAW)     fn += (size_t)snprintf(fl + fn, sizeof(fl) - fn, ",raw");
        if (in->flags & SMP_IF_FUSE)    snprintf(fl + fn, sizeof(fl) - fn, ",fuse");
        /* Ширина под самую длинную реальную комбинацию: v256,ftz,raw,fuse.
         * Колонка на то и колонка, чтобы не разъезжаться от лишнего флага. */
        fprintf(out, "%s[%-18s]%s ", dc(color, D_DIM), fl, dc(color, D_RESET));

        fprintf(out, "%s%-8s%s ", dc(color, D_OP), def->mnemonic, dc(color, D_RESET));

        switch (def->fmt) {
            case SMP_FMT_NONE: break;
            case SMP_FMT_D:
                fprintf(out, "%sr%u%s", dc(color, D_REG), in->d, dc(color, D_RESET));
                break;
            case SMP_FMT_D_A:
                fprintf(out, "%sr%u, r%u%s", dc(color, D_REG), in->d, in->a, dc(color, D_RESET));
                break;
            case SMP_FMT_D_A_B:
                fprintf(out, "%sr%u, r%u, r%u%s", dc(color, D_REG), in->d, in->a, in->b,
                        dc(color, D_RESET));
                break;
            case SMP_FMT_D_T:
                fprintf(out, "%sr%u%s, ", dc(color, D_REG), in->d, dc(color, D_RESET));
                print_tensor(out, m, in->k, color);
                break;
            case SMP_FMT_T_A:
                print_tensor(out, m, in->k, color);
                fprintf(out, ", %sr%u%s", dc(color, D_REG), in->a, dc(color, D_RESET));
                break;
            case SMP_FMT_D_K:
                fprintf(out, "%sr%u%s, K%u", dc(color, D_REG), in->d, dc(color, D_RESET), in->k);
                if (in->k < m->n_consts)
                    fprintf(out, " %s(%s %g)%s", dc(color, D_DIM),
                            smp_dtype_name((SmpDType)in->aux),
                            smp_const_as_double(m->consts[in->k], in->aux),
                            dc(color, D_RESET));
                break;
            case SMP_FMT_D_A_K:
                fprintf(out, "%sr%u, r%u%s, K%u", dc(color, D_REG), in->d, in->a,
                        dc(color, D_RESET), in->k);
                if (in->k < m->n_consts)
                    fprintf(out, " %s(%s %g)%s", dc(color, D_DIM),
                            smp_dtype_name((SmpDType)in->aux),
                            smp_const_as_double(m->consts[in->k], in->aux),
                            dc(color, D_RESET));
                break;
            case SMP_FMT_D_A_X:
                fprintf(out, "%sr%u, r%u%s  %s(%s)%s", dc(color, D_REG), in->d,
                        in->a, dc(color, D_RESET), dc(color, D_DIM),
                        smp_emit_fmt_name(in->aux), dc(color, D_RESET));
                break;
            case SMP_FMT_D_A_B_K:
                fprintf(out, "%sr%u, r%u, r%u%s, K%u", dc(color, D_REG), in->d, in->a,
                        in->b, dc(color, D_RESET), in->k);
                if (in->k < m->n_consts)
                    fprintf(out, " %s(шаг %llu Б)%s", dc(color, D_DIM),
                            (unsigned long long)m->consts[in->k].u, dc(color, D_RESET));
                break;
        }
        fputc('\n', out);
    }
}
