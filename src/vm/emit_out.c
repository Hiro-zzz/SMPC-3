/* SMPC3 :: emit_out.c -- вывод тензора в поток.
 *
 * Философия не нарушена ни в одном месте:
 *   - ни одного malloc: буфер фиксированный, на стеке;
 *   - формат выбран на компиляции и лежит в поле aux, рантайм не разбирает
 *     никаких строк формата;
 *   - обход идёт по шагам, так что печатать можно и срез со шагом;
 *   - недопустимая кодовая точка не заменяется вопросиком, а роняет процесс.
 */
#include "smpc3/vm.h"

#include <string.h>
#include <stdio.h>
#include <inttypes.h>

const char *smp_emit_fmt_name(uint8_t f)
{
    switch (f) {
        case SMP_EMIT_TEXT: return "text";
        case SMP_EMIT_LINE: return "line";
        case SMP_EMIT_DEC:  return "dec";
        case SMP_EMIT_HEX:  return "hex";
        case SMP_EMIT_BITS: return "bits";
        case SMP_EMIT_NUM:  return "num";
        default:            return "?";
    }
}

/* ========================================================================== */
/*  Буфер                                                                     */
/* ========================================================================== */

/* Пишем не по одному байту в FILE*, а пачками: на матрице 1024x1024 разница
 * между putc и fwrite измеряется секундами. */
#define OUTBUF 4096u

typedef struct {
    FILE   *f;
    char    b[OUTBUF];
    size_t  n;
} Out;

static void out_flush(Out *o)
{
    if (o->n) { fwrite(o->b, 1, o->n, o->f); o->n = 0; }
}

static void out_byte(Out *o, char c)
{
    if (o->n == OUTBUF) out_flush(o);
    o->b[o->n++] = c;
}

static void out_str(Out *o, const char *s, size_t n)
{
    for (size_t i = 0; i < n; i++) out_byte(o, s[i]);
}

/* ========================================================================== */
/*  Кодирование UTF-8                                                         */
/* ========================================================================== */

/* Возвращает число байт или 0, если точка недопустима. */
static uint32_t utf8_put(Out *o, uint32_t cp)
{
    if (cp > 0x10FFFFu)                    return 0;
    if (cp >= 0xD800u && cp <= 0xDFFFu)    return 0;   /* суррогаты */

    if (cp < 0x80u) {
        out_byte(o, (char)cp);
        return 1;
    }
    if (cp < 0x800u) {
        out_byte(o, (char)(0xC0u | (cp >> 6)));
        out_byte(o, (char)(0x80u | (cp & 0x3Fu)));
        return 2;
    }
    if (cp < 0x10000u) {
        out_byte(o, (char)(0xE0u | (cp >> 12)));
        out_byte(o, (char)(0x80u | ((cp >> 6) & 0x3Fu)));
        out_byte(o, (char)(0x80u | (cp & 0x3Fu)));
        return 3;
    }
    out_byte(o, (char)(0xF0u | (cp >> 18)));
    out_byte(o, (char)(0x80u | ((cp >> 12) & 0x3Fu)));
    out_byte(o, (char)(0x80u | ((cp >> 6) & 0x3Fu)));
    out_byte(o, (char)(0x80u | (cp & 0x3Fu)));
    return 4;
}

/* ========================================================================== */
/*  Обход тензора                                                             */
/* ========================================================================== */

static double elem_get(const void *p, SmpDType dt, size_t i)
{
    switch (dt) {
        case SMP_DT_F32: return (double)((const float    *)p)[i];
        case SMP_DT_F64: return           ((const double  *)p)[i];
        case SMP_DT_I32: return (double)((const int32_t  *)p)[i];
        case SMP_DT_U64: return (double)((const uint64_t *)p)[i];
        default:         return 0.0;
    }
}

static size_t elem_off(const SmpTensor *t, const uint32_t *idx)
{
    size_t off = 0;
    for (uint32_t i = 0; i < t->rank; i++)
        off += (size_t)idx[i] * (size_t)t->stride[i];
    return off;
}

static bool idx_next(const SmpTensor *t, uint32_t *idx)
{
    for (uint32_t i = t->rank; i-- > 0; ) {
        if (++idx[i] < t->shape[i]) return true;
        idx[i] = 0;
    }
    return false;
}

/* ========================================================================== */
/*  Печать одного элемента                                                    */
/* ========================================================================== */

static void put_bits(Out *o, uint64_t v)
{
    /* Ведущие нули не печатаем, но ноль печатаем как «0», а не как пустоту. */
    if (v == 0) { out_byte(o, '0'); return; }
    char tmp[64];
    int  n = 0;
    while (v) { tmp[n++] = (char)('0' + (int)(v & 1u)); v >>= 1; }
    while (n--) out_byte(o, tmp[n]);
}

/* false — вывод прерван диагностикой. */
static bool put_elem(Out *o, uint8_t fmt, double v, uint32_t *out_cp)
{
    char tmp[64];
    int  n;

    switch (fmt) {
        case SMP_EMIT_TEXT:
        case SMP_EMIT_LINE: {
            const uint32_t cp = (uint32_t)(v < 0.0 ? 0.0 : v);
            if (out_cp) *out_cp = cp;
            return utf8_put(o, cp) != 0;
        }
        case SMP_EMIT_DEC:
            n = snprintf(tmp, sizeof tmp, "%lld", (long long)v);
            out_str(o, tmp, (size_t)(n > 0 ? n : 0));
            return true;

        case SMP_EMIT_HEX:
            n = snprintf(tmp, sizeof tmp, "0x%llX", (unsigned long long)(v < 0.0 ? 0.0 : v));
            out_str(o, tmp, (size_t)(n > 0 ? n : 0));
            return true;

        case SMP_EMIT_BITS:
            put_bits(o, (uint64_t)(v < 0.0 ? 0.0 : v));
            return true;

        default:
            n = snprintf(tmp, sizeof tmp, "%g", v);
            out_str(o, tmp, (size_t)(n > 0 ? n : 0));
            return true;
    }
}

/* ========================================================================== */

SmpStatus smp_vm_emit(FILE *dst, const SmpBuf *src, uint8_t fmt,
                      uint64_t *n_written, uint32_t *bad_cp)
{
    Out o;
    o.f = dst ? dst : stdout;
    o.n = 0;

    const SmpTensor *t  = src->t;
    const SmpDType   dt = (SmpDType)t->dtype;
    const bool separated = (fmt != SMP_EMIT_TEXT && fmt != SMP_EMIT_LINE);

    uint64_t count = 0;
    uint32_t idx[SMP_MAX_RANK] = { 0, 0, 0, 0 };
    bool     ok = true;

    if (t->nelem) {
        do {
            if (separated && count) out_byte(&o, ' ');
            if (!put_elem(&o, fmt, elem_get(src->p, dt, elem_off(t, idx)),
                          bad_cp)) {
                ok = false;
                break;
            }
            count++;
        } while (t->rank && idx_next(t, idx));
    }

    if (ok && (separated || fmt == SMP_EMIT_LINE)) out_byte(&o, '\n');

    out_flush(&o);
    fflush(o.f);

    if (n_written) *n_written = count;
    return ok ? SMP_OK : SMP_ERR_TYPE;
}
