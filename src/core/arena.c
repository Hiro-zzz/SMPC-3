/* SMPC3 :: arena.c */
#include "smpc3/arena.h"
#include "smpc3/plat.h"
#include <string.h>
#include <stdio.h>

/* В debug-сборке свежая и откатанная память забивается ядовитым паттерном:
 * чтение неинициализированного тензора должно ломаться громко, а не «почти
 * работать». */
#ifndef NDEBUG
#  define SMP_ARENA_POISON 1
#else
#  define SMP_ARENA_POISON 0
#endif
#define SMP_POISON_BYTE 0xCD

SmpStatus smp_arena_init(SmpArena *a, size_t cap, uint32_t id, const char *label)
{
    if (!a || cap == 0) return SMP_ERR_INTERNAL;

    memset(a, 0, sizeof(*a));
    cap = SMP_ALIGN_UP(cap, SMP_ARENA_ALIGN);

    uint8_t *mem = (uint8_t *)smp_plat_pages(cap);
    if (!mem) return SMP_ERR_OOM;

    /* Слой платформы обещает ≥4096, но проверяем инвариант явно:
     * вся SIMD-часть рантайма опирается на него без дальнейших сомнений. */
    if (!SMP_IS_ALIGNED(mem, SMP_ARENA_ALIGN)) {
        smp_plat_pages_free(mem, cap);
        return SMP_ERR_ALIGN;
    }

#if SMP_ARENA_POISON
    memset(mem, SMP_POISON_BYTE, cap);
#endif

    a->base       = mem;
    a->cap        = cap;
    a->used       = 0;
    a->peak       = 0;
    a->id         = id;
    a->n_allocs   = 0;
    a->label      = label ? label : "unnamed";
    a->owns_pages = true;
    return SMP_OK;
}

SmpStatus smp_arena_wrap(SmpArena *a, void *mem, size_t cap, uint32_t id, const char *label)
{
    if (!a || !mem || cap == 0)              return SMP_ERR_INTERNAL;
    if (!SMP_IS_ALIGNED(mem, SMP_ARENA_ALIGN)) return SMP_ERR_ALIGN;

    memset(a, 0, sizeof(*a));
    a->base       = (uint8_t *)mem;
    a->cap        = SMP_ALIGN_DOWN(cap, SMP_ARENA_ALIGN);
    a->id         = id;
    a->label      = label ? label : "wrapped";
    a->owns_pages = false;
    return SMP_OK;
}

void smp_arena_release(SmpArena *a)
{
    if (!a || !a->base) return;
    if (a->owns_pages) smp_plat_pages_free(a->base, a->cap);
    memset(a, 0, sizeof(*a));
}

void *smp_arena_push(SmpArena *a, size_t bytes, size_t align)
{
    /* Проверяем то, что запросил вызывающий, ДО подъёма до минимума. Иначе
     * align=48 молча превратился бы в 64, и арена «починила» бы чужой баг —
     * ровно то поведение, которого в этом языке быть не должно. */
    if (SMP_UNLIKELY(!SMP_IS_POW2(align))) return NULL;
    if (align < SMP_ARENA_ALIGN) align = SMP_ARENA_ALIGN;

    size_t cursor = SMP_ALIGN_UP(a->used, align);

    /* Переполнение size_t на 64-битной машине маловероятно, но арена — это
     * последний рубеж перед сырыми указателями, здесь дешевле проверить. */
    if (SMP_UNLIKELY(cursor > a->cap || bytes > a->cap - cursor)) return NULL;

    uint8_t *p = a->base + cursor;
    a->used    = cursor + bytes;
    if (a->used > a->peak) a->peak = a->used;
    a->n_allocs++;
    return p;
}

void *smp_arena_push_raw(SmpArena *a, size_t bytes, size_t align)
{
    if (SMP_UNLIKELY(!SMP_IS_POW2(align))) return NULL;

    size_t cursor = SMP_ALIGN_UP(a->used, align);
    if (SMP_UNLIKELY(cursor > a->cap || bytes > a->cap - cursor)) return NULL;

    uint8_t *p = a->base + cursor;
    a->used    = cursor + bytes;
    if (a->used > a->peak) a->peak = a->used;
    a->n_allocs++;
    return p;
}

void *smp_arena_push_zero(SmpArena *a, size_t bytes, size_t align)
{
    void *p = smp_arena_push(a, bytes, align);
    if (p) memset(p, 0, bytes);
    return p;
}

uint64_t smp_arena_push_off(SmpArena *a, size_t bytes, size_t align)
{
    void *p = smp_arena_push(a, bytes, align);
    if (!p) return SMP_ARENA_NIL;
    return (uint64_t)((uint8_t *)p - a->base);
}

void smp_arena_rewind(SmpArena *a, SmpArenaMark m)
{
    if (m > a->used) return;          /* откат «вперёд» — это не откат */
#if SMP_ARENA_POISON
    memset(a->base + m, SMP_POISON_BYTE, a->used - m);
#endif
    a->used = m;
}

void smp_arena_reset(SmpArena *a)
{
    smp_arena_rewind(a, 0);
    a->n_allocs = 0;
}

bool smp_arena_owns(const SmpArena *a, const void *p)
{
    const uint8_t *q = (const uint8_t *)p;
    return q >= a->base && q < a->base + a->cap;
}

static void smp__fmt_bytes(size_t n, char *buf, size_t cap)
{
    static const char *unit[] = { "B", "KiB", "MiB", "GiB", "TiB" };
    double v = (double)n;
    size_t u = 0;
    while (v >= 1024.0 && u + 1 < SMP_ARRLEN(unit)) { v /= 1024.0; u++; }
    if (u == 0) snprintf(buf, cap, "%zu %s", n, unit[0]);
    else        snprintf(buf, cap, "%.1f %s", v, unit[u]);
}

char *smp_arena_report(const SmpArena *a, char *buf, size_t cap)
{
    char u[32], c[32], p[32];
    smp__fmt_bytes(a->used, u, sizeof u);
    smp__fmt_bytes(a->cap,  c, sizeof c);
    smp__fmt_bytes(a->peak, p, sizeof p);
    snprintf(buf, cap, "arena#%u '%s' %s / %s (peak %s, %u allocs)",
             a->id, a->label, u, c, p, a->n_allocs);
    return buf;
}
