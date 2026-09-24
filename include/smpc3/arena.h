/* SMPC3 :: arena.h -- Flat Arena. Никаких malloc в рантайме.
 *
 * Вся память программы выделяется ОДИН раз на старте. Дальше только bump.
 * Освобождение — только откат к отметке (marker) или полный reset.
 * Любой указатель, выданный ареной, выровнен минимум на 64 байта. */
#ifndef SMPC3_ARENA_H
#define SMPC3_ARENA_H

#include "smpc3/common.h"

typedef struct SmpArena {
    uint8_t    *base;      /* начало блока, всегда выровнено на 64          */
    size_t      cap;       /* полная ёмкость, байт                          */
    size_t      used;      /* текущая отметка                               */
    size_t      peak;      /* максимум used за всё время (для отчёта)       */
    uint32_t    id;        /* номер арены: [#arena:N]                       */
    uint32_t    n_allocs;  /* счётчик выдач                                 */
    const char *label;
    bool        owns_pages;/* память получена от ОС (иначе — заимствована)  */
} SmpArena;

typedef size_t SmpArenaMark;

/* Резервирует cap байт у ОС, выравнивает базу на 64. cap округляется вверх.
 * Единственная точка в системе, где вообще происходит выделение памяти. */
SmpStatus     smp_arena_init(SmpArena *a, size_t cap, uint32_t id, const char *label);

/* Оборачивает уже существующий буфер (например, статический). Не владеет им. */
SmpStatus     smp_arena_wrap(SmpArena *a, void *mem, size_t cap, uint32_t id, const char *label);

void          smp_arena_release(SmpArena *a);

/* Bump-выделение. align обязан быть степенью двойки. NULL при исчерпании —
 * вызывающий обязан превратить это в диагностику, а не в тихий фолбэк. */
void         *smp_arena_push(SmpArena *a, size_t bytes, size_t align);
void         *smp_arena_push_zero(SmpArena *a, size_t bytes, size_t align);

/* Служебное выделение для структур самого компилятора (токены, узлы AST,
 * таблицы символов). В отличие от smp_arena_push НЕ поднимает выравнивание до
 * 64 байт: 32-байтный дескриптор рядом с 48-байтным токеном не должен стоить
 * кэш-линию каждый. Данные тензоров через это НЕ выделяются никогда.
 *
 * Последовательные вызовы с одним и тем же align и размером, кратным align,
 * дают непрерывный массив — на этом построена сборка потока токенов. */
void         *smp_arena_push_raw(SmpArena *a, size_t bytes, size_t align);

/* То же, но возвращает смещение (для SmpTensor.off). SMP_ARENA_NIL при OOM. */
#define SMP_ARENA_NIL UINT64_MAX
uint64_t      smp_arena_push_off(SmpArena *a, size_t bytes, size_t align);

/* Отметка / откат — единственная форма «освобождения». */
SMP_INLINE SmpArenaMark smp_arena_mark(const SmpArena *a) { return a->used; }
void          smp_arena_rewind(SmpArena *a, SmpArenaMark m);
void          smp_arena_reset(SmpArena *a);

SMP_INLINE size_t smp_arena_avail(const SmpArena *a) { return a->cap - a->used; }
SMP_INLINE void  *smp_arena_at(const SmpArena *a, uint64_t off) { return a->base + off; }

/* Принадлежит ли указатель этой арене (для проверок сырых указателей). */
bool          smp_arena_owns(const SmpArena *a, const void *p);

/* Строка отчёта: "arena#0 'main' 12.5 MiB / 64.0 MiB (peak 13.1 MiB, 42 allocs)" */
char         *smp_arena_report(const SmpArena *a, char *buf, size_t cap);

#endif /* SMPC3_ARENA_H */
