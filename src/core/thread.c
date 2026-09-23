/* SMPC3 :: thread.c
 *
 * Сами потоки поднимает слой платформы; здесь — то, что от ОС не зависит. */
#include "smpc3/thread.h"
#include "smpc3/cpu.h"
#include "smpc3/plat.h"

#if defined(_MSC_VER) && !defined(__clang__)
#  include <intrin.h>
#endif

/* Потолок параллельности. Выше него пул всё равно перестаёт иметь смысл, а
 * WaitForMultipleObjects больше 64 объектов ждать не умеет. */
#define SMP_MAX_THREADS 64u

SmpStatus smp_threads_run(SmpThreadFn fn, void *ctx, uint32_t n)
{
    if (!fn) return SMP_ERR_INTERNAL;

    /* Один поток — это просто вызов. Поднимать ради него ОС незачем, и заодно
     * отладка однопоточного случая остаётся линейной. */
    if (n <= 1) { fn(ctx, 0); return SMP_OK; }
    if (n > SMP_MAX_THREADS) n = SMP_MAX_THREADS;

    return smp_plat_threads(fn, ctx, n);
}

int32_t smp_atomic_fetch_add(volatile int32_t *p, int32_t v)
{
#if defined(__clang__) || defined(__GNUC__)
    return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST);
#elif defined(_MSC_VER)
    return (int32_t)_InterlockedExchangeAdd((volatile long *)p, (long)v);
#else
#   error "нет атомарных операций для этого компилятора"
#endif
}

bool smp_thread_pin(uint32_t cpu) { return smp_plat_pin(cpu); }
void smp_thread_unpin(void)       { smp_plat_unpin(); }

uint32_t smp_threads_default(void)
{
    const uint32_t n = smp_cpu()->n_logical;
    return n ? (n > SMP_MAX_THREADS ? SMP_MAX_THREADS : n) : 1u;
}
