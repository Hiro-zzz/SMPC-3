/* SMPC3 :: thread.c */
#include "smpc3/thread.h"
#include "smpc3/cpu.h"

#include <string.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <pthread.h>
#endif

/* WaitForMultipleObjects умеет ждать не больше 64 объектов, и это ровно тот
 * потолок, выше которого пул всё равно перестаёт иметь смысл. */
#define SMP_MAX_THREADS 64u

typedef struct {
    SmpThreadFn fn;
    void       *ctx;
    uint32_t    idx;
} Slot;

#if defined(_WIN32)
static DWORD WINAPI trampoline(LPVOID p)
{
    Slot *s = (Slot *)p;
    s->fn(s->ctx, s->idx);
    return 0;
}
#else
static void *trampoline(void *p)
{
    Slot *s = (Slot *)p;
    s->fn(s->ctx, s->idx);
    return NULL;
}
#endif

SmpStatus smp_threads_run(SmpThreadFn fn, void *ctx, uint32_t n)
{
    if (!fn) return SMP_ERR_INTERNAL;

    /* Один поток — это просто вызов. Поднимать ради него ОС незачем, и заодно
     * отладка однопоточного случая остаётся линейной. */
    if (n <= 1) { fn(ctx, 0); return SMP_OK; }
    if (n > SMP_MAX_THREADS) n = SMP_MAX_THREADS;

    Slot slots[SMP_MAX_THREADS];
    for (uint32_t i = 0; i < n; i++) { slots[i].fn = fn; slots[i].ctx = ctx; slots[i].idx = i; }

#if defined(_WIN32)
    HANDLE h[SMP_MAX_THREADS];
    uint32_t started = 0;

    for (uint32_t i = 0; i < n; i++) {
        h[i] = CreateThread(NULL, 0, trampoline, &slots[i], 0, NULL);
        if (!h[i]) break;
        started++;
    }

    /* Не поднялось сколько-то потоков — их долю доделываем сами, а не
     * оставляем работу невыполненной. */
    for (uint32_t i = started; i < n; i++) fn(ctx, i);

    if (started) {
        WaitForMultipleObjects(started, h, TRUE, INFINITE);
        for (uint32_t i = 0; i < started; i++) CloseHandle(h[i]);
    }
    return started == n ? SMP_OK : SMP_ERR_INTERNAL;
#else
    pthread_t th[SMP_MAX_THREADS];
    uint32_t started = 0;

    for (uint32_t i = 0; i < n; i++) {
        if (pthread_create(&th[i], NULL, trampoline, &slots[i]) != 0) break;
        started++;
    }
    for (uint32_t i = started; i < n; i++) fn(ctx, i);
    for (uint32_t i = 0; i < started; i++) pthread_join(th[i], NULL);
    return started == n ? SMP_OK : SMP_ERR_INTERNAL;
#endif
}

int32_t smp_atomic_fetch_add(volatile int32_t *p, int32_t v)
{
#if defined(__clang__) || defined(__GNUC__)
    return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST);
#elif defined(_WIN32)
    return (int32_t)InterlockedExchangeAdd((volatile LONG *)p, (LONG)v);
#else
#   error "нет атомарных операций для этой платформы"
#endif
}

#if defined(_WIN32)
static DWORD_PTR g_saved_affinity;
#endif

bool smp_thread_pin(uint32_t cpu)
{
#if defined(_WIN32)
    if (cpu >= 64u) return false;
    const DWORD_PTR mask = (DWORD_PTR)1u << cpu;
    const DWORD_PTR prev = SetThreadAffinityMask(GetCurrentThread(), mask);
    if (!prev) return false;
    if (!g_saved_affinity) g_saved_affinity = prev;
    return true;
#else
    SMP_UNUSED(cpu);
    return false;
#endif
}

void smp_thread_unpin(void)
{
#if defined(_WIN32)
    if (g_saved_affinity) {
        SetThreadAffinityMask(GetCurrentThread(), g_saved_affinity);
        g_saved_affinity = 0;
    }
#endif
}

uint32_t smp_threads_default(void)
{
    const uint32_t n = smp_cpu()->n_logical;
    return n ? (n > SMP_MAX_THREADS ? SMP_MAX_THREADS : n) : 1u;
}
