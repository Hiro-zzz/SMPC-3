/* SMPC3 :: plat/host.c -- слой платформы для Windows и POSIX.
 *
 * Всё, что раньше было разбросано по arena.c, cpu.c, thread.c и diag.c под
 * #if defined(_WIN32), собрано здесь. Остальной код об ОС не знает. */
#include "smpc3/plat.h"

#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <io.h>
#  include <fcntl.h>
#else
#  include <pthread.h>
#  include <sys/mman.h>
#  include <time.h>
#  include <unistd.h>
#endif

/* --- Память --------------------------------------------------------------- */

void *smp_plat_pages(size_t bytes)
{
    if (bytes == 0) return NULL;
#if defined(_WIN32)
    /* VirtualAlloc отдаёт обнулённые страницы, выровненные на 64 KiB. */
    return VirtualAlloc(NULL, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return (p == MAP_FAILED) ? NULL : p;
#endif
}

void smp_plat_pages_free(void *p, size_t bytes)
{
    if (!p) return;
#if defined(_WIN32)
    SMP_UNUSED(bytes);
    VirtualFree(p, 0, MEM_RELEASE);
#else
    munmap(p, bytes);
#endif
}

/* --- Процессоры и потоки -------------------------------------------------- */

uint32_t smp_plat_cpus(void)
{
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (uint32_t)si.dwNumberOfProcessors;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (uint32_t)n : 1u;
#endif
}

/* WaitForMultipleObjects умеет ждать не больше 64 объектов, и это ровно тот
 * потолок, выше которого пул всё равно перестаёт иметь смысл. */
#define SMP_PLAT_MAX_THREADS 64u

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

SmpStatus smp_plat_threads(SmpThreadFn fn, void *ctx, uint32_t n)
{
    if (n > SMP_PLAT_MAX_THREADS) n = SMP_PLAT_MAX_THREADS;

    Slot slots[SMP_PLAT_MAX_THREADS];
    for (uint32_t i = 0; i < n; i++) { slots[i].fn = fn; slots[i].ctx = ctx; slots[i].idx = i; }

#if defined(_WIN32)
    HANDLE h[SMP_PLAT_MAX_THREADS];
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
    pthread_t th[SMP_PLAT_MAX_THREADS];
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

#if defined(_WIN32)
static DWORD_PTR g_saved_affinity;
#endif

bool smp_plat_pin(uint32_t cpu)
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

void smp_plat_unpin(void)
{
#if defined(_WIN32)
    if (g_saved_affinity) {
        SetThreadAffinityMask(GetCurrentThread(), g_saved_affinity);
        g_saved_affinity = 0;
    }
#endif
}

/* --- Время и окружение ---------------------------------------------------- */

double smp_plat_seconds(void)
{
#if defined(_WIN32)
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

const char *smp_plat_env(const char *name)
{
    return getenv(name);
}

/* --- Консоль -------------------------------------------------------------- */

void smp_plat_console_setup(void)
{
#if defined(_WIN32)
    /* Текстовый режим Windows подменяет каждый \n на \r\n. Для @emit.text это
     * недопустимо: язык обещает вывести ровно те кодовые точки, которые
     * посчитаны, а не «примерно те же плюс возврат каретки». Переводим stdout
     * в двоичный режим — терминалы одиночный \n понимают прекрасно. */
    _setmode(_fileno(stdout), _O_BINARY);

    SetConsoleOutputCP(CP_UTF8);
    DWORD mode;
    HANDLE h[2] = { GetStdHandle(STD_OUTPUT_HANDLE), GetStdHandle(STD_ERROR_HANDLE) };
    for (int i = 0; i < 2; i++) {
        if (h[i] && h[i] != INVALID_HANDLE_VALUE && GetConsoleMode(h[i], &mode))
            SetConsoleMode(h[i], mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    }
#endif
    /* Диагностика должна доходить до пользователя целиком даже если процесс
     * умирает следующей инструкцией. */
    setvbuf(stderr, NULL, _IONBF, 0);
}

bool smp_plat_isatty(FILE *f)
{
#if defined(_WIN32)
    return _isatty(_fileno(f)) != 0;
#else
    return isatty(fileno(f)) != 0;
#endif
}
