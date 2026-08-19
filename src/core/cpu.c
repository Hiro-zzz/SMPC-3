/* SMPC3 :: cpu.c */
#include "smpc3/cpu.h"
#include <string.h>
#include <stdio.h>

#if defined(_MSC_VER) || defined(__clang__)
#  include <intrin.h>
#  include <immintrin.h>
#endif

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <unistd.h>
#  include <time.h>
#endif

static SmpCpu  g_cpu;
static bool    g_cpu_ready = false;

static void smp__cpuidex(int out[4], int leaf, int sub)
{
#if defined(_MSC_VER) || defined(__clang__)
    __cpuidex(out, leaf, sub);
#else
    __asm__ __volatile__("cpuid"
        : "=a"(out[0]), "=b"(out[1]), "=c"(out[2]), "=d"(out[3])
        : "a"(leaf), "c"(sub));
#endif
}

static uint64_t smp__xgetbv0(void)
{
#if defined(_MSC_VER) || defined(__clang__)
    return _xgetbv(0);
#else
    uint32_t lo, hi;
    __asm__ __volatile__(".byte 0x0f,0x01,0xd0" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((uint64_t)hi << 32) | lo;
#endif
}

static uint32_t smp__n_logical(void)
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

static void smp__detect(void)
{
    int r[4];
    uint32_t isa = 0;

    memset(&g_cpu, 0, sizeof(g_cpu));
    g_cpu.cacheline = SMP_CACHELINE;
    g_cpu.n_logical = smp__n_logical();

    smp__cpuidex(r, 0, 0);
    const int max_leaf = r[0];

    /* Brand string: три листа по 16 байт. */
    smp__cpuidex(r, 0x80000000, 0);
    if ((unsigned)r[0] >= 0x80000004u) {
        for (int i = 0; i < 3; i++)
            smp__cpuidex((int *)(g_cpu.brand + i * 16), 0x80000002 + i, 0);
        g_cpu.brand[48] = '\0';
    } else {
        snprintf(g_cpu.brand, sizeof g_cpu.brand, "unknown x86-64");
    }

    if (max_leaf >= 1) {
        smp__cpuidex(r, 1, 0);
        const uint32_t ecx = (uint32_t)r[2], edx = (uint32_t)r[3];
        if (edx & (1u << 26)) isa |= SMP_ISA_SSE2;
        if (ecx & (1u << 20)) isa |= SMP_ISA_SSE42;

        /* AVX годен только если ОС реально сохраняет YMM (XCR0 биты 1|2).
         * Флаг CPUID сам по себе ничего не гарантирует. */
        const bool osxsave = (ecx & (1u << 27)) != 0;
        const bool avx_cpu = (ecx & (1u << 28)) != 0;
        uint64_t xcr0 = 0;
        if (osxsave) xcr0 = smp__xgetbv0();

        const bool ymm_ok = osxsave && ((xcr0 & 0x6u) == 0x6u);
        const bool zmm_ok = ymm_ok  && ((xcr0 & 0xE0u) == 0xE0u);

        if (avx_cpu && ymm_ok) {
            isa |= SMP_ISA_AVX;
            if (ecx & (1u << 12)) isa |= SMP_ISA_FMA;
        }

        if (max_leaf >= 7) {
            smp__cpuidex(r, 7, 0);
            const uint32_t ebx = (uint32_t)r[1];
            if ((isa & SMP_ISA_AVX) && (ebx & (1u << 5))) isa |= SMP_ISA_AVX2;
            if (zmm_ok) {
                if (ebx & (1u << 16)) isa |= SMP_ISA_AVX512F;
                if (ebx & (1u << 17)) isa |= SMP_ISA_AVX512DQ;
                if (ebx & (1u << 30)) isa |= SMP_ISA_AVX512BW;
                if (ebx & (1u << 31)) isa |= SMP_ISA_AVX512VL;
            }
        }
    }

    g_cpu.isa = isa;
    if      (isa & SMP_ISA_AVX512F) g_cpu.max_vec_bits = 512;
    else if (isa & SMP_ISA_AVX)     g_cpu.max_vec_bits = 256;
    else if (isa & SMP_ISA_SSE2)    g_cpu.max_vec_bits = 128;
    else                            g_cpu.max_vec_bits = 64;

    g_cpu_ready = true;
}

const SmpCpu *smp_cpu(void)
{
    if (SMP_UNLIKELY(!g_cpu_ready)) smp__detect();
    return &g_cpu;
}

double smp_now_sec(void)
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

const char *smp_vec_name(uint32_t vec_bits)
{
    switch (vec_bits) {
        case 128: return "v128";
        case 256: return "v256";
        case 512: return "v512";
        default:  return "scalar";
    }
}

/* --- MXCSR ---------------------------------------------------------------- */
#define SMP_MXCSR_FTZ (1u << 15)
#define SMP_MXCSR_DAZ (1u << 6)

uint32_t smp_fpu_get_mxcsr(void)
{
#if defined(__SSE__) || defined(_M_X64) || defined(__x86_64__)
    return _mm_getcsr();
#else
    return 0;
#endif
}

void smp_fpu_set_mxcsr(uint32_t v)
{
#if defined(__SSE__) || defined(_M_X64) || defined(__x86_64__)
    _mm_setcsr(v);
#else
    SMP_UNUSED(v);
#endif
}

void smp_fpu_set_ftz(bool on)
{
    uint32_t v = smp_fpu_get_mxcsr();
    if (on) v |=  (SMP_MXCSR_FTZ | SMP_MXCSR_DAZ);
    else    v &= ~(SMP_MXCSR_FTZ | SMP_MXCSR_DAZ);
    smp_fpu_set_mxcsr(v);
}
