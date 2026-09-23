/* SMPC3 :: cpu.c */
#include "smpc3/cpu.h"
#include "smpc3/plat.h"
#include <string.h>
#include <stdio.h>
#include <immintrin.h>

/* Интринсики __cpuidex и _xgetbv есть только там, где есть MSVC-совместимый
 * <intrin.h>. На голом таргете (ядро NablaOS) его нет, и работает ассемблер. */
#if defined(_MSC_VER)
#  include <intrin.h>
#endif

static SmpCpu  g_cpu;
static bool    g_cpu_ready = false;

static void smp__cpuidex(int out[4], int leaf, int sub)
{
#if defined(_MSC_VER)
    __cpuidex(out, leaf, sub);
#else
    __asm__ __volatile__("cpuid"
        : "=a"(out[0]), "=b"(out[1]), "=c"(out[2]), "=d"(out[3])
        : "a"(leaf), "c"(sub));
#endif
}

static uint64_t smp__xgetbv0(void)
{
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    uint32_t lo, hi;
    __asm__ __volatile__(".byte 0x0f,0x01,0xd0" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((uint64_t)hi << 32) | lo;
#endif
}

/* Детерминированные параметры кэшей: лист 4 у Intel, 0x8000001D у AMD —
 * формат подлистов одинаковый.
 *
 *   eax[4:0]   тип   (0 — подлистов больше нет, 1 — данные, 2 — код, 3 — общий)
 *   eax[7:5]   уровень
 *   ebx[11:0]  длина линии - 1
 *   ebx[21:12] разделов - 1
 *   ebx[31:22] путей - 1
 *   ecx        наборов - 1
 *
 * Размер = пути * разделы * линия * наборы. */
static void smp__detect_caches(SmpCpu *c, int leaf)
{
    for (int sub = 0; sub < 16; sub++) {
        int r[4];
        smp__cpuidex(r, leaf, sub);

        const uint32_t type = (uint32_t)r[0] & 0x1Fu;
        if (type == 0u) break;                     /* подлисты кончились */
        if (type == 2u) continue;                  /* кэш кода не интересует */

        const uint32_t level = ((uint32_t)r[0] >> 5) & 0x7u;
        const uint64_t line  = (((uint32_t)r[1]) & 0xFFFu) + 1u;
        const uint64_t parts = ((((uint32_t)r[1]) >> 12) & 0x3FFu) + 1u;
        const uint64_t ways  = ((((uint32_t)r[1]) >> 22) & 0x3FFu) + 1u;
        const uint64_t sets  = (uint64_t)(uint32_t)r[2] + 1u;

        const uint64_t bytes = ways * parts * line * sets;
        if (bytes == 0u || bytes > 0xFFFFFFFFull) continue;

        switch (level) {
            case 1: if (type == 1u || type == 3u) c->l1d_bytes = (uint32_t)bytes; break;
            case 2: c->l2_bytes = (uint32_t)bytes; break;
            case 3: c->l3_bytes = (uint32_t)bytes; break;
            default: break;
        }
    }
}

static void smp__detect(void)
{
    int r[4];
    uint32_t isa = 0;

    memset(&g_cpu, 0, sizeof(g_cpu));
    g_cpu.cacheline = SMP_CACHELINE;
    g_cpu.n_logical = smp_plat_cpus();

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

    /* Кэши: сперва лист 4, при неудаче — расширенный лист AMD. */
    if (max_leaf >= 4) smp__detect_caches(&g_cpu, 4);
    if (!g_cpu.l2_bytes) {
        smp__cpuidex(r, 0x80000000, 0);
        if ((unsigned)r[0] >= 0x8000001Du) smp__detect_caches(&g_cpu, 0x8000001D);
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

double smp_now_sec(void) { return smp_plat_seconds(); }

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
