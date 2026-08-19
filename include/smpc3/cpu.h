/* SMPC3 :: cpu.h -- runtime-детект ISA. Компилируем один бинарь, а ветку
 * кернела выбираем по факту, а не по надеждам. */
#ifndef SMPC3_CPU_H
#define SMPC3_CPU_H

#include "smpc3/common.h"

enum SmpIsaBits {
    SMP_ISA_SSE2     = 1u << 0,
    SMP_ISA_SSE42    = 1u << 1,
    SMP_ISA_AVX      = 1u << 2,
    SMP_ISA_AVX2     = 1u << 3,
    SMP_ISA_FMA      = 1u << 4,
    SMP_ISA_AVX512F  = 1u << 5,
    SMP_ISA_AVX512DQ = 1u << 6,
    SMP_ISA_AVX512BW = 1u << 7,
    SMP_ISA_AVX512VL = 1u << 8
};

/* Полный профиль векторного блока v512 (не только F). */
#define SMP_ISA_V512_FULL \
    (SMP_ISA_AVX512F | SMP_ISA_AVX512DQ | SMP_ISA_AVX512BW | SMP_ISA_AVX512VL)

typedef struct SmpCpu {
    uint32_t isa;              /* битовая маска SmpIsaBits                   */
    uint32_t max_vec_bits;     /* 128 / 256 / 512 — что реально исполнимо    */
    uint32_t n_logical;        /* логических процессоров                     */
    uint32_t cacheline;

    /* Размеры кэшей одного ядра, байт; 0 — определить не вышло. Нужны GEMM:
     * блокировка держит обе упакованные панели в L2, а «сколько это» зависит
     * от процессора и на разных машинах отличается вчетверо. */
    uint32_t l1d_bytes;
    uint32_t l2_bytes;
    uint32_t l3_bytes;

    char     brand[49];
} SmpCpu;

/* Идемпотентно. Первый вызов делает CPUID, дальше отдаёт кэш. */
const SmpCpu *smp_cpu(void);

SMP_INLINE bool smp_cpu_has(uint32_t bits) { return (smp_cpu()->isa & bits) == bits; }

/* Монотонное время в секундах — для замеров пропускной способности. */
double      smp_now_sec(void);

/* Имя ширины: 128 -> "v128". Для диагностики. */
const char *smp_vec_name(uint32_t vec_bits);

/* Управление денормалами: [~flush-to-zero] выставляет FTZ+DAZ в MXCSR. */
uint32_t smp_fpu_get_mxcsr(void);
void     smp_fpu_set_mxcsr(uint32_t v);
void     smp_fpu_set_ftz(bool on);   /* FTZ (bit15) + DAZ (bit6) разом */

#endif /* SMPC3_CPU_H */
