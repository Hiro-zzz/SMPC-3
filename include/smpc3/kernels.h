/* SMPC3 :: kernels.h -- вычислительные ядра.
 *
 * Все ядра работают по дескрипторам и базовым указателям арен, а не по
 * плоским массивам: шаги нужны для срезов и транспозиции без копии.
 *
 * Здесь объявлен только скалярный эталон. Ф6 добавит рядом ветки AVX2+FMA и
 * диспетчер по CPUID; эталон останется — на нём проверяется численная
 * правильность быстрых версий.
 */
#ifndef SMPC3_KERNELS_H
#define SMPC3_KERNELS_H

#include "smpc3/common.h"
#include "smpc3/types.h"

/* Плоский вид тензора для ядра: база уже смещена на t->off. */
typedef struct SmpBuf {
    void           *p;       /* первый элемент                              */
    const SmpTensor *t;      /* форма и шаги                                */
} SmpBuf;

/* Поэлементные: dst[i] = f(src[i]). Формы совпадают, шаги могут различаться. */
void smp_k_relu (const SmpBuf *dst, const SmpBuf *src);
void smp_k_abs  (const SmpBuf *dst, const SmpBuf *src);
void smp_k_scale(const SmpBuf *dst, const SmpBuf *src, double k);
void smp_k_copy (const SmpBuf *dst, const SmpBuf *src);
void smp_k_fill (const SmpBuf *dst, double v);
void smp_k_zero (const SmpBuf *dst);

/* Бинарные поэлементные. */
void smp_k_add(const SmpBuf *dst, const SmpBuf *a, const SmpBuf *b);
void smp_k_mul(const SmpBuf *dst, const SmpBuf *a, const SmpBuf *b);

/* Приведение типа с копированием. */
void smp_k_cast(const SmpBuf *dst, const SmpBuf *src);

/* Свёртки в скаляр. Возвращают результат в double независимо от типа входа;
 * VM приводит его обратно. */
double smp_k_reduce_add(const SmpBuf *src);
double smp_k_reduce_max(const SmpBuf *src);

/* --- Рабочая память ядер ---------------------------------------------------
 *
 * Блочный GEMM пакует панели A и B во временные буферы. Раньше они лежали в
 * статических массивах внутри avx2.c — и это делало ядро непригодным для
 * нескольких потоков: два инстанса затёрли бы панели друг друга и получили бы
 * молча неверные числа, без падения и без диагностики.
 *
 * Теперь рабочая память принадлежит вызывающему. У каждого инстанса VM она
 * своя, выделяется один раз при подъёме и на исполнении не растёт. */
typedef struct SmpKScratch {
    float *apack;
    float *bpack;
    size_t bytes;      /* 0 — рабочей памяти нет, ядро уйдёт в эталон */
} SmpKScratch;

size_t smp_k_scratch_bytes(void);
void   smp_k_scratch_bind(SmpKScratch *s, void *mem, size_t bytes);

/* C[M,N] = A[M,K] x B[K,N]. Все три — по своим шагам.
 * scratch может быть NULL: тогда считает скалярный эталон. */
void smp_k_gemm(const SmpBuf *c, const SmpBuf *a, const SmpBuf *b,
                SmpKScratch *scratch);

/* --- Выбор ветки ----------------------------------------------------------- */
typedef enum SmpKernelBackend {
    SMP_KB_AUTO = 0,   /* по CPUID                                            */
    SMP_KB_SCALAR,     /* принудительно эталон — нужен тестам для сверки       */
    SMP_KB_AVX2
} SmpKernelBackend;

/* false, если запрошенная ветка недоступна на этом процессоре. */
bool        smp_kernels_select(SmpKernelBackend b);

/* Имя выбранной реализации — для `smpc3 info` и отчётов. */
const char *smp_kernels_name(void);

#endif /* SMPC3_KERNELS_H */
