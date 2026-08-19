/* SMPC3 :: kernels/impl.h -- внутренний интерфейс между ветками ядер.
 *
 * Наружу (в kernels.h) торчит один набор функций; здесь их реализации живут
 * парами: скалярный эталон и векторная версия. Диспетчер в dispatch.c
 * выбирает между ними по CPUID и по тому, применима ли векторная ветка к
 * конкретным дескрипторам.
 *
 * Этот заголовок не входит в include/: снаружи знать про ветки незачем.
 */
#ifndef SMPC3_KERNELS_IMPL_H
#define SMPC3_KERNELS_IMPL_H

#include "smpc3/kernels.h"

/* --- Скалярный эталон ----------------------------------------------------- */
void   smp_ks_relu (const SmpBuf *dst, const SmpBuf *src);
void   smp_ks_abs  (const SmpBuf *dst, const SmpBuf *src);
void   smp_ks_scale(const SmpBuf *dst, const SmpBuf *src, double k);
void   smp_ks_copy (const SmpBuf *dst, const SmpBuf *src);
void   smp_ks_cast (const SmpBuf *dst, const SmpBuf *src);
void   smp_ks_fill (const SmpBuf *dst, double v);
void   smp_ks_zero (const SmpBuf *dst);
void   smp_ks_add  (const SmpBuf *dst, const SmpBuf *a, const SmpBuf *b);
void   smp_ks_mul  (const SmpBuf *dst, const SmpBuf *a, const SmpBuf *b);
void   smp_ks_fuse (const SmpBuf *dst, const SmpBuf *src,
                    const SmpFuseStep *steps, uint32_t nsteps);
double smp_ks_reduce_add(const SmpBuf *src);
double smp_ks_reduce_max(const SmpBuf *src);
void   smp_ks_gemm (const SmpBuf *c, const SmpBuf *a, const SmpBuf *b);

/* --- AVX2 + FMA ----------------------------------------------------------- */
/* Все принимают только f32 с плотной раскладкой; проверку делает диспетчер. */
void   smp_ka_relu (float *dst, const float *src, size_t n);
void   smp_ka_abs  (float *dst, const float *src, size_t n);
void   smp_ka_scale(float *dst, const float *src, size_t n, float k);
void   smp_ka_add  (float *dst, const float *a, const float *b, size_t n);
void   smp_ka_mul  (float *dst, const float *a, const float *b, size_t n);
void   smp_ka_fuse (float *dst, const float *src, size_t n,
                    const SmpFuseStep *steps, uint32_t nsteps);
double smp_ka_reduce_add(const float *src, size_t n);
double smp_ka_reduce_max(const float *src, size_t n);

/* C[M,N] = A[M,K] x B[K,N], все три row-major с произвольным ведущим шагом.
 * Панели пакуются в scratch, принадлежащий вызывающему. */
void   smp_ka_gemm(float *C, size_t ldc, const float *A, size_t lda,
                   const float *B, size_t ldb, size_t M, size_t N, size_t K,
                   float *apack, float *bpack);

#endif /* SMPC3_KERNELS_IMPL_H */
