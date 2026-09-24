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

/* --- Слияние поэлементных стадий -------------------------------------------
 *
 * Цепочка вида @add -> @relu -> @scale делала проход по памяти на каждую
 * стадию. Пока данные помещаются в L3, это почти бесплатно, но дальше каждая
 * стадия честно платит DRAM: на тензорах по 16 МиБ три стадии стоили 6250 мкс
 * против 2500 у одной.
 *
 * Слитое ядро читает элемент один раз, прогоняет через всю цепочку в
 * регистрах и один раз пишет. Промежуточные буферы не трогаются вовсе. */
#define SMP_FUSE_MAX 8u

typedef enum SmpFuseOp {
    SMP_FOP_RELU = 0,
    SMP_FOP_ABS,
    SMP_FOP_SCALE,   /* k    */
    SMP_FOP_ADD,     /* b    */
    SMP_FOP_MUL      /* b    */
} SmpFuseOp;

typedef struct SmpFuseStep {
    uint8_t op;      /* SmpFuseOp                                    */
    double  k;       /* SMP_FOP_SCALE                                */
    SmpBuf  b;       /* SMP_FOP_ADD, SMP_FOP_MUL                     */
} SmpFuseStep;

/* Свёртка, закрывающая цепочку. Промежуточного буфера тогда нет вовсе:
 * @relu -> @reduce.add читает вход один раз и отдаёт скаляр. */
typedef enum SmpFuseRed {
    SMP_FRED_NONE = 0,
    SMP_FRED_ADD,
    SMP_FRED_MAX
} SmpFuseRed;

/* dst = chain(src) за один проход. Формы обязаны совпадать. */
void smp_k_fuse(const SmpBuf *dst, const SmpBuf *src,
                const SmpFuseStep *steps, uint32_t nsteps);

/* reduce(chain(src)) за один проход, без промежуточного буфера. */
double smp_k_fuse_reduce(const SmpBuf *src, const SmpFuseStep *steps,
                         uint32_t nsteps, uint8_t red);

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
    float   *apack;      /* комплект 0; комплект t — со сдвигом t*set_floats  */
    float   *bpack;
    size_t   bytes;      /* 0 — рабочей памяти нет, ядро уйдёт в эталон       */

    /* Комплектов панелей — столько потоков может считать один GEMM разом.
     * Каждый поток пакует в свой комплект; общий комплект на двоих — ровно
     * та гонка, ради которой рабочая память и стала чужой. */
    uint32_t sets;
    size_t   set_floats;
} SmpKScratch;

/* Размер под один комплект и под sets комплектов. Сколько комплектов
 * получилось, bind выводит из размера сам. */
size_t smp_k_scratch_bytes(void);
size_t smp_k_scratch_bytes_for(uint32_t sets);
void   smp_k_scratch_bind(SmpKScratch *s, void *mem, size_t bytes);

/* C[M,N] = A[M,K] x B[K,N]. Все три — по своим шагам.
 * scratch может быть NULL: тогда считает скалярный эталон. */
void smp_k_gemm(const SmpBuf *c, const SmpBuf *a, const SmpBuf *b,
                SmpKScratch *scratch);

/* То же с эпилогом: C = chain(A x B) за один проход по C.
 *
 * Цепочка применяется прямо в цикле выгрузки тайла и только на последнем
 * k-блоке: до него в C идёт накопление через +=, и применять @relu к
 * незаконченной сумме было бы просто неверно. Тайл в этот момент лежит в L1,
 * так что эпилог достаётся почти даром — отдельный проход по C стоил бы
 * полной ширины памяти.
 *
 * nsteps == 0 эквивалентно smp_k_gemm. Векторная ветка требует того же, что и
 * обычное слияние: f32 и плотная раскладка у каждого операнда цепочки. Не
 * сошлось — эталон посчитает GEMM, а цепочку доложит отдельным проходом:
 * результат тот же, экономии нет. */
void smp_k_gemm_ep(const SmpBuf *c, const SmpBuf *a, const SmpBuf *b,
                   SmpKScratch *scratch,
                   const SmpFuseStep *steps, uint32_t nsteps);

/* --- Умножение на транспонированную и Q8_0 --------------------------------
 *
 * C[M,N] = A[M,K] x B[N,K]^T — линейный слой: веса лежат строками выхода.
 * B в f32 или f64 умножается обычным GEMM по транспонированному виду. B в
 * q8_0 распаковывается на лету, в регистрах; A и C тогда f32. Векторная
 * ветка делит строки B между потоками рабочей памяти (scratch->sets): каждый
 * выход считает ровно один поток в одном и том же порядке, так что результат
 * бит в бит тот же, что на одном потоке. */
void smp_k_mmul_t(const SmpBuf *c, const SmpBuf *a, const SmpBuf *b,
                  SmpKScratch *scratch);

/* Ниже этого M*N*K раздача строк q8_0 потокам не окупается. */
void smp_k_gemm_q8_par_min(uint64_t work);

/* Половинная точность: к ближайшему, при равенстве — к чётному. */
float    smp_f16_to_f32(uint16_t h);
uint16_t smp_f32_to_f16(float f);

/* n кратно SMP_Q8_0_BLOCK. Квантование — как эталонное в GGUF. */
void  smp_q8_0_quantize(void *dst, const float *src, size_t n);
void  smp_q8_0_dequantize(float *dst, const void *src, size_t n);
float smp_q8_0_get(const void *src, size_t i);

/* --- Блокировка GEMM -------------------------------------------------------
 *
 * Упаковка панелей A и B имеет смысл ровно пока обе держатся в L2. Значения по
 * умолчанию (MC=96, KC=256, NC=256) просят 352 КиБ и выбраны развёрткой на
 * Raptor Lake, где L2 равен 1.25 МиБ. На ядре с L2 в 256 КиБ те же числа не
 * помещаются, панели начинают вытеснять друг друга, и пропускная способность
 * падает вдвое — развёртка это показывает наглядно.
 *
 * Поэтому блокировка ужимается под реальный L2, а не подбирается заново: на
 * машине, где значения по умолчанию влезают, ничего не меняется. */
void smp_k_gemm_block(uint32_t *mc, uint32_t *kc, uint32_t *nc);

/* --- Параллельный GEMM -----------------------------------------------------
 *
 * Если у рабочей памяти больше одного комплекта, крупный GEMM делится на
 * плитки C и раздаётся потокам (smp_threads_run). Каждая плитка считается тем
 * же блочным циклом с тем же разбиением по K, поэтому порядок сложения у
 * каждого элемента прежний, и результат бит в бит совпадает с однопоточным.
 * Режим MXCSR вызывающего потока (FTZ) переносится на исполнителей.
 *
 * Делится только то, где M*N*K не меньше порога: на мелочи раздача дороже
 * самой арифметики. Порог можно сдвинуть — ноль делит всё, это для тестов. */
void smp_k_gemm_par_min(uint64_t work);

/* Тот же расчёт, но для произвольного L2 (0 — размер неизвестен, берутся
 * значения по умолчанию). Вынесен наружу, чтобы проверять на числах, которых
 * у машины под рукой нет. */
void smp_k_gemm_block_for(uint32_t l2_bytes,
                          uint32_t *mc, uint32_t *kc, uint32_t *nc);

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
