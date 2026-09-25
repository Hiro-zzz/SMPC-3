/* SMPC3 :: types.h -- система типов и 32-байтный дескриптор тензора. */
#ifndef SMPC3_TYPES_H
#define SMPC3_TYPES_H

#include "smpc3/common.h"

/* --- Скалярные типы ------------------------------------------------------- */
/* Порядок фиксирован: он кодируется в байткоде .s3b, менять нельзя. */
typedef enum SmpDType {
    SMP_DT_INVALID = 0,
    SMP_DT_F32     = 1,
    SMP_DT_F64     = 2,
    SMP_DT_I32     = 3,
    SMP_DT_U64     = 4,
    SMP_DT_RAW_PTR = 5,
    SMP_DT_Q8_0    = 6,
    SMP_DT__COUNT
} SmpDType;

/* Q8_0, как в GGUF: блок из 32 весов i8 со своим масштабом f16, 34 байта
 * (сначала масштаб, потом веса); значение — масштаб × вес. Отдельный
 * элемент блочного типа не адресуется, поэтому размер элемента у него ноль,
 * а байты считает smp_dtype_bytes. Последняя ось такого тензора кратна 32,
 * и строка всегда состоит из целых блоков. */
#define SMP_Q8_0_BLOCK 32u
#define SMP_Q8_0_BYTES 34u

#define SMP_MAX_RANK 4

const char *smp_dtype_name(SmpDType dt);
uint32_t    smp_dtype_size(SmpDType dt);      /* байт на элемент; 0 у блочных */
uint64_t    smp_dtype_bytes(SmpDType dt, uint64_t nelem);  /* байт на nelem */
bool        smp_dtype_is_block(SmpDType dt);
SmpDType    smp_dtype_parse(const char *s, size_t len);
bool        smp_dtype_is_float(SmpDType dt);

/* Сколько элементов данного типа влезает в векторный регистр. */
uint32_t    smp_dtype_lanes(SmpDType dt, uint32_t vec_bits);

/* --- Флаги дескриптора ---------------------------------------------------- */
enum SmpTensorFlags {
    SMP_TF_NONE      = 0,
    SMP_TF_ALIGN64   = 1u << 0,  /* база гарантированно выровнена на 64      */
    SMP_TF_ALIGN32   = 1u << 1,  /* ...на 32                                 */
    SMP_TF_VIEW      = 1u << 2,  /* не владеет памятью (срез)                */
    SMP_TF_NO_ALIAS  = 1u << 3,  /* объявлен [!no-alias]                     */
    SMP_TF_CONTIG    = 1u << 4,  /* плотная укладка, stride выводим          */
    SMP_TF_READONLY  = 1u << 5,
    SMP_TF_TRANSPOSED= 1u << 6,  /* логическая транспозиция без копии        */
    SMP_TF_FTZ       = 1u << 7,  /* [~flush-to-zero] в области определения   */
    SMP_TF_DYNOFF    = 1u << 8,  /* смещение известно только в рантайме      */
    SMP_TF_EXTERN    = 1u << 9   /* вид: лежит не в арене, а там, куда его
                                    положил @load; off — в пространстве видов */
};

/* Старшие три бита flags хранят номер арены. Дескриптор обязан оставаться
 * 32-байтным, а отдельного поля под это нет: восемь арен ровно укладываются
 * в биты 13..15, которые иначе простаивали бы. */
#define SMP_TF_ARENA_SHIFT 13u
#define SMP_TF_ARENA_MASK  0xE000u

SMP_INLINE uint32_t smp_tf_arena(uint16_t flags)
{
    return (uint32_t)((flags & SMP_TF_ARENA_MASK) >> SMP_TF_ARENA_SHIFT);
}
SMP_INLINE uint16_t smp_tf_with_arena(uint16_t flags, uint32_t arena)
{
    return (uint16_t)((flags & (uint16_t)~SMP_TF_ARENA_MASK) |
                      ((arena & 7u) << SMP_TF_ARENA_SHIFT));
}

/* --- Дескриптор тензора: ровно 64 байта, одна строка кэша ----------------- */
/* Было 32 байта с осями в uint16_t, и ось не вмещала словарь языковой модели:
 * у Qwen2.5 их 151 936. Оси и шаги стали uint32_t, смещение — uint64_t, чтобы
 * арена не упиралась в 4 ГиБ. Число элементов осталось uint32_t: тензор до
 * 4 294 967 295 элементов, и шаг любой оси тогда тоже влезает в uint32_t.
 * Хвост — нули: дескрипторы сравниваются побайтово, дыр быть не должно. */
typedef struct SmpTensor {
    uint64_t off;                    /*  0: смещение в арене, байты          */
    uint32_t nelem;                  /*  8: полное число элементов           */
    uint32_t name_id;                /* 12: символ в таблице имён (диагност.)*/
    uint32_t shape[SMP_MAX_RANK];    /* 16: размерности                      */
    uint32_t stride[SMP_MAX_RANK];   /* 32: шаги В ЭЛЕМЕНТАХ, не в байтах    */
    uint8_t  dtype;                  /* 48: SmpDType                         */
    uint8_t  rank;                   /* 49: 1..4                             */
    uint16_t flags;                  /* 50: SmpTensorFlags                   */
    uint8_t  reserved[12];           /* 52: нули                             */
} SmpTensor;

SMP_STATIC_ASSERT(sizeof(SmpTensor) == 64, tensor_descriptor_is_64_bytes);
SMP_STATIC_ASSERT(offsetof(SmpTensor, dtype) == 48, tensor_dtype_offset);

#define SMP_DIM_MAX   0xFFFFFFFFu   /* потолок одной размерности (uint32_t)  */
#define SMP_NELEM_MAX 0xFFFFFFFFu   /* потолок числа элементов (uint32_t)    */

/* --- Операции над дескрипторами ------------------------------------------- */

/* Заполнить плотный (contiguous, row-major) дескриптор. Не выделяет память. */
void     smp_tensor_dense(SmpTensor *t, SmpDType dt, uint32_t rank, const uint32_t *shape);

/* Полный размер в байтах логического содержимого. */
uint64_t smp_tensor_bytes(const SmpTensor *t);

/* Плотный ли по факту (strides совпадают с выведенными из shape). */
bool     smp_tensor_is_contiguous(const SmpTensor *t);

/* Совместимы ли по форме поэлементно (broadcast НЕ поддерживается: слишком
 * дипломатично для этого языка). */
bool     smp_tensor_same_shape(const SmpTensor *a, const SmpTensor *b);

/* Человекочитаемая форма: "f32:1024,1024" -> буфер. Возвращает buf. */
char    *smp_tensor_sig(const SmpTensor *t, char *buf, size_t cap);

/* Число для человека: целые — целыми, дробные — кратчайшей записью, которая
 * читается обратно в то же значение своего типа. Возвращает buf. */
char    *smp_num_fmt(double v, SmpDType dt, char *buf, size_t cap);

#endif /* SMPC3_TYPES_H */
