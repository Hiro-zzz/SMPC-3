/* SMPC3 :: sema.h -- семантический анализ.
 *
 * Здесь дерево превращается в нечто, что можно исполнить: имена связываются с
 * символами, типы протаскиваются через конвейер, тензорам назначаются
 * смещения в арене, а обещания вроде [!no-alias] проверяются на честность.
 *
 * Проверки, которые нельзя сделать здесь, не переносятся в рантайм молча —
 * на них выдаётся диагностика прямо сейчас.
 */
#ifndef SMPC3_SEMA_H
#define SMPC3_SEMA_H

#include "smpc3/ast.h"
#include "smpc3/arena.h"
#include "smpc3/diag.h"

#define SMP_MAX_SYMS   512u

/* --- Операции ------------------------------------------------------------- */
/*   X(суффикс, имя, мин.арг, макс.арг, краткое описание)                     */
#define SMP_OPS(X)                                                             \
    X(ALLOC,      "alloc",      0, 0, "занять память под объявленный тензор")  \
    X(FILL,       "fill",       1, 1, "заполнить скаляром")                    \
    X(FILL_INST,  "fill.instance", 0, 0, "заполнить номером инстанса")         \
    X(LOAD,       "load",       0, 0, "прочитать привязанный файл в тензор")   \
    X(MMUL,       "mmul",       1, 1, "матричное умножение [M,K]x[K,N]")       \
    X(TRANSPOSE,  "transpose",  0, 0, "логическая транспозиция без копии")     \
    X(PACK,       "pack",       0, 0, "материализовать срез в плотный буфер")  \
    X(RELU,       "relu",       0, 0, "поэлементно max(x, 0)")                 \
    X(ABS,        "abs",        0, 0, "поэлементно |x|")                       \
    X(SCALE,      "scale",      1, 1, "умножить поэлементно на скаляр")        \
    X(ADD,        "add",        1, 1, "поэлементное сложение")                 \
    X(MUL,        "mul",        1, 1, "поэлементное умножение")                \
    X(REDUCE_ADD, "reduce.add", 0, 0, "свернуть в скаляр суммой")              \
    X(REDUCE_MAX, "reduce.max", 0, 0, "свернуть в скаляр максимумом")          \
    X(CAST_F32,   "cast.f32",   0, 0, "привести к f32")                        \
    X(CAST_F64,   "cast.f64",   0, 0, "привести к f64")                        \
    X(CAST_I32,   "cast.i32",   0, 0, "привести к i32")                        \
    X(CAST_U64,   "cast.u64",   0, 0, "привести к u64")                        \
    X(EMIT_TEXT,  "emit.text",  0, 0, "вывести как кодовые точки UTF-8")       \
    X(EMIT_LINE,  "emit.line",  0, 0, "то же плюс перевод строки")             \
    X(EMIT_DEC,   "emit.dec",   0, 0, "вывести целыми через пробел")           \
    X(EMIT_HEX,   "emit.hex",   0, 0, "вывести шестнадцатеричными")            \
    X(EMIT_BITS,  "emit.bits",  0, 0, "вывести двоичными")                     \
    X(EMIT_NUM,   "emit.num",   0, 0, "вывести вещественными")                 \
    X(STORE,      "store",      0, 0, "записать тензор в привязанный файл")

#define SMP_OP_ENUM(id, name, lo, hi, desc) SMP_OP_##id,
typedef enum SmpOpKind {
    SMP_OPS(SMP_OP_ENUM)
    SMP_OP__COUNT,
    SMP_OP__UNKNOWN = 0xFFFF
} SmpOpKind;
#undef SMP_OP_ENUM

typedef struct SmpOpDef {
    const char *name;
    uint8_t     min_args, max_args;
    const char *desc;
} SmpOpDef;

const SmpOpDef *smp_op_def(SmpOpKind k);
SmpOpKind       smp_op_lookup(SmpName n);

/* Перечисление известных атрибутов — для CLI и для подсказок «ты имел в виду». */
void smp_attrs_list(FILE *out);
void smp_ops_list(FILE *out);

/* --- Значение, текущее по конвейеру --------------------------------------- */
#define SMP_SYM_NONE 0xFFFFFFFFu

typedef struct SmpValue {
    bool     is_scalar;                  /* результат @reduce и литералы      */
    SmpDType dtype;
    uint32_t rank;
    uint32_t shape[SMP_MAX_RANK];
    uint32_t stride[SMP_MAX_RANK];       /* в элементах                       */
    uint16_t flags;                      /* SmpTensorFlags                    */
    uint32_t sym;                        /* тензор-источник или SMP_SYM_NONE  */
    uint64_t byte_off;                   /* смещение с учётом среза           */
} SmpValue;

/* --- Символ --------------------------------------------------------------- */
typedef enum SmpSymKind { SMP_SYM_TENSOR = 0, SMP_SYM_REG } SmpSymKind;

typedef struct SmpSym {
    SmpSymKind kind;
    SmpName    name;
    SmpSpan    decl_span;
    SmpValue   val;

    /* только для тензоров */
    uint32_t   arena_id;
    uint64_t   offset;      /* байт от начала своей арены, выровнено на 64    */
    uint64_t   bytes;
    bool       initialized; /* прошёл @alloc или побывал приёмником           */

    uint32_t   n_reads;
    uint32_t   n_writes;
    SmpSpan    last_write;
} SmpSym;

/* --- Разобранная инструкция для Ф4 ---------------------------------------- */
typedef struct SmpStmtInfo {
    uint32_t  arena_id;
    uint32_t  req_vec_bits;   /* что просили в #simd                          */
    uint32_t  vec_bits;       /* что реально будет исполнено                  */
    bool      strict;         /* ?strict                                      */
    bool      no_alias;       /* !no-alias                                    */
    bool      ftz;            /* ~flush-to-zero                               */
    bool      raw;            /* ^raw                                         */
    bool      ok;             /* инструкция прошла проверку                   */

    SmpOpKind ops[SMP_MAX_STAGES];
    SmpValue  result;
    uint32_t  dest_sym;

    /* Типы всех промежуточных значений. Эмиттеру нужны формы и шаги каждого
     * операнда; выводить их второй раз в Ф4 значило бы держать две копии одних
     * и тех же правил и однажды их рассинхронизировать. */
    SmpValue  src_val;
    SmpValue  arg_val[SMP_MAX_STAGES];   /* аргумент стадии, если он есть   */
    SmpValue  stage_out[SMP_MAX_STAGES]; /* что стадия вернула              */
    SmpValue  dest_val;
    bool      dest_is_tensor;
} SmpStmtInfo;

/* --- Результат ------------------------------------------------------------ */
typedef struct SmpSemaResult {
    SmpSym      *syms;
    uint32_t     nsyms;
    SmpStmtInfo *info;                        /* по одному на инструкцию      */
    uint32_t     ninfo;
    uint64_t     arena_bytes[SMP_MAX_ARENAS]; /* сколько занять на старте     */
    uint32_t     max_arena_id;
} SmpSemaResult;

typedef struct SmpSema {
    SmpArena        *arena;
    SmpDiagCtx      *diag;
    const SmpSource *src;
    uint32_t         host_vec_bits;   /* что умеет железо; 0 -> спросить CPUID */
    uint32_t         n_errors;
    uint32_t         n_warnings;

    /* Какой группе повторов претензия уже высказана. Развёртка размножает одну
     * написанную строку в N инструкций, и правило «одна претензия на
     * инструкцию» обязано считать по написанному: иначе опечатка внутри
     * [#repeat:1024] выдала бы тысячу одинаковых сообщений об одной строке.
     * SMP_REPEAT_NONE — ещё ни одной. */
    uint32_t         repeat_told;
    uint32_t         repeat_warned;
} SmpSema;

#define SMP_REPEAT_NONE 0xFFFFFFFFu

void      smp_sema_init(SmpSema *sm, SmpArena *arena, SmpDiagCtx *diag,
                        const SmpSource *src);
SmpStatus smp_sema_run(SmpSema *sm, const SmpAstProgram *prog, SmpSemaResult *res);

void      smp_sema_dump(FILE *out, const SmpSemaResult *res);

#endif /* SMPC3_SEMA_H */
