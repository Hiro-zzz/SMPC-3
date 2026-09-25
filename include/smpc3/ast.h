/* SMPC3 :: ast.h -- дерево разбора.
 *
 * Форма инструкции жёсткая:
 *
 *     [ПРЕФИКС]  ИСТОЧНИК -> @стадия -> @стадия => ПРИЁМНИК  [СУФФИКС] ;
 *
 * Узлы живут в арене и не владеют ничем: имена — указатели внутрь исходного
 * текста, массивы — непрерывные куски арены. Освобождения нет, только откат.
 *
 * Все ограничения ниже — жёсткие. Язык, который обещает укладываться в
 * регистры, не может позволить себе конвейер неограниченной длины.
 */
#ifndef SMPC3_AST_H
#define SMPC3_AST_H

#include "smpc3/arena.h"
#include "smpc3/common.h"
#include "smpc3/diag.h"
#include "smpc3/types.h"

#define SMP_MAX_PREFIX  8u    /* элементов префикса на инструкцию            */
#define SMP_MAX_SUFFIX  8u    /* элементов суффикса на инструкцию            */
#define SMP_MAX_STAGES 32u    /* стадий в конвейере                          */
#define SMP_MAX_ARGS    8u    /* аргументов у стадии                         */

/* Потолок развёртки [#repeat:N]. Это не цикл: тело дублируется в байткоде, и
 * потолок здесь — прямая цена в размере модуля, а не произвольное число. */
#define SMP_MAX_REPEAT 4096u

/* --- Имя, указывающее внутрь исходника ------------------------------------ */
typedef struct SmpName {
    const char *p;
    uint32_t    len;
} SmpName;

bool smp_name_is(SmpName n, const char *s);

/* --- Индекс среза: 0, .., $r1 --------------------------------------------- */
typedef enum SmpIdxKind {
    SMP_IDX_INT = 0,   /* конкретный номер                                   */
    SMP_IDX_ALL,       /* ..                                                 */
    SMP_IDX_REG        /* $r1                                                */
} SmpIdxKind;

typedef struct SmpAstIndex {
    SmpIdxKind kind;
    SmpSpan    span;
    uint64_t   ival;   /* SMP_IDX_INT                                        */
    SmpName    reg;    /* SMP_IDX_REG                                        */
} SmpAstIndex;

/* --- Ссылка на тензор: *&C<f32:1024,1024>[0, ..] -------------------------- */
typedef struct SmpAstTensor {
    SmpSpan  span;         /* вся ссылка целиком                             */
    SmpName  name;

    bool     has_type;     /* был ли <...> — то есть объявление, а не ссылка */
    SmpSpan  type_span;
    SmpDType dtype;
    uint32_t rank;
    uint32_t dims[SMP_MAX_RANK];

    bool        has_index; /* был ли [...]                                   */
    SmpSpan     index_span;
    uint32_t    nidx;
    SmpAstIndex idx[SMP_MAX_RANK];
} SmpAstTensor;

/* --- Литерал тензора: [1, 2, 3], [[1, 2], [3, 4]] ------------------------- */

/* Числа литерала едут в пул констант подряд, и пул общий на программу:
 * потолок держит одну таблицу чисел от того, чтобы съесть его целиком. */
#define SMP_MAX_LIST 1024u

typedef struct SmpAstNum {
    bool     is_float;
    uint64_t ival;             /* целое; отрицательное — в дополнительном коде */
    double   fval;
} SmpAstNum;

typedef struct SmpAstList {
    uint32_t   rank;           /* глубина вложенности, 1..4                  */
    uint32_t   dims[SMP_MAX_RANK];
    uint32_t   n;              /* чисел всего, строка за строкой             */
    SmpAstNum *vals;
    bool       any_float;      /* есть ли хоть одно дробное                  */
} SmpAstList;

/* --- Операнд -------------------------------------------------------------- */
typedef enum SmpOperandKind {
    SMP_OPD_NONE = 0,
    SMP_OPD_REG,
    SMP_OPD_TENSOR,
    SMP_OPD_INT,
    SMP_OPD_FLOAT,
    SMP_OPD_LIST
} SmpOperandKind;

typedef struct SmpAstOperand {
    SmpOperandKind kind;
    SmpSpan        span;
    SmpName        reg;        /* SMP_OPD_REG                                */
    SmpAstTensor  *tensor;     /* SMP_OPD_TENSOR                             */
    uint64_t       ival;       /* SMP_OPD_INT                                */
    double         fval;       /* SMP_OPD_FLOAT                              */
    SmpAstList    *list;       /* SMP_OPD_LIST                               */
} SmpAstOperand;

/* --- Стадия конвейера: @mmul($r2), @reduce.add ---------------------------- */
typedef struct SmpAstStage {
    SmpSpan        span;       /* @имя(аргументы) целиком                    */
    SmpSpan        name_span;
    SmpName        name;       /* "mmul", "reduce.add" — без '@'             */
    uint32_t       nargs;
    SmpAstOperand *args;
} SmpAstStage;

/* --- Префикс: [#arena:0], [#simd:v256], [^raw] ---------------------------- */
typedef enum SmpPrefixKind {
    SMP_PFX_DIRECTIVE = 0,     /* #имя[:значение]                            */
    SMP_PFX_MODE               /* ^имя                                       */
} SmpPrefixKind;

typedef struct SmpAstPrefix {
    SmpPrefixKind kind;
    SmpSpan       span;
    SmpName       name;
    bool          has_value;
    bool          value_is_int;
    SmpSpan       value_span;
    uint64_t      ival;        /* #arena:0                                   */
    SmpName       sval;        /* #simd:v256                                 */
} SmpAstPrefix;

/* --- Суффикс: [!no-alias, ?strict, ~flush-to-zero] ------------------------ */
typedef enum SmpSuffixKind {
    SMP_SFX_ASSERT = 0,        /* !                                          */
    SMP_SFX_QUERY,             /* ?                                          */
    SMP_SFX_FPMODE             /* ~                                          */
} SmpSuffixKind;

typedef struct SmpAstSuffix {
    SmpSuffixKind kind;
    SmpSpan       span;
    SmpName       name;
} SmpAstSuffix;

/* --- Инструкция ----------------------------------------------------------- */
typedef struct SmpAstStmt {
    SmpSpan        span;       /* от первого токена до ';'                   */

    uint32_t       nprefix;
    SmpAstPrefix  *prefix;

    SmpAstOperand  source;

    uint32_t       nstages;
    SmpAstStage   *stages;

    bool           has_dest;
    SmpAstOperand  dest;
    SmpSpan        arrow_span; /* позиция '=>' — для диагностик о приёмнике  */

    uint32_t       nsuffix;
    SmpAstSuffix  *suffix;

    /* Копия, порождённая [#repeat:N]. Диагностике эти поля нужны буквально:
     * все копии указывают на одну и ту же строку исходника, и без номера
     * повтора N одинаковых сообщений о ней не различить. */
    bool           from_repeat;
    uint32_t       repeat_idx;   /* 0 .. repeat_n-1                           */
    uint32_t       repeat_n;
    uint32_t       repeat_of;    /* номер оригинала до развёртки              */
} SmpAstStmt;

/* --- Программа ------------------------------------------------------------ */
typedef struct SmpAstProgram {
    const SmpSource *src;
    uint32_t         nstmts;
    SmpAstStmt      *stmts;
} SmpAstProgram;

/* --- Поиск в префиксе/суффиксе (пригодится в Ф3) -------------------------- */
const SmpAstPrefix *smp_stmt_directive(const SmpAstStmt *s, const char *name);
const SmpAstSuffix *smp_stmt_suffix(const SmpAstStmt *s, SmpSuffixKind k,
                                    const char *name);
bool                smp_stmt_has_mode(const SmpAstStmt *s, const char *name);

/* --- Развёртка ------------------------------------------------------------ */

/* Разворачивает [#repeat:N] в N копий, подставляя вместо регистра из #index
 * номер повтора.
 *
 * Делается ДО семантики намеренно. Только тогда каждая копия проверяется со
 * своим конкретным индексом, и выход за границу оси или поехавшее под #simd
 * выравнивание ловятся на компиляции — а не фаталом на седьмой итерации, где
 * их уже не отличить от настоящей ошибки в данных.
 *
 * Программа переписывается на месте, память берётся из той же арены. Дерево
 * после этого не содержит ни одной копии, помнящей про регистр-индекс: он
 * существовал только на время развёртки. */
SmpStatus smp_ast_expand(SmpAstProgram *prog, SmpArena *arena,
                         SmpDiagCtx *diag, uint32_t *n_errors);

/* --- Печать дерева -------------------------------------------------------- */
void smp_ast_dump(FILE *out, const SmpAstProgram *prog, bool color);

#endif /* SMPC3_AST_H */
