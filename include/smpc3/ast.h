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

#include "smpc3/common.h"
#include "smpc3/diag.h"
#include "smpc3/types.h"

#define SMP_MAX_PREFIX  8u    /* элементов префикса на инструкцию            */
#define SMP_MAX_SUFFIX  8u    /* элементов суффикса на инструкцию            */
#define SMP_MAX_STAGES 32u    /* стадий в конвейере                          */
#define SMP_MAX_ARGS    8u    /* аргументов у стадии                         */

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
    uint16_t dims[SMP_MAX_RANK];

    bool        has_index; /* был ли [...]                                   */
    SmpSpan     index_span;
    uint32_t    nidx;
    SmpAstIndex idx[SMP_MAX_RANK];
} SmpAstTensor;

/* --- Операнд -------------------------------------------------------------- */
typedef enum SmpOperandKind {
    SMP_OPD_NONE = 0,
    SMP_OPD_REG,
    SMP_OPD_TENSOR,
    SMP_OPD_INT,
    SMP_OPD_FLOAT
} SmpOperandKind;

typedef struct SmpAstOperand {
    SmpOperandKind kind;
    SmpSpan        span;
    SmpName        reg;        /* SMP_OPD_REG                                */
    SmpAstTensor  *tensor;     /* SMP_OPD_TENSOR                             */
    uint64_t       ival;       /* SMP_OPD_INT                                */
    double         fval;       /* SMP_OPD_FLOAT                              */
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

/* --- Печать дерева -------------------------------------------------------- */
void smp_ast_dump(FILE *out, const SmpAstProgram *prog, bool color);

#endif /* SMPC3_AST_H */
