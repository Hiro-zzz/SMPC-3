/* SMPC3 :: bytecode.h -- набор инструкций и формат .s3b.
 *
 * Инструкция ровно 8 байт и декодируется без ветвлений: это условие прямой
 * диспетчеризации в Ф5. Переменная длина сэкономила бы файл и стоила бы
 * такта на каждой команде.
 *
 * Файл секционный: заголовок, таблица секций, дальше содержимое. Порядок
 * байтов — little-endian, как у целевой архитектуры; читатель это проверяет
 * и не пытается «догадаться».
 */
#ifndef SMPC3_BYTECODE_H
#define SMPC3_BYTECODE_H

#include "smpc3/common.h"
#include "smpc3/types.h"

/* ========================================================================== */
/*  Опкоды                                                                    */
/* ========================================================================== */

/* Формат операндов — нужен дизассемблеру и валидатору.
 *   D    приёмник        A,B  источники        K    индекс в пуле констант
 *   T    индекс в таблице дескрипторов                                       */
typedef enum SmpOpFmt {
    SMP_FMT_NONE = 0,
    SMP_FMT_D,
    SMP_FMT_D_A,
    SMP_FMT_D_A_B,
    SMP_FMT_D_T,
    SMP_FMT_D_K,
    SMP_FMT_D_A_K,
    SMP_FMT_D_A_B_K,
    SMP_FMT_T_A,
    SMP_FMT_D_A_X    /* D, A, а в aux — вариант операции */
} SmpOpFmt;

/*   X(суффикс, мнемоника, формат, описание)                                  */
#define SMP_BC_OPS(X)                                                          \
    X(HALT,   "halt",    SMP_FMT_NONE,    "останов")                           \
    X(NOP,    "nop",     SMP_FMT_NONE,    "ничего")                            \
    X(LOADT,  "loadt",   SMP_FMT_D_T,     "rD <- дескриптор T")                \
    X(LOADK,  "loadk",   SMP_FMT_D_K,     "rD <- константа K")                 \
    X(MOVE,   "move",    SMP_FMT_D_A,     "rD <- rA")                          \
    X(STORET, "storet",  SMP_FMT_T_A,     "T <- rA (копия в именованный)")     \
    X(SLICED, "sliced",  SMP_FMT_D_A_B_K, "rD <- rA + rB*K байт (динам. срез)")\
    X(ALLOC,  "alloc",   SMP_FMT_D,       "занулить область rD")               \
    X(FILL,   "fill",    SMP_FMT_D_A_K,   "rD <- rA, заполненный константой K")\
    X(FILLI,  "filli",   SMP_FMT_D,       "rD <- заполнен номером инстанса")    \
    X(MMUL,   "mmul",    SMP_FMT_D_A_B,   "rD <- rA x rB")                     \
    X(TRANS,  "trans",   SMP_FMT_D_A,     "rD <- вид rA с переставленными осями")\
    X(PACK,   "pack",    SMP_FMT_D_A,     "rD <- плотная копия rA")            \
    X(RELU,   "relu",    SMP_FMT_D_A,     "rD <- max(rA, 0)")                  \
    X(ABS,    "abs",     SMP_FMT_D_A,     "rD <- |rA|")                        \
    X(SCALE,  "scale",   SMP_FMT_D_A_K,   "rD <- rA * K")                      \
    X(ADD,    "add",     SMP_FMT_D_A_B,   "rD <- rA + rB поэлементно")         \
    X(MUL,    "mul",     SMP_FMT_D_A_B,   "rD <- rA * rB поэлементно")         \
    X(REDADD, "redadd",  SMP_FMT_D_A,     "rD <- сумма всех элементов rA")     \
    X(REDMAX, "redmax",  SMP_FMT_D_A,     "rD <- максимум элементов rA")       \
    X(CVTF32, "cvt.f32", SMP_FMT_D_A,     "rD <- (f32)rA")                     \
    X(CVTF64, "cvt.f64", SMP_FMT_D_A,     "rD <- (f64)rA")                     \
    X(CVTI32, "cvt.i32", SMP_FMT_D_A,     "rD <- (i32)rA")                     \
    X(CVTU64, "cvt.u64", SMP_FMT_D_A,     "rD <- (u64)rA")                     \
    X(EMIT,   "emit",    SMP_FMT_D_A_X,   "вывести rA; rD <- сколько выведено")\
    X(LOAD,   "load",    SMP_FMT_D,       "rD <- содержимое привязанного файла") \
    X(STORE,  "store",   SMP_FMT_D_A,     "rA -> привязанный файл; rD <- байт")

#define SMP_BC_ENUM(id, mn, fmt, desc) SMP_BC_##id,
typedef enum SmpOpcode {
    SMP_BC_OPS(SMP_BC_ENUM)
    SMP_BC__COUNT
} SmpOpcode;
#undef SMP_BC_ENUM

typedef struct SmpOpcodeDef {
    const char *mnemonic;
    SmpOpFmt    fmt;
    const char *desc;
} SmpOpcodeDef;

const SmpOpcodeDef *smp_opcode_def(SmpOpcode op);

/* ========================================================================== */
/*  Инструкция                                                                */
/* ========================================================================== */

/* Байт флагов: ширина вектора и режимы, действующие на этой инструкции. */
enum SmpInstrFlags {
    SMP_IF_VEC_MASK = 0x03u,   /* 0=скаляр, 1=v128, 2=v256, 3=v512          */
    SMP_IF_FTZ      = 0x04u,
    SMP_IF_STRICT   = 0x08u,
    SMP_IF_NOALIAS  = 0x10u,
    SMP_IF_RAW      = 0x20u,

    /* Результат этой инструкции — временный буфер, который читает ТОЛЬКО
     * следующая инструкция, и она тоже поэлементная. Значит, писать его в
     * память незачем: обе стадии считаются за один проход.
     *
     * Утверждение исходит от компилятора — он один знает, что буфер больше
     * никем не читается. VM ничего не выводит сама: снимет флаг — получит
     * прежнее поведение, инструкция за инструкцией. Поэтому старый рантайм
     * исполнит новый модуль правильно, просто медленнее. */
    SMP_IF_FUSE     = 0x40u
};

SMP_INLINE uint8_t  smp_vec_code(uint32_t bits)
{
    return (uint8_t)(bits >= 512 ? 3u : bits >= 256 ? 2u : bits >= 128 ? 1u : 0u);
}
SMP_INLINE uint32_t smp_vec_bits(uint8_t code)
{
    static const uint32_t t[4] = { 64u, 128u, 256u, 512u };
    return t[code & 3u];
}

typedef struct SmpInstr {
    uint8_t  op;
    uint8_t  flags;
    uint8_t  d;
    uint8_t  a;
    uint8_t  b;

    /* Смысл зависит от опкода. Там, где инструкция читает пул констант, здесь
     * лежит SmpDType этой константы: пул хранит сырые 8 байт, и без пометки
     * читатель не знает, целое там или double. Раньше не знал и читал всё
     * как double — целые литералы молча превращались в ноль. */
    uint8_t  aux;

    uint16_t k;      /* индекс в таблице дескрипторов или в пуле констант */
} SmpInstr;

SMP_STATIC_ASSERT(sizeof(SmpInstr) == 8, instruction_is_8_bytes);

#define SMP_MAX_REGS 256u

/* ========================================================================== */
/*  Константа пула                                                            */
/* ========================================================================== */

typedef union SmpConst {
    uint64_t u;
    int64_t  i;
    double   f;
} SmpConst;

SMP_STATIC_ASSERT(sizeof(SmpConst) == 8, const_slot_is_8_bytes);

/* ========================================================================== */
/*  Форматы вывода                                                            */
/* ========================================================================== */

/* Вариант операции emit. Едет в поле aux — новых полей инструкция не растит.
 * Формат выбирается на компиляции: рантайм ничего не разбирает и не гадает. */
typedef enum SmpEmitFmt {
    SMP_EMIT_TEXT = 1,   /* кодовые точки как UTF-8, без разделителей */
    SMP_EMIT_LINE,       /* то же плюс перевод строки                 */
    SMP_EMIT_DEC,        /* целые через пробел, в конце перевод строки*/
    SMP_EMIT_HEX,
    SMP_EMIT_BITS,
    SMP_EMIT_NUM,        /* вещественные                              */
    SMP_EMIT__COUNT
} SmpEmitFmt;

const char *smp_emit_fmt_name(uint8_t f);

/* Значение константы как double, с учётом её типа из поля aux. */
SMP_INLINE double smp_const_as_double(SmpConst c, uint8_t dtype)
{
    switch (dtype) {
        case SMP_DT_F32:
        case SMP_DT_F64: return c.f;
        case SMP_DT_I32: return (double)c.i;
        case SMP_DT_U64: return (double)c.u;
        default:         return c.f;
    }
}

/* ========================================================================== */
/*  Файл .s3b                                                                 */
/* ========================================================================== */

#define SMP_S3B_MAGIC0 0x53u  /* 'S' */
#define SMP_S3B_MAGIC1 0x4Du  /* 'M' */
#define SMP_S3B_MAGIC2 0x50u  /* 'P' */
#define SMP_S3B_MAGIC3 0x33u  /* '3' */

#define SMP_S3B_VER_MAJOR 0u
#define SMP_S3B_VER_MINOR 2u

enum SmpS3bFlags {
    SMP_S3B_LITTLE_ENDIAN = 1u << 0,
    SMP_S3B_HAS_DEBUG     = 1u << 1
};

typedef struct SmpS3bHeader {
    uint8_t  magic[4];      /*  0: 53 4D 50 33                               */
    uint16_t ver_major;     /*  4                                            */
    uint16_t ver_minor;     /*  6                                            */
    uint32_t flags;         /*  8                                            */
    uint32_t n_sections;    /* 12                                            */
    uint32_t total_size;    /* 16: весь файл, байт                           */
    uint32_t checksum;      /* 20: FNV-1a 32 по всему, что после заголовка   */
    uint32_t reserved[2];   /* 24                                            */
} SmpS3bHeader;

SMP_STATIC_ASSERT(sizeof(SmpS3bHeader) == 32, s3b_header_is_32_bytes);

typedef struct SmpS3bSection {
    uint8_t  tag[4];        /* "AREN", "TENS", "CONS", "CODE", "STRS", "DBGL" */
    uint32_t offset;        /* от начала файла                                */
    uint32_t size;          /* байт                                           */
    uint32_t count;         /* элементов                                      */
} SmpS3bSection;

SMP_STATIC_ASSERT(sizeof(SmpS3bSection) == 16, s3b_section_is_16_bytes);

/* Строка отладочной таблицы: какая инструкция из какого места исходника. */
typedef struct SmpDbgLine {
    uint32_t instr;         /* индекс инструкции в CODE                      */
    uint32_t line;
    uint32_t col;
} SmpDbgLine;

SMP_STATIC_ASSERT(sizeof(SmpDbgLine) == 12, dbgline_is_12_bytes);

/* --- Загруженный модуль --------------------------------------------------- */
typedef struct SmpModule {
    const SmpInstr   *code;
    uint32_t          n_code;

    const SmpTensor  *tens;
    uint32_t          n_tens;

    const SmpConst   *consts;
    uint32_t          n_consts;

    const uint64_t   *arena_bytes;
    uint32_t          n_arenas;

    const char       *strs;
    uint32_t          strs_size;

    const SmpDbgLine *dbg;
    uint32_t          n_dbg;

    /* Имена именованных регистров: смещения в STRS по номеру регистра.
     * Нужны и дизассемблеру, и дампу при фатальной ошибке — без них падение
     * сообщает «r7», а пользователь писал «$total». */
    const uint32_t   *reg_names;
    uint32_t          n_reg_names;

    uint32_t          n_regs;   /* сколько регистров реально нужно */
} SmpModule;

/* Имя из строковой таблицы; при выходе за границы — "<нет>". */
const char *smp_module_str(const SmpModule *m, uint32_t off);

/* Позиция в исходнике для инструкции; {0,0,0} если отладочных данных нет. */
SmpDbgLine  smp_module_dbg(const SmpModule *m, uint32_t instr);

#endif /* SMPC3_BYTECODE_H */
