/* SMPC3 :: common.h -- базовые макросы, атрибуты, выравнивание.
 * SuperMegaPlexCalc3000. C99. Zero-dependency. */
#ifndef SMPC3_COMMON_H
#define SMPC3_COMMON_H

#include <stddef.h>
#include <stdint.h>
#include <stdarg.h>

#ifndef __cplusplus
#  include <stdbool.h>
#endif

#define SMPC3_VERSION_MAJOR 0
#define SMPC3_VERSION_MINOR 1
#define SMPC3_VERSION_PATCH 0
#define SMPC3_VERSION_STR   "0.1.0"

/* --- Геометрия памяти ----------------------------------------------------- */
#define SMP_CACHELINE      64u   /* обязательное выравнивание арены          */
#define SMP_ALIGN_V256     32u
#define SMP_ALIGN_V512     64u
#define SMP_ARENA_ALIGN    SMP_CACHELINE

#define SMP_ALIGN_UP(x, a)    (((size_t)(x) + ((size_t)(a) - 1u)) & ~((size_t)(a) - 1u))
#define SMP_ALIGN_DOWN(x, a)  ((size_t)(x) & ~((size_t)(a) - 1u))
#define SMP_IS_POW2(a)        ((a) != 0u && (((a) & ((a) - 1u)) == 0u))
#define SMP_IS_ALIGNED(p, a)  ((((uintptr_t)(p)) & ((uintptr_t)(a) - 1u)) == 0u)

/* Сколько независимых арен знает язык: [#arena:0] .. [#arena:7]. Значение
 * общее для компилятора и рантайма — и вдобавок ровно укладывается в три
 * старших бита flags дескриптора (см. types.h). */
#define SMP_MAX_ARENAS 8u

#define SMP_ARRLEN(a)  (sizeof(a) / sizeof((a)[0]))
#define SMP_MIN(a, b)  ((a) < (b) ? (a) : (b))
#define SMP_MAX(a, b)  ((a) > (b) ? (a) : (b))
#define SMP_UNUSED(x)  ((void)(x))

/* --- Атрибуты компилятора ------------------------------------------------- */
#if defined(__GNUC__) || defined(__clang__)
#  define SMP_LIKELY(x)     __builtin_expect(!!(x), 1)
#  define SMP_UNLIKELY(x)   __builtin_expect(!!(x), 0)
#  define SMP_NORETURN      __attribute__((noreturn))
#  define SMP_ALIGNED(n)    __attribute__((aligned(n)))
#  define SMP_INLINE        static inline __attribute__((always_inline))
#  define SMP_NOINLINE      __attribute__((noinline))
#  define SMP_HOT           __attribute__((hot))
#  define SMP_COLD          __attribute__((cold))
#  define SMP_PURE          __attribute__((pure))
#  define SMP_PRINTF(f, a)  __attribute__((format(printf, f, a)))
#  define SMP_RESTRICT      __restrict__
#  define SMP_UNREACHABLE() __builtin_unreachable()
#  define SMP_ASSUME_ALIGNED(p, a) __builtin_assume_aligned((p), (a))
#  define SMP_TRAP()        __builtin_trap()
#else
#  define SMP_LIKELY(x)     (x)
#  define SMP_UNLIKELY(x)   (x)
#  define SMP_NORETURN      __declspec(noreturn)
#  define SMP_ALIGNED(n)    __declspec(align(n))
#  define SMP_INLINE        static __forceinline
#  define SMP_NOINLINE      __declspec(noinline)
#  define SMP_HOT
#  define SMP_COLD
#  define SMP_PURE
#  define SMP_PRINTF(f, a)
#  define SMP_RESTRICT      __restrict
#  define SMP_UNREACHABLE() __assume(0)
#  define SMP_ASSUME_ALIGNED(p, a) (p)
#  define SMP_TRAP()        __debugbreak()
#endif

/* Прямая диспетчеризация возможна только там, где есть computed goto.
 * Без неё VM собирается в режиме switch-fallback (медленнее, но корректно). */
#if defined(__GNUC__) || defined(__clang__)
#  define SMP_HAS_COMPUTED_GOTO 1
#else
#  define SMP_HAS_COMPUTED_GOTO 0
#endif

/* --- Статические проверки (C99, без static_assert) ------------------------ */
#define SMP__CAT2(a, b) a##b
#define SMP__CAT(a, b)  SMP__CAT2(a, b)
#define SMP_STATIC_ASSERT(cond, tag) \
    typedef char SMP__CAT(smp_static_assert_##tag##_, __LINE__)[(cond) ? 1 : -1]

/* --- Результаты ----------------------------------------------------------- */
typedef enum SmpStatus {
    SMP_OK = 0,
    SMP_ERR_OOM,          /* арена исчерпана                                 */
    SMP_ERR_ALIGN,        /* нарушено выравнивание                           */
    SMP_ERR_IO,
    SMP_ERR_SYNTAX,
    SMP_ERR_TYPE,
    SMP_ERR_SHAPE,
    SMP_ERR_ISA,          /* инструкция не поддержана железом                */
    SMP_ERR_INTERNAL
} SmpStatus;

const char *smp_status_str(SmpStatus s);

#endif /* SMPC3_COMMON_H */
