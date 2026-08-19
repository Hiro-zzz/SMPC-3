/* SMPC3 :: tests/harness.h -- микро-обвязка для тестов.
 * Каждый тест — отдельный бинарь, поэтому статические счётчики в заголовке
 * никому не мешают. */
#ifndef SMPC3_TEST_HARNESS_H
#define SMPC3_TEST_HARNESS_H

#include <stdio.h>

static unsigned g_pass = 0, g_fail = 0;

#define CHECK(cond, ...)                                                    \
    do {                                                                    \
        if (cond) { g_pass++; }                                             \
        else {                                                              \
            g_fail++;                                                       \
            fprintf(stderr, "  FAIL %s:%d  ", __FILE__, __LINE__);          \
            fprintf(stderr, __VA_ARGS__);                                   \
            fputc('\n', stderr);                                            \
        }                                                                   \
    } while (0)

#define SECTION(name) fprintf(stderr, "-- %s\n", name)

#define REPORT()                                                            \
    (fprintf(stderr, "\n%u прошло, %u провалено\n", g_pass, g_fail),         \
     g_fail ? 1 : 0)

#endif /* SMPC3_TEST_HARNESS_H */
