/* SMPC3 :: test_core.c -- проверки Ф0: арена, типы, диагностика. */
#include "smpc3/arena.h"
#include "smpc3/cpu.h"
#include "smpc3/diag.h"
#include "smpc3/types.h"

#include "harness.h"

#include <stdio.h>
#include <string.h>

/* ========================================================================== */

static void test_arena(void)
{
    SECTION("arena");

    SmpArena a;
    CHECK(smp_arena_init(&a, 1u << 20, 3, "test") == SMP_OK, "init провалился");
    CHECK(SMP_IS_ALIGNED(a.base, 64), "база арены не выровнена на 64");
    CHECK(a.used == 0, "свежая арена не пуста: used=%zu", a.used);

    /* Каждая выдача выровнена минимум на 64, даже если попросили меньше. */
    void *p1 = smp_arena_push(&a, 1, 1);
    void *p2 = smp_arena_push(&a, 1, 1);
    CHECK(p1 && p2, "push вернул NULL на пустой арене");
    CHECK(SMP_IS_ALIGNED(p1, 64) && SMP_IS_ALIGNED(p2, 64), "выдача не выровнена");
    CHECK(p2 != p1, "две выдачи совпали");

    /* push_zero действительно обнуляет (важно: poison-заливка не должна течь). */
    unsigned char *z = (unsigned char *)smp_arena_push_zero(&a, 256, 64);
    CHECK(z != NULL, "push_zero вернул NULL");
    if (z) {
        int nonzero = 0;
        for (int i = 0; i < 256; i++) if (z[i]) nonzero++;
        CHECK(nonzero == 0, "push_zero оставил %d ненулевых байт", nonzero);
    }

    /* Откат к отметке. */
    SmpArenaMark m = smp_arena_mark(&a);
    void *p3 = smp_arena_push(&a, 4096, 64);
    CHECK(p3 != NULL, "push 4 KiB провалился");
    smp_arena_rewind(&a, m);
    CHECK(a.used == m, "rewind не вернул used: %zu != %zu", a.used, m);
    void *p4 = smp_arena_push(&a, 4096, 64);
    CHECK(p4 == p3, "после rewind адрес не переиспользован");

    /* peak переживает откат. */
    CHECK(a.peak >= m + 4096, "peak занижен: %zu", a.peak);

    /* Исчерпание — NULL, а не UB и не тихое расширение. */
    void *huge = smp_arena_push(&a, 1u << 30, 64);
    CHECK(huge == NULL, "арена выдала больше своей ёмкости");

    /* Невалидное выравнивание отвергается. */
    CHECK(smp_arena_push(&a, 16, 48) == NULL, "принято выравнивание не степени двойки");

    CHECK(smp_arena_owns(&a, p1), "owns не признал собственный указатель");
    CHECK(!smp_arena_owns(&a, (void *)&a), "owns признал чужой указатель");

    smp_arena_reset(&a);
    CHECK(a.used == 0, "reset не обнулил used");

    smp_arena_release(&a);
    CHECK(a.base == NULL, "release не затёр структуру");
}

/* ========================================================================== */

static void test_types(void)
{
    SECTION("types");

    CHECK(sizeof(SmpTensor) == 32, "дескриптор %zu байт вместо 32", sizeof(SmpTensor));

    CHECK(smp_dtype_size(SMP_DT_F32) == 4, "f32 не 4 байта");
    CHECK(smp_dtype_size(SMP_DT_F64) == 8, "f64 не 8 байт");
    CHECK(smp_dtype_size(SMP_DT_U64) == 8, "u64 не 8 байт");
    CHECK(smp_dtype_parse("f32", 3) == SMP_DT_F32, "разбор f32");
    CHECK(smp_dtype_parse("raw_ptr", 7) == SMP_DT_RAW_PTR, "разбор raw_ptr");
    CHECK(smp_dtype_parse("f16", 3) == SMP_DT_INVALID, "f16 не должен разбираться");
    CHECK(smp_dtype_parse("f3", 2) == SMP_DT_INVALID, "префикс не должен совпадать");
    CHECK(smp_dtype_is_float(SMP_DT_F64) && !smp_dtype_is_float(SMP_DT_I32), "is_float");

    CHECK(smp_dtype_lanes(SMP_DT_F32, 256) == 8, "f32 дорожек в v256");
    CHECK(smp_dtype_lanes(SMP_DT_F64, 512) == 8, "f64 дорожек в v512");

    uint16_t shape[3] = { 2, 3, 4 };
    SmpTensor t;
    smp_tensor_dense(&t, SMP_DT_F32, 3, shape);

    CHECK(t.rank == 3, "ранг %u", t.rank);
    CHECK(t.nelem == 24, "nelem %u", t.nelem);
    CHECK(smp_tensor_bytes(&t) == 96, "bytes %llu", (unsigned long long)smp_tensor_bytes(&t));
    /* row-major: последняя ось самая быстрая */
    CHECK(t.stride[0] == 12 && t.stride[1] == 4 && t.stride[2] == 1,
          "strides %u,%u,%u", t.stride[0], t.stride[1], t.stride[2]);
    CHECK(smp_tensor_is_contiguous(&t), "плотный тензор не признан плотным");

    /* Срез строки: шаг по ведущей оси ломает плотность. */
    SmpTensor v = t;
    v.rank = 1; v.shape[0] = 4; v.stride[0] = 3;
    CHECK(!smp_tensor_is_contiguous(&v), "разреженный срез признан плотным");

    SmpTensor u;
    smp_tensor_dense(&u, SMP_DT_F32, 3, shape);
    CHECK(smp_tensor_same_shape(&t, &u), "одинаковые формы не совпали");
    u.shape[1] = 9;
    CHECK(!smp_tensor_same_shape(&t, &u), "разные формы совпали");

    char sig[64];
    smp_tensor_sig(&t, sig, sizeof sig);
    CHECK(strcmp(sig, "f32:2,3,4") == 0, "сигнатура '%s'", sig);
}

/* ========================================================================== */

static void test_cpu(void)
{
    SECTION("cpu");

    const SmpCpu *c = smp_cpu();
    CHECK(c == smp_cpu(), "детект не кэшируется");
    CHECK(c->n_logical >= 1, "n_logical = %u", c->n_logical);
    CHECK(c->max_vec_bits >= 128, "max_vec_bits = %u", c->max_vec_bits);
    CHECK(c->brand[0] != '\0', "brand пуст");

    /* Иерархия расширений обязана быть непротиворечивой. */
    if (c->isa & SMP_ISA_AVX2)     CHECK(c->isa & SMP_ISA_AVX,  "AVX2 без AVX");
    if (c->isa & SMP_ISA_AVX512F)  CHECK(c->isa & SMP_ISA_AVX2, "AVX512F без AVX2");
    if (c->isa & SMP_ISA_AVX512F)  CHECK(c->max_vec_bits == 512, "AVX512F, но vec=%u", c->max_vec_bits);
    else                           CHECK(c->max_vec_bits != 512, "нет AVX512F, но vec=512");

    /* FTZ/DAZ переключаются и восстанавливаются. */
    uint32_t saved = smp_fpu_get_mxcsr();
    smp_fpu_set_ftz(true);
    CHECK((smp_fpu_get_mxcsr() & ((1u << 15) | (1u << 6))) == ((1u << 15) | (1u << 6)),
          "FTZ/DAZ не выставились");
    smp_fpu_set_ftz(false);
    CHECK((smp_fpu_get_mxcsr() & ((1u << 15) | (1u << 6))) == 0, "FTZ/DAZ не снялись");
    smp_fpu_set_mxcsr(saved);
    CHECK(smp_fpu_get_mxcsr() == saved, "MXCSR не восстановлен");
}

/* ========================================================================== */

static const char g_src[] =
    "alpha\n"
    "[#simd:v256]  $r1 -> @mmul($r2) => *&C;\n"
    "gamma\n";

static void test_diag(void)
{
    SECTION("diag");

    /* Реестр консистентен: коды уникальны и находятся по тексту. */
    for (unsigned i = 0; i < SMP_DIAG__COUNT; i++) {
        const SmpDiagInfo *inf = smp_diag_info((SmpDiagCode)i);
        CHECK(inf->text && inf->title && inf->fix, "код #%u неполон", i);
        CHECK(smp_diag_lookup(inf->text) == (SmpDiagCode)i, "поиск '%s'", inf->text);
        CHECK((unsigned)inf->cat < SMP_CAT__COUNT, "категория кода '%s'", inf->text);
    }
    CHECK(smp_diag_lookup("E9999") == SMP_DIAG__COUNT, "несуществующий код найден");
    CHECK(smp_diag_info(SMP_E0418)->sev == SMP_SEV_FATAL, "E0418 должен быть FATAL");
    CHECK(smp_diag_info(SMP_W0512)->sev == SMP_SEV_WARN,  "W0512 должен быть WARN");

    SmpSource src = { "t.smpc", g_src, sizeof(g_src) - 1u };

    /* Пересчёт байтового смещения в строку/колонку. */
    SmpSpan sp = smp_span_from_offset(&src, 6, 1);       /* первый байт строки 2 */
    CHECK(sp.line == 2 && sp.col == 1, "offset->span = %u:%u", sp.line, sp.col);
    sp = smp_span_from_offset(&src, 0, 1);
    CHECK(sp.line == 1 && sp.col == 1, "offset 0 -> %u:%u", sp.line, sp.col);

    /* Вывод в файл: цвет выключается, содержимое проверяемо. */
    FILE *f = tmpfile();
    CHECK(f != NULL, "tmpfile недоступен");
    if (!f) return;

    SmpDiagCtx D;
    smp_diag_init(&D, &src, f);
    CHECK(!D.color, "цвет включён для не-терминала");
    CHECK(D.deterministic, "по умолчанию должен быть детерминированный режим");

    smp_diag_emit(&D, &(SmpDiagMsg){
        .code = SMP_E0419, .span = (SmpSpan){ 2, 15, 4 },
        .details = smp_fmt(&D, "K = %u против %u", 1024u, 512u)
    });
    CHECK(D.n_fatal == 1 && D.n_warn == 0, "счётчики: fatal=%u warn=%u", D.n_fatal, D.n_warn);

    smp_diag_emit(&D, &(SmpDiagMsg){ .code = SMP_W0512, .span = (SmpSpan){ 2, 2, 10 } });
    CHECK(D.n_fatal == 1 && D.n_warn == 1, "счётчики после warn");

    /* Тот же код в том же месте обязан дать тот же ДИАГНОЗ. */
    long end = ftell(f);
    rewind(f);
    static char out[8192];
    size_t n = fread(out, 1, sizeof(out) - 1, f);
    out[n] = '\0';
    fclose(f);
    CHECK(end > 0 && n > 0, "диагностика ничего не напечатала");

    CHECK(strstr(out, "[FATAL SKILL ISSUE :: E0419]") != NULL, "нет шапки FATAL");
    CHECK(strstr(out, "[SKILL ISSUE :: W0512]") != NULL, "нет шапки WARN");
    CHECK(strstr(out, "t.smpc:2:15") != NULL, "нет позиции");
    CHECK(strstr(out, "ОШИБКА") && strstr(out, "ДЕТАЛИ") &&
          strstr(out, "ДИАГНОЗ") && strstr(out, "ИСПРАВЛЕНИЕ"), "нет всех полей");
    CHECK(strstr(out, "K = 1024 против 512") != NULL, "smp_fmt не подставился");
    CHECK(strstr(out, "\x1b[") == NULL, "ANSI-коды просочились в файл");
    /* Исходная строка процитирована целиком. */
    CHECK(strstr(out, "$r1 -> @mmul($r2) => *&C;") != NULL, "нет цитаты исходника");
    /* Каретка стоит под колонкой 15 (1-based) -> 14 пробелов после "| ". */
    CHECK(strstr(out, "|               ^~~~") != NULL, "каретка не на месте");

    /* Кольцо smp_fmt: слоты переиспользуются, но не рвут строку. */
    SmpDiagCtx D2;
    smp_diag_init(&D2, &src, stderr);
    const char *first = smp_fmt(&D2, "slot-%d", 0);
    CHECK(strcmp(first, "slot-0") == 0, "smp_fmt вернул '%s'", first);
    /* Слот 0 переживает ровно SMP_FMT_SLOTS-1 последующих вызовов... */
    for (unsigned i = 1; i < SMP_FMT_SLOTS; i++) (void)smp_fmt(&D2, "slot-%u", i);
    CHECK(strcmp(first, "slot-0") == 0, "слот затёрт раньше срока: '%s'", first);
    /* ...и затирается на следующем. */
    (void)smp_fmt(&D2, "slot-%u", SMP_FMT_SLOTS);
    CHECK(strcmp(first, "slot-0") != 0, "кольцо не провернулось за %u слотов", SMP_FMT_SLOTS);

    char sum[128];
    smp_diag_summary(&D, sum, sizeof sum);
    CHECK(strcmp(sum, "1 фатальная ошибка, 1 предупреждение") == 0, "итог: '%s'", sum);

    SmpDiagCtx D3;
    smp_diag_init(&D3, &src, stderr);
    smp_diag_summary(&D3, sum, sizeof sum);
    CHECK(strcmp(sum, "замечаний нет") == 0, "пустой итог: '%s'", sum);

    /* Русские числительные. */
    D3.n_fatal = 2;  smp_diag_summary(&D3, sum, sizeof sum);
    CHECK(strstr(sum, "2 фатальные ошибки") != NULL, "склонение 2: '%s'", sum);
    D3.n_fatal = 5;  smp_diag_summary(&D3, sum, sizeof sum);
    CHECK(strstr(sum, "5 фатальных ошибок") != NULL, "склонение 5: '%s'", sum);
    D3.n_fatal = 11; smp_diag_summary(&D3, sum, sizeof sum);
    CHECK(strstr(sum, "11 фатальных ошибок") != NULL, "склонение 11: '%s'", sum);
    D3.n_fatal = 21; smp_diag_summary(&D3, sum, sizeof sum);
    CHECK(strstr(sum, "21 фатальная ошибка") != NULL, "склонение 21: '%s'", sum);
}

/* ========================================================================== */

/* ========================================================================== */
/*  Пулы диагнозов                                                            */
/* ========================================================================== */

static void test_diag_pools(void)
{
    SECTION("пулы диагнозов");

    static const char *const cat_name[SMP_CAT__COUNT] = {
        "LEX", "PARSE", "TYPE", "MEM", "SIMD", "RUNTIME", "INTERNAL"
    };

    unsigned total = 0;

    for (unsigned c = 0; c < SMP_CAT__COUNT; c++) {
        const unsigned n = smp_diag_pool_size((SmpDiagCat)c);
        CHECK(n >= 2u, "%s: диагнозов всего %u", cat_name[c], n);
        total += n;

        for (unsigned i = 0; i < n; i++) {
            const char *a = smp_diag_pool_at((SmpDiagCat)c, i);
            CHECK(a != NULL && a[0] != '\0', "%s[%u]: пустой диагноз", cat_name[c], i);
            if (!a) continue;

            /* Дубликат внутри пула молча сокращает разнообразие: выбор идёт по
             * остатку от деления, и две одинаковые строки просто съедают долю. */
            for (unsigned j = i + 1u; j < n; j++) {
                const char *b = smp_diag_pool_at((SmpDiagCat)c, j);
                CHECK(b && strcmp(a, b) != 0,
                      "%s: диагнозы %u и %u совпадают", cat_name[c], i, j);
            }
        }
    }
    CHECK(total >= 60u, "диагнозов на все категории всего %u", total);

    /* Выход за границы — NULL, а не чужая память. */
    CHECK(smp_diag_pool_at(SMP_CAT_LEX, smp_diag_pool_size(SMP_CAT_LEX)) == NULL,
          "индекс за пулом не отсечён");
    CHECK(smp_diag_pool_at((SmpDiagCat)SMP_CAT__COUNT, 0) == NULL,
          "несуществующая категория не отсечена");
    CHECK(smp_diag_pool_size((SmpDiagCat)SMP_CAT__COUNT) == 0u,
          "размер несуществующей категории не ноль");

    /* Пул должен реально перебираться, а не отдавать одну строку на всё.
     * Один код, разные позиции в исходнике — диагнозы обязаны различаться. */
    SmpSource src = { "t.smpc", "x", 1 };
    SmpDiagCtx D;
    FILE *f = tmpfile();
    if (!f) { CHECK(0, "tmpfile недоступен"); return; }

    smp_diag_init(&D, &src, f);
    D.deterministic = true;

    /* Строки копируются, а не запоминаются указателем: буфер чтения один на
     * все итерации, и указатель в него после следующего прогона показывает уже
     * не то, что запомнили. */
    static char seen[64][48];
    unsigned    nseen = 0;
    for (uint32_t line = 1; line <= 60u && nseen < SMP_ARRLEN(seen); line++) {
        rewind(f);
        smp_diag_emit(&D, &(SmpDiagMsg){
            .code = SMP_E0421, .span = (SmpSpan){ line, 1, 1 }
        });

        static char buf[4096];
        const long end = ftell(f);
        rewind(f);
        const size_t got = fread(buf, 1, sizeof(buf) - 1u, f);
        buf[got] = '\0';
        if (end <= 0 || got == 0) continue;

        const char *d = strstr(buf, "ДИАГНОЗ: ");
        if (!d) continue;
        d += strlen("ДИАГНОЗ: ");

        bool known = false;
        for (unsigned i = 0; i < nseen; i++)
            if (strncmp(seen[i], d, sizeof(seen[0]) - 1u) == 0) known = true;
        if (!known && nseen < SMP_ARRLEN(seen)) {
            memcpy(seen[nseen], d, sizeof(seen[0]) - 1u);
            seen[nseen][sizeof(seen[0]) - 1u] = '\0';
            nseen++;
        }
    }
    fclose(f);

    /* E0421 — категория MEM, своего диагноза у него нет. Хотя бы половина пула
     * обязана встретиться на шестидесяти позициях, иначе выбор перекошен. */
    CHECK(nseen * 2u >= smp_diag_pool_size(SMP_CAT_MEM),
          "на 60 позиций выпало лишь %u диагнозов из %u",
          nseen, smp_diag_pool_size(SMP_CAT_MEM));
}

static void test_diag_own(void)
{
    SECTION("собственный диагноз кода");

    unsigned n_own = 0, n_pool = 0;
    for (unsigned i = 0; i < SMP_DIAG__COUNT; i++) {
        const SmpDiagInfo *inf = smp_diag_info((SmpDiagCode)i);
        CHECK(inf->title && inf->title[0], "код %u без строки ОШИБКА", i);
        CHECK(inf->fix && inf->fix[0], "код %u без строки ИСПРАВЛЕНИЕ", i);
        if (inf->diagnosis) {
            CHECK(inf->diagnosis[0] != '\0', "код %s: пустой свой диагноз", inf->text);
            n_own++;
        } else {
            n_pool++;
        }
    }
    CHECK(n_own >= 5u, "кодов со своим диагнозом всего %u", n_own);
    CHECK(n_pool > n_own, "своих диагнозов больше, чем взятых из пула");

    /* Свой диагноз перебивает пул и не зависит от позиции в исходнике: он на
     * то и свой, чтобы говорить именно про эту беду, а не про категорию. */
    SmpSource src = { "t.smpc", "x", 1 };
    SmpDiagCtx D;
    FILE *f = tmpfile();
    if (!f) { CHECK(0, "tmpfile недоступен"); return; }

    smp_diag_init(&D, &src, f);
    D.deterministic = true;

    const SmpDiagInfo *inf = smp_diag_info(SMP_E0419);
    CHECK(inf->diagnosis != NULL, "у E0419 нет своего диагноза");

    if (inf->diagnosis) {
        for (uint32_t line = 1; line <= 20u; line++) {
            rewind(f);
            smp_diag_emit(&D, &(SmpDiagMsg){
                .code = SMP_E0419, .span = (SmpSpan){ line, 1, 1 }
            });

            static char b2[4096];
            const long end = ftell(f);
            rewind(f);
            const size_t got = fread(b2, 1, sizeof(b2) - 1u, f);
            b2[got] = '\0';
            if (end <= 0 || got == 0) { CHECK(0, "нет вывода на строке %u", line); break; }

            CHECK(strstr(b2, inf->diagnosis) != NULL,
                  "на строке %u свой диагноз подменён пулом", line);
        }
    }
    fclose(f);
}


static void test_diag_mute(void)
{
    SECTION("заглушённые коды");

    SmpSource src = { "t.smpc", g_src, sizeof(g_src) - 1u };
    FILE *f = tmpfile();
    CHECK(f != NULL, "tmpfile недоступен");
    if (!f) return;

    SmpDiagCtx D;
    smp_diag_init(&D, &src, f);
    CHECK(smp_diag_mute(&D, SMP_W0301), "предупреждение заглушить можно");
    CHECK(!smp_diag_mute(&D, SMP_E0419), "фатальную ошибку заглушить нельзя");

    smp_diag_emit(&D, &(SmpDiagMsg){ .code = SMP_W0301, .span = (SmpSpan){ 1, 1, 1 } });
    CHECK(D.n_warn == 0, "заглушённое предупреждение посчитано: %u", D.n_warn);
    CHECK(ftell(f) == 0, "заглушённое предупреждение напечатано");

    /* Соседний код не задет, а фатальная ошибка проходит как была. */
    smp_diag_emit(&D, &(SmpDiagMsg){ .code = SMP_W0512, .span = (SmpSpan){ 1, 1, 1 } });
    smp_diag_emit(&D, &(SmpDiagMsg){ .code = SMP_E0419, .span = (SmpSpan){ 1, 1, 1 } });
    CHECK(D.n_warn == 1 && D.n_fatal == 1, "warn=%u fatal=%u", D.n_warn, D.n_fatal);
    CHECK(ftell(f) > 0, "незаглушённое не напечатано");

    /* Новый контекст ничего не помнит. */
    smp_diag_init(&D, &src, f);
    smp_diag_emit(&D, &(SmpDiagMsg){ .code = SMP_W0301, .span = (SmpSpan){ 1, 1, 1 } });
    CHECK(D.n_warn == 1, "после init W0301 всё ещё заглушён");
    fclose(f);
}

int main(void)
{
    smp_console_setup();
    fprintf(stderr, "SMPC3 tests :: Ф0\n\n");

    test_arena();
    test_types();
    test_cpu();
    test_diag();
    test_diag_pools();
    test_diag_own();
    test_diag_mute();

    return REPORT();
}
