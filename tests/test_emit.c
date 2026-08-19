/* SMPC3 :: test_emit.c -- проверки Ф4: эмиттер, контейнер .s3b, дизассемблер. */
#include "smpc3/emit.h"
#include "smpc3/io.h"
#include "smpc3/lex.h"
#include "smpc3/parse.h"
#include "smpc3/sema.h"

#include "harness.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ========================================================================== */
/*  Обвязка                                                                   */
/* ========================================================================== */

static SmpArena   g_arena;
static SmpDiagCtx g_diag;
static SmpSource  g_src;
static FILE      *g_sink;
static char       g_out[16384];

#define TMP_S3B "build/test_emit_tmp.s3b"

/* Полный проход до модуля. Возвращает true, если дошли без ошибок. */
static bool build_str(const char *text, SmpModule *mod)
{
    smp_arena_reset(&g_arena);
    rewind(g_sink);
    g_out[0] = '\0';
    memset(mod, 0, sizeof *mod);

    smp_source_from_memory(&g_src, "t.smpc", text, strlen(text));
    smp_diag_init(&g_diag, &g_src, g_sink);

    SmpLexer lx;
    smp_lex_init(&lx, &g_src, &g_diag);
    SmpToken *toks = NULL;
    uint32_t  ntok = 0;
    smp_lex_all(&lx, &g_arena, &toks, &ntok);
    if (lx.n_errors) { CHECK(0, "лексер: %s", text); return false; }

    SmpParser P;
    smp_parse_init(&P, &g_arena, &g_diag, &g_src, toks, ntok);
    SmpAstProgram prog;
    smp_parse(&P, &prog);
    if (P.n_errors) { CHECK(0, "парсер: %s", text); return false; }

    SmpSema sm;
    smp_sema_init(&sm, &g_arena, &g_diag, &g_src);
    sm.host_vec_bits = 256;
    SmpSemaResult res;
    smp_sema_run(&sm, &prog, &res);
    if (sm.n_errors) { CHECK(0, "семантика: %s", text); return false; }

    SmpEmitter em;
    smp_emit_init(&em, &g_arena, &g_diag, &g_src);
    const SmpStatus st = smp_emit(&em, &prog, &res, mod);

    rewind(g_sink);
    const size_t n = fread(g_out, 1, sizeof(g_out) - 1u, g_sink);
    g_out[n] = '\0';

    CHECK(st == SMP_OK && em.n_errors == 0, "эмиттер: %s", text);
    return st == SMP_OK && em.n_errors == 0;
}

/* Сколько инструкций данного опкода в модуле. */
static uint32_t count_op(const SmpModule *m, SmpOpcode op)
{
    uint32_t n = 0;
    for (uint32_t i = 0; i < m->n_code; i++) if (m->code[i].op == op) n++;
    return n;
}

/* Структурные инварианты, обязательные для любого модуля. */
static void check_wellformed(const SmpModule *m, const char *label)
{
    CHECK(m->n_code > 0, "%s: пустой код", label);
    if (!m->n_code) return;

    CHECK(m->code[m->n_code - 1u].op == SMP_BC_HALT,
          "%s: последняя инструкция не halt", label);

    for (uint32_t i = 0; i < m->n_code; i++) {
        const SmpInstr *in = &m->code[i];
        CHECK(in->op < SMP_BC__COUNT, "%s: инстр. %u опкод %u", label, i, in->op);
        const SmpOpFmt f = smp_opcode_def((SmpOpcode)in->op)->fmt;

        /* aux осмыслен только там, где инструкция читает пул констант; в
         * остальных случаях он обязан быть нулём, а не мусором. */
        const bool uses_aux = (f == SMP_FMT_D_K || f == SMP_FMT_D_A_K ||
                               f == SMP_FMT_D_A_B_K || f == SMP_FMT_D_A_X);
        if (uses_aux)
            CHECK(in->aux != 0, "%s: инстр. %u читает пул, но не помечен типом",
                  label, i);
        else
            CHECK(in->aux == 0, "%s: инстр. %u мусор в aux", label, i);

        if (f == SMP_FMT_D_T || f == SMP_FMT_T_A)
            CHECK(in->k < m->n_tens, "%s: инстр. %u -> дескриптор %u из %u",
                  label, i, in->k, m->n_tens);
        if (f == SMP_FMT_D_K || f == SMP_FMT_D_A_K || f == SMP_FMT_D_A_B_K)
            CHECK(in->k < m->n_consts, "%s: инстр. %u -> константа %u из %u",
                  label, i, in->k, m->n_consts);

        CHECK(in->d < m->n_regs && in->a < m->n_regs && in->b < m->n_regs,
              "%s: инстр. %u выходит за %u регистров", label, i, m->n_regs);
    }

    /* Каждый дескриптор обязан целиком лежать внутри своей арены. */
    for (uint32_t i = 0; i < m->n_tens; i++) {
        const SmpTensor *t = &m->tens[i];
        const uint64_t end = (uint64_t)t->off + smp_tensor_bytes(t);
        bool fits = false;
        for (uint32_t a = 0; a < m->n_arenas; a++)
            if (end <= m->arena_bytes[a]) { fits = true; break; }
        CHECK(fits, "%s: дескриптор %u (%s) выходит за арену: конец %llu",
              label, i, smp_module_str(m, t->name_id), (unsigned long long)end);
        CHECK(t->name_id < m->strs_size, "%s: дескриптор %u -> имя вне таблицы", label, i);
    }

    /* Отладочная таблица обязана покрывать код. */
    for (uint32_t i = 0; i < m->n_dbg; i++)
        CHECK(m->dbg[i].instr < m->n_code, "%s: отладочная строка вне кода", label);
}

/* ========================================================================== */

static void test_shape(void)
{
    SECTION("форма модуля");

    SmpModule m;
    if (!build_str("*&A<f32:4,4> -> @alloc => $a;", &m)) return;

    check_wellformed(&m, "минимальный");
    CHECK(count_op(&m, SMP_BC_LOADT) == 1, "loadt %u", count_op(&m, SMP_BC_LOADT));
    CHECK(count_op(&m, SMP_BC_ALLOC) == 1, "alloc %u", count_op(&m, SMP_BC_ALLOC));
    CHECK(count_op(&m, SMP_BC_HALT)  == 1, "halt %u",  count_op(&m, SMP_BC_HALT));
    CHECK(m.n_arenas >= 1 && m.arena_bytes[0] >= 64, "арена %llu",
          m.n_arenas ? (unsigned long long)m.arena_bytes[0] : 0);
    CHECK(m.arena_bytes[0] % 64u == 0, "размер арены не кратен 64: %llu",
          (unsigned long long)m.arena_bytes[0]);

    /* Имя тензора доехало до строковой таблицы. */
    bool found_a = false;
    for (uint32_t i = 0; i < m.n_tens; i++)
        if (strcmp(smp_module_str(&m, m.tens[i].name_id), "A") == 0) found_a = true;
    CHECK(found_a, "имя 'A' не попало в строковую таблицу");
}

static void test_pipeline(void)
{
    SECTION("конвейер");

    SmpModule m;

    /* Пример из спецификации целиком. */
    if (build_str("[#arena:0]  *&A<f32:1024,1024>  ->  @alloc  =>  $r1;\n"
                  "[#arena:0]  *&B<f32:1024,1024>  ->  @alloc  =>  $r2;\n"
                  "[#simd:v256]  $r1 -> @mmul($r2) => *&C<f32:1024,1024> [!no-alias, ?strict];\n"
                  "[^raw]  *&C[0, ..] -> @relu -> @reduce.add => $f0 [~flush-to-zero];\n", &m)) {
        check_wellformed(&m, "пример спеки");
        CHECK(count_op(&m, SMP_BC_MMUL)   == 1, "нет mmul");
        CHECK(count_op(&m, SMP_BC_RELU)   == 1, "нет relu");
        CHECK(count_op(&m, SMP_BC_REDADD) == 1, "нет redadd");

        /* Флаги инструкции должны нести режимы своей строки. */
        bool saw_strict = false, saw_ftz = false, saw_raw = false, saw_na = false;
        for (uint32_t i = 0; i < m.n_code; i++) {
            const uint8_t f = m.code[i].flags;
            if (m.code[i].op == SMP_BC_MMUL) {
                saw_strict = (f & SMP_IF_STRICT) != 0;
                saw_na     = (f & SMP_IF_NOALIAS) != 0;
                CHECK(smp_vec_bits(f & SMP_IF_VEC_MASK) == 256,
                      "у mmul ширина %u", smp_vec_bits(f & SMP_IF_VEC_MASK));
            }
            if (m.code[i].op == SMP_BC_REDADD) {
                saw_ftz = (f & SMP_IF_FTZ) != 0;
                saw_raw = (f & SMP_IF_RAW) != 0;
            }
        }
        CHECK(saw_strict, "?strict не доехал до флагов mmul");
        CHECK(saw_na,     "!no-alias не доехал до флагов mmul");
        CHECK(saw_ftz,    "~flush-to-zero не доехал до флагов redadd");
        CHECK(saw_raw,    "^raw не доехал до флагов redadd");
    }

    /* Срез должен стать отдельным дескриптором с тем же базовым адресом. */
    if (build_str("*&A<f32:8,8> -> @alloc => $a;\n"
                  "*&A[1, ..] -> @relu => *&R<f32:8>;\n", &m)) {
        const SmpTensor *base = NULL, *view = NULL;
        for (uint32_t i = 0; i < m.n_tens; i++) {
            if (strcmp(smp_module_str(&m, m.tens[i].name_id), "A") != 0) continue;
            if (m.tens[i].rank == 2) base = &m.tens[i];
            if (m.tens[i].rank == 1) view = &m.tens[i];
        }
        CHECK(base && view, "срез не стал отдельным дескриптором");
        if (base && view) {
            CHECK(view->off == base->off + 8u * 4u,
                  "смещение среза %u, ждали %u", view->off, base->off + 32u);
            CHECK(view->shape[0] == 8 && view->stride[0] == 1, "форма среза");
        }
    }
}

static void test_direct_write(void)
{
    SECTION("запись напрямую в приёмник");

    SmpModule m;

    /* Приёмник не читается — копия в конце не нужна. */
    if (build_str("*&A<f32:4,4> -> @alloc => $a;\n"
                  "*&B<f32:4,4> -> @alloc => $b;\n"
                  "$a -> @mmul($b) => *&C<f32:4,4>;\n", &m)) {
        CHECK(count_op(&m, SMP_BC_STORET) == 0,
              "лишняя копия storet: приёмник нигде не читается");
    }

    /* Приёмник читается той же инструкцией — писать в него по ходу нельзя. */
    if (build_str("*&A<f32:4,4> -> @alloc => $a;\n"
                  "$a -> @relu => *&A;\n", &m)) {
        CHECK(count_op(&m, SMP_BC_STORET) == 1,
              "нет storet: результат обязан собраться во временном буфере");
        check_wellformed(&m, "запись в себя");
    }
}

static void test_pools(void)
{
    SECTION("пулы констант и строк");

    SmpModule m;
    if (!build_str("*&A<f32:4,4> -> @alloc => $a;\n"
                   "$a -> @scale(2.5) => *&B<f32:4,4>;\n"
                   "*&B -> @scale(2.5) => *&C<f32:4,4>;\n"
                   "*&C -> @scale(1.5) => *&D<f32:4,4>;\n", &m)) return;

    check_wellformed(&m, "константы");
    CHECK(m.n_consts == 2, "констант %u, ждали 2 (2.5 переиспользована)", m.n_consts);

    bool has25 = false, has15 = false;
    for (uint32_t i = 0; i < m.n_consts; i++) {
        if (m.consts[i].f == 2.5) has25 = true;
        if (m.consts[i].f == 1.5) has15 = true;
    }
    CHECK(has25 && has15, "константы записаны неверно");

    /* Строковая таблица: нулевое смещение — пустая строка. */
    CHECK(m.strs_size > 0 && m.strs[0] == '\0',
          "смещение 0 обязано быть пустым именем");
    CHECK(m.strs[m.strs_size - 1u] == '\0', "таблица строк без терминатора");
}

/* ========================================================================== */
/*  Контейнер                                                                 */
/* ========================================================================== */

static bool write_tmp(const SmpModule *m)
{
    return smp_s3b_write(m, TMP_S3B) == SMP_OK;
}

static SmpStatus read_tmp(SmpModule *out)
{
    rewind(g_sink);
    g_out[0] = '\0';
    smp_diag_init(&g_diag, &g_src, g_sink);
    const SmpStatus st = smp_s3b_read(out, TMP_S3B, &g_arena, &g_diag);
    rewind(g_sink);
    const size_t n = fread(g_out, 1, sizeof(g_out) - 1u, g_sink);
    g_out[n] = '\0';
    return st;
}

/* Меняет один байт в файле. */
static void poke(long at, uint8_t val)
{
    FILE *f = fopen(TMP_S3B, "r+b");
    if (!f) return;
    fseek(f, at, SEEK_SET);
    fwrite(&val, 1, 1, f);
    fclose(f);
}

static long file_size(const char *p)
{
    FILE *f = fopen(p, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fclose(f);
    return n;
}

static void test_container(void)
{
    SECTION("контейнер .s3b");

    SmpModule m;
    if (!build_str("[#arena:0] *&A<f32:16,16> -> @alloc => $a;\n"
                   "[#arena:1] *&B<f32:16,16> -> @alloc => $b;\n"
                   "[#simd:v256] $a -> @scale(3.0) -> @relu => *&C<f32:16,16>;\n"
                   "*&C -> @reduce.add => $s;\n", &m)) return;

    CHECK(write_tmp(&m), "запись .s3b провалилась");

    /* Магия на месте и ровно та, что в спецификации. */
    {
        FILE *f = fopen(TMP_S3B, "rb");
        CHECK(f != NULL, "файл не открылся");
        if (f) {
            uint8_t mg[4] = { 0 };
            const size_t got = fread(mg, 1, 4, f);
            fclose(f);
            CHECK(got == 4 && mg[0] == 0x53 && mg[1] == 0x4D &&
                  mg[2] == 0x50 && mg[3] == 0x33,
                  "magic %02X %02X %02X %02X, ждали 53 4D 50 33",
                  mg[0], mg[1], mg[2], mg[3]);
        }
    }

    /* Круговой рейс: всё совпадает до байта. */
    SmpModule r;
    CHECK(read_tmp(&r) == SMP_OK, "чтение провалилось: %s", g_out);

    CHECK(r.n_code   == m.n_code,   "инструкций %u vs %u", r.n_code, m.n_code);
    CHECK(r.n_tens   == m.n_tens,   "дескрипторов %u vs %u", r.n_tens, m.n_tens);
    CHECK(r.n_consts == m.n_consts, "констант %u vs %u", r.n_consts, m.n_consts);
    CHECK(r.n_arenas == m.n_arenas, "арен %u vs %u", r.n_arenas, m.n_arenas);
    CHECK(r.n_dbg    == m.n_dbg,    "отладочных строк %u vs %u", r.n_dbg, m.n_dbg);

    CHECK(r.n_code && memcmp(r.code, m.code, r.n_code * sizeof(SmpInstr)) == 0,
          "код после круга не совпал");
    CHECK(r.n_tens && memcmp(r.tens, m.tens, r.n_tens * sizeof(SmpTensor)) == 0,
          "дескрипторы после круга не совпали");
    CHECK(r.n_consts == 0 ||
          memcmp(r.consts, m.consts, r.n_consts * sizeof(SmpConst)) == 0,
          "константы после круга не совпали");
    CHECK(memcmp(r.arena_bytes, m.arena_bytes, r.n_arenas * sizeof(uint64_t)) == 0,
          "размеры арен после круга не совпали");
    CHECK(strcmp(smp_module_str(&r, 0), "") == 0, "нулевая строка после круга");

    check_wellformed(&r, "после чтения");
}

static void test_hostile(void)
{
    SECTION("враждебный .s3b");

    SmpModule m;
    if (!build_str("*&A<f32:8,8> -> @alloc => $a;\n"
                   "$a -> @scale(2.0) => *&B<f32:8,8>;\n", &m)) return;

    const long sz = (write_tmp(&m), file_size(TMP_S3B));
    CHECK(sz > 0, "файл не записался");

    SmpModule r;

    /* Подмена сигнатуры. */
    poke(0, 'X');
    CHECK(read_tmp(&r) != SMP_OK, "подменённая сигнатура принята");
    CHECK(strstr(g_out, "53 4D 50 33") != NULL, "в сообщении нет ожидаемой сигнатуры");
    write_tmp(&m);

    /* Чужая версия. */
    poke(6, 0x7F);
    CHECK(read_tmp(&r) != SMP_OK, "чужая версия принята");
    CHECK(strstr(g_out, "версии") != NULL, "нет упоминания версии");
    write_tmp(&m);

    /* Флаг порядка байтов снят. */
    poke(8, 0x00);
    CHECK(read_tmp(&r) != SMP_OK, "big-endian файл принят");
    write_tmp(&m);

    /* Заявленный размер не совпадает с фактическим. */
    poke(16, 0xFF);
    CHECK(read_tmp(&r) != SMP_OK, "неверный total_size принят");
    write_tmp(&m);

    /* Порча содержимого ловится контрольной суммой. */
    poke(sz - 4, 0xAB);
    CHECK(read_tmp(&r) != SMP_OK, "испорченное содержимое принято");
    CHECK(strstr(g_out, "сумма") != NULL, "не сработала контрольная сумма");
    write_tmp(&m);

    /* Обрезанный файл. */
    {
        FILE *f = fopen(TMP_S3B, "rb");
        uint8_t *buf = (uint8_t *)malloc((size_t)sz);
        if (f && buf) {
            const size_t got = fread(buf, 1, (size_t)sz, f);
            fclose(f);
            f = fopen(TMP_S3B, "wb");
            if (f) { fwrite(buf, 1, got / 2u, f); fclose(f); }
        }
        free(buf);
        CHECK(read_tmp(&r) != SMP_OK, "обрезанный файл принят");
    }

    /* Файл короче заголовка. */
    {
        FILE *f = fopen(TMP_S3B, "wb");
        if (f) { fputs("SMP", f); fclose(f); }
        CHECK(read_tmp(&r) != SMP_OK, "огрызок принят");
    }

    remove(TMP_S3B);
}

/* ========================================================================== */

static void test_disasm(void)
{
    SECTION("дизассемблер");

    SmpModule m;
    if (!build_str("[#simd:v256] *&A<f32:4,4> -> @alloc => $a;\n"
                   "$a -> @transpose => $t;\n"
                   "$a -> @scale(2.0) => *&B<f32:4,4> [!no-alias];\n", &m)) return;

    FILE *f = tmpfile();
    CHECK(f != NULL, "tmpfile недоступен");
    if (!f) return;

    smp_disasm(f, &m, false);
    rewind(f);
    static char txt[16384];
    const size_t n = fread(txt, 1, sizeof(txt) - 1u, f);
    txt[n] = '\0';
    fclose(f);

    CHECK(strstr(txt, "loadt") != NULL, "нет loadt");
    CHECK(strstr(txt, "trans") != NULL, "нет trans");
    CHECK(strstr(txt, "scale") != NULL, "нет scale");
    CHECK(strstr(txt, "halt")  != NULL, "нет halt");
    CHECK(strstr(txt, "v256")  != NULL, "не показана ширина вектора");
    CHECK(strstr(txt, ",na")   != NULL, "не показан !no-alias");
    CHECK(strstr(txt, "A<f32:4,4>") != NULL, "не показан дескриптор с именем");
    CHECK(strstr(txt, "строка") != NULL, "нет привязки к строкам исходника");
    CHECK(strstr(txt, "\x1b[") == NULL, "ANSI-коды при color=false");

    /* Все мнемоники реестра непусты и различны. */
    for (unsigned i = 0; i < SMP_BC__COUNT; i++) {
        const SmpOpcodeDef *d = smp_opcode_def((SmpOpcode)i);
        CHECK(d->mnemonic && d->mnemonic[0], "опкод %u без мнемоники", i);
        for (unsigned j = i + 1u; j < SMP_BC__COUNT; j++)
            CHECK(strcmp(d->mnemonic, smp_opcode_def((SmpOpcode)j)->mnemonic) != 0,
                  "мнемоника '%s' повторяется", d->mnemonic);
    }
    CHECK(strcmp(smp_opcode_def((SmpOpcode)SMP_BC__COUNT)->mnemonic, "<?>") == 0,
          "выход за реестр опкодов не отловлен");
}

static void test_formats(void)
{
    SECTION("размеры структур");

    CHECK(sizeof(SmpInstr)      ==  8, "инструкция %zu Б", sizeof(SmpInstr));
    CHECK(sizeof(SmpTensor)     == 32, "дескриптор %zu Б", sizeof(SmpTensor));
    CHECK(sizeof(SmpConst)      ==  8, "константа %zu Б", sizeof(SmpConst));
    CHECK(sizeof(SmpS3bHeader)  == 32, "заголовок %zu Б", sizeof(SmpS3bHeader));
    CHECK(sizeof(SmpS3bSection) == 16, "секция %zu Б", sizeof(SmpS3bSection));
    CHECK(sizeof(SmpDbgLine)    == 12, "отладочная строка %zu Б", sizeof(SmpDbgLine));

    CHECK(smp_vec_bits(smp_vec_code(256)) == 256, "кодирование v256");
    CHECK(smp_vec_bits(smp_vec_code(512)) == 512, "кодирование v512");
    CHECK(smp_vec_bits(smp_vec_code(128)) == 128, "кодирование v128");
    CHECK(smp_vec_bits(smp_vec_code(64))  ==  64, "кодирование скаляра");
}

/* ========================================================================== */

int main(void)
{
    smp_console_setup();
    fprintf(stderr, "SMPC3 tests :: Ф4\n\n");

    if (smp_arena_init(&g_arena, 64u << 20, 9, "emittest") != SMP_OK) {
        fprintf(stderr, "арена не поднялась\n");
        return 70;
    }
    g_sink = tmpfile();
    if (!g_sink) { fprintf(stderr, "tmpfile недоступен\n"); return 70; }

    test_formats();
    test_shape();
    test_pipeline();
    test_direct_write();
    test_pools();
    test_container();
    test_hostile();
    test_disasm();

    fclose(g_sink);
    smp_arena_release(&g_arena);
    return REPORT();
}
