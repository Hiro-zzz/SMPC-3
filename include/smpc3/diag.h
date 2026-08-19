/* SMPC3 :: diag.h -- Zero-Diplomacy Diagnostics.
 *
 * Движок не маскирует UB, не «чинит» ошибку и не предлагает компромиссов.
 * Он печатает место, факт, причину, оценку квалификации автора и способ
 * исправления. После FATAL процесс не продолжается.
 *
 * Аллокаций нет: форматирование идёт в кольцевой буфер внутри контекста.
 */
#ifndef SMPC3_DIAG_H
#define SMPC3_DIAG_H

#include "smpc3/common.h"
#include "smpc3/diag_codes.h"

#include <stdio.h>

/* --- Коды ----------------------------------------------------------------- */
#define SMP_DIAG_ENUM(id, text, sev, cat, title, fix, diag) id,
typedef enum SmpDiagCode {
    SMP_DIAG_CODES(SMP_DIAG_ENUM)
    SMP_DIAG__COUNT
} SmpDiagCode;
#undef SMP_DIAG_ENUM

typedef enum SmpSeverity {
    SMP_SEV_NOTE = 0,
    SMP_SEV_WARN,
    SMP_SEV_FATAL
} SmpSeverity;

typedef struct SmpDiagInfo {
    const char *text;      /* "E0418"                                        */
    SmpSeverity sev;
    SmpDiagCat  cat;
    const char *title;     /* строка ОШИБКА по умолчанию                     */
    const char *fix;       /* строка ИСПРАВЛЕНИЕ по умолчанию                */

    /* Собственная строка ДИАГНОЗ. NULL — брать из пула категории. Пул даёт
     * разнообразие ценой общности формулировки; там, где код обозначает одну
     * конкретную беду, точная строка полезнее случайной. */
    const char *diagnosis;
} SmpDiagInfo;

const SmpDiagInfo *smp_diag_info(SmpDiagCode c);

/* Пул строк ДИАГНОЗ по категории. Наружу торчит ради тестов: дубликат в пуле
 * молча сокращает разнообразие, и заметить это иначе нечем. */
unsigned    smp_diag_pool_size(SmpDiagCat cat);
const char *smp_diag_pool_at(SmpDiagCat cat, unsigned i);
SmpDiagCode        smp_diag_lookup(const char *text); /* "E0418" -> код; SMP_DIAG__COUNT если нет */

/* --- Исходник и позиция --------------------------------------------------- */
typedef struct SmpSource {
    const char *path;      /* "kernel.smpc"                                  */
    const char *text;      /* весь файл, не обязан быть \0-терминирован      */
    size_t      len;
} SmpSource;

/* Позиция: 1-based строка, 1-based колонка В БАЙТАХ, длина подчёркивания
 * в байтах. Каретка при печати пересчитывается в кодовые точки UTF-8. */
typedef struct SmpSpan {
    uint32_t line;
    uint32_t col;
    uint32_t len;
} SmpSpan;

#define SMP_SPAN_NONE ((SmpSpan){ 0, 0, 0 })
SMP_INLINE bool smp_span_valid(SmpSpan s) { return s.line != 0; }

/* --- Контекст ------------------------------------------------------------- */
#define SMP_FMT_SLOTS   8u     /* кольцо временных строк                     */
#define SMP_FMT_SLOTLEN 1024u

/* --- Журнал в памяти -------------------------------------------------------
 *
 * Пулу нужен отдельный журнал на инстанс, иначе диагностика шестнадцати
 * потоков склеится в кашу. Раньше это был tmpfile(), то есть открытый
 * дескриптор на инстанс — и пул упирался в лимит CRT (512 на Windows, минус
 * stdin/stdout/stderr), не поднимаясь с 510 при заявленных 1024.
 *
 * Буфер выделяется один раз при подъёме пула и не растёт: обещание про
 * отсутствие аллокаций на исполнении остаётся в силе. Если написанное не
 * влезло, журнал НЕ продолжает писать молча — он поднимает truncated, и отчёт
 * об этом сообщает. */
typedef struct SmpLog {
    char  *buf;
    size_t cap;
    size_t len;
    bool   truncated;
} SmpLog;

void smp_log_bind(SmpLog *l, char *buf, size_t cap);
void smp_log_reset(SmpLog *l);
void smp_log_write(SmpLog *l, const char *p, size_t n);

/* Хук дампа регистров: VM подставляет свой, компилятор оставляет NULL.
 * Печатает через smp_diag_write, а не в FILE*, чтобы дамп уходил туда же,
 * куда и остальное сообщение — в том числе в журнал в памяти.
 *
 * Тег объявляется заранее: контекст определён ниже, а typedef на него в C99
 * повторить нельзя. */
struct SmpDiagCtx;
typedef void (*SmpRegDumpFn)(struct SmpDiagCtx *d, void *user, bool color);

typedef struct SmpDiagCtx {
    const SmpSource *src;

    /* Куда идёт вывод. Если задан log, пишем в него, а out не трогаем. */
    FILE            *out;
    SmpLog          *log;
    bool             color;
    bool             deterministic;  /* ДИАГНОЗ выбирается детерминированно */
    uint64_t         rng;

    uint32_t         n_fatal;
    uint32_t         n_warn;
    uint32_t         n_note;

    SmpRegDumpFn     regdump;
    void            *regdump_user;

    /* кольцевой буфер для smp_fmt() */
    char             fmtbuf[SMP_FMT_SLOTS][SMP_FMT_SLOTLEN];
    uint32_t         fmtslot;
} SmpDiagCtx;

void smp_diag_init(SmpDiagCtx *d, const SmpSource *src, FILE *out);

/* Перенаправить вывод в память. Цвет при этом гасится: журнал читает не
 * терминал, а отчёт пула. */
void smp_diag_set_log(SmpDiagCtx *d, SmpLog *log);

/* Единая точка вывода диагностики: в журнал, если он задан, иначе в out.
 * Публична, потому что через неё печатает и хук дампа регистров. */
SMP_PRINTF(2, 3) void smp_diag_write(SmpDiagCtx *d, const char *fmt, ...);

/* Включает UTF-8 и ANSI-последовательности в консоли Windows. Вызвать один
 * раз на старте процесса до любой печати. */
void smp_console_setup(void);

/* --- Сообщение ------------------------------------------------------------ */
typedef struct SmpDiagMsg {
    SmpDiagCode code;
    SmpSpan     span;
    const char *details;    /* строка ДЕТАЛИ; '\n' даёт продолжение с отступом */
    const char *diagnosis;  /* строка ДИАГНОЗ; NULL -> из пула по категории    */
    const char *fix;        /* строка ИСПРАВЛЕНИЕ; NULL -> из реестра          */
    bool        dump_regs;  /* печатать дамп регистров, если хук установлен    */
} SmpDiagMsg;

/* Печатает сообщение. Не завершает процесс даже для FATAL: решение о смерти
 * принимает вызывающий, чтобы успеть собрать несколько ошибок за проход. */
void smp_diag_emit(SmpDiagCtx *d, const SmpDiagMsg *m);

/* Печатает и немедленно убивает процесс с кодом 70 (EX_SOFTWARE). */
SMP_NORETURN void smp_diag_die(SmpDiagCtx *d, const SmpDiagMsg *m);

/* Форматирование во временный слот кольца. Указатель живёт до тех пор, пока
 * не будет израсходовано SMP_FMT_SLOTS новых слотов. Достаточно, чтобы
 * собрать одно сообщение целиком. */
SMP_PRINTF(2, 3) const char *smp_fmt(SmpDiagCtx *d, const char *fmt, ...);

/* Итог прохода: "3 фатальных, 1 предупреждение". Возвращает buf. */
char *smp_diag_summary(const SmpDiagCtx *d, char *buf, size_t cap);

/* Вычислить SmpSpan по байтовому смещению в исходнике (для лексера). */
SmpSpan smp_span_from_offset(const SmpSource *src, size_t off, uint32_t len);

#endif /* SMPC3_DIAG_H */
