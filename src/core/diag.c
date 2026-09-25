/* SMPC3 :: diag.c -- рендер диагностики. */
#include "smpc3/diag.h"
#include "smpc3/plat.h"

#include <string.h>
#include <stdlib.h>

/* ========================================================================== */
/*  Реестр                                                                    */
/* ========================================================================== */

#define SMP_DIAG_ROW(id, text, sev, cat, title, fix, diag) \
    { text, SMP_SEV_##sev, SMP_CAT_##cat, title, fix, diag },

static const SmpDiagInfo g_info[SMP_DIAG__COUNT] = {
    SMP_DIAG_CODES(SMP_DIAG_ROW)
};
#undef SMP_DIAG_ROW

static const SmpDiagInfo g_info_bogus =
    { "E????", SMP_SEV_FATAL, SMP_CAT_INTERNAL,
      "Запрошен несуществующий диагностический код.",
      "Это баг самого компилятора.",
      "Сломался инструмент, а не пользователь." };

const SmpDiagInfo *smp_diag_info(SmpDiagCode c)
{
    if ((unsigned)c >= (unsigned)SMP_DIAG__COUNT) return &g_info_bogus;
    return &g_info[c];
}

SmpDiagCode smp_diag_lookup(const char *text)
{
    for (unsigned i = 0; i < SMP_DIAG__COUNT; i++)
        if (strcmp(g_info[i].text, text) == 0) return (SmpDiagCode)i;
    return SMP_DIAG__COUNT;
}

/* ========================================================================== */
/*  Пулы ДИАГНОЗОВ                                                            */
/* ========================================================================== */

static const char *const g_pool_lex[] = {
    "Ты не осилил алфавит. Алфавит — это первое, что осваивают.",
    "Твой редактор пишет быстрее, чем твой мозг проверяет.",
    "Символы кончились, а самоуверенность нет.",
    "Лексер — самая простая фаза. Ты не прошёл и её.",
    "В языке конечное число символов. Ты нашёл способ выйти за него.",
    "Это даже не синтаксис. Это буквы.",
    "Токенизатор споткнулся раньше, чем успел начать работу.",
    "Файл не дочитан до конца, потому что ты его не дописал.",
    "Набор символов задан спецификацией, а не настроением.",
    "Ошибка на уровне букв. До смысла мы ещё не добрались."
};
static const char *const g_pool_parse[] = {
    "Грамматика описана на одной странице. Ты не дочитал до конца.",
    "Форма [ПРЕФИКС] ТЕЛО [СУФФИКС] держится на трёх правилах. Ты нарушил одно.",
    "Парсер не телепат. Он читает ровно то, что ты написал.",
    "LL(2) значит два токена вперёд. Тебе не хватило и одного.",
    "Структура инструкции задана жёстко. Ты собрал её как-то иначе.",
    "Ты написал нечто, что не разбирается ни одним правилом языка.",
    "Каждая конструкция языка описана. Твоя среди них не значится.",
    "Восстановление после ошибки съело твою инструкцию целиком.",
    "Синтаксис фиксирован. Импровизация предусмотрена в других языках.",
    "Разбор остановился здесь. Дальше идёт то, чего ты не имел в виду.",
    "Ты изобрёл конструкцию, которой нет. Изобретение отклонено.",
    "Тебе кажется, что написанное очевидно. Парсеру не кажется."
};
static const char *const g_pool_type[] = {
    "Ты складываешь сущности, у которых нет общей природы.",
    "Система типов существует именно для таких, как ты.",
    "Неявных приведений нет. Придётся думать.",
    "Тип — не пожелание, а обязательство. Ты его нарушил.",
    "Пять типов. Ты умудрился перепутать два из них.",
    "Компилятор знает тип каждого выражения. Ты, похоже, нет.",
    "Приведение пишется явно. Молча округлять язык не станет.",
    "f32 и i32 различаются не только буквой.",
    "Ранг и форма — часть типа. Отмахнуться от них не выйдет.",
    "Ты требуешь операции, которая для этого типа не определена.",
    "Тензор и скаляр — разные вещи. Одно вместо другого не подставляется.",
    "Вывод типов сработал. Не сработал ты."
};
static const char *const g_pool_mem[] = {
    "Твои руки не приспособлены для линейной алгебры.",
    "Ты умножил корову на радиоприемник.",
    "Раскладка памяти — не то, что можно угадать интуицией.",
    "Указатель — это адрес, а не пожелание.",
    "Границы буфера ты воспринимаешь как рекомендацию. Кремний — нет.",
    "Формы либо совпадают, либо это ошибка. Broadcast здесь не предусмотрен.",
    "Арена конечна. Твоя фантазия, к сожалению, нет.",
    "Ты дал компилятору обещание, которого не собирался держать.",
    "За твоим тензором начинается чужая память. Она этого не переживёт.",
    "Размерности сходятся или не сходятся. Третьего не дано.",
    "Выравнивание на 64 байта — условие, а не украшение.",
    "Срез описывает часть тензора, а не то, что тебе хотелось бы видеть."
};
static const char *const g_pool_simd[] = {
    "Ты требуешь от процессора инструкций, которых в нём нет.",
    "Векторные регистры не растягиваются под твои амбиции.",
    "Выравнивание — это не суеверие, это условие корректности.",
    "CPUID отвечает честно. Это ты его не спросил.",
    "Ширина вектора определяется железом, а не строчкой в исходнике.",
    "v512 на процессоре без v512 — это заклинание, а не код.",
    "Кремний не расширяется под требования исходного текста.",
    "Инструкция существует в документации. В этом процессоре — нет.",
    "Ты писал под другой процессор. Возможно, под воображаемый.",
    "SIMD прощает многое, но не невыровненный адрес."
};
static const char *const g_pool_runtime[] = {
    "Программа дошла до выполнения. Дальше повезло меньше.",
    "Ты проверил всё, кроме того, что сломалось.",
    "Арифметика не прощает оптимизма.",
    "Компиляция прошла. Это был максимум твоих достижений.",
    "Числа стали известны только сейчас. Лучше от этого не стало.",
    "Бесконечность — не результат. Это признание поражения.",
    "Ты сам просил останавливаться на таком. Останавливаемся.",
    "Всё шло хорошо ровно до этой инструкции.",
    "Диапазон типа конечен. Твоё значение — нет.",
    "Рантайм остановился здесь, чтобы ты не увидел, что было бы дальше."
};
static const char *const g_pool_internal[] = {
    "Инвариант нарушен внутри компилятора. Здесь ты ни при чём.",
    "Сломался инструмент, а не пользователь. Редкий случай.",
    "Это ошибка компилятора. Впервые за сегодня виноват не ты.",
    "Внутреннее состояние разошлось с ожидаемым. Об этом стоит сообщить.",
    "Здесь сломалось то, что ломаться не должно было."
};

typedef struct { const char *const *items; unsigned n; } SmpPool;

static const SmpPool g_pools[SMP_CAT__COUNT] = {
    { g_pool_lex,      (unsigned)SMP_ARRLEN(g_pool_lex)      },
    { g_pool_parse,    (unsigned)SMP_ARRLEN(g_pool_parse)    },
    { g_pool_type,     (unsigned)SMP_ARRLEN(g_pool_type)     },
    { g_pool_mem,      (unsigned)SMP_ARRLEN(g_pool_mem)      },
    { g_pool_simd,     (unsigned)SMP_ARRLEN(g_pool_simd)     },
    { g_pool_runtime,  (unsigned)SMP_ARRLEN(g_pool_runtime)  },
    { g_pool_internal, (unsigned)SMP_ARRLEN(g_pool_internal) }
};

/* Детерминированный выбор: одно и то же место в коде всегда получает один и
 * тот же диагноз. Это делает вывод воспроизводимым для golden-тестов. */
static uint64_t smp__mix(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

unsigned smp_diag_pool_size(SmpDiagCat cat)
{
    return (unsigned)cat < SMP_CAT__COUNT ? g_pools[cat].n : 0u;
}

const char *smp_diag_pool_at(SmpDiagCat cat, unsigned i)
{
    if ((unsigned)cat >= SMP_CAT__COUNT || i >= g_pools[cat].n) return NULL;
    return g_pools[cat].items[i];
}

static const char *smp__diagnosis(SmpDiagCtx *d, SmpDiagCode code, SmpSpan sp)
{
    const SmpDiagInfo *inf = smp_diag_info(code);

    /* Своя строка кода перебивает пул: пул придуман для разнообразия, а не
     * чтобы подменять точную формулировку случайной. */
    if (inf->diagnosis) return inf->diagnosis;

    const SmpPool *p = &g_pools[(unsigned)inf->cat < SMP_CAT__COUNT ? inf->cat : SMP_CAT_INTERNAL];
    if (p->n == 0) return "Диагноз не сформулирован.";

    /* Затравка берётся из ТЕКСТА кода, а не из его номера в перечислении.
     * Номер сдвигается, стоит вставить новый код в середину реестра, — и
     * диагноз меняется у всех кодов после него, хотя их никто не трогал.
     * Один раз это уже переписало 21 снимок вывода из 31, и заметить такое
     * можно было только по объёму диффа. Текст кода не сдвигается никогда. */
    uint64_t h = 1469598103934665603ull;              /* FNV-1a 64 */
    for (const char *t = inf->text; *t; t++) {
        h ^= (unsigned char)*t;
        h *= 1099511628211ull;
    }

    uint64_t seed = d->deterministic
        ? smp__mix(h ^ ((uint64_t)sp.line << 16) ^ sp.col)
        : smp__mix(d->rng += 0x2545F4914F6CDD1Dull);
    return p->items[seed % p->n];
}

/* ========================================================================== */
/*  UTF-8 и работа со строками                                                */
/* ========================================================================== */

/* Число кодовых точек в первых nbytes байтах. Продолжения (10xxxxxx) не в счёт. */
static size_t smp__u8len(const char *s, size_t nbytes)
{
    size_t n = 0;
    for (size_t i = 0; i < nbytes; i++)
        if (((unsigned char)s[i] & 0xC0u) != 0x80u) n++;
    return n;
}

/* ========================================================================== */
/*  Цвета                                                                     */
/* ========================================================================== */

#define C_RESET "\x1b[0m"
#define C_BOLD  "\x1b[1m"
#define C_DIM   "\x1b[2m"
#define C_RED   "\x1b[91m"
#define C_YEL   "\x1b[93m"
#define C_CYN   "\x1b[96m"
#define C_BLU   "\x1b[94m"
#define C_GRN   "\x1b[92m"
#define C_MAG   "\x1b[95m"

static const char *smp__c(const SmpDiagCtx *d, const char *seq)
{
    return d->color ? seq : "";
}

static const char *smp__sev_color(const SmpDiagCtx *d, SmpSeverity s)
{
    if (!d->color) return "";
    switch (s) {
        case SMP_SEV_FATAL: return C_RED;
        case SMP_SEV_WARN:  return C_YEL;
        default:            return C_CYN;
    }
}

static const char *smp__sev_banner(SmpSeverity s)
{
    switch (s) {
        case SMP_SEV_FATAL: return "FATAL SKILL ISSUE";
        case SMP_SEV_WARN:  return "SKILL ISSUE";
        default:            return "NOTE";
    }
}

/* ========================================================================== */
/*  Контекст                                                                  */
/* ========================================================================== */

/* ========================================================================== */
/*  Журнал в памяти                                                           */
/* ========================================================================== */

void smp_log_bind(SmpLog *l, char *buf, size_t cap)
{
    l->buf = buf; l->cap = cap; l->len = 0; l->truncated = false;
}

void smp_log_reset(SmpLog *l) { l->len = 0; l->truncated = false; }

/* Дописывает сколько влезло и поднимает truncated, если влезло не всё. Молча
 * терять хвост диагностики этот язык не станет — про обрезку скажет отчёт. */
void smp_log_write(SmpLog *l, const char *p, size_t n)
{
    if (!l->buf || n == 0) return;

    const size_t room = (l->len < l->cap) ? l->cap - l->len : 0u;
    if (n > room) { l->truncated = true; n = room; }
    if (n) { memcpy(l->buf + l->len, p, n); l->len += n; }
}

void smp_diag_set_log(SmpDiagCtx *d, SmpLog *log)
{
    d->log = log;
    if (log) d->color = false;   /* журнал читает отчёт пула, а не терминал */
}

void smp_diag_write(SmpDiagCtx *d, const char *fmt, ...)
{
    va_list ap;

    if (!d->log) {
        va_start(ap, fmt);
        vfprintf(d->out, fmt, ap);
        va_end(ap);
        return;
    }

    /* Через промежуточный буфер: писать сразу в хвост журнала нельзя, пока не
     * известно, влезет ли — vsnprintf усечёт по своему разумению, а решать это
     * должен журнал. */
    char tmp[SMP_FMT_SLOTLEN];
    va_start(ap, fmt);
    const int n = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    if (n <= 0) return;

    size_t len = (size_t)n;
    if (len >= sizeof tmp) { len = sizeof tmp - 1u; d->log->truncated = true; }
    smp_log_write(d->log, tmp, len);
}

/* ========================================================================== */

void smp_diag_init(SmpDiagCtx *d, const SmpSource *src, FILE *out)
{
    memset(d, 0, sizeof(*d));
    d->src           = src;
    d->out           = out ? out : stderr;
    d->color         = smp_plat_isatty(d->out);
    d->deterministic = true;
    d->rng           = 0x123456789ABCDEFull;

    /* Короткая форма снаружи — переменной, чтобы инструментам вокруг модели
     * не нужен был свой флаг у каждой команды CLI. */
    const char *mode = smp_plat_env("SMPC3_DIAG");
    d->compact = mode && strcmp(mode, "compact") == 0;
}

void smp_console_setup(void) { smp_plat_console_setup(); }

const char *smp_fmt(SmpDiagCtx *d, const char *fmt, ...)
{
    char *slot = d->fmtbuf[d->fmtslot];
    d->fmtslot = (d->fmtslot + 1u) % SMP_FMT_SLOTS;

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(slot, SMP_FMT_SLOTLEN, fmt, ap);
    va_end(ap);
    return slot;
}

/* ========================================================================== */
/*  Извлечение строки исходника                                               */
/* ========================================================================== */

typedef struct { const char *p; size_t len; } SmpLineRef;

static bool smp__line_at(const SmpSource *src, uint32_t line, SmpLineRef *out)
{
    if (!src || !src->text || line == 0) return false;

    const char *s = src->text, *end = src->text + src->len;
    uint32_t cur = 1;
    const char *ls = s;

    while (ls < end && cur < line) {
        const char *nl = (const char *)memchr(ls, '\n', (size_t)(end - ls));
        if (!nl) return false;
        ls = nl + 1;
        cur++;
    }
    if (cur != line || ls > end) return false;

    const char *nl = (const char *)memchr(ls, '\n', (size_t)(end - ls));
    const char *le = nl ? nl : end;
    if (le > ls && le[-1] == '\r') le--;

    out->p   = ls;
    out->len = (size_t)(le - ls);
    return true;
}

SmpSpan smp_span_from_offset(const SmpSource *src, size_t off, uint32_t len)
{
    SmpSpan sp = { 1, 1, len };
    if (!src || !src->text) return sp;
    if (off > src->len) off = src->len;

    uint32_t line = 1;
    size_t   ls   = 0;
    for (size_t i = 0; i < off; i++) {
        if (src->text[i] == '\n') { line++; ls = i + 1; }
    }
    sp.line = line;
    sp.col  = (uint32_t)(off - ls) + 1u;
    return sp;
}

/* ========================================================================== */
/*  Рендер сниппета                                                           */
/* ========================================================================== */

#define SMP_TABW      4u
#define SMP_LINE_CAP  2048u   /* байт исходной строки, дальше — обрезка       */
#define SMP_VIEW_COLS 132u    /* ширина окна вывода в кодовых точках          */

/* Разворачивает табуляции и считает позицию каретки в кодовых точках. */
typedef struct {
    char     buf[SMP_LINE_CAP * 2u];
    size_t   nbytes;
    uint32_t ncols;         /* всего кодовых точек                            */
    uint32_t cp_off[SMP_LINE_CAP + 1u]; /* байтовое смещение каждой к.точки   */
    uint32_t caret_col;     /* 0-based, в кодовых точках                      */
    uint32_t caret_len;     /* в кодовых точках, >=1                          */
} SmpRendered;

static void smp__render_line(const SmpLineRef *ln, const SmpSpan *sp, SmpRendered *r)
{
    r->nbytes    = 0;
    r->ncols     = 0;
    r->caret_col = 0;
    r->caret_len = sp->len ? sp->len : 1u;

    const size_t n = ln->len > SMP_LINE_CAP ? SMP_LINE_CAP : ln->len;
    const size_t caret_byte = sp->col ? (size_t)sp->col - 1u : 0u;
    const size_t caret_end  = caret_byte + (sp->len ? sp->len : 1u);

    uint32_t caret_col_set = 0, caret_end_col = 0;

    for (size_t i = 0; i <= n; i++) {
        if (i == caret_byte) { r->caret_col = r->ncols; caret_col_set = 1; }
        if (i == caret_end)  { caret_end_col = r->ncols; }
        if (i == n) break;

        const unsigned char c = (unsigned char)ln->p[i];

        if (c == '\t') {
            uint32_t pad = SMP_TABW - (r->ncols % SMP_TABW);
            for (uint32_t k = 0; k < pad && r->ncols < SMP_LINE_CAP; k++) {
                r->cp_off[r->ncols++] = (uint32_t)r->nbytes;
                r->buf[r->nbytes++]   = ' ';
            }
            continue;
        }
        if ((c & 0xC0u) != 0x80u) {                 /* начало кодовой точки  */
            if (r->ncols < SMP_LINE_CAP) r->cp_off[r->ncols++] = (uint32_t)r->nbytes;
        }
        /* Управляющие символы заменяем точкой, чтобы не рвать разметку. */
        r->buf[r->nbytes++] = (c < 0x20u) ? '.' : (char)c;
    }

    if (!caret_col_set) r->caret_col = r->ncols;
    if (caret_end_col > r->caret_col) r->caret_len = caret_end_col - r->caret_col;
    if (r->caret_len == 0) r->caret_len = 1;

    r->cp_off[r->ncols] = (uint32_t)r->nbytes;
    r->buf[r->nbytes]   = '\0';
}

/* ========================================================================== */
/*  Печать блоков                                                             */
/* ========================================================================== */

/* Поле вида
 *     |-- ЛЕЙБЛ: первая строка
 *     |          продолжение
 * Продолжения выравниваются под первую букву текста. corner — '|' или '+'. */
static void smp__field(SmpDiagCtx *d, uint32_t gutter, char corner,
                       const char *label, const char *body)
{
    const char *dim  = smp__c(d, C_DIM);
    const char *rst  = smp__c(d, C_RESET);
    const char *bold = smp__c(d, C_BOLD);

    /* ширина "-- ЛЕЙБЛ: " в кодовых точках */
    const int lead = (int)smp__u8len(label, strlen(label)) + 5;

    const char *p     = body ? body : "";
    bool        first = true;

    for (;;) {
        const char *nl  = strchr(p, '\n');
        const int   seg = nl ? (int)(nl - p) : (int)strlen(p);

        if (first) {
            smp_diag_write(d, "%s%*s%c--%s %s%s:%s %.*s\n",
                           dim, (int)gutter, "", corner, rst, bold,
                           label, rst, seg, p);
        } else {
            smp_diag_write(d, "%s%*s|%s%*s%.*s\n",
                           dim, (int)gutter, "", rst, lead, "", seg, p);
        }

        if (!nl) break;
        p     = nl + 1;
        first = false;
    }
}

static uint32_t smp__ndigits(uint32_t v)
{
    uint32_t n = 1;
    while (v >= 10u) { v /= 10u; n++; }
    return n;
}

static void smp__snippet(SmpDiagCtx *d, SmpSpan sp, SmpSeverity sev, uint32_t gutter)
{
    SmpLineRef ln;
    if (!smp__line_at(d->src, sp.line, &ln)) return;

    static SmpRendered r;   /* ~10 KiB: держим в .bss, а не на стеке */
    smp__render_line(&ln, &sp, &r);

    const char *dim  = smp__c(d, C_DIM);
    const char *rst  = smp__c(d, C_RESET);
    const char *sevc = smp__sev_color(d, sev);

    /* Окно вывода: длинные строки подрезаем вокруг каретки. */
    uint32_t vstart = 0;
    if (r.ncols > SMP_VIEW_COLS) {
        const uint32_t half = SMP_VIEW_COLS / 2u;
        if (r.caret_col > half) vstart = r.caret_col - half;
        if (vstart + SMP_VIEW_COLS > r.ncols) vstart = r.ncols - SMP_VIEW_COLS;
    }
    uint32_t vend = vstart + SMP_VIEW_COLS;
    if (vend > r.ncols) vend = r.ncols;

    const char  *seg    = r.buf + r.cp_off[vstart];
    const int    seglen = (int)(r.cp_off[vend] - r.cp_off[vstart]);
    const char  *lead   = (vstart > 0)      ? "\xE2\x80\xA6" : "";  /* … */
    const char  *trail  = (vend < r.ncols)  ? "\xE2\x80\xA6" : "";

    smp_diag_write(d, "%s%*u |%s %s%.*s%s\n",
                   dim, (int)gutter - 1, sp.line, rst, lead, seglen, seg, trail);

    /* Каретка. */
    uint32_t cc = (r.caret_col >= vstart) ? r.caret_col - vstart : 0u;
    if (vstart > 0) cc += 1u;                       /* поправка на многоточие */
    uint32_t cl = r.caret_len;
    if (cc + cl > SMP_VIEW_COLS + 1u) cl = SMP_VIEW_COLS + 1u - cc;
    if (cl == 0) cl = 1;

    smp_diag_write(d, "%s%*s|%s %*s%s%s", dim, (int)gutter, "", rst, (int)cc, "",
                   smp__c(d, C_BOLD), sevc);
    smp_diag_write(d, "^");
    for (uint32_t i = 1; i < cl; i++) smp_diag_write(d, "~");
    smp_diag_write(d, "%s\n", rst);
}

static bool smp__muted(const SmpDiagCtx *d, SmpDiagCode c)
{
    return (d->muted[c / 64u] >> (c % 64u)) & 1u;
}

bool smp_diag_mute(SmpDiagCtx *d, SmpDiagCode code)
{
    if ((unsigned)code >= SMP_DIAG__COUNT) return false;
    if (smp_diag_info(code)->sev == SMP_SEV_FATAL) return false;
    d->muted[code / 64u] |= 1ull << (code % 64u);
    return true;
}

/* Поле короткой формы: продолжения строк ДЕТАЛИ — с отступом под текстом. */
static void smp__compact_field(SmpDiagCtx *d, const char *label, const char *body)
{
    const char *p = body ? body : "";
    smp_diag_write(d, "  %s: ", label);
    for (;;) {
        const char *nl = strchr(p, '\n');
        smp_diag_write(d, "%.*s\n", nl ? (int)(nl - p) : (int)strlen(p), p);
        if (!nl) break;
        p = nl + 1;
        smp_diag_write(d, "    ");
    }
}

static void smp__emit_compact(SmpDiagCtx *d, const SmpDiagMsg *m, const SmpDiagInfo *inf)
{
    const bool has_span = smp_span_valid(m->span);
    if (has_span)
        smp_diag_write(d, "%s %u:%u %s\n", inf->text, m->span.line, m->span.col, inf->title);
    else
        smp_diag_write(d, "%s %s\n", inf->text, inf->title);

    SmpLineRef ln;
    if (has_span && smp__line_at(d->src, m->span.line, &ln)) {
        while (ln.len && (*ln.p == ' ' || *ln.p == '\t')) { ln.p++; ln.len--; }
        smp_diag_write(d, "  %u | %.*s\n", m->span.line,
                       (int)(ln.len < 200u ? ln.len : 200u), ln.p);
    }
    if (m->details && m->details[0]) smp__compact_field(d, "ДЕТАЛИ", m->details);
    smp__compact_field(d, "ИСПРАВЛЕНИЕ", m->fix ? m->fix : inf->fix);
    if (!d->log) fflush(d->out);
}

static void smp__emit(SmpDiagCtx *d, const SmpDiagMsg *m)
{
    const SmpDiagInfo *inf = smp_diag_info(m->code);
    if (smp__muted(d, m->code)) return;

    switch (inf->sev) {
        case SMP_SEV_FATAL: d->n_fatal++; break;
        case SMP_SEV_WARN:  d->n_warn++;  break;
        default:            d->n_note++;  break;
    }

    if (d->compact) {
        smp__emit_compact(d, m, inf);
        return;
    }

    const char *dim  = smp__c(d, C_DIM);
    const char *rst  = smp__c(d, C_RESET);
    const char *bold = smp__c(d, C_BOLD);
    const char *sevc = smp__sev_color(d, inf->sev);

    const bool     has_span = smp_span_valid(m->span) && d->src && d->src->text;
    const uint32_t gutter   = has_span ? (2u + smp__ndigits(m->span.line) + 1u) : 5u;

    /* --- шапка --- */
    smp_diag_write(d, "\n");
    smp_diag_write(d, "%s%s[%s :: %s]%s",
                   bold, sevc, smp__sev_banner(inf->sev), inf->text, rst);

    if (has_span) {
        smp_diag_write(d, " %sin%s %s:%u:%u\n", dim, rst,
                       d->src->path ? d->src->path : "<memory>",
                       m->span.line, m->span.col);
        smp__snippet(d, m->span, inf->sev, gutter);
    } else if (d->src && d->src->path) {
        smp_diag_write(d, " %sin%s %s\n", dim, rst, d->src->path);
    } else {
        smp_diag_write(d, "\n");
    }

    /* --- тело --- */
    smp__field(d, gutter, '|', "ОШИБКА", inf->title);

    if (m->details && m->details[0])
        smp__field(d, gutter, '|', "ДЕТАЛИ", m->details);

    smp__field(d, gutter, '|', "ДИАГНОЗ",
               m->diagnosis ? m->diagnosis : smp__diagnosis(d, m->code, m->span));

    smp__field(d, gutter, '+', "ИСПРАВЛЕНИЕ", m->fix ? m->fix : inf->fix);

    if (m->dump_regs && d->regdump) {
        smp_diag_write(d, "\n");
        d->regdump(d, d->regdump_user, d->color);
    }

    if (!d->log) fflush(d->out);
}

void smp_diag_emit(SmpDiagCtx *d, const SmpDiagMsg *m)
{
    smp__emit(d, m);
}

/* ========================================================================== */
/*  Итог                                                                      */
/* ========================================================================== */

static const char *smp__plural_ru(uint32_t n, const char *one,
                                  const char *few, const char *many)
{
    const uint32_t n100 = n % 100u, n10 = n % 10u;
    if (n100 >= 11u && n100 <= 14u) return many;
    if (n10 == 1u)                  return one;
    if (n10 >= 2u && n10 <= 4u)     return few;
    return many;
}

char *smp_diag_summary(const SmpDiagCtx *d, char *buf, size_t cap)
{
    if (d->n_fatal == 0 && d->n_warn == 0) {
        snprintf(buf, cap, "замечаний нет");
        return buf;
    }

    size_t n = 0;
    int    w;

    if (d->n_fatal) {
        w = snprintf(buf, cap, "%u %s", d->n_fatal,
                     smp__plural_ru(d->n_fatal, "фатальная ошибка",
                                    "фатальные ошибки", "фатальных ошибок"));
        if (w > 0) n = (size_t)w;
    }
    if (d->n_warn && n < cap) {
        w = snprintf(buf + n, cap - n, "%s%u %s", n ? ", " : "", d->n_warn,
                     smp__plural_ru(d->n_warn, "предупреждение",
                                    "предупреждения", "предупреждений"));
        if (w > 0) n += (size_t)w;
    }
    return buf;
}
