/* SMPC3 :: emit.c -- сборка байткода из дерева и результатов семантики. */
#include "smpc3/emit.h"

#include <string.h>
#include <stdio.h>

/* Жёсткие потолки. Они не «на всякий случай»: каждый заранее выделен в арене
 * одним куском, потому что рантайм не выделяет память, а компилятор не
 * пересобирает массивы на лету. */
#define EM_MAX_CODE   65536u
#define EM_MAX_TENS    4096u
#define EM_MAX_CONST   4096u
#define EM_MAX_STRS   65536u

typedef struct Em {
    SmpEmitter          *e;
    const SmpAstProgram *prog;
    const SmpSemaResult *sema;

    SmpInstr   *code;    uint32_t n_code;
    SmpTensor  *tens;    uint32_t n_tens;
    SmpConst   *consts;  uint32_t n_consts;
    SmpDbgLine *dbg;     uint32_t n_dbg;
    char       *strs;    uint32_t n_strs;
    uint32_t   *rnames;

    uint64_t arena_bytes[SMP_MAX_ARENAS];
    uint32_t max_arena;

    /* Распределение регистров. */
    uint16_t sym_reg[SMP_MAX_SYMS];    /* именованный регистр -> номер       */
    bool     sym_has_reg[SMP_MAX_SYMS];

    uint32_t n_named;                  /* сколько номеров занято под имена   */
    uint32_t next_temp;                /* курсор временных, сбрасывается     */
    uint32_t max_reg;                  /* максимум за всю программу          */
    bool     reg_scratch[SMP_MAX_REGS];/* регистр держит наш буфер           */

    SmpSpan  cur_span;                 /* для отладочной таблицы             */
} Em;

/* ========================================================================== */
/*  Ошибки                                                                    */
/* ========================================================================== */

static void eerr(Em *m, SmpDiagCode code, SmpSpan sp, const char *details)
{
    m->e->n_errors++;
    if (!m->e->diag) return;
    SmpDiagMsg d;
    memset(&d, 0, sizeof d);
    d.code = code; d.span = sp; d.details = details;
    smp_diag_emit(m->e->diag, &d);
}

static const char *efmt(Em *m, const char *f, ...) SMP_PRINTF(2, 3);
static const char *efmt(Em *m, const char *f, ...)
{
    if (!m->e->diag) return NULL;
    static char scratch[SMP_FMT_SLOTLEN];
    va_list ap;
    va_start(ap, f);
    vsnprintf(scratch, sizeof scratch, f, ap);
    va_end(ap);
    return smp_fmt(m->e->diag, "%s", scratch);
}

/* ========================================================================== */
/*  Пулы                                                                      */
/* ========================================================================== */

static uint32_t intern_str(Em *m, const char *p, uint32_t len)
{
    /* Линейный поиск: имён в программе десятки, экономить не на чем. */
    for (uint32_t i = 0; i < m->n_strs; ) {
        const uint32_t l = (uint32_t)strlen(m->strs + i);
        if (l == len && memcmp(m->strs + i, p, len) == 0) return i;
        i += l + 1u;
    }
    if (m->n_strs + len + 1u > EM_MAX_STRS) return 0;

    const uint32_t off = m->n_strs;
    memcpy(m->strs + off, p, len);
    m->strs[off + len] = '\0';
    m->n_strs += len + 1u;
    return off;
}

static uint32_t intern_str_name(Em *m, SmpName n) { return intern_str(m, n.p, n.len); }

static uint32_t intern_const(Em *m, SmpConst v)
{
    for (uint32_t i = 0; i < m->n_consts; i++)
        if (m->consts[i].u == v.u) return i;
    if (m->n_consts >= EM_MAX_CONST) return 0;
    m->consts[m->n_consts] = v;
    return m->n_consts++;
}

static uint32_t const_u64(Em *m, uint64_t u) { SmpConst c; c.u = u; return intern_const(m, c); }
static uint32_t const_f64(Em *m, double f)   { SmpConst c; c.f = f; return intern_const(m, c); }

/* Запись в таблице дескрипторов. Возвращает индекс или UINT32_MAX. */
static uint32_t add_tens(Em *m, const SmpValue *v, uint32_t abs_off,
                         uint32_t name_id, uint32_t arena_id)
{
    if (m->n_tens >= EM_MAX_TENS) {
        eerr(m, SMP_E0207, m->cur_span,
             efmt(m, "Дескрипторов больше %u.", EM_MAX_TENS));
        return 0xFFFFFFFFu;
    }
    SmpTensor *t = &m->tens[m->n_tens];
    memset(t, 0, sizeof *t);
    t->off     = abs_off;
    t->dtype   = (uint8_t)v->dtype;
    t->rank    = (uint8_t)v->rank;
    t->flags   = smp_tf_with_arena(v->flags, arena_id);
    t->name_id = name_id;

    uint32_t n = 1;
    for (uint32_t i = 0; i < v->rank; i++) {
        t->shape[i]  = v->shape[i];
        t->stride[i] = v->stride[i];
        n *= v->shape[i];
    }
    t->nelem = v->rank ? n : 1u;
    return m->n_tens++;
}

/* Дескриптор для значения, привязанного к символу (в т.ч. срез). */
static uint32_t tens_for_value(Em *m, const SmpValue *v, uint32_t fallback_arena)
{
    uint32_t base_off = 0, name_id = 0, arena = fallback_arena;
    if (v->sym != SMP_SYM_NONE) {
        const SmpSym *s = &m->sema->syms[v->sym];
        base_off = s->offset;
        name_id  = intern_str_name(m, s->name);
        arena    = s->arena_id;
    }
    return add_tens(m, v, base_off + v->byte_off, name_id, arena);
}

/* Свежий буфер в арене под промежуточный результат. */
static uint32_t tens_scratch(Em *m, const SmpValue *v, uint32_t arena_id)
{
    if (arena_id >= SMP_MAX_ARENAS) arena_id = 0;

    uint64_t nelem = 1;
    for (uint32_t i = 0; i < v->rank; i++) nelem *= v->shape[i];
    if (v->rank == 0) nelem = 1;
    const uint64_t sz = nelem * smp_dtype_size(v->dtype);

    uint64_t *cur = &m->arena_bytes[arena_id];
    *cur = SMP_ALIGN_UP(*cur, SMP_CACHELINE);
    if (*cur + sz > 0xFFFFFFFFull) {
        eerr(m, SMP_E0401, m->cur_span,
             efmt(m, "Арена #%u перевалила за 4 ГиБ при размещении временных.", arena_id));
        return 0xFFFFFFFFu;
    }
    const uint32_t off = (uint32_t)*cur;
    *cur += sz;
    if (arena_id > m->max_arena) m->max_arena = arena_id;

    char nm[32];
    snprintf(nm, sizeof nm, "%%t%u", m->n_tens);

    SmpValue tmp = *v;
    tmp.flags |= SMP_TF_ALIGN64 | SMP_TF_ALIGN32 | SMP_TF_CONTIG;
    return add_tens(m, &tmp, off, intern_str(m, nm, (uint32_t)strlen(nm)), arena_id);
}

/* ========================================================================== */
/*  Регистры и код                                                            */
/* ========================================================================== */

static uint32_t alloc_temp(Em *m)
{
    if (m->next_temp >= SMP_MAX_REGS) {
        eerr(m, SMP_E0603, m->cur_span,
             efmt(m, "Регистров всего %u, а инструкции нужен ещё один.", SMP_MAX_REGS));
        return 0;
    }
    const uint32_t r = m->next_temp++;
    if (m->next_temp > m->max_reg) m->max_reg = m->next_temp;
    m->reg_scratch[r] = false;
    return r;
}

static void emit_aux(Em *m, SmpOpcode op, uint8_t flags,
                     uint32_t d, uint32_t a, uint32_t b, uint32_t k, uint8_t aux)
{
    if (m->n_code >= EM_MAX_CODE) {
        eerr(m, SMP_E0207, m->cur_span,
             efmt(m, "Инструкций больше %u.", EM_MAX_CODE));
        return;
    }
    SmpInstr *i = &m->code[m->n_code];
    i->op    = (uint8_t)op;
    i->flags = flags;
    i->d     = (uint8_t)d;
    i->a     = (uint8_t)a;
    i->b     = (uint8_t)b;
    i->aux   = aux;
    i->k     = (uint16_t)k;

    if (m->n_dbg < EM_MAX_CODE) {
        m->dbg[m->n_dbg].instr = m->n_code;
        m->dbg[m->n_dbg].line  = m->cur_span.line;
        m->dbg[m->n_dbg].col   = m->cur_span.col;
        m->n_dbg++;
    }
    m->n_code++;
}

/* Большинству инструкций пул констант не нужен — им aux безразличен. */
static void emit(Em *m, SmpOpcode op, uint8_t flags,
                 uint32_t d, uint32_t a, uint32_t b, uint32_t k)
{
    emit_aux(m, op, flags, d, a, b, k, 0);
}

static uint8_t stmt_flags(const SmpStmtInfo *in)
{
    uint8_t f = smp_vec_code(in->vec_bits);
    if (in->ftz)      f |= SMP_IF_FTZ;
    if (in->strict)   f |= SMP_IF_STRICT;
    if (in->no_alias) f |= SMP_IF_NOALIAS;
    if (in->raw)      f |= SMP_IF_RAW;
    return f;
}

/* ========================================================================== */
/*  Операнды                                                                  */
/* ========================================================================== */

/* Кладёт значение в регистр и возвращает его номер. */
static uint32_t load_value(Em *m, const SmpAstOperand *o, const SmpValue *v,
                           uint8_t flags, uint32_t arena_id)
{
        if (o->kind == SMP_OPD_REG) {
        const SmpSym *s = NULL;
        for (uint32_t i = 0; i < m->sema->nsyms; i++) {
            const SmpSym *c = &m->sema->syms[i];
            if (c->kind == SMP_SYM_REG && c->name.len == o->reg.len &&
                memcmp(c->name.p, o->reg.p, o->reg.len) == 0) { s = c; break; }
        }
        if (s) {
            const uint32_t idx = (uint32_t)(s - m->sema->syms);
            if (m->sym_has_reg[idx]) return m->sym_reg[idx];
        }
        return alloc_temp(m);
    }

    if (o->kind == SMP_OPD_INT || o->kind == SMP_OPD_FLOAT) {
        const uint32_t r = alloc_temp(m);
        const bool     is_int = (o->kind == SMP_OPD_INT);
        const uint32_t k = is_int ? const_u64(m, o->ival) : const_f64(m, o->fval);
        emit_aux(m, SMP_BC_LOADK, flags, r, 0, 0, k,
                 (uint8_t)(is_int ? SMP_DT_I32 : SMP_DT_F64));
        return r;
    }

    /* Тензор, возможно со срезом. */
    const uint32_t t = tens_for_value(m, v, arena_id);
    if (t == 0xFFFFFFFFu) return 0;

    const uint32_t r = alloc_temp(m);
    emit(m, SMP_BC_LOADT, flags, r, 0, 0, t);

    /* Динамический индекс: статическая часть смещения уже в дескрипторе,
     * а вклад регистра добавляется на исполнении. */
    const SmpAstTensor *at = o->tensor;
    if (at && at->has_index && v->sym != SMP_SYM_NONE) {
        const SmpSym  *base = &m->sema->syms[v->sym];
        const uint32_t esz  = smp_dtype_size(base->val.dtype);
        for (uint32_t ax = 0; ax < at->nidx; ax++) {
            if (at->idx[ax].kind != SMP_IDX_REG) continue;

            uint32_t ridx = 0; bool found = false;
            for (uint32_t i = 0; i < m->sema->nsyms; i++) {
                const SmpSym *c = &m->sema->syms[i];
                if (c->kind == SMP_SYM_REG && c->name.len == at->idx[ax].reg.len &&
                    memcmp(c->name.p, at->idx[ax].reg.p, c->name.len) == 0) {
                    if (m->sym_has_reg[i]) { ridx = m->sym_reg[i]; found = true; }
                    break;
                }
            }
            if (!found) continue;
            const uint32_t stride_bytes = base->val.stride[ax] * esz;
            emit_aux(m, SMP_BC_SLICED, flags, r, r, ridx,
                     const_u64(m, stride_bytes), (uint8_t)SMP_DT_U64);
        }
    }
    return r;
}

/* ========================================================================== */
/*  Стадии                                                                    */
/* ========================================================================== */

/* Какому опкоду соответствует операция семантики. */
static SmpOpcode op_to_bc(SmpOpKind k)
{
    switch (k) {
        case SMP_OP_ALLOC:      return SMP_BC_ALLOC;
        case SMP_OP_FILL:       return SMP_BC_FILL;
        case SMP_OP_FILL_INST:  return SMP_BC_FILLI;
        case SMP_OP_MMUL:       return SMP_BC_MMUL;
        case SMP_OP_TRANSPOSE:  return SMP_BC_TRANS;
        case SMP_OP_PACK:       return SMP_BC_PACK;
        case SMP_OP_RELU:       return SMP_BC_RELU;
        case SMP_OP_ABS:        return SMP_BC_ABS;
        case SMP_OP_SCALE:      return SMP_BC_SCALE;
        case SMP_OP_ADD:        return SMP_BC_ADD;
        case SMP_OP_MUL:        return SMP_BC_MUL;
        case SMP_OP_REDUCE_ADD: return SMP_BC_REDADD;
        case SMP_OP_REDUCE_MAX: return SMP_BC_REDMAX;
        case SMP_OP_EMIT_TEXT:
        case SMP_OP_EMIT_LINE:
        case SMP_OP_EMIT_DEC:
        case SMP_OP_EMIT_HEX:
        case SMP_OP_EMIT_BITS:
        case SMP_OP_EMIT_NUM:   return SMP_BC_EMIT;
        case SMP_OP_CAST_F32:   return SMP_BC_CVTF32;
        case SMP_OP_CAST_F64:   return SMP_BC_CVTF64;
        case SMP_OP_CAST_I32:   return SMP_BC_CVTI32;
        default:                return SMP_BC_CVTU64;
    }
}

/* Нужен ли стадии собственный буфер под результат. */
static bool needs_buffer(SmpOpKind k)
{
    switch (k) {
        case SMP_OP_MMUL: case SMP_OP_PACK:
        case SMP_OP_CAST_F32: case SMP_OP_CAST_F64:
        case SMP_OP_CAST_I32: case SMP_OP_CAST_U64:
        case SMP_OP_RELU: case SMP_OP_ABS: case SMP_OP_SCALE:
        case SMP_OP_ADD:  case SMP_OP_MUL:
            return true;
        default:
            return false;   /* alloc, fill, transpose, reduce — на месте */
    }
}

/* ========================================================================== */
/*  Инструкция целиком                                                        */
/* ========================================================================== */

static void emit_stmt(Em *m, const SmpAstStmt *s, const SmpStmtInfo *in)
{
    if (!in->ok) return;

    m->cur_span   = s->span;
    m->next_temp  = m->n_named;
    const uint8_t flags = stmt_flags(in);

    /* Символы, которые инструкция ЧИТАЕТ. Писать результат прямо в приёмник
     * можно только если он ни разу не читается: иначе @mmul затрёт свой же
     * вход на полпути. */
    uint32_t reads[SMP_MAX_STAGES + 1];
    uint32_t n_reads = 0;
    if (in->src_val.sym != SMP_SYM_NONE) reads[n_reads++] = in->src_val.sym;
    for (uint32_t i = 0; i < s->nstages && n_reads < SMP_ARRLEN(reads); i++)
        if (in->arg_val[i].sym != SMP_SYM_NONE) reads[n_reads++] = in->arg_val[i].sym;

    const uint32_t dest_sym = in->dest_is_tensor ? in->dest_val.sym : SMP_SYM_NONE;
    bool dest_is_read = false;
    for (uint32_t i = 0; i < n_reads; i++)
        if (reads[i] == dest_sym) dest_is_read = true;

    /* --- источник --- */
    uint32_t r = load_value(m, &s->source, &in->src_val, flags, in->arena_id);

    /* --- стадии --- */
    for (uint32_t i = 0; i < s->nstages; i++) {
        const SmpAstStage *st = &s->stages[i];
        const SmpOpKind    k  = in->ops[i];
        const SmpOpcode    bc = op_to_bc(k);
        m->cur_span = st->span;

        uint32_t a = r, b = 0, kk = 0, d = r;
        uint8_t  kaux = 0;

        /* Аргумент стадии. */
        if (st->nargs >= 1) {
            if (k == SMP_OP_SCALE || k == SMP_OP_FILL) {
                const SmpAstOperand *ao = &st->args[0];
                if (ao->kind == SMP_OPD_INT) {
                    kk = const_u64(m, ao->ival); kaux = (uint8_t)SMP_DT_I32;
                } else if (ao->kind == SMP_OPD_FLOAT) {
                    kk = const_f64(m, ao->fval); kaux = (uint8_t)SMP_DT_F64;
                } else {
                    b = load_value(m, ao, &in->arg_val[i], flags, in->arena_id);
                }
            } else {
                b = load_value(m, &st->args[0], &in->arg_val[i], flags, in->arena_id);
            }
        }

        /* Вариант emit кодируется в aux — ровно так же, как тип константы
         * у инструкций, читающих пул. Новых полей инструкция не растит. */
        switch (k) {
            case SMP_OP_EMIT_TEXT: kaux = SMP_EMIT_TEXT; break;
            case SMP_OP_EMIT_LINE: kaux = SMP_EMIT_LINE; break;
            case SMP_OP_EMIT_DEC:  kaux = SMP_EMIT_DEC;  break;
            case SMP_OP_EMIT_HEX:  kaux = SMP_EMIT_HEX;  break;
            case SMP_OP_EMIT_BITS: kaux = SMP_EMIT_BITS; break;
            case SMP_OP_EMIT_NUM:  kaux = SMP_EMIT_NUM;  break;
            default: break;
        }

        if (needs_buffer(k)) {
            const bool last = (i + 1u == s->nstages);
            uint32_t t;

            if (last && in->dest_is_tensor && !dest_is_read &&
                (in->dest_val.flags & SMP_TF_CONTIG)) {
                /* Пишем прямо в приёмник — копия в конце не понадобится. */
                t = tens_for_value(m, &in->dest_val, in->arena_id);
            } else if (m->reg_scratch[a] && k != SMP_OP_MMUL && k != SMP_OP_PACK) {
                /* Вход уже наш временный буфер нужной формы — работаем на месте. */
                t = 0xFFFFFFFEu;
            } else {
                t = tens_scratch(m, &in->stage_out[i], in->arena_id);
            }

            if (t == 0xFFFFFFFFu) return;
            if (t != 0xFFFFFFFEu) {
                d = alloc_temp(m);
                emit(m, SMP_BC_LOADT, flags, d, 0, 0, t);
                m->reg_scratch[d] = true;
            } else {
                d = a;
            }
        } else if (k == SMP_OP_REDUCE_ADD || k == SMP_OP_REDUCE_MAX ||
                   (k >= SMP_OP_EMIT_TEXT && k <= SMP_OP_EMIT_NUM)) {
            d = alloc_temp(m);
            m->reg_scratch[d] = false;
        } else if (k == SMP_OP_TRANSPOSE) {
            d = alloc_temp(m);
            m->reg_scratch[d] = m->reg_scratch[a];
        }

        emit_aux(m, bc, flags, d, a, b, kk, kaux);
        r = d;
    }

    /* --- приёмник --- */
    m->cur_span = s->dest.span;

    if (!in->dest_is_tensor) {
        uint32_t dr = 0;
        for (uint32_t i = 0; i < m->sema->nsyms; i++) {
            const SmpSym *c = &m->sema->syms[i];
            if (c->kind == SMP_SYM_REG && c->name.len == s->dest.reg.len &&
                memcmp(c->name.p, s->dest.reg.p, c->name.len) == 0) {
                if (m->sym_has_reg[i]) dr = m->sym_reg[i];
                break;
            }
        }
        if (dr != r) emit(m, SMP_BC_MOVE, flags, dr, r, 0, 0);
        return;
    }

    /* Если последняя стадия уже писала в приёмник, копировать нечего. */
    const SmpOpKind lastk = s->nstages ? in->ops[s->nstages - 1u] : SMP_OP__COUNT;
    const bool wrote_direct = s->nstages && needs_buffer(lastk) &&
                              !dest_is_read && (in->dest_val.flags & SMP_TF_CONTIG);
    if (wrote_direct) return;

    const uint32_t dt = tens_for_value(m, &in->dest_val, in->arena_id);
    if (dt == 0xFFFFFFFFu) return;
    emit(m, SMP_BC_STORET, flags, 0, r, 0, dt);
}

/* ========================================================================== */
/*  Точка входа                                                               */
/* ========================================================================== */

void smp_emit_init(SmpEmitter *e, SmpArena *arena, SmpDiagCtx *diag,
                   const SmpSource *src)
{
    memset(e, 0, sizeof *e);
    e->arena = arena;
    e->diag  = diag;
    e->src   = src;
}

SmpStatus smp_emit(SmpEmitter *e, const SmpAstProgram *prog,
                   const SmpSemaResult *sema, SmpModule *out)
{
    memset(out, 0, sizeof *out);

    Em m;
    memset(&m, 0, sizeof m);
    m.e = e; m.prog = prog; m.sema = sema;

    m.code   = (SmpInstr   *)smp_arena_push_raw(e->arena, EM_MAX_CODE  * sizeof(SmpInstr), 8);
    m.tens   = (SmpTensor  *)smp_arena_push_raw(e->arena, EM_MAX_TENS  * sizeof(SmpTensor), 8);
    m.consts = (SmpConst   *)smp_arena_push_raw(e->arena, EM_MAX_CONST * sizeof(SmpConst), 8);
    m.dbg    = (SmpDbgLine *)smp_arena_push_raw(e->arena, EM_MAX_CODE  * sizeof(SmpDbgLine), 8);
    m.strs   = (char       *)smp_arena_push_raw(e->arena, EM_MAX_STRS, 8);
    m.rnames = (uint32_t   *)smp_arena_push_raw(e->arena, SMP_MAX_REGS * sizeof(uint32_t), 8);
    if (!m.code || !m.tens || !m.consts || !m.dbg || !m.strs || !m.rnames)
        return SMP_ERR_OOM;
    memset(m.rnames, 0, SMP_MAX_REGS * sizeof(uint32_t));

    /* Нулевое смещение строковой таблицы занимает пустая строка: name_id == 0
     * должен означать «имени нет», а не указывать на чужое имя. */
    m.strs[0] = '\0';
    m.n_strs  = 1;

    memcpy(m.arena_bytes, sema->arena_bytes, sizeof m.arena_bytes);
    m.max_arena = sema->max_arena_id;

    /* Именованные регистры получают постоянные номера на всю программу. */
    for (uint32_t i = 0; i < sema->nsyms; i++) {
        if (sema->syms[i].kind != SMP_SYM_REG) continue;
        if (m.n_named >= SMP_MAX_REGS) {
            eerr(&m, SMP_E0603, sema->syms[i].decl_span,
                 efmt(&m, "Именованных регистров больше %u.", SMP_MAX_REGS));
            return SMP_ERR_INTERNAL;
        }
        m.rnames[m.n_named] = intern_str_name(&m, sema->syms[i].name);
        m.sym_reg[i]        = (uint16_t)m.n_named++;
        m.sym_has_reg[i]    = true;
    }
    m.max_reg = m.n_named;

    for (uint32_t i = 0; i < prog->nstmts && i < sema->ninfo; i++)
        emit_stmt(&m, &prog->stmts[i], &sema->info[i]);

    m.cur_span = (SmpSpan){ 0, 0, 0 };
    emit(&m, SMP_BC_HALT, 0, 0, 0, 0, 0);

    out->code        = m.code;      out->n_code   = m.n_code;
    out->tens        = m.tens;      out->n_tens   = m.n_tens;
    out->consts      = m.consts;    out->n_consts = m.n_consts;
    out->dbg         = m.dbg;       out->n_dbg    = m.n_dbg;
    out->strs        = m.strs;      out->strs_size = m.n_strs;
    out->reg_names   = m.rnames;    out->n_reg_names = m.n_named;
    out->n_regs      = m.max_reg;

    uint64_t *ab = (uint64_t *)smp_arena_push_raw(e->arena,
                        (m.max_arena + 1u) * sizeof(uint64_t), 8);
    if (!ab) return SMP_ERR_OOM;
    for (uint32_t i = 0; i <= m.max_arena; i++)
        ab[i] = SMP_ALIGN_UP(m.arena_bytes[i], SMP_CACHELINE);
    out->arena_bytes = ab;
    out->n_arenas    = m.max_arena + 1u;

    return e->n_errors ? SMP_ERR_INTERNAL : SMP_OK;
}
