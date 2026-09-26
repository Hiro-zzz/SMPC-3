/* SMPC3 :: vm.c -- прямая диспетчеризация байткода. */
#include "smpc3/vm.h"
#include "smpc3/cpu.h"
#include "smpc3/fmath.h"
#include "smpc3/kernels.h"
#include "smpc3/plat.h"
#include "smpc3/thread.h"

#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

/* ========================================================================== */
/*  Инициализация                                                             */
/* ========================================================================== */

SmpStatus smp_vm_init(SmpVM *vm, const SmpModule *mod, SmpDiagCtx *diag)
{
    return smp_vm_init_ex(vm, mod, diag, smp_threads_default());
}

SmpStatus smp_vm_init_ex(SmpVM *vm, const SmpModule *mod, SmpDiagCtx *diag,
                         uint32_t par)
{
    memset(vm, 0, sizeof *vm);
    vm->mod  = mod;
    vm->diag = diag;

    vm->n_arenas = mod->n_arenas < SMP_MAX_ARENAS ? mod->n_arenas : SMP_MAX_ARENAS;

    for (uint32_t i = 0; i < vm->n_arenas; i++) {
        /* Пустую арену всё равно поднимаем: инвариант «база выровнена на 64»
         * должен держаться и там, где ничего не выделено. */
        const size_t sz = mod->arena_bytes[i] ? (size_t)mod->arena_bytes[i] : SMP_CACHELINE;
        if (smp_arena_init(&vm->arenas[i], sz, i, "vm") != SMP_OK) {
            for (uint32_t j = 0; j < i; j++) smp_arena_release(&vm->arenas[j]);
            return SMP_ERR_OOM;
        }
        /* Вся арена занята сразу: раскладку посчитал компилятор, курсор здесь
         * никому не нужен, а границы проверяются по cap. */
        (void)smp_arena_push(&vm->arenas[i], sz, SMP_CACHELINE);
    }

    vm->n_regs = mod->n_regs < SMP_MAX_REGS ? mod->n_regs : SMP_MAX_REGS;

    /* Рабочая память ядер выделяется здесь, на подъёме, и больше не растёт.
     * У каждого инстанса она своя — без этого два потока в GEMM затёрли бы
     * панели друг друга. Комплектов — по одному на поток крупного GEMM. */
    vm->scratch_bytes = smp_k_scratch_bytes_for(par ? par : 1u);
    vm->scratch_mem   = smp_plat_pages(vm->scratch_bytes);
    if (!vm->scratch_mem) {
        for (uint32_t j = 0; j < vm->n_arenas; j++) smp_arena_release(&vm->arenas[j]);
        return SMP_ERR_OOM;
    }
    smp_k_scratch_bind(&vm->scratch, vm->scratch_mem, vm->scratch_bytes);

    vm->entry_mxcsr  = smp_fpu_get_mxcsr();
    vm->cur_fp_flags = 0;
    vm->live         = true;
    return SMP_OK;
}

void smp_vm_release(SmpVM *vm)
{
    /* Идемпотентно и безопасно на обнулённой структуре: см. комментарий к
     * полю live в vm.h — восстановление MXCSR из нуля снимает маски всех
     * исключений FPU. */
    if (!vm->live) return;

    smp_fpu_set_mxcsr(vm->entry_mxcsr);
    for (uint32_t i = 0; i < vm->n_arenas; i++) smp_arena_release(&vm->arenas[i]);
    vm->n_arenas = 0;

    smp_plat_pages_free(vm->scratch_mem, vm->scratch_bytes);
    vm->scratch_mem   = NULL;
    vm->scratch_bytes = 0;
    memset(&vm->scratch, 0, sizeof vm->scratch);

    if (vm->view_mem) smp_plat_pages_free(vm->view_mem, (size_t)vm->mod->view_bytes);
    vm->view_mem = NULL;
    vm->n_views  = 0;

    vm->live = false;
}

const char *smp_vm_backend(void) { return smp_kernels_name(); }

/* ========================================================================== */
/*  Диагностика времени исполнения                                            */
/* ========================================================================== */

/* Профиль: время с прошлой диспетчеризации — инструкции, которая только
 * что кончилась. Слитая цепочка записывается на свою первую инструкцию. */
static void vm_prof(SmpVM *vm)
{
    const double now = smp_plat_seconds();
    if (vm->prof_started && vm->pc < vm->mod->n_code)
        vm->prof[vm->pc] += now - vm->prof_last;
    vm->prof_last    = now;
    vm->prof_started = true;
}

static void vm_fatal(SmpVM *vm, SmpDiagCode code, const char *details,
                     const char *fix)
{
    vm->trapped = true;
    if (!vm->diag) return;

    const SmpDbgLine dl = smp_module_dbg(vm->mod, vm->pc);

    SmpDiagMsg m;
    memset(&m, 0, sizeof m);
    m.code      = code;
    m.span      = (SmpSpan){ dl.line, dl.col, 1 };
    m.details   = details;
    m.fix       = fix;
    m.dump_regs = true;
    smp_diag_emit(vm->diag, &m);
}

static const char *vfmt(SmpVM *vm, const char *f, ...) SMP_PRINTF(2, 3);
static const char *vfmt(SmpVM *vm, const char *f, ...)
{
    if (!vm->diag) return NULL;
    /* Буфер локальный, а не статический: в пуле инстансов несколько потоков
     * могут собирать диагностику одновременно. */
    char scratch[SMP_FMT_SLOTLEN];
    va_list ap;
    va_start(ap, f);
    vsnprintf(scratch, sizeof scratch, f, ap);
    va_end(ap);
    return smp_fmt(vm->diag, "%s", scratch);
}

/* Имя регистра из модуля; для временных — просто rN. */
static const char *reg_label(const SmpVM *vm, uint32_t i, char *buf, size_t cap)
{
    const SmpModule *m = vm->mod;
    if (m->reg_names && i < m->n_reg_names && m->reg_names[i]) {
        snprintf(buf, cap, "$%s", smp_module_str(m, m->reg_names[i]));
    } else {
        snprintf(buf, cap, "r%u", i);
    }
    return buf;
}

void smp_vm_regdump(SmpDiagCtx *d, void *user, bool color)
{
    const SmpVM *vm = (const SmpVM *)user;
    const char *dim = color ? "\x1b[2m" : "";
    const char *rst = color ? "\x1b[0m" : "";

    smp_diag_write(d, "%s  --- ДАМП РЕГИСТРОВ (инструкция %u) ------------------%s\n",
                   dim, vm->pc, rst);

    for (uint32_t i = 0; i < vm->n_regs; i++) {
        const SmpReg *r = &vm->regs[i];
        if (!r->is_tensor && r->dtype == 0) continue;   /* никогда не писался */

        char lbl[64];
        reg_label(vm, i, lbl, sizeof lbl);

        if (r->is_tensor) {
            char sig[80];
            smp_tensor_sig(&r->t, sig, sizeof sig);
            smp_diag_write(d, "   %-8s = tensor{ a%u:0x%08llX  %-18s stride=",
                           lbl, smp_tf_arena(r->t.flags),
                           (unsigned long long)r->t.off, sig);
            for (uint32_t k = 0; k < r->t.rank; k++)
                smp_diag_write(d, "%s%u", k ? "," : "", (unsigned)r->t.stride[k]);
            smp_diag_write(d, "  flags=%s%s%s }\n",
                    (r->t.flags & SMP_TF_CONTIG)     ? "CONTIG" : "STRIDED",
                    (r->t.flags & SMP_TF_ALIGN64)    ? "|A64"   : "",
                    (r->t.flags & SMP_TF_TRANSPOSED) ? "|T"     : "");
        } else {
            smp_diag_write(d, "   %-8s = %-8s %.9g\n", lbl,
                           smp_dtype_name((SmpDType)r->dtype), r->s.f);
        }
    }

    const SmpInstr *in = (vm->pc < vm->mod->n_code) ? &vm->mod->code[vm->pc] : NULL;
    smp_diag_write(d, "   MXCSR = 0x%04X   ядра = %s   исполнено = %llu\n",
                   smp_fpu_get_mxcsr(), smp_kernels_name(),
                   (unsigned long long)vm->n_executed);
    if (in)
        smp_diag_write(d, "   опкод = %s   ширина = %s\n",
                       smp_opcode_def((SmpOpcode)in->op)->mnemonic,
                       smp_vec_name(smp_vec_bits(in->flags & SMP_IF_VEC_MASK)));
}

/* ========================================================================== */
/*  Доступ к памяти                                                           */
/* ========================================================================== */

static SmpView *view_find(SmpVM *vm, uint32_t name_id)
{
    for (uint32_t i = 0; i < vm->n_views; i++)
        if (vm->views[i].name_id == name_id) return &vm->views[i];
    return NULL;
}

/* @store заменил объект под именем вида. Адрес берётся заново, если объект
 * остался того же размера (*&W -> @load => *&W; *&W -> @store => $n;), а
 * иначе забывается: читать вид дальше можно только после нового @load. */
static void view_remap(SmpVM *vm, uint32_t name_id)
{
    SmpView *v = view_find(vm, name_id);
    if (!v) return;

    const SmpStore *st   = vm->store;
    const char     *name = smp_module_str(vm->mod, name_id);
    SmpTensor have;
    memset(&have, 0, sizeof have);
    const void *p = NULL;
    if (st && st->map && st->find(st->ctx, name, &have)) {
        uint64_t n = 1;
        for (uint32_t i = 0; i < have.rank; i++) n *= have.shape[i];
        if (smp_dtype_bytes((SmpDType)have.dtype, n) == v->bytes) p = st->map(st->ctx, name);
    }
    if (p) v->p = (const uint8_t *)p;
    else   *v = vm->views[--vm->n_views];
}

/* Вид или его срез: по адресу, который отдало хранилище, либо в своей копии
 * пространства видов. span — сколько байт от t->off занимает тензор. */
static void *view_base(SmpVM *vm, const SmpTensor *t, uint64_t span)
{
    const char    *name = smp_module_str(vm->mod, t->name_id);
    const SmpView *v    = view_find(vm, t->name_id);
    if (v) {
        if (t->off < v->off || t->off - v->off > v->bytes || span > v->bytes - (t->off - v->off)) {
            vm_fatal(vm, SMP_E0410,
                     vfmt(vm, "Срез вида '%s' требует байт до %llu, а объект занимает %llu.",
                          name, (unsigned long long)(t->off - v->off + span),
                          (unsigned long long)v->bytes), NULL);
            return NULL;
        }
        return (void *)(uintptr_t)(v->p + (t->off - v->off));
    }
    if (vm->view_mem) {
        if (t->off + span > vm->mod->view_bytes) {
            vm_fatal(vm, SMP_E0410,
                     vfmt(vm, "Вид '%s' требует байт до %llu, а видам отведено %llu.",
                          name, (unsigned long long)(t->off + span),
                          (unsigned long long)vm->mod->view_bytes), NULL);
            return NULL;
        }
        return vm->view_mem + t->off;
    }
    vm_fatal(vm, SMP_E0611,
             vfmt(vm, "Вид '%s' читают, а адреса у него на этом прогоне нет: @load "
                      "его ещё не было, либо объект с тех пор заменили через @store.",
                  name),
             vfmt(vm, "Поставь *&%s -> @load => *&%s; до первого чтения.", name, name));
    return NULL;
}

/* Указатель на первый элемент тензора с проверкой границ арены.
 * NULL означает, что диагностика уже выдана. */
static void *tensor_base(SmpVM *vm, const SmpTensor *t)
{
    /* Самый дальний элемент по шагам — именно он, а не nelem*esz, задаёт
     * реальный хвост у среза с шагом. */
    uint64_t last = 0;
    for (uint32_t i = 0; i < t->rank; i++)
        last += (uint64_t)(t->shape[i] ? t->shape[i] - 1u : 0u) * t->stride[i];
    const uint64_t span = smp_dtype_bytes((SmpDType)t->dtype, last + 1u);

    if (t->flags & SMP_TF_EXTERN) return view_base(vm, t, span);

    const uint32_t a = smp_tf_arena(t->flags);
    if (a >= vm->n_arenas) {
        vm_fatal(vm, SMP_E0604,
                 vfmt(vm, "Дескриптор ссылается на арену #%u, а их поднято %u.",
                      a, vm->n_arenas), NULL);
        return NULL;
    }

    const uint64_t need = t->off + span;
    if (need > vm->arenas[a].cap) {
        vm_fatal(vm, SMP_E0410,
                 vfmt(vm, "Тензор требует байт до %llu, а арена #%u занимает %zu.\n"
                          "Дескриптор описывает область за её пределами.",
                      (unsigned long long)need, a, vm->arenas[a].cap), NULL);
        return NULL;
    }
    return vm->arenas[a].base + t->off;
}

static bool make_buf(SmpVM *vm, SmpBuf *b, const SmpTensor *t)
{
    b->t = t;
    b->p = tensor_base(vm, t);
    return b->p != NULL;
}

/* Буфер, в который инструкция пишет. Вид смотрит в чужую память — в объект
 * хранилища, — и компилятор видом делает только то, во что программа не
 * пишет. Сюда доходит лишь собранный вручную или испорченный байткод. */
static bool make_dst(SmpVM *vm, SmpBuf *b, const SmpTensor *t)
{
    if (t->flags & SMP_TF_EXTERN) {
        const char *name = smp_module_str(vm->mod, t->name_id);
        vm_fatal(vm, SMP_E0612,
                 vfmt(vm, "Инструкция пишет в '%s', а это вид: он только для чтения.", name),
                 NULL);
        return false;
    }
    return make_buf(vm, b, t);
}

/* Операнд бинарной поэлементной операции в форме результата. Ось размера 1,
 * которая у результата длиннее, растягивается шагом 0 — тот же элемент на
 * всю ось, без копии; такой вид не плотный, и считает скалярный эталон.
 * Совпадает ли что с чем, проверила сема; здесь только сборка вида. */
static bool make_bcast(SmpVM *vm, SmpBuf *b, const SmpTensor *t, const SmpTensor *like,
                       SmpTensor *tmp)
{
    if (t->nelem == like->nelem) return make_buf(vm, b, t);
    *tmp = *t;
    for (uint32_t i = 0; i < t->rank; i++) {
        if (t->shape[i] == 1 && like->shape[i] != 1) {
            tmp->shape[i]  = like->shape[i];
            tmp->stride[i] = 0;
        }
    }
    tmp->nelem  = like->nelem;
    tmp->flags &= (uint16_t)~(SMP_TF_CONTIG | SMP_TF_ALIGN64 | SMP_TF_ALIGN32);
    return make_buf(vm, b, tmp);
}

/* ========================================================================== */
/*  Умножение по живой длине                                                  */
/* ========================================================================== */

/* Длина из регистра k (aux = 1): сколько первых строк B живые у @mmul.t,
 * сколько первых элементов оси K — у @mmul. Меньше единицы или не число —
 * ноль, больше оси — вся ось; как у @softmax. */
static uint32_t live_len(const SmpVM *vm, const SmpInstr *in, uint32_t full)
{
    const SmpReg *r = &vm->regs[in->k];
    const double  n = r->is_tensor ? 0.0 : r->s.f;
    return n >= (double)full ? full : n >= 1.0 ? (uint32_t)n : 0u;
}

/* Первые n столбцов C = A x (первые n строк B)^T, остальные — нули. Каждый
 * столбец — своё скалярное произведение, так что живые считаются ровно как
 * без длины; мёртвые не считаются вовсе. */
static void mmul_t_live(SmpVM *vm, const SmpInstr *in, const SmpBuf *c,
                        const SmpBuf *a, const SmpBuf *b)
{
    const uint32_t M = c->t->shape[0], N = c->t->shape[1];
    const uint32_t n = live_len(vm, in, N);
    if (n > 0) {
        SmpTensor ct = *c->t, bt = *b->t;
        ct.shape[1] = n;
        ct.nelem    = M * n;
        bt.shape[0] = n;
        bt.nelem    = n * bt.shape[1];
        if (n < N) ct.flags = (uint16_t)(ct.flags & ~SMP_TF_CONTIG);
        const SmpBuf c2 = { c->p, &ct }, b2 = { b->p, &bt };
        smp_k_mmul_t(&c2, a, &b2, &vm->scratch);
    }
    const size_t esz = smp_dtype_size((SmpDType)c->t->dtype);
    for (uint32_t m = 0; m < M && n < N; m++)
        memset((uint8_t *)c->p + ((size_t)m * c->t->stride[0] + n) * esz, 0,
               (size_t)(N - n) * esz);
}

/* C = (первые k столбцов A) x (первые k строк B): ось K обрезана. */
static void mmul_live(SmpVM *vm, const SmpInstr *in, const SmpBuf *c,
                      const SmpBuf *a, const SmpBuf *b)
{
    const uint32_t K = a->t->shape[1];
    const uint32_t k = live_len(vm, in, K);
    if (k == 0) {
        smp_k_fill(c, 0.0);
        return;
    }
    SmpTensor at = *a->t, bt = *b->t;
    at.shape[1] = k;
    at.nelem    = at.shape[0] * k;
    bt.shape[0] = k;
    bt.nelem    = k * bt.shape[1];
    if (k < K) at.flags = (uint16_t)(at.flags & ~SMP_TF_CONTIG);
    const SmpBuf a2 = { a->p, &at }, b2 = { b->p, &bt };
    smp_k_gemm(c, &a2, &b2, &vm->scratch);
}

/* ========================================================================== */
/*  Режимы с плавающей точкой                                                 */
/* ========================================================================== */

/* MXCSR трогаем, только когда режим действительно сменился: запись в него
 * сериализует конвейер, делать это на каждой инструкции незачем. */
static void apply_fp(SmpVM *vm, uint8_t flags)
{
    const uint8_t want = flags & SMP_IF_FTZ;
    if (want == vm->cur_fp_flags) return;
    smp_fpu_set_ftz(want != 0);
    vm->cur_fp_flags = want;
}

/* Под ?strict вырожденный результат — фатальная ошибка, а не «ну как-то так». */
static bool strict_check(SmpVM *vm, uint8_t flags, double v, const char *what)
{
    if (!(flags & SMP_IF_STRICT)) return true;
    if (!isnan(v) && !isinf(v))   return true;

    vm_fatal(vm, SMP_E0602,
             vfmt(vm, "%s дало %s.\nАктивен ?strict, поэтому продолжения не будет.",
                  what, isnan(v) ? "NaN" : "бесконечность"), NULL);
    return false;
}

/* ========================================================================== */
/*  Числа                                                                     */
/* ========================================================================== */

/* Число из регистра: скаляр как есть или единственный элемент тензора — так
 * приходит *&v[2], вид на один элемент. false — диагностика уже выдана. */
static bool reg_num(SmpVM *vm, const SmpReg *r, double *out)
{
    if (!r->is_tensor) { *out = r->s.f; return true; }
    SmpBuf b;
    if (r->t.nelem != 1u) {
        vm_fatal(vm, SMP_E0604, "Число ждали в регистре, а там тензор.", NULL);
        return false;
    }
    if (!make_buf(vm, &b, &r->t)) return false;
    *out = smp_k_reduce_add(&b);
    return true;
}

static uint8_t reg_dtype(const SmpReg *r)
{
    return r->is_tensor ? r->t.dtype : r->dtype;
}

/* Число, которым заполняет @fill и умножает @scale: константа из пула, а при
 * aux == SMP_AUX_REG — регистр b. */
static bool stage_num(SmpVM *vm, const SmpInstr *in, double *out)
{
    if (in->aux != SMP_AUX_REG) {
        *out = smp_const_as_double(vm->mod->consts[in->k], in->aux);
        return true;
    }
    return reg_num(vm, &vm->regs[in->b], out);
}

/* Арифметика над числами (aux == SMP_AUX_SCALAR). Скаляр в регистре —
 * double с логическим типом; результат округляется в этот тип ровно так,
 * как посчитало бы железо: для f32 двойное округление через double
 * безвредно, у double на 53 бита запаса больше чем вдвое. Целые не
 * насыщаются: переполнение — ошибка, а не молча другое число. Целого
 * деления на железе здесь нет вовсе, поэтому и #DE в Ring 0 взяться неоткуда:
 * деление на ноль ловится до него. */
static bool scalar_op(SmpVM *vm, const SmpInstr *in)
{
    SmpReg *R = vm->regs;
    double  x = 0.0, y = 0.0, r;
    if (!reg_num(vm, &R[in->a], &x)) return false;
    const bool two = in->op != SMP_BC_ABS && in->op != SMP_BC_SQRT;
    if (two && !reg_num(vm, &R[in->b], &y)) return false;

    const SmpDType dt     = (SmpDType)reg_dtype(&R[in->a]);
    const bool     is_int = (dt == SMP_DT_I32 || dt == SMP_DT_U64);

    switch (in->op) {
        case SMP_BC_ADD:  r = x + y; break;
        case SMP_BC_SUB:  r = x - y; break;
        case SMP_BC_MUL:  r = x * y; break;
        case SMP_BC_ABS:  r = x < 0.0 ? -x : x; break;
        case SMP_BC_SQRT: r = smp_sqrt(x); break;
        default:
            if (is_int && y == 0.0) {
                vm_fatal(vm, SMP_E0601,
                         vfmt(vm, "%.17g делится на ноль в %s.", x, smp_dtype_name(dt)),
                         NULL);
                return false;
            }
            r = x / y;
            /* Целое деление отбрасывает дробную часть, к нулю — как в C. */
            if (is_int) r = r < 0.0 ? -(double)(uint64_t)(-r) : (double)(uint64_t)r;
            break;
    }

    if (dt == SMP_DT_F32) r = (double)(float)r;
    if ((dt == SMP_DT_I32 && (r < -2147483648.0 || r > 2147483647.0)) ||
        (dt == SMP_DT_U64 && (r < 0.0 || r >= 18446744073709551616.0))) {
        vm_fatal(vm, SMP_E0613,
                 vfmt(vm, "Получилось %.17g, а это %s.", r, smp_dtype_name(dt)), NULL);
        return false;
    }
    if (!strict_check(vm, in->flags, r, smp_opcode_def((SmpOpcode)in->op)->mnemonic))
        return false;

    R[in->d].is_tensor = false;
    R[in->d].s.f       = r;
    R[in->d].dtype     = (uint8_t)dt;
    return true;
}

/* ========================================================================== */
/*  Обмен с файлами                                                           */
/* ========================================================================== */

void smp_vm_bind(SmpVM *vm, const SmpBind *binds, uint32_t n)
{
    vm->binds   = binds;
    vm->n_binds = n;
}

void smp_vm_store(SmpVM *vm, const SmpStore *store)
{
    vm->store = store;
}

/* Привязка ищется по имени тензора и направлению. Вход и выход разведены
 * намеренно: одно и то же имя может быть привязано и на чтение, и на запись —
 * прочитать вектор, посчитать, положить обратно рядом. */
static const SmpBind *bind_find(const SmpVM *vm, const char *name, bool write)
{
    for (uint32_t i = 0; i < vm->n_binds; i++)
        if (vm->binds[i].write == write && vm->binds[i].name &&
            strcmp(vm->binds[i].name, name) == 0)
            return &vm->binds[i];
    return NULL;
}

/* Общая часть @load и @store: найти привязку либо объяснить, чего не хватает. */
static const SmpBind *bind_or_fatal(SmpVM *vm, const SmpTensor *t, bool write)
{
    const char    *name = smp_module_str(vm->mod, t->name_id);
    const SmpBind *b    = bind_find(vm, name, write);
    if (b) return b;

    vm_fatal(vm, SMP_E0606,
             vfmt(vm, "Тензор '%s' участвует в @%s, но к файлу не привязан.",
                  name, write ? "store" : "load"),
             vfmt(vm, "Добавь к запуску: --%s %s=путь",
                  write ? "out" : "in", name));
    return NULL;
}

/* Чтение файла прямо в арену: промежуточного буфера нет, и обещание про
 * отсутствие аллокаций на исполнении остаётся в силе — место уже выделено
 * компилятором, мы лишь заполняем его байтами. */
static bool file_load(SmpVM *vm, const SmpTensor *t, const SmpBuf *buf)
{
    const SmpBind *b = bind_or_fatal(vm, t, false);
    if (!b) return false;

    const uint64_t want = smp_tensor_bytes(t);

    FILE *f = fopen(b->path, "rb");
    if (!f) {
        vm_fatal(vm, SMP_E0607,
                 vfmt(vm, "Файл '%s' не открывается на чтение.", b->path), NULL);
        return false;
    }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); goto bad_size; }
    {
        const long sz = ftell(f);
        rewind(f);
        if (sz < 0 || (uint64_t)sz != want) {
            char sig[80];
            vm_fatal(vm, SMP_E0607,
                     vfmt(vm, "Файл '%s' занимает %lld байт, а тензор %s просит %llu.",
                          b->path, (long long)sz, smp_tensor_sig(t, sig, sizeof sig),
                          (unsigned long long)want),
                     "Размер обязан совпасть байт в байт: заголовка у файла нет, "
                     "и подогнать его не по чему.");
            fclose(f);
            return false;
        }
    }
    {
        const size_t got = want ? fread(buf->p, 1, (size_t)want, f) : 0u;
        fclose(f);
        if (got != (size_t)want) goto bad_size;
    }
    return true;

bad_size:
    vm_fatal(vm, SMP_E0607,
             vfmt(vm, "Файл '%s' не дочитался до конца.", b->path), NULL);
    return false;
}

static bool file_store(SmpVM *vm, const SmpTensor *t, const SmpBuf *buf,
                       uint64_t *written)
{
    const SmpBind *b = bind_or_fatal(vm, t, true);
    if (!b) return false;

    const uint64_t want = smp_tensor_bytes(t);

    FILE *f = fopen(b->path, "wb");
    if (!f) {
        vm_fatal(vm, SMP_E0607,
                 vfmt(vm, "Файл '%s' не открывается на запись.", b->path), NULL);
        return false;
    }
    const size_t put = want ? fwrite(buf->p, 1, (size_t)want, f) : 0u;
    const int    err = fclose(f);
    if (put != (size_t)want || err != 0) {
        vm_fatal(vm, SMP_E0607,
                 vfmt(vm, "В файл '%s' записано %llu байт из %llu.",
                      b->path, (unsigned long long)put, (unsigned long long)want),
                 "Проверь права и свободное место.");
        return false;
    }
    *written = want;
    return true;
}

/* ========================================================================== */
/*  Обмен с хранилищем                                                        */
/* ========================================================================== */

static bool same_kind(const SmpTensor *a, const SmpTensor *b)
{
    if (a->dtype != b->dtype || a->rank != b->rank) return false;
    for (uint32_t i = 0; i < a->rank; i++)
        if (a->shape[i] != b->shape[i]) return false;
    return true;
}

/* Есть ли объект и тот ли он. Сверяется не размер, а тип и форма: объект
 * хранилища их помнит, и различать f32:8 и f64:4 здесь есть чем. */
static bool store_check(SmpVM *vm, const SmpTensor *t)
{
    const SmpStore *st   = vm->store;
    const char     *name = smp_module_str(vm->mod, t->name_id);

    SmpTensor have;
    memset(&have, 0, sizeof have);
    if (!st->find(st->ctx, name, &have)) {
        vm_fatal(vm, SMP_E0608,
                 vfmt(vm, "@load ищет '%s', а такого объекта в хранилище нет.", name),
                 vfmt(vm, "Сначала положи его туда: *&%s -> @store => $n;", name));
        return false;
    }
    if (!same_kind(&have, t)) {
        char a[80], b[80];
        vm_fatal(vm, SMP_E0609,
                 vfmt(vm, "В хранилище лежит %s<%s>, а объявлен %s<%s>.",
                      name, smp_tensor_sig(&have, a, sizeof a),
                      name, smp_tensor_sig(t, b, sizeof b)),
                 NULL);
        return false;
    }
    return true;
}

/* Как у файла — прямо в арену. */
static bool store_load(SmpVM *vm, const SmpTensor *t, const SmpBuf *buf)
{
    if (!store_check(vm, t)) return false;
    vm->store->read(vm->store->ctx, smp_module_str(vm->mod, t->name_id),
                    buf->p, smp_tensor_bytes(t));
    return true;
}

/* Вид: адрес объекта вместо копии. Держится до конца прогона, до @store
 * того же имени или до следующего @load. */
static bool store_map(SmpVM *vm, const SmpTensor *t)
{
    if (!store_check(vm, t)) return false;

    const char *name = smp_module_str(vm->mod, t->name_id);
    const void *p    = vm->store->map(vm->store->ctx, name);
    SmpView    *v    = view_find(vm, t->name_id);
    if (!p || (!v && vm->n_views == SMP_MAX_VIEWS)) {
        vm_fatal(vm, SMP_E0611,
                 vfmt(vm, "Хранилище нашло '%s', но адреса не дало.", name), NULL);
        return false;
    }
    if (!v) v = &vm->views[vm->n_views++];
    v->name_id = t->name_id;
    v->off     = t->off;
    v->bytes   = smp_tensor_bytes(t);
    v->p       = (const uint8_t *)p;
    return true;
}

static bool store_store(SmpVM *vm, const SmpTensor *t, const SmpBuf *buf,
                        uint64_t *written)
{
    const SmpStore *st   = vm->store;
    const char     *name = smp_module_str(vm->mod, t->name_id);
    const uint64_t  want = smp_tensor_bytes(t);

    if (!st->write(st->ctx, name, t, buf->p, want)) {
        char sig[80];
        vm_fatal(vm, SMP_E0610,
                 vfmt(vm, "'%s' <%s> — %llu байт — положить некуда.",
                      name, smp_tensor_sig(t, sig, sizeof sig),
                      (unsigned long long)want),
                 NULL);
        return false;
    }
    *written = want;
    return true;
}

/* ========================================================================== */
/*  Слияние поэлементных стадий                                               */
/* ========================================================================== */

/* Какой стадии слитой цепочки соответствует опкод; -1 — не сливается. */
static int fuse_op_of(uint8_t bc)
{
    switch (bc) {
        case SMP_BC_RELU:  return (int)SMP_FOP_RELU;
        case SMP_BC_ABS:   return (int)SMP_FOP_ABS;
        case SMP_BC_SCALE: return (int)SMP_FOP_SCALE;
        case SMP_BC_ADD:   return (int)SMP_FOP_ADD;
        case SMP_BC_MUL:   return (int)SMP_FOP_MUL;
        default:           return -1;
    }
}

/* Какой свёрткой закрывается цепочка; SMP_FRED_NONE — не свёртка. */
static int fuse_red_of(uint8_t bc)
{
    switch (bc) {
        case SMP_BC_REDADD: return (int)SMP_FRED_ADD;
        case SMP_BC_REDMAX: return (int)SMP_FRED_MAX;
        default:            return (int)SMP_FRED_NONE;
    }
}

/* Собирает цепочку, размеченную SMP_IF_FUSE, и считает её за один проход.
 *
 * false означает «слить не вышло» — вызывающий тогда исполняет инструкцию
 * обычным путём, как будто флага не было. Поэтому все проверки идут ДО первого
 * обращения к памяти: отказ на полпути оставил бы половину работы сделанной.
 *
 * Флаг — утверждение компилятора, а не догадка рантайма. Но границы цепочки VM
 * всё равно перепроверяет: опкод обязан быть поэлементным, операнды —
 * тензорами. Не сошлось — считаем по-старому, а не молча иначе. */
static bool vm_fuse(SmpVM *vm, const SmpInstr **ipp, const SmpInstr *first)
{
    const SmpModule *mod = vm->mod;
    SmpReg          *R   = vm->regs;

    const SmpInstr *chain[SMP_FUSE_MAX];
    const SmpInstr *ip   = *ipp;
    const SmpInstr *cur  = first;
    const SmpInstr *tail = NULL;      /* свёртка, закрывающая цепочку */
    uint8_t         red  = (uint8_t)SMP_FRED_NONE;
    uint32_t        n    = 0;

    for (;;) {
        const int r = fuse_red_of(cur->op);
        if (r != (int)SMP_FRED_NONE) {
            /* Свёртка обязана быть последней: продолжать цепочку скаляром
             * нечем, и флаг на ней означал бы испорченный модуль. */
            if (cur->flags & SMP_IF_FUSE) return false;
            red  = (uint8_t)r;
            tail = cur;
            break;
        }
        if (n >= SMP_FUSE_MAX)             return false;
        if (fuse_op_of(cur->op) < 0)       return false;
        chain[n++] = cur;
        if (!(cur->flags & SMP_IF_FUSE))   break;
        if (ip >= mod->code + mod->n_code) return false;
        cur = ip++;
    }

    /* Без свёртки цепочка из одной стадии смысла не имеет: экономить нечего. */
    if (n < (tail ? 1u : 2u)) return false;

    /* Источник цепочки — вход первой стадии; приёмник — выход последней, а со
     * свёрткой его нет вовсе: результат скаляр. */
    if (!R[first->a].is_tensor) return false;
    if (!tail && !R[chain[n - 1u]->d].is_tensor) return false;
    for (uint32_t i = 0; i < n; i++) {
        const int f = fuse_op_of(chain[i]->op);
        if ((f == (int)SMP_FOP_ADD || f == (int)SMP_FOP_MUL) &&
            !R[chain[i]->b].is_tensor) return false;
    }

    SmpBuf      src, dst;
    SmpFuseStep steps[SMP_FUSE_MAX];

    /* Дальше отказы уже фатальные: диагностика выдана, и повторять работу
     * обычным путём не нужно — отсюда true, а не false.
     *
     * pc по ходу двигается на ту стадию, которую разбираем: слияние не должно
     * стоить точности диагностики. Иначе дамп регистров при падении на свёртке
     * показывал бы первую инструкцию цепочки, а пользователь читал бы про
     * @reduce.add рядом с опкодом relu. */
    if (!make_buf(vm, &src, &R[first->a].t)) return true;
    if (!tail && !make_dst(vm, &dst, &R[chain[n - 1u]->d].t)) return true;

    for (uint32_t i = 0; i < n; i++) {
        const SmpInstr *c = chain[i];
        const int       f = fuse_op_of(c->op);

        memset(&steps[i], 0, sizeof steps[i]);
        steps[i].op = (uint8_t)f;

        if (f == (int)SMP_FOP_SCALE) {
            vm->pc = (uint32_t)(c - mod->code);
            if (!stage_num(vm, c, &steps[i].k)) return true;
        } else if (f == (int)SMP_FOP_ADD || f == (int)SMP_FOP_MUL) {
            vm->pc = (uint32_t)(c - mod->code);
            if (!make_buf(vm, &steps[i].b, &R[c->b].t)) return true;
        }
    }

    apply_fp(vm, first->flags);

    if (tail) {
        vm->pc = (uint32_t)(tail - mod->code);
        const double v = smp_k_fuse_reduce(&src, steps, n, red);

        /* Проверка на вырождение та же, что у обычной свёртки: слияние меняет
         * число проходов по памяти, а не то, за что ругается ?strict. */
        if (!strict_check(vm, tail->flags, v,
                          red == (uint8_t)SMP_FRED_MAX ? "@reduce.max"
                                                       : "@reduce.add"))
            return true;

        /* Тип берётся у промежуточного буфера, а не у источника: ровно так же
         * поступает несливаемый путь, и дескриптор буфера в регистре есть —
         * его loadt эмиттер поднял выше цепочки. */
        R[tail->d].is_tensor = false;
        R[tail->d].s.f       = v;
        R[tail->d].dtype     = R[tail->a].is_tensor ? R[tail->a].t.dtype
                                                    : (uint8_t)SMP_DT_F64;
        n += 1u;   /* свёртка тоже израсходована */
    } else {
        smp_k_fuse(&dst, &src, steps, n);
    }

    /* Диспетчеризация была одна, а инструкций израсходовано n: счётчик должен
     * показывать исполненное, а не продиспетчеризованное. */
    vm->n_executed += n - 1u;
    *ipp = ip;
    return true;
}

/* ========================================================================== */
/*  Эпилог GEMM                                                               */
/* ========================================================================== */

/* Сколько байт адресует дескриптор от первого элемента до последнего
 * включительно. Считается по шагам, а не по nelem: у среза с шагом занятый
 * диапазон шире, чем его содержимое, и для проверки перекрытия важен именно
 * диапазон. */
static size_t buf_span(const SmpBuf *b)
{
    const SmpTensor *t   = b->t;
    size_t           far = 0;

    for (uint32_t i = 0; i < t->rank; i++)
        if (t->shape[i]) far += (size_t)(t->shape[i] - 1u) * (size_t)t->stride[i];
    return (size_t)smp_dtype_bytes((SmpDType)t->dtype, far + 1u);
}

static bool buf_overlap(const SmpBuf *x, const SmpBuf *y)
{
    const unsigned char *px = (const unsigned char *)x->p;
    const unsigned char *py = (const unsigned char *)y->p;
    return px < py + buf_span(y) && py < px + buf_span(x);
}

/* Собирает поэлементную цепочку за @mmul и считает её в выгрузке тайла.
 *
 * Возврат false означает «слить не вышло»: вызывающий тогда исполняет mmul
 * обычным путём, а стадии цепочки сольются между собой сами — флаги на них
 * никуда не делись. Поэтому отказ здесь ничего не портит и ничего не теряет,
 * кроме одного прохода по C.
 *
 * Свёртка цепочку не закрывает: @reduce после @mmul отдаёт скаляр, приёмника
 * у GEMM тогда нет вовсе, и это уже другое ядро. Такая цепочка сюда попадёт и
 * получит отказ — дальше её разберёт обычное слияние. */
static bool vm_fuse_gemm(SmpVM *vm, const SmpInstr **ipp, const SmpInstr *first)
{
    const SmpModule *mod = vm->mod;
    SmpReg          *R   = vm->regs;

    const SmpInstr *chain[SMP_FUSE_MAX];
    const SmpInstr *ip = *ipp;
    const SmpInstr *cur;
    uint32_t        n = 0;

    if (ip >= mod->code + mod->n_code) return false;
    cur = ip++;

    for (;;) {
        if (n >= SMP_FUSE_MAX)             return false;
        if (fuse_op_of(cur->op) < 0)       return false;
        chain[n++] = cur;
        if (!(cur->flags & SMP_IF_FUSE))   break;
        if (ip >= mod->code + mod->n_code) return false;
        cur = ip++;
    }

    /* Границы цепочки VM перепроверяет сама: флаг — утверждение компилятора о
     * мёртвом буфере, а не разрешение читать что попало. */
    if (!R[first->a].is_tensor || !R[first->b].is_tensor) return false;
    if (!R[chain[n - 1u]->d].is_tensor)                   return false;
    for (uint32_t i = 0; i < n; i++) {
        const int f = fuse_op_of(chain[i]->op);
        if ((f == (int)SMP_FOP_ADD || f == (int)SMP_FOP_MUL) &&
            !R[chain[i]->b].is_tensor) return false;
    }

    const SmpTensor *tc = &R[chain[n - 1u]->d].t;
    const SmpTensor *ta = &R[first->a].t;
    const SmpTensor *tb = &R[first->b].t;

    /* Приёмник цепочки становится приёмником самого GEMM, так что его форма
     * обязана быть формой произведения, а не просто совпадать поэлементно с
     * промежуточным буфером. */
    if (tc->rank != 2u || ta->rank != 2u || tb->rank != 2u) return false;
    if (tc->shape[0] != ta->shape[0] || tc->shape[1] != tb->shape[1]) return false;

    SmpBuf      dst, ba, bb;
    SmpFuseStep steps[SMP_FUSE_MAX];

    /* Дальше отказы фатальные: диагностика выдана, повторять работу обычным
     * путём не нужно — отсюда true, а не false. */
    if (!make_dst(vm, &dst, tc) ||
        !make_buf(vm, &ba,  ta) ||
        !make_buf(vm, &bb,  tb)) return true;

    for (uint32_t i = 0; i < n; i++) {
        const SmpInstr *c = chain[i];
        const int       f = fuse_op_of(c->op);

        memset(&steps[i], 0, sizeof steps[i]);
        steps[i].op = (uint8_t)f;

        if (f == (int)SMP_FOP_SCALE) {
            vm->pc = (uint32_t)(c - mod->code);
            if (!stage_num(vm, c, &steps[i].k)) return true;
        } else if (f == (int)SMP_FOP_ADD || f == (int)SMP_FOP_MUL) {
            vm->pc = (uint32_t)(c - mod->code);
            if (!make_buf(vm, &steps[i].b, &R[c->b].t)) return true;
        }
    }

    /* Неслитый GEMM писал в свой временный буфер, а цепочка читала его отдельно
     * от A, B и операндов @add. Слитый пишет в приёмник ПОСРЕДИ счёта, и если
     * тот перекрывается с чьим-то входом — читаться будет уже испорченное.
     * Компилятор про это знать не обязан: срез с динамическим индексом сходится
     * с чужой памятью только в рантайме. */
    if (buf_overlap(&dst, &ba) || buf_overlap(&dst, &bb)) return false;
    for (uint32_t i = 0; i < n; i++)
        if ((steps[i].op == SMP_FOP_ADD || steps[i].op == SMP_FOP_MUL) &&
            buf_overlap(&dst, &steps[i].b)) return false;

    apply_fp(vm, first->flags);
    smp_k_gemm_ep(&dst, &ba, &bb, &vm->scratch, steps, n);

    /* Диспетчеризация была одна, а инструкций израсходовано n сверх самого
     * mmul: счётчик показывает исполненное, а не продиспетчеризованное. */
    vm->n_executed += n;
    *ipp = ip;
    return true;
}

/* ========================================================================== */
/*  Главный цикл                                                              */
/* ========================================================================== */

SmpStatus smp_vm_run(SmpVM *vm)
{
    /* MXCSR — регистр потока, а не процесса. Снимок берём здесь: инстанс мог
     * быть поднят на одном потоке, а исполняться на другом. */
    vm->entry_mxcsr = smp_fpu_get_mxcsr();
    vm->cur_fp_flags = 0;

    /* Адреса видов живут один прогон: между прогонами хозяин мог заменить
     * или удалить объект. Хранилищу без адресов — своя копия, заводится
     * здесь, до первой инструкции, и остаётся до smp_vm_release. */
    vm->n_views = 0;
    vm->prof_started = false;
    if (vm->mod->view_bytes && !vm->view_mem && !(vm->store && vm->store->map)) {
        vm->view_mem = (uint8_t *)smp_plat_pages((size_t)vm->mod->view_bytes);
        if (!vm->view_mem) return SMP_ERR_OOM;
    }

    const SmpModule *mod  = vm->mod;
    const SmpInstr  *code = mod->code;
    SmpReg          *R    = vm->regs;

#if SMP_HAS_COMPUTED_GOTO
    /* Таблица строится из того же X-макроса, что и enum: разъехаться они не
     * могут даже при добавлении опкода посередине. */
#   define OPLABEL(id, mn, fmt, desc) &&L_##id,
    static const void *const dispatch[] = { SMP_BC_OPS(OPLABEL) };
#   undef OPLABEL

    /* Соответствие таблицы и реестра проверяется на компиляции: если метка
     * для какого-то опкода не написана, делитель обнулится и сборка встанет.
     * Через enum, а не typedef, — чтобы не плодить неиспользуемых имён. */
    enum { dispatch_covers_all_opcodes =
               1 / (int)(SMP_ARRLEN(dispatch) == SMP_BC__COUNT) };

#   define VM_NEXT()                                                          \
        do {                                                                  \
            if (SMP_UNLIKELY(vm->trapped)) return SMP_ERR_INTERNAL;           \
            vm->n_executed++;                                                 \
            if (SMP_UNLIKELY(vm->prof != NULL)) vm_prof(vm);                  \
            vm->pc = (uint32_t)(ip - code);                                   \
            in = ip++;                                                        \
            goto *dispatch[in->op];                                           \
        } while (0)
#   define VM_CASE(id) L_##id:
#else
#   define VM_NEXT()                                                          \
        do {                                                                  \
            if (SMP_UNLIKELY(vm->trapped)) return SMP_ERR_INTERNAL;           \
            vm->n_executed++;                                                 \
            if (SMP_UNLIKELY(vm->prof != NULL)) vm_prof(vm);                  \
            vm->pc = (uint32_t)(ip - code);                                   \
            in = ip++;                                                        \
            goto dispatch_switch;                                             \
        } while (0)
#   define VM_CASE(id) case SMP_BC_##id:
#endif

    const SmpInstr *ip = code;
    const SmpInstr *in = NULL;

    SmpBuf bd, ba, bb;
    SmpTensor ta, tb;                   /* виды растянутых операндов */

#if SMP_HAS_COMPUTED_GOTO
    VM_NEXT();
#else
    VM_NEXT();
dispatch_switch:
    switch (in->op) {
#endif

    VM_CASE(HALT)
        if (vm->prof) vm_prof(vm);
        return SMP_OK;

    VM_CASE(NOP)
        VM_NEXT();

    VM_CASE(LOADT)
        R[in->d].is_tensor = true;
        R[in->d].t         = mod->tens[in->k];
        R[in->d].dtype     = mod->tens[in->k].dtype;
        VM_NEXT();

    VM_CASE(LOADK)
        /* Скалярный регистр всегда держит double в .f, а dtype хранит
         * ЛОГИЧЕСКИЙ тип значения. Единая договорённость: иначе каждый
         * потребитель гадал бы, как читать восемь байт. */
        R[in->d].is_tensor = false;
        R[in->d].s.f       = smp_const_as_double(mod->consts[in->k], in->aux);
        R[in->d].dtype     = in->aux ? in->aux : (uint8_t)SMP_DT_F64;
        /* Число f32 обязано быть числом f32: 0.1 в пуле — double. */
        if (in->aux == SMP_DT_F32) R[in->d].s.f = (double)(float)R[in->d].s.f;
        VM_NEXT();

    VM_CASE(MOVE)
        R[in->d] = R[in->a];
        VM_NEXT();

    /* То же, что storet, но приёмник — вид в регистре: срез, смещение
     * которого стало известно только на исполнении (sliced). */
    VM_CASE(STORER)
    VM_CASE(STORET) {
        const SmpTensor dt = in->op == SMP_BC_STORER ? R[in->d].t : mod->tens[in->k];
        if (!R[in->a].is_tensor) {
            /* Скаляр в тензор: заполняем целиком. */
            if (!make_dst(vm, &bd, &dt)) return SMP_ERR_INTERNAL;
            smp_k_fill(&bd, R[in->a].s.f);
            VM_NEXT();
        }
        /* Тензор в самого себя — так кончается *&W -> @load => *&W. Копировать
         * нечего, а у вида и некуда. */
        if (memcmp(&dt, &R[in->a].t, sizeof dt) == 0) VM_NEXT();
        if (!make_dst(vm, &bd, &dt) || !make_buf(vm, &ba, &R[in->a].t))
            return SMP_ERR_INTERNAL;
        smp_k_copy(&bd, &ba);
        VM_NEXT();
    }

    VM_CASE(SLICED) {
        /* Динамический срез: к смещению добавляется значение регистра,
         * умноженное на шаг в байтах. */
        R[in->d] = R[in->a];
        const int64_t  n    = R[in->b].is_tensor ? 0 : (int64_t)R[in->b].s.f;
        const uint64_t step = mod->consts[in->k].u;

        /* Здесь только арифметика не должна переполниться; попадание в
         * арену проверит tensor_base там, где срез читают. */
        if (n < 0 || (step && (uint64_t)n > (UINT64_MAX / 2 - R[in->a].t.off) / step)) {
            vm_fatal(vm, SMP_E0410,
                     vfmt(vm, "Динамический индекс %lld даёт смещение вне тензора.",
                          (long long)n), NULL);
            return SMP_ERR_INTERNAL;
        }
        R[in->d].t.off = R[in->a].t.off + (uint64_t)n * step;

        /* Срез мог сбить выравнивание, которое компилятор проверить не мог:
         * индекс стал известен только сейчас. Требовать его можно лишь там,
         * где ширину попросили явно (#simd), — без этого строка f32:3 по
         * индексу-регистру падала на любой машине с AVX2. У q8_0 строка —
         * целые блоки по 34 байта, выровненной она не бывает, и ядра читают
         * её без требований к выравниванию. */
        const uint32_t bits = smp_vec_bits(in->flags & SMP_IF_VEC_MASK);
        if ((in->flags & SMP_IF_SIMD) && bits >= 256 &&
            !smp_dtype_is_block((SmpDType)R[in->d].t.dtype) &&
            (R[in->d].t.off % (bits / 8u)) != 0) {
            vm_fatal(vm, SMP_E0402,
                     vfmt(vm, "Динамический индекс %lld дал смещение %llu байт, "
                              "а %s требует кратности %u.",
                          (long long)n, (unsigned long long)R[in->d].t.off,
                          smp_vec_name(bits), bits / 8u),
                     "Выровняй индекс или сними #simd с этой инструкции.");
            return SMP_ERR_INTERNAL;
        }
        VM_NEXT();
    }

    VM_CASE(ALLOC)
        if (!R[in->d].is_tensor) VM_NEXT();
        if (!make_dst(vm, &bd, &R[in->d].t)) return SMP_ERR_INTERNAL;
        smp_k_zero(&bd);
        VM_NEXT();

    VM_CASE(FILL) {
        double v;
        apply_fp(vm, in->flags);
        if (!stage_num(vm, in, &v) || !make_dst(vm, &bd, &R[in->d].t))
            return SMP_ERR_INTERNAL;
        smp_k_fill(&bd, v);
        VM_NEXT();
    }

    VM_CASE(FILLI)
        if (!make_dst(vm, &bd, &R[in->d].t)) return SMP_ERR_INTERNAL;
        smp_k_fill(&bd, (double)vm->instance);
        VM_NEXT();

    VM_CASE(LOAD)
        if (!R[in->d].is_tensor) {
            vm_fatal(vm, SMP_E0606, "@load ждёт тензор, а в регистре скаляр.", NULL);
            return SMP_ERR_INTERNAL;
        }
        /* Вид у хранилища с адресами: не копия, а адрес объекта. */
        if ((R[in->d].t.flags & SMP_TF_EXTERN) && vm->store && vm->store->map) {
            if (!store_map(vm, &R[in->d].t)) return SMP_ERR_INTERNAL;
            VM_NEXT();
        }
        /* Иначе — байты в арену либо в свою копию видов: единственное
         * место, где вид пишется. */
        if (!make_buf(vm, &bd, &R[in->d].t)) return SMP_ERR_INTERNAL;
        if (!(vm->store ? store_load(vm, &R[in->d].t, &bd)
                        : file_load(vm, &R[in->d].t, &bd))) return SMP_ERR_INTERNAL;
        VM_NEXT();

    VM_CASE(STORE) {
        if (!R[in->a].is_tensor) {
            vm_fatal(vm, SMP_E0606, "@store ждёт тензор, а в регистре скаляр.", NULL);
            return SMP_ERR_INTERNAL;
        }
        if (!make_buf(vm, &ba, &R[in->a].t)) return SMP_ERR_INTERNAL;
        uint64_t put = 0;
        if (!(vm->store ? store_store(vm, &R[in->a].t, &ba, &put)
                        : file_store(vm, &R[in->a].t, &ba, &put))) return SMP_ERR_INTERNAL;
        /* Объект под этим именем хранилище только что заменило: прежний
         * адрес вида больше не его. */
        view_remap(vm, R[in->a].t.name_id);

        /* Отдаём число записанных байт — ровно так же, как @emit отдаёт число
         * выведенных элементов. Отдельной операции без результата в языке нет. */
        /* Скалярный регистр держит double в .f, а dtype — логический тип
         * значения. Договорённость единая на весь рантайм; записать сюда
         * .u значило бы отдать дампу и @emit битовый мусор. */
        R[in->d].is_tensor = false;
        R[in->d].s.f       = (double)put;
        R[in->d].dtype     = (uint8_t)SMP_DT_U64;
        VM_NEXT();
    }

    VM_CASE(MMUL)
        /* Эпилог: @mmul -> @relu -> @add считается в выгрузке тайла, пока тот
         * лежит в L1. Не сошлось — обычный путь, а цепочка сольётся сама. */
        if (SMP_UNLIKELY(in->flags & SMP_IF_FUSE) && in->aux == 0 &&
            vm_fuse_gemm(vm, &ip, in))
            VM_NEXT();
        apply_fp(vm, in->flags);
        if (!make_dst(vm, &bd, &R[in->d].t) ||
            !make_buf(vm, &ba, &R[in->a].t) ||
            !make_buf(vm, &bb, &R[in->b].t)) return SMP_ERR_INTERNAL;
        if (in->aux == 1) mmul_live(vm, in, &bd, &ba, &bb);
        else              smp_k_gemm(&bd, &ba, &bb, &vm->scratch);
        VM_NEXT();

    VM_CASE(SUB)
        if (in->aux == SMP_AUX_SCALAR) {
            if (!scalar_op(vm, in)) return SMP_ERR_INTERNAL;
            VM_NEXT();
        }
        apply_fp(vm, in->flags);
        if (!make_dst(vm, &bd, &R[in->d].t) ||
            !make_bcast(vm, &ba, &R[in->a].t, &R[in->d].t, &ta) ||
            !make_bcast(vm, &bb, &R[in->b].t, &R[in->d].t, &tb)) return SMP_ERR_INTERNAL;
        smp_k_sub(&bd, &ba, &bb);
        VM_NEXT();

    VM_CASE(SILU)
        apply_fp(vm, in->flags);
        if (!make_dst(vm, &bd, &R[in->d].t) ||
            !make_buf(vm, &ba, &R[in->a].t)) return SMP_ERR_INTERNAL;
        smp_k_silu(&bd, &ba);
        VM_NEXT();

    VM_CASE(RMSN)
        apply_fp(vm, in->flags);
        if (!make_dst(vm, &bd, &R[in->d].t) ||
            !make_buf(vm, &ba, &R[in->a].t)) return SMP_ERR_INTERNAL;
        smp_k_rmsnorm(&bd, &ba, smp_const_as_double(mod->consts[in->k], in->aux));
        VM_NEXT();

    VM_CASE(SOFTMX) {
        /* Длина из регистра: сколько первых элементов строки живые. Всё,
         * что меньше нуля или не число, — ноль; больше строки — вся строка. */
        uint64_t len = UINT64_MAX;
        if (in->aux) {
            const double n = R[in->b].is_tensor ? 0.0 : R[in->b].s.f;
            len = n >= 1.8e19 ? UINT64_MAX : n >= 0.0 ? (uint64_t)n : 0u;
        }
        apply_fp(vm, in->flags);
        if (!make_dst(vm, &bd, &R[in->d].t) ||
            !make_buf(vm, &ba, &R[in->a].t)) return SMP_ERR_INTERNAL;
        smp_k_softmax(&bd, &ba, len);
        VM_NEXT();
    }

    VM_CASE(ROPE) {
        const double pos = R[in->b].is_tensor ? 0.0 : R[in->b].s.f;
        if (!(pos >= 0.0 && pos <= 16777216.0)) {
            vm_fatal(vm, SMP_E0410,
                     vfmt(vm, "Позиция @rope %g вне 0..2^24: угол считается в f32, "
                              "и дальше позиции уже неразличимы.", pos), NULL);
            return SMP_ERR_INTERNAL;
        }
        apply_fp(vm, in->flags);
        if (!make_dst(vm, &bd, &R[in->d].t) ||
            !make_buf(vm, &ba, &R[in->a].t)) return SMP_ERR_INTERNAL;
        smp_k_rope(&bd, &ba, pos, smp_const_as_double(mod->consts[in->k], in->aux));
        VM_NEXT();
    }

    VM_CASE(ARGMAX) {
        uint64_t idx = 0;
        if (R[in->a].is_tensor) {
            if (!make_buf(vm, &ba, &R[in->a].t)) return SMP_ERR_INTERNAL;
            idx = smp_k_argmax(&ba);
        }
        R[in->d].is_tensor = false;
        R[in->d].s.f       = (double)idx;
        R[in->d].dtype     = (uint8_t)SMP_DT_U64;
        VM_NEXT();
    }

    VM_CASE(RESHP) {
        /* Та же память, другая форма: смещение, арена и имя — от rA. */
        const SmpTensor *nt = &mod->tens[in->k];
        R[in->d] = R[in->a];
        SmpTensor *t = &R[in->d].t;
        t->rank  = nt->rank;
        t->nelem = nt->nelem;
        memcpy(t->shape, nt->shape, sizeof t->shape);
        memcpy(t->stride, nt->stride, sizeof t->stride);
        t->flags = (uint16_t)((t->flags & (uint16_t)~SMP_TF_TRANSPOSED) | SMP_TF_CONTIG);
        VM_NEXT();
    }

    VM_CASE(MMULT)
        apply_fp(vm, in->flags);
        if (!make_dst(vm, &bd, &R[in->d].t) ||
            !make_buf(vm, &ba, &R[in->a].t) ||
            !make_buf(vm, &bb, &R[in->b].t)) return SMP_ERR_INTERNAL;
        if (in->aux == 1) mmul_t_live(vm, in, &bd, &ba, &bb);
        else              smp_k_mmul_t(&bd, &ba, &bb, &vm->scratch);
        VM_NEXT();

    VM_CASE(TRANS) {
        R[in->d] = R[in->a];
        SmpTensor *t = &R[in->d].t;
        const uint32_t s0 = t->shape[0], s1 = t->shape[1];
        const uint32_t d0 = t->stride[0], d1 = t->stride[1];
        t->shape[0] = s1; t->shape[1] = s0;
        t->stride[0] = d1; t->stride[1] = d0;
        t->flags = (uint16_t)((t->flags & (uint16_t)~SMP_TF_CONTIG) | SMP_TF_TRANSPOSED);
        VM_NEXT();
    }

    VM_CASE(PACK)
        if (!make_dst(vm, &bd, &R[in->d].t) ||
            !make_buf(vm, &ba, &R[in->a].t)) return SMP_ERR_INTERNAL;
        smp_k_copy(&bd, &ba);
        VM_NEXT();

    VM_CASE(RELU)
        if (SMP_UNLIKELY(in->flags & SMP_IF_FUSE) && vm_fuse(vm, &ip, in))
            VM_NEXT();
        apply_fp(vm, in->flags);
        if (!make_dst(vm, &bd, &R[in->d].t) ||
            !make_buf(vm, &ba, &R[in->a].t)) return SMP_ERR_INTERNAL;
        smp_k_relu(&bd, &ba);
        VM_NEXT();

    VM_CASE(ABS)
        if (in->aux == SMP_AUX_SCALAR) {
            if (!scalar_op(vm, in)) return SMP_ERR_INTERNAL;
            VM_NEXT();
        }
        if (SMP_UNLIKELY(in->flags & SMP_IF_FUSE) && vm_fuse(vm, &ip, in))
            VM_NEXT();
        apply_fp(vm, in->flags);
        if (!make_dst(vm, &bd, &R[in->d].t) ||
            !make_buf(vm, &ba, &R[in->a].t)) return SMP_ERR_INTERNAL;
        smp_k_abs(&bd, &ba);
        VM_NEXT();

    VM_CASE(SCALE) {
        if (SMP_UNLIKELY(in->flags & SMP_IF_FUSE) && vm_fuse(vm, &ip, in))
            VM_NEXT();
        double k;
        apply_fp(vm, in->flags);
        if (!stage_num(vm, in, &k) ||
            !make_dst(vm, &bd, &R[in->d].t) ||
            !make_buf(vm, &ba, &R[in->a].t)) return SMP_ERR_INTERNAL;
        smp_k_scale(&bd, &ba, k);
        VM_NEXT();
    }

    VM_CASE(ADD)
        if (in->aux == SMP_AUX_SCALAR) {
            if (!scalar_op(vm, in)) return SMP_ERR_INTERNAL;
            VM_NEXT();
        }
        if (SMP_UNLIKELY(in->flags & SMP_IF_FUSE) && vm_fuse(vm, &ip, in))
            VM_NEXT();
        apply_fp(vm, in->flags);
        if (!make_dst(vm, &bd, &R[in->d].t) ||
            !make_bcast(vm, &ba, &R[in->a].t, &R[in->d].t, &ta) ||
            !make_bcast(vm, &bb, &R[in->b].t, &R[in->d].t, &tb)) return SMP_ERR_INTERNAL;
        smp_k_add(&bd, &ba, &bb);
        VM_NEXT();

    VM_CASE(MUL)
        if (in->aux == SMP_AUX_SCALAR) {
            if (!scalar_op(vm, in)) return SMP_ERR_INTERNAL;
            VM_NEXT();
        }
        if (SMP_UNLIKELY(in->flags & SMP_IF_FUSE) && vm_fuse(vm, &ip, in))
            VM_NEXT();
        apply_fp(vm, in->flags);
        if (!make_dst(vm, &bd, &R[in->d].t) ||
            !make_bcast(vm, &ba, &R[in->a].t, &R[in->d].t, &ta) ||
            !make_bcast(vm, &bb, &R[in->b].t, &R[in->d].t, &tb)) return SMP_ERR_INTERNAL;
        smp_k_mul(&bd, &ba, &bb);
        VM_NEXT();

    VM_CASE(DIV)
        if (in->aux == SMP_AUX_SCALAR) {
            if (!scalar_op(vm, in)) return SMP_ERR_INTERNAL;
            VM_NEXT();
        }
        apply_fp(vm, in->flags);
        if (!make_dst(vm, &bd, &R[in->d].t) ||
            !make_bcast(vm, &ba, &R[in->a].t, &R[in->d].t, &ta) ||
            !make_bcast(vm, &bb, &R[in->b].t, &R[in->d].t, &tb)) return SMP_ERR_INTERNAL;
        smp_k_div(&bd, &ba, &bb);
        VM_NEXT();

    VM_CASE(SQRT)
        if (in->aux == SMP_AUX_SCALAR) {
            if (!scalar_op(vm, in)) return SMP_ERR_INTERNAL;
            VM_NEXT();
        }
        apply_fp(vm, in->flags);
        if (!make_dst(vm, &bd, &R[in->d].t) ||
            !make_buf(vm, &ba, &R[in->a].t)) return SMP_ERR_INTERNAL;
        smp_k_sqrt(&bd, &ba);
        VM_NEXT();

    VM_CASE(LIT) {
        /* Числа литерала лежат в пуле подряд, с k; сколько их — знает
         * дескриптор в rD. Кладутся по его шагам, так что годится и столбец. */
        const SmpTensor *t = &R[in->d].t;
        if (!R[in->d].is_tensor || (uint64_t)in->k + t->nelem > mod->n_consts) {
            vm_fatal(vm, SMP_E0604, "Литерал выходит за пул констант.", NULL);
            return SMP_ERR_INTERNAL;
        }
        if (!make_dst(vm, &bd, t)) return SMP_ERR_INTERNAL;
        for (uint32_t i = 0; i < t->nelem; i++)
            smp_k_set(&bd, i, smp_const_as_double(mod->consts[in->k + i], in->aux));
        VM_NEXT();
    }

    VM_CASE(REDADD) {
        apply_fp(vm, in->flags);
        /* Регистр уже держит скаляр — сворачивать нечего. */
        if (!R[in->a].is_tensor) { R[in->d] = R[in->a]; VM_NEXT(); }
        if (!make_buf(vm, &ba, &R[in->a].t)) return SMP_ERR_INTERNAL;
        const double v = smp_k_reduce_add(&ba);
        if (!strict_check(vm, in->flags, v, "@reduce.add")) return SMP_ERR_INTERNAL;
        R[in->d].is_tensor = false;
        R[in->d].s.f       = v;
        R[in->d].dtype     = R[in->a].t.dtype;
        VM_NEXT();
    }

    VM_CASE(REDMAX) {
        apply_fp(vm, in->flags);
        if (!R[in->a].is_tensor) { R[in->d] = R[in->a]; VM_NEXT(); }
        if (!make_buf(vm, &ba, &R[in->a].t)) return SMP_ERR_INTERNAL;
        const double v = smp_k_reduce_max(&ba);
        if (!strict_check(vm, in->flags, v, "@reduce.max")) return SMP_ERR_INTERNAL;
        R[in->d].is_tensor = false;
        R[in->d].s.f       = v;
        R[in->d].dtype     = R[in->a].t.dtype;
        VM_NEXT();
    }

    VM_CASE(EMIT) {
        if (!R[in->a].is_tensor) {
            /* Скаляр печатается как тензор из одного элемента: заводить ради
             * этого второй путь незачем. */
            SmpTensor one;
            memset(&one, 0, sizeof one);
            one.dtype = R[in->a].dtype ? R[in->a].dtype : (uint8_t)SMP_DT_F64;
            one.rank  = 0;
            one.nelem = 1;
            double v  = R[in->a].s.f;
            SmpBuf b  = { &v, &one };
            /* Значение лежит как double, а дескриптор объявляет свой тип —
             * согласуем: печатаем через f64. */
            one.dtype = (uint8_t)SMP_DT_F64;
            uint64_t n = 0;
            smp_vm_emit(vm->out, vm->out_log, &b, in->aux, &n, NULL);
            R[in->d].is_tensor = false;
            R[in->d].s.f       = (double)n;
            R[in->d].dtype     = (uint8_t)SMP_DT_U64;
            VM_NEXT();
        }

        if (!make_buf(vm, &ba, &R[in->a].t)) return SMP_ERR_INTERNAL;

        uint64_t n   = 0;
        uint32_t bad = 0;
        if (smp_vm_emit(vm->out, vm->out_log, &ba, in->aux, &n, &bad) != SMP_OK) {
            vm_fatal(vm, SMP_E0605,
                     vfmt(vm, "Элемент #%llu равен 0x%X. Это %s.",
                          (unsigned long long)n, bad,
                          (bad >= 0xD800u && bad <= 0xDFFFu)
                              ? "суррогат, в UTF-8 он не кодируется"
                              : "за пределами Unicode"), NULL);
            return SMP_ERR_INTERNAL;
        }
        R[in->d].is_tensor = false;
        R[in->d].s.f       = (double)n;
        R[in->d].dtype     = (uint8_t)SMP_DT_U64;
        VM_NEXT();
    }

    VM_CASE(CVTF32)
    VM_CASE(CVTF64)
    VM_CASE(CVTI32)
    VM_CASE(CVTU64)
    VM_CASE(CVTQ80) {
        const SmpDType to = in->op == SMP_BC_CVTF32 ? SMP_DT_F32
                          : in->op == SMP_BC_CVTF64 ? SMP_DT_F64
                          : in->op == SMP_BC_CVTI32 ? SMP_DT_I32
                          : in->op == SMP_BC_CVTU64 ? SMP_DT_U64 : SMP_DT_Q8_0;
        if (!R[in->a].is_tensor) {
            /* Ярлыка мало: приведение к целому обязано отбросить дробную
             * часть, иначе @cast.i32 над 2.7 давал бы 2.7 с надписью i32. */
            double v = R[in->a].s.f;
            if (v != v) v = 0.0;                      /* NaN в целое — ноль */
            if (to == SMP_DT_I32) {
                v = v < -2147483648.0 ? -2147483648.0
                  : (v > 2147483647.0 ? 2147483647.0 : v);
                v = (double)(int32_t)v;
            } else if (to == SMP_DT_U64) {
                v = v < 0.0 ? 0.0
                  : (v > 18446744073709549568.0 ? 18446744073709549568.0 : v);
                v = (double)(uint64_t)v;
            } else if (to == SMP_DT_F32) {
                v = (double)(float)v;
            }
            R[in->d].is_tensor = false;
            R[in->d].s.f       = v;
            R[in->d].dtype     = (uint8_t)to;
            VM_NEXT();
        }
        if (!make_dst(vm, &bd, &R[in->d].t) ||
            !make_buf(vm, &ba, &R[in->a].t)) return SMP_ERR_INTERNAL;
        smp_k_cast(&bd, &ba);
        VM_NEXT();
    }

#if !SMP_HAS_COMPUTED_GOTO
    default:
        vm_fatal(vm, SMP_E0604,
                 vfmt(vm, "Опкод 0x%02X не имеет обработчика.", in->op), NULL);
        return SMP_ERR_INTERNAL;
    }
#endif

#undef VM_NEXT
#undef VM_CASE
}

/* ========================================================================== */
/*  Печать результатов                                                        */
/* ========================================================================== */

const void *smp_vm_tensor_data(const SmpVM *vm, const SmpTensor *t)
{
    if (t->flags & SMP_TF_EXTERN) {
        for (uint32_t i = 0; i < vm->n_views; i++) {
            const SmpView *v = &vm->views[i];
            if (v->name_id == t->name_id && t->off >= v->off)
                return v->p + (t->off - v->off);
        }
        return vm->view_mem ? vm->view_mem + t->off : NULL;
    }
    const uint32_t a = smp_tf_arena(t->flags);
    return a < vm->n_arenas ? vm->arenas[a].base + t->off : NULL;
}

void smp_vm_dump_tensors(FILE *out, const SmpVM *vm, uint32_t max_elems)
{
    const SmpModule *mod = vm->mod;

    /* Скалярные регистры: их значения — тоже результат программы, а из
     * тензорной таблицы их не видно. */
    for (uint32_t i = 0; i < vm->n_regs && i < mod->n_reg_names; i++) {
        const SmpReg *r = &vm->regs[i];
        if (r->is_tensor || r->dtype == 0 || !mod->reg_names[i]) continue;
        char num[40];
        fprintf(out, "$%-12s <%s>  %s\n",
                smp_module_str(mod, mod->reg_names[i]),
                smp_dtype_name((SmpDType)r->dtype),
                smp_num_fmt(r->s.f, (SmpDType)r->dtype, num, sizeof num));
    }

    for (uint32_t i = 0; i < mod->n_tens; i++) {
        const SmpTensor *t    = &mod->tens[i];
        const char      *name = smp_module_str(mod, t->name_id);
        if (name[0] == '\0' || name[0] == '%') continue;   /* временные пропускаем */

        /* Один и тот же тензор попадает в таблицу несколько раз (срезы);
         * печатаем только полный вид. */
        bool is_view = false;
        for (uint32_t j = 0; j < i; j++)
            if (strcmp(smp_module_str(mod, mod->tens[j].name_id), name) == 0)
                is_view = true;
        if (is_view) continue;

        const uint8_t *base = (const uint8_t *)smp_vm_tensor_data(vm, t);
        if (!base) continue;

        char sig[80];
        fprintf(out, "*&%-11s <%s>  ", name, smp_tensor_sig(t, sig, sizeof sig));

        const uint32_t n    = t->nelem < max_elems ? t->nelem : max_elems;
        fputc('[', out);
        for (uint32_t e = 0; e < n; e++) {
            double v = 0.0;
            switch ((SmpDType)t->dtype) {
                case SMP_DT_F32: v = ((const float    *)base)[e]; break;
                case SMP_DT_F64: v = ((const double   *)base)[e]; break;
                case SMP_DT_I32: v = ((const int32_t  *)base)[e]; break;
                case SMP_DT_U64: v = (double)((const uint64_t *)base)[e]; break;
                case SMP_DT_Q8_0: v = smp_q8_0_get(base, e); break;
                default: break;
            }
            char num[40];
            fprintf(out, "%s%s", e ? ", " : "",
                    smp_num_fmt(v, (SmpDType)t->dtype, num, sizeof num));
        }
        if (t->nelem > n) fprintf(out, ", ... (+%u)", t->nelem - n);
        fprintf(out, "]\n");
    }
}
