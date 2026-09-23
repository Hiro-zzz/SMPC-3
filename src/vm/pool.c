/* SMPC3 :: pool.c */
#include "smpc3/pool.h"
#include "smpc3/thread.h"
#include "smpc3/cpu.h"
#include "smpc3/kernels.h"
#include "smpc3/plat.h"

#include <string.h>
#include <stdlib.h>

/* ========================================================================== */
/*  Подъём                                                                    */
/* ========================================================================== */

SmpStatus smp_vm_pool_init(SmpVMPool *p, const SmpModule *mod,
                           uint32_t n_inst, uint32_t n_threads)
{
    memset(p, 0, sizeof *p);
    if (!mod || n_inst == 0) return SMP_ERR_INTERNAL;
    if (n_inst > SMP_POOL_MAX_INSTANCES) n_inst = SMP_POOL_MAX_INSTANCES;

    /* Ветку ядер и профиль процессора разрешаем ЗДЕСЬ, до появления потоков.
     * Оба кэша ленивые и записываются при первом обращении; сделай это первым
     * обращением из рабочего потока — и получишь гонку на ровном месте. */
    (void)smp_cpu();
    (void)smp_kernels_name();

    p->mod       = mod;
    p->n_inst    = n_inst;
    p->n_threads = n_threads ? n_threads : smp_threads_default();
    if (p->n_threads > n_inst) p->n_threads = n_inst;

    p->inst   = (SmpVM      *)smp_plat_pages(n_inst * sizeof(SmpVM));
    p->diag   = (SmpDiagCtx *)smp_plat_pages(n_inst * sizeof(SmpDiagCtx));
    p->log    = (SmpLog     *)smp_plat_pages(n_inst * sizeof(SmpLog));
    p->status = (SmpStatus  *)smp_plat_pages(n_inst * sizeof(SmpStatus));

    /* Журналы одним блоком: тысяча отдельных выделений здесь не нужна, а
     * освобождать проще один указатель. */
    p->logmem = (char *)smp_plat_pages((size_t)n_inst * SMP_POOL_LOG_BYTES);

    if (!p->inst || !p->diag || !p->log || !p->status || !p->logmem) {
        smp_vm_pool_release(p);
        return SMP_ERR_OOM;
    }

    for (uint32_t i = 0; i < n_inst; i++) {
        /* Свой журнал: диагностика шестнадцати потоков в один поток вывода
         * склеилась бы в нечитаемую кашу. Печатаем позже и по очереди. */
        smp_log_bind(&p->log[i], p->logmem + (size_t)i * SMP_POOL_LOG_BYTES,
                     SMP_POOL_LOG_BYTES);

        smp_diag_init(&p->diag[i], NULL, NULL);
        smp_diag_set_log(&p->diag[i], &p->log[i]);

        if (smp_vm_init(&p->inst[i], mod, &p->diag[i]) != SMP_OK) {
            smp_vm_pool_release(p);
            return SMP_ERR_OOM;
        }
        p->inst[i].instance = i;
        p->inst[i].out_log  = &p->log[i];   /* @emit тоже в свой журнал */

        p->diag[i].regdump      = smp_vm_regdump;
        p->diag[i].regdump_user = &p->inst[i];
    }
    return SMP_OK;
}

void smp_vm_pool_release(SmpVMPool *p)
{
    if (p->inst) {
        for (uint32_t i = 0; i < p->n_inst; i++) smp_vm_release(&p->inst[i]);
        smp_plat_pages_free(p->inst, p->n_inst * sizeof(SmpVM));
    }
    smp_plat_pages_free(p->log,    p->n_inst * sizeof(SmpLog));
    smp_plat_pages_free(p->logmem, (size_t)p->n_inst * SMP_POOL_LOG_BYTES);
    smp_plat_pages_free(p->diag,   p->n_inst * sizeof(SmpDiagCtx));
    smp_plat_pages_free(p->status, p->n_inst * sizeof(SmpStatus));
    memset(p, 0, sizeof *p);
}

/* ========================================================================== */
/*  Исполнение                                                                */
/* ========================================================================== */

/* Задания разбираются динамически, а не нарезаются заранее: инстансы могут
 * стоить по-разному, и статическая нарезка оставила бы часть ядер простаивать
 * в ожидании самого медленного. */
static void worker(void *ctx, uint32_t idx)
{
    SMP_UNUSED(idx);
    SmpVMPool *p = (SmpVMPool *)ctx;

    for (;;) {
        const int32_t i = smp_atomic_fetch_add(&p->cursor, 1);
        if (i < 0 || (uint32_t)i >= p->n_inst) return;
        p->status[i] = smp_vm_run(&p->inst[i]);
    }
}

SmpStatus smp_vm_pool_run(SmpVMPool *p)
{
    if (!p->inst) return SMP_ERR_INTERNAL;

    p->cursor     = 0;
    p->n_executed = 0;
    p->n_failed   = 0;

    const double t0 = smp_now_sec();
    smp_threads_run(worker, p, p->n_threads);
    p->seconds = smp_now_sec() - t0;

    for (uint32_t i = 0; i < p->n_inst; i++) {
        p->n_executed += p->inst[i].n_executed;
        if (p->status[i] != SMP_OK) p->n_failed++;
    }
    return p->n_failed ? SMP_ERR_INTERNAL : SMP_OK;
}

/* ========================================================================== */
/*  Отчёт                                                                     */
/* ========================================================================== */

void smp_vm_pool_report(FILE *out, SmpVMPool *p)
{
    for (uint32_t i = 0; i < p->n_inst; i++) {
        if (p->status[i] == SMP_OK) continue;
        if (!p->log || !p->log[i].len) continue;

        fprintf(out, "\n=== инстанс #%u ===\n", i);
        fwrite(p->log[i].buf, 1, p->log[i].len, out);

        /* Про обрезку говорим прямо: молча отдать половину диагностики хуже,
         * чем не отдать ничего. */
        if (p->log[i].truncated)
            fprintf(out, "\n[журнал инстанса #%u обрезан: не влез в %u байт]\n",
                    i, (unsigned)SMP_POOL_LOG_BYTES);
    }
    fflush(out);
}

/* ========================================================================== */

void *smp_vm_pool_tensor(const SmpVMPool *p, uint32_t inst,
                         const char *name, const SmpTensor **desc)
{
    if (!p->mod || inst >= p->n_inst) return NULL;

    for (uint32_t i = 0; i < p->mod->n_tens; i++) {
        const SmpTensor *t = &p->mod->tens[i];
        if (strcmp(smp_module_str(p->mod, t->name_id), name) != 0) continue;

        const uint32_t a = smp_tf_arena(t->flags);
        if (a >= p->inst[inst].n_arenas) return NULL;
        if (desc) *desc = t;
        return p->inst[inst].arenas[a].base + t->off;
    }
    return NULL;
}
