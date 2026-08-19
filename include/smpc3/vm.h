/* SMPC3 :: vm.h -- Direct Threading VM.
 *
 * Диспетчеризация идёт через goto *dispatch[op] на прямых метках: у switch на
 * горячем цикле один общий переход, который предсказатель ветвлений угадать
 * не может, а здесь у каждой инструкции свой — история переходов работает.
 * Там, где computed goto недоступен, собирается корректная switch-версия.
 *
 * Память: арены поднимаются один раз в smp_vm_init по размерам из модуля.
 * За время исполнения не выделяется ни байта.
 */
#ifndef SMPC3_VM_H
#define SMPC3_VM_H

#include "smpc3/arena.h"
#include "smpc3/bytecode.h"
#include "smpc3/diag.h"
#include "smpc3/kernels.h"

/* Регистр держит либо дескриптор тензора, либо скаляр. */
typedef struct SmpReg {
    bool      is_tensor;
    SmpTensor t;
    SmpConst  s;
    uint8_t   dtype;    /* тип скаляра */
} SmpReg;

typedef struct SmpVM {
    const SmpModule *mod;
    SmpDiagCtx      *diag;

    SmpArena  arenas[SMP_MAX_ARENAS];
    uint32_t  n_arenas;

    /* Регистровый файл лежит прямо в структуре: глобальный массив сделал бы
     * две VM в одном процессе неотличимыми друг от друга. */
    SmpReg    regs[SMP_MAX_REGS];
    uint32_t  n_regs;

    uint32_t  instance;        /* номер инстанса в пуле; вне пула ноль      */
    uint32_t  pc;              /* текущая инструкция — для дампа            */
    uint64_t  n_executed;
    uint32_t  entry_mxcsr;     /* состояние MXCSR до запуска                */
    uint8_t   cur_fp_flags;    /* какие fp-режимы сейчас в MXCSR            */
    bool      trapped;

    /* Была ли VM поднята. Без этого признака smp_vm_release на обнулённой
     * структуре записал бы в MXCSR ноль, то есть СНЯЛ БЫ МАСКИ всех
     * исключений FPU — и следующая же неточная операция с плавающей точкой
     * убила бы процесс аппаратной ловушкой в произвольном месте. */
    bool      live;

    /* Куда пишет @emit. NULL означает stdout. */
    FILE     *out;

    /* Рабочая память ядер — своя у каждого инстанса. Именно она делает
     * несколько VM в разных потоках безопасными. */
    void        *scratch_mem;
    size_t       scratch_bytes;
    SmpKScratch  scratch;
} SmpVM;

SmpStatus smp_vm_init(SmpVM *vm, const SmpModule *mod, SmpDiagCtx *diag);
SmpStatus smp_vm_run(SmpVM *vm);
void      smp_vm_release(SmpVM *vm);

/* Дамп регистров — тот самый, который спецификация обещает при фатальной
 * ошибке. Подходит под SmpRegDumpFn и ставится в диагностический контекст. */
void      smp_vm_regdump(FILE *out, void *vm, bool color);

/* Показать содержимое именованных тензоров после прогона. */
void      smp_vm_dump_tensors(FILE *out, const SmpVM *vm, uint32_t max_elems);

/* Вывод тензора в поток. Живёт отдельным файлом: к диспетчеризации отношения
 * не имеет, а вот к философии — прямое. */
SmpStatus smp_vm_emit(FILE *dst, const struct SmpBuf *src, uint8_t fmt,
                      uint64_t *n_written, uint32_t *bad_cp);

/* Строка «какая ветка ядер выбрана» — для отчётов. */
const char *smp_vm_backend(void);

#endif /* SMPC3_VM_H */
