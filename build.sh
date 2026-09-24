#!/usr/bin/env bash
# SMPC3 :: build.sh -- сборка голым clang, без cmake/make.
#
#   ./build.sh              debug-сборка
#   ./build.sh release      -O3, без poison-заливки арены
#   ./build.sh test         собрать и прогнать тесты
#   ./build.sh release test
#   ./build.sh clean
#
# Файлы с SIMD-кернелами собираются с -mavx2 -mfma отдельно; всё остальное
# остаётся на базовом x86-64, потому что выбор ветки делает runtime-детект,
# а не компилятор.

set -euo pipefail
cd "$(dirname "$0")"

CLANG_DIR="/c/Program Files/LLVM/bin"
[[ -d "$CLANG_DIR" ]] && PATH="$CLANG_DIR:$PATH"
command -v clang >/dev/null || { echo "clang не найден"; exit 1; }

MODE="${1:-debug}"
OUT="build"
BIN="$OUT/smpc3.exe"

if [[ "$MODE" == "clean" ]]; then
    rm -rf "$OUT"/*.o "$OUT"/*.d "$OUT"/*.exe "$OUT"/*.pdb "$OUT"/*.ilk
    echo "очищено"
    exit 0
fi

WARN=(-Wall -Wextra -Wshadow -Wvla -Wpointer-arith -Wstrict-prototypes
      -Wmissing-prototypes -Wno-unused-parameter)
BASE=(-std=c99 -Iinclude -D_CRT_SECURE_NO_WARNINGS "${WARN[@]}")

if [[ "$MODE" == "release" ]]; then
    BASE+=(-O3 -DNDEBUG -fno-math-errno)
else
    BASE+=(-O1 -g -fno-omit-frame-pointer)
fi

# Кернелы, которым нужен полный AVX2+FMA на этапе компиляции.
SIMD_SRCS=(src/kernels)

mkdir -p "$OUT"

# Нужна ли пересборка. Кроме самого .c проверяются все заголовки из depfile,
# который clang пишет по -MMD. Без этого правка diag_codes.h не пересобирала
# diag.o, и enum расходился с таблицей описаний — молча, до первого запуска.
needs_rebuild() {
    local src="$1" obj="$2" dep="${obj%.o}.d"
    [[ -f "$obj" ]]      || return 0
    [[ "$src" -nt "$obj" ]] && return 0
    [[ -f "$dep" ]]      || return 0          # depfile пропал — не рискуем
    local f
    # -MMD не включает системные заголовки, поэтому путей с пробелами тут нет.
    for f in $(sed -e 's/\\$//' -e '1s/^[^:]*://' "$dep"); do
        [[ -f "$f" && "$f" -nt "$obj" ]] && return 0
    done
    return 1
}

compile() {   # compile <src> <obj> [доп. флаги...]
    local src="$1" obj="$2"; shift 2
    echo "  CC  $src"
    clang "${BASE[@]}" "$@" -MMD -MF "${obj%.o}.d" -c "$src" -o "$obj"
}

OBJS=()
NCOMPILED=0

for src in $(find src -name '*.c' | sort); do
    obj="$OUT/$(echo "${src#src/}" | tr '/' '_' | sed 's/\.c$/.o/')"
    EXTRA=()
    for d in "${SIMD_SRCS[@]}"; do
        [[ "$src" == "$d"/* ]] && EXTRA+=(-mavx2 -mfma)
    done

    if needs_rebuild "$src" "$obj"; then
        compile "$src" "$obj" ${EXTRA[@]+"${EXTRA[@]}"}
        NCOMPILED=$((NCOMPILED + 1))
    fi
    OBJS+=("$obj")
done

if [[ $NCOMPILED -gt 0 || ! -f "$BIN" ]]; then
    echo "  LD  $BIN"
    clang "${OBJS[@]}" -o "$BIN"
fi

echo "готово: $BIN ($MODE, ${#OBJS[@]} объектных файлов)"

# --- тесты ------------------------------------------------------------------
# Каждый tests/test_*.c — отдельный бинарь со своим main(); линкуется со всеми
# объектниками кроме src/tool (там лежит main() самого компилятора).
if [[ "$MODE" == "test" || "${2:-}" == "test" ]]; then
    CORE=()
    for o in "${OBJS[@]}"; do
        [[ "$o" == *"tool_"* ]] || CORE+=("$o")
    done

    RC=0
    for t in $(find tests -name 'test_*.c' | sort); do
        name="$(basename "$t" .c)"
        obj="$OUT/$name.o"
        if needs_rebuild "$t" "$obj"; then
            compile "$t" "$obj"
        fi
        clang "$obj" "${CORE[@]}" -o "$OUT/$name.exe"
        echo
        "./$OUT/$name.exe" || RC=1
        echo
    done

    # Снимки вывода: ловят то, чего не видят тесты на структурах — уехавшую
    # каретку, потерянную подсказку, изменившуюся формулировку.
    echo "SMPC3 tests :: снимки вывода"
    echo
    ./tests/golden.sh || RC=1

    # Шаг Qwen2 против эталона на Python. Нужен интерпретатор, который
    # действительно запускается: python3 на Windows бывает заглушкой.
    PY=""
    for p in python python3; do
        if "$p" -c "import sys" >/dev/null 2>&1; then PY="$p"; break; fi
    done
    echo
    echo "SMPC3 tests :: шаг Qwen2 против эталона"
    echo
    if [[ -n "$PY" ]]; then
        "$PY" -X utf8 tests/qwen_step.py "$BIN" || RC=1
    else
        echo "  пропущено: нет Python"
    fi

    exit $RC
fi
