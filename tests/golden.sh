#!/usr/bin/env bash
# SMPC3 :: golden.sh -- снимки вывода компилятора.
#
# Диагностика — главная особенность языка, и ломается она незаметно: код
# собирается, тесты на структурах проходят, а формулировка или каретка уехали.
# Здесь вывод сравнивается с эталонными снимками целиком, посимвольно.
#
#   ./tests/golden.sh            проверить
#   ./tests/golden.sh --update   перезаписать снимки (после осознанной правки)

set -uo pipefail
cd "$(dirname "$0")/.."

BIN=build/smpc3.exe

# Снимки сняты для машины с v256. Без этого на процессоре с AVX-512
# предупреждение W0512 просто не возникнет, и половина снимков разойдётся.
export SMPC3_VEC_BITS=256
DIR=tests/golden
TMP=build/golden_tmp
UPDATE=0
[[ "${1:-}" == "--update" ]] && UPDATE=1

[[ -x "$BIN" ]] || { echo "нет $BIN — сначала ./build.sh"; exit 1; }
mkdir -p "$DIR" "$TMP"

# Вывод обязан быть одинаковым на любой машине и при любом уровне оптимизации.
# Три вещи от этого зависят:
#   - имя процессора;
#   - путь к временному файлу;
#   - биты СОСТОЯНИЯ в MXCSR. Они липкие и копятся от любой операции с
#     плавающей точкой в процессе, включая libc, так что между -O1 и -O3
#     честно расходятся. Само наличие MXCSR в дампе снимок проверяет,
#     конкретное значение — нет.
#   - показания часов у пула инстансов: они меняются от запуска к запуску по
#     определению, и фиксировать их значило бы делать тест мигающим.
normalize() {
    sed -e 's/«[^»]*»/«ПРОЦЕССОР»/g' \
        -e 's/MXCSR = 0x[0-9A-Fa-f]*/MXCSR = 0x????/g' \
        -e 's/^время     : .*/время     : ?/' \
        -e 's/^темп      : .*/темп      : ?/' \
        -e "s|$TMP|<tmp>|g"
}

PASS=0
FAIL=0

check() {                       # check <имя> <аргументы smpc3...>
    local name="$1"; shift
    local want="$DIR/$name.txt"
    local got="$TMP/$name.out"

    "$BIN" "$@" > "$got.raw" 2>&1
    local rc=$?
    { echo "\$ smpc3 $*"; echo "--- код возврата: $rc"; cat "$got.raw"; } \
        | normalize > "$got"
    rm -f "$got.raw"

    if [[ $UPDATE -eq 1 ]]; then
        cp "$got" "$want"
        echo "  ЗАПИСАН  $name"
        return
    fi
    if [[ ! -f "$want" ]]; then
        echo "  НЕТ СНИМКА  $name  (запусти с --update)"
        FAIL=$((FAIL + 1))
        return
    fi
    if diff -q "$want" "$got" >/dev/null 2>&1; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
        echo "  РАСХОЖДЕНИЕ  $name"
        diff -u "$want" "$got" | head -30 | sed 's/^/      /'
    fi
}

# --- справочники -------------------------------------------------------------
check help
check codes         codes
check ops           ops
check attrs         attrs
check explain_0418  explain E0418
check explain_bad   explain E9999

# --- диагностика ------------------------------------------------------------
check demo          demo

# --- фронтенд ---------------------------------------------------------------
check lex_kernel    lex   examples/kernel.smpc
check lex_broken    lex   examples/broken.smpc
check parse_kernel  parse examples/kernel.smpc
check parse_bad     parse examples/bad_syntax.smpc
check check_kernel  check examples/kernel.smpc -v
check check_bad     check examples/bad_sema.smpc
check check_bad_rep check examples/bad_repeat.smpc
check check_rank    check examples/rank.smpc

# --- бэкенд и рантайм -------------------------------------------------------
check build_kernel  build examples/kernel.smpc -o "$TMP/kernel.s3b" -S
check dis_kernel    dis   "$TMP/kernel.s3b"
check run_hello     run   examples/hello.smpc
check run_hello2    run   examples/hello2.smpc
check run_table     run   examples/table.smpc
check run_bars      run   examples/bars.smpc
check run_pool      run   examples/pool.smpc
check run_pool_n8   run   examples/pool.smpc -n 8 -j 2
check run_dotprod   run   examples/dotprod.smpc
check run_slices    run   examples/slices.smpc
check run_types     run   examples/types.smpc
check run_gemm      run   examples/gemm.smpc
check run_forward   run   examples/forward.smpc
check check_forward check examples/forward.smpc -v
check run_repeat    run   examples/repeat.smpc
check run_io        run   examples/io.smpc --out V="$TMP/io.bin" --in W="$TMP/io.bin"
check build_repeat  build examples/repeat.smpc -o "$TMP/repeat.s3b" -S
check run_trap      run   examples/trap.smpc
check run_s3b       run   "$TMP/kernel.s3b"

# --- ошибки CLI -------------------------------------------------------------
check missing_file  run   examples/нет-такого.smpc
check bad_command   нетакой

rm -rf "$TMP"

if [[ $UPDATE -eq 1 ]]; then
    echo "снимки обновлены"
    exit 0
fi

echo
echo "$PASS снимков совпало, $FAIL разошлось"
[[ $FAIL -eq 0 ]]
