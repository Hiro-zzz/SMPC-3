#!/usr/bin/env python3
"""SMPC3 tests :: шаг Qwen2 против эталона.

Слой Qwen2 в миниатюре — та же архитектура, что у Qwen2.5-0.5B, только узкая:
RMSNorm, q/k/v со смещениями, RoPE с θ = 10^6, GQA (4 головы запроса на 2
головы K/V), KV-кэш, SwiGLU, общие эмбеддинги со слоем выхода. Веса
случайные, квантуются в Q8_0 здесь же — тем же алгоритмом, что C, бит в бит.

Скрипт пишет программу на SMPC_3 на три шага с KV-кэшем (токены заданы,
как при чтении подсказки), прогоняет её через `smpc3 run` с привязкой
файлов и сверяет логиты каждого шага с прямым проходом, посчитанным здесь в
double. Только стандартная библиотека.

    python tests/qwen_step.py [путь к smpc3]
"""

import math
import os
import random
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- форма модели ---------------------------------------------------------------

D, HQ, HK, HD, F, V, T = 64, 4, 2, 16, 128, 96, 8   # ширина, головы, FFN, словарь, кэш
G = HQ // HK                                       # голов запроса на голову K/V
EPS, THETA = 1e-6, 1000000.0
TOKENS = [5, 17, 42]                               # позиции 0, 1, 2
TOL = 1e-4                                         # от наибольшего |логита|

# --- f32 и f16 так, как их округляет C --------------------------------------------


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def f16_bits(x):
    """К ближайшему, при равенстве — к чётному: struct делает ровно это."""
    return struct.unpack("<H", struct.pack("<e", x))[0]


def f16_value(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def round_half_away(v):
    return int(math.floor(v + 0.5)) if v >= 0 else -int(math.floor(-v + 0.5))


def quantize_q8_0(row):
    """Как smp_q8_0_quantize: d = max|x| / 127 и 1/d во f32, веса — округлением
    x * (1/d) от нуля. Частное и произведение двух f32 точны в double, так что
    округление их до f32 здесь даёт то же, что арифметика f32 в C."""
    out, deq = bytearray(), []
    for b in range(0, len(row), 32):
        x = [f32(v) for v in row[b:b + 32]]
        amax = max(abs(v) for v in x)
        d = f32(amax / 127.0)
        inv = f32(1.0 / d) if d != 0.0 else 0.0
        h = f16_bits(d)
        dq = f16_value(h)
        out += struct.pack("<H", h)
        for v in x:
            q = max(-127, min(127, round_half_away(f32(v * inv))))
            out += struct.pack("<b", q)
            deq.append(dq * q)
    return bytes(out), deq


# --- веса ------------------------------------------------------------------------


def make_weights(rng):
    def mat(n, k, s):
        return [[rng.gauss(0.0, s) for _ in range(k)] for _ in range(n)]

    def vec(n, s, base=0.0):
        return [f32(base + rng.gauss(0.0, s)) for _ in range(n)]

    w = {
        "E":  mat(V, D, 0.5),
        "Wq": mat(HQ * HD, D, 0.2), "Wk": mat(HK * HD, D, 0.2), "Wv": mat(HK * HD, D, 0.2),
        "Wo": mat(D, HQ * HD, 0.2),
        "Wg": mat(F, D, 0.2), "Wu": mat(F, D, 0.2), "Wd": mat(D, F, 0.2),
        "bq": vec(HQ * HD, 0.1), "bk": vec(HK * HD, 0.1), "bv": vec(HK * HD, 0.1),
        "g1": vec(D, 0.1, 1.0), "g2": vec(D, 0.1, 1.0), "gf": vec(D, 0.1, 1.0),
    }
    files, used = {}, {}
    for name, value in w.items():
        if name.startswith(("E", "W")):
            blob, rows = bytearray(), []
            for r in value:
                b, dq = quantize_q8_0(r)
                blob += b
                rows.append(dq)
            files[name], used[name] = bytes(blob), rows
        else:
            files[name], used[name] = struct.pack(f"<{len(value)}f", *value), value
    return files, used


# --- эталон ----------------------------------------------------------------------


def rmsnorm(x, g):
    inv = 1.0 / math.sqrt(sum(v * v for v in x) / len(x) + EPS)
    return [v * inv * gv for v, gv in zip(x, g)]


def linear(x, w, b=None):
    y = [sum(a * c for a, c in zip(x, row)) for row in w]
    return [v + bv for v, bv in zip(y, b)] if b else y


def rope(head, pos):
    """Как Qwen2 в HF: частота и угол во f32, пары (i, i + n/2)."""
    n, half = len(head), len(head) // 2
    out = list(head)
    for i in range(half):
        inv = f32(1.0 / f32(THETA ** f32((2 * i) / n)))
        ang = f32(f32(float(pos)) * inv)
        c, s = f32(math.cos(ang)), f32(math.sin(ang))
        out[i] = head[i] * c - head[i + half] * s
        out[i + half] = head[i + half] * c + head[i] * s
    return out


def reference(w):
    kc, vc, logits = [], [], []
    for pos, tok in enumerate(TOKENS):
        x = list(w["E"][tok])
        xn = rmsnorm(x, w["g1"])
        q = linear(xn, w["Wq"], w["bq"])
        k = linear(xn, w["Wk"], w["bk"])
        v = linear(xn, w["Wv"], w["bv"])
        kc.append([rope(k[h * HD:(h + 1) * HD], pos) for h in range(HK)])
        vc.append([v[h * HD:(h + 1) * HD] for h in range(HK)])
        o = []
        for h in range(HQ):
            qh, g = rope(q[h * HD:(h + 1) * HD], pos), h // G
            s = [sum(a * b for a, b in zip(qh, kc[t][g])) / math.sqrt(HD) for t in range(pos + 1)]
            m = max(s)
            e = [math.exp(v_ - m) for v_ in s]
            z = sum(e)
            o += [sum(e[t] / z * vc[t][g][d] for t in range(pos + 1)) for d in range(HD)]
        x = [a + b for a, b in zip(x, linear(o, w["Wo"]))]
        xn = rmsnorm(x, w["g2"])
        gate = [a / (1.0 + math.exp(-a)) for a in linear(xn, w["Wg"])]
        up = linear(xn, w["Wu"])
        x = [a + b for a, b in zip(x, linear([a * b for a, b in zip(gate, up)], w["Wd"]))]
        logits.append(linear(rmsnorm(x, w["gf"]), w["E"]))
    return logits


# --- программа на SMPC_3 ---------------------------------------------------------


def program():
    lines = [
        "// Шаг Qwen2 в миниатюре: сгенерировано tests/qwen_step.py.",
        f"[#arena:1] *&E<q8_0:{V},{D}> -> @load => *&E;",
    ]
    for n, (r, c) in {"Wq": (HQ * HD, D), "Wk": (HK * HD, D), "Wv": (HK * HD, D),
                      "Wo": (D, HQ * HD), "Wg": (F, D), "Wu": (F, D), "Wd": (D, F)}.items():
        lines.append(f"[#arena:1] *&{n}<q8_0:{r},{c}> -> @load => *&{n};")
    for n, c in {"bq": HQ * HD, "bk": HK * HD, "bv": HK * HD,
                 "g1": D, "g2": D, "gf": D}.items():
        lines.append(f"[#arena:1] *&{n}<f32:1,{c}> -> @load => *&{n};")
    lines += [
        f"[#arena:2] *&Kc<f32:{T},{HK},{HD}> -> @alloc => $kc;",
        f"[#arena:2] *&Vc<f32:{T},{HK},{HD}> -> @alloc => $vc;",
        f"*&x<f32:1,{D}> -> @alloc => $x0;",
        f"*&xn<f32:1,{D}> -> @alloc => $xn0;",
        f"*&q<f32:{HK},{G},{HD}> -> @alloc => $q0;",
        f"*&o<f32:{HK},{G},{HD}> -> @alloc => $o0;",
        f"*&gt<f32:1,{F}> -> @alloc => $gt0;",
    ]
    scale = 1.0 / math.sqrt(HD)
    for pos, tok in enumerate(TOKENS):
        lines += [
            f"// --- позиция {pos}, токен {tok}",
            f"{pos} => $pos;",
            f"{pos + 1} => $len;",
            f"*&E[{tok}, ..] -> @cast.f32 -> @reshape(1, {D}) => *&x;",
            f"*&x -> @rmsnorm({EPS!r}) -> @mul(*&g1) => *&xn;",
            f"*&xn -> @mmul.t(*&Wq) -> @add(*&bq) -> @reshape({HK}, {G}, {HD})"
            f" -> @rope($pos, {THETA!r}) => *&q;",
            f"*&xn -> @mmul.t(*&Wk) -> @add(*&bk) -> @reshape({HK}, {HD})"
            f" -> @rope($pos, {THETA!r}) => *&Kc[$pos, .., ..];",
            f"*&xn -> @mmul.t(*&Wv) -> @add(*&bv) -> @reshape({HK}, {HD}) => *&Vc[$pos, .., ..];",
            f"[#repeat:{HK}, #index:g] *&q[$g, .., ..] -> @mmul.t(*&Kc[.., $g, ..])"
            f" -> @scale({scale!r}) -> @softmax($len) -> @mmul(*&Vc[.., $g, ..]) => *&o[$g, .., ..];",
            f"*&o -> @reshape(1, {HQ * HD}) -> @mmul.t(*&Wo) -> @add(*&x) => *&x;",
            f"*&x -> @rmsnorm({EPS!r}) -> @mul(*&g2) => *&xn;",
            "*&xn -> @mmul.t(*&Wg) -> @silu => *&gt;",
            "*&xn -> @mmul.t(*&Wu) -> @mul(*&gt) -> @mmul.t(*&Wd) -> @add(*&x) => *&x;",
            f"*&x -> @rmsnorm({EPS!r}) -> @mul(*&gf) -> @mmul.t(*&E) => *&L{pos}<f32:1,{V}>;",
            f"*&L{pos} -> @store => $n{pos};",
            f"*&L{pos} -> @argmax => $next{pos};",
        ]
    return "\n".join(lines) + "\n"


# --- прогон ----------------------------------------------------------------------


def main():
    exe = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "build", "smpc3.exe" if os.name == "nt" else "smpc3"))
    files, used = make_weights(random.Random(20260924))
    want = reference(used)

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "qwen_step.smpc")
        with open(src, "w", encoding="utf-8") as f:
            f.write(program())
        args = [exe, "run", src]
        for name, blob in files.items():
            path = os.path.join(tmp, name + ".bin")
            with open(path, "wb") as f:
                f.write(blob)
            args += ["--in", f"{name}={path}"]
        outs = [os.path.join(tmp, f"L{p}.bin") for p in range(len(TOKENS))]
        for p, path in enumerate(outs):
            args += ["--out", f"L{p}={path}"]

        run = subprocess.run(args, capture_output=True)
        if run.returncode != 0:
            print(run.stdout.decode("utf-8", "replace")[-3000:])
            print(run.stderr.decode("utf-8", "replace")[-3000:])
            print("smpc3 run упал")
            return 1
        got = []
        for path in outs:
            with open(path, "rb") as f:
                got.append(list(struct.unpack(f"<{V}f", f.read())))

    failed = 0
    for p, (w_, g_) in enumerate(zip(want, got)):
        scale = max(abs(v) for v in w_)
        err = max(abs(a - b) for a, b in zip(w_, g_)) / scale
        top_w = max(range(V), key=lambda i: w_[i])
        top_g = max(range(V), key=lambda i: g_[i])
        ok = err < TOL and top_w == top_g
        failed += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  позиция {p}: отн. ошибка {err:.2e},"
              f" argmax {top_g} (эталон {top_w})")
    print(f"шаг Qwen2: {len(TOKENS)} позиций, {failed} мимо")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
