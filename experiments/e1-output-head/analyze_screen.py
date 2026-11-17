#!/usr/bin/env python3
"""E1: cabeza de salida en dos etapas (see docs/RESULTS.md).

Con estados finales reales del modelo (volcados con DAFO_SWARM_DUMP_FINAL_HIDDEN) y la matriz de salida del GGUF, mide para
varias cribas compactas del vocabulario:
  - si la criba sola ya acierta el token de mayor logit, y en qué puesto queda ese token (recall@C);
  - cuántas filas exactas hay que leer para GARANTIZAR el mismo token (cota de error de la criba, kappa = 1), y cuántas con
    cotas más ajustadas (kappa < 1: ya no es garantía, se comprueba contra el resultado exacto);
  - los bytes leídos por token frente a leer la matriz entera.

  analyze_screen.py --gguf MODELO.gguf --states 'DIR/states-*.bin' --out resultado.json
"""
import argparse
import glob
import json
import time

import numpy as np

import gguf_q8

KAPPAS = (1.0, 0.5, 0.25, 0.125)
RECALLS = (1, 16, 64, 256, 1024, 4096)


def load_states(paths, width, prefill_sample, seed):
    """Estados (N, anchura) y su clase: 0 = posición del prompt, 1 = token generado."""
    rng = np.random.default_rng(seed)
    states, kinds = [], []
    for path in paths:
        raw = np.memmap(path, dtype=np.uint8, mode="r")
        offset, first = 0, True
        while offset < raw.size:
            count = int(raw[offset:offset + 4].view("<u4")[0])
            rows = raw[offset + 4:offset + 4 + count * width * 4].view("<f4").reshape(count, width)
            if first and count > 1:
                keep = np.sort(rng.choice(count, size=min(prefill_sample, count), replace=False))
                states.append(np.array(rows[keep]))
                kinds.append(np.zeros(len(keep), np.uint8))
            else:
                states.append(np.array(rows))
                kinds.append(np.ones(count, np.uint8))
            first = False
            offset += 4 + count * width * 4
    return np.concatenate(states), np.concatenate(kinds)


class BlockScreen:
    """Criba que cuantiza cada bloque de 32 pesos con `bits` por peso y una escala por bloque."""

    def __init__(self, bits):
        self.bits = bits
        self.name = f"{bits} bit"

    def row_bytes(self, width):
        return width * self.bits // 8 + width // gguf_q8.BLOCK * 2

    def prepare(self, states):
        self.transposed = np.ascontiguousarray(states.T)
        self.norms = np.linalg.norm(states, axis=1)

    def approximate(self, weights):
        blocks = weights.reshape(weights.shape[0], -1, gguf_q8.BLOCK)
        if self.bits == 1:
            codes = np.where(blocks >= 0, np.float32(1), np.float32(-1))
        else:
            half = 1 << (self.bits - 1)
            step = np.abs(blocks).max(axis=2, keepdims=True) / np.float32(half) + np.float32(1e-12)
            codes = np.clip(np.floor(blocks / step), -half, half - 1) + np.float32(0.5)
        # Escala por bloque de mínimos cuadrados para esos códigos.
        scale = (blocks * codes).sum(axis=2, keepdims=True) / (codes * codes).sum(axis=2, keepdims=True)
        return (codes * scale).reshape(weights.shape).astype(np.float32)

    def scores(self, weights):
        approximation = self.approximate(weights)
        return approximation @ self.transposed, (np.linalg.norm(weights - approximation, axis=1),)

    def epsilon(self, errors):
        return errors[0][:, None] * self.norms[None, :]


class ProjectionScreen:
    """Criba de baja dimensión: las filas proyectadas a las `rank` direcciones principales, a 8 bits."""

    def __init__(self, basis):
        self.basis = basis
        self.name = f"proyección {basis.shape[1]}"

    def row_bytes(self, _width):
        return self.basis.shape[1] + 2

    def prepare(self, states):
        self.inside = self.basis.T @ states.T
        inside_norms = np.linalg.norm(self.inside, axis=0)
        self.inside_norms = inside_norms
        self.outside_norms = np.sqrt(np.maximum(np.linalg.norm(states, axis=1) ** 2 - inside_norms ** 2, 0))

    def scores(self, weights):
        coefficients = weights @ self.basis
        scale = np.abs(coefficients).max(axis=1, keepdims=True) / np.float32(127) + np.float32(1e-12)
        stored = np.round(coefficients / scale) * scale
        outside = np.sqrt(np.maximum((weights * weights).sum(axis=1) - (coefficients * coefficients).sum(axis=1), 0))
        return stored @ self.inside, (outside, np.linalg.norm(coefficients - stored, axis=1))

    def epsilon(self, errors):
        outside, rounding = errors
        return outside[:, None] * self.outside_norms[None, :] + rounding[:, None] * self.inside_norms[None, :]


def chunks(blocks, size):
    for start in range(0, blocks.shape[0], size):
        yield start, gguf_q8.dequantize(blocks[start:start + size])


def principal_basis(blocks, size, ranks):
    width = blocks.shape[1] * gguf_q8.BLOCK
    gram = np.zeros((width, width), np.float64)
    for _, weights in chunks(blocks, size):
        gram += weights.T @ weights
    _, vectors = np.linalg.eigh(gram)
    return {rank: np.ascontiguousarray(vectors[:, ::-1][:, :rank]).astype(np.float32) for rank in ranks}


def summarize(values, mask):
    selected = values[mask]
    return {
        "media": float(selected.mean()),
        "mediana": float(np.median(selected)),
        "p90": float(np.quantile(selected, 0.90)),
        "p99": float(np.quantile(selected, 0.99)),
        "max": float(selected.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--states", required=True, help="patrón de los volcados de estados")
    parser.add_argument("--out", required=True)
    parser.add_argument("--prefill-sample", type=int, default=400, help="posiciones de prompt por volcado")
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--rows-limit", type=int, default=0, help="solo para depurar: primeras filas de la matriz")
    parser.add_argument("--bits", default="1,2,4", help="cribas por bloque: bits por peso")
    parser.add_argument("--projections", default="128,256")
    parser.add_argument("--seed", type=int, default=7)
    arguments = parser.parse_args()

    started = time.time()
    blocks, name = gguf_q8.output_matrix(arguments.gguf)
    if arguments.rows_limit:
        blocks = blocks[:arguments.rows_limit]
    vocabulary, width = blocks.shape[0], blocks.shape[1] * gguf_q8.BLOCK
    full_row = blocks.shape[1] * gguf_q8.Q8_BLOCK.itemsize
    states, kinds = load_states(sorted(glob.glob(arguments.states)), width, arguments.prefill_sample, arguments.seed)
    count = states.shape[0]
    print(f"{name}: {vocabulary} x {width}, {vocabulary * full_row / 2**20:.0f} MiB; estados: {count} "
          f"({int(kinds.sum())} generados, {int((kinds == 0).sum())} de prompt)", flush=True)

    ranks = [int(rank) for rank in arguments.projections.split(",") if rank]
    bases = principal_basis(blocks, arguments.chunk, ranks) if ranks else {}
    bits = [int(value) for value in arguments.bits.split(",") if value]
    screens = [BlockScreen(value) for value in bits] + [ProjectionScreen(bases[rank]) for rank in ranks]
    for screen in screens:
        screen.prepare(states)
    transposed = np.ascontiguousarray(states.T)
    columns = np.arange(count)
    print(f"bases listas ({time.time() - started:.0f} s)", flush=True)

    # Pasada A: resultado exacto (mejor y segundo logit) y el candidato que elige cada criba sola.
    best = np.full(count, -np.inf, np.float32)
    second = np.full(count, -np.inf, np.float32)
    winner = np.zeros(count, np.int64)
    screen_best = [np.full(count, -np.inf, np.float32) for _ in screens]
    screen_winner = [np.zeros(count, np.int64) for _ in screens]
    for start, weights in chunks(blocks, arguments.chunk):
        logits = weights @ transposed
        index = logits.argmax(axis=0)
        top = logits[index, columns]
        logits[index, columns] = -np.inf
        runner = logits.max(axis=0)
        second = np.maximum(np.maximum(second, runner), np.minimum(best, top))
        update = top > best
        winner[update] = start + index[update]
        best = np.maximum(best, top)
        for position, screen in enumerate(screens):
            approximate, _ = screen.scores(weights)
            index = approximate.argmax(axis=0)
            top = approximate[index, columns]
            update = top > screen_best[position]
            screen_winner[position][update] = start + index[update]
            screen_best[position] = np.maximum(screen_best[position], top)
    print(f"pasada A lista ({time.time() - started:.0f} s)", flush=True)

    def rows_of(indices):
        unique, inverse = np.unique(indices, return_inverse=True)
        return gguf_q8.dequantize(blocks[unique])[inverse]

    winner_rows = rows_of(winner)
    # Por criba: valor y cota de la criba en la fila ganadora exacta, y logit exacto de su propio candidato (la referencia).
    winner_score, winner_epsilon, reference = [], [], []
    for position, screen in enumerate(screens):
        approximate, errors = screen.scores(winner_rows)
        winner_score.append(approximate[columns, columns])
        winner_epsilon.append(screen.epsilon(errors)[columns, columns])
        reference.append((rows_of(screen_winner[position]) * states).sum(axis=1))

    # Pasada B: puesto del ganador en la criba y filas que hay que leer con cada cota.
    rank = [np.zeros(count, np.int64) for _ in screens]
    needed = [{kappa: np.zeros(count, np.int64) for kappa in KAPPAS} for _ in screens]
    tightness = [np.zeros(count, np.float32) for _ in screens]
    for start, weights in chunks(blocks, arguments.chunk):
        logits = weights @ transposed
        for position, screen in enumerate(screens):
            approximate, errors = screen.scores(weights)
            epsilon = screen.epsilon(errors)
            rank[position] += (approximate > winner_score[position][None, :]).sum(axis=0)
            for kappa in KAPPAS:
                needed[position][kappa] += (approximate + kappa * epsilon >= reference[position][None, :]).sum(axis=0)
            ratio = np.abs(logits - approximate) / np.maximum(epsilon, 1e-20)
            tightness[position] = np.maximum(tightness[position], ratio.max(axis=0))
    print(f"pasada B lista ({time.time() - started:.0f} s)", flush=True)

    full_bytes = vocabulary * full_row
    masks = {"todos": np.ones(count, bool), "generados": kinds == 1, "prompt": kinds == 0}
    result = {
        "matriz": name, "vocabulario": vocabulary, "anchura": width, "bytes_completa": full_bytes, "estados": count,
        "margen_exacto": {key: summarize(best - second, mask) for key, mask in masks.items() if mask.any()},
        "cribas": [],
    }
    for position, screen in enumerate(screens):
        screen_bytes = vocabulary * screen.row_bytes(width)
        entry = {"criba": screen.name, "bytes_criba": screen_bytes, "fraccion_criba": screen_bytes / full_bytes, "por_clase": {}}
        for key, mask in masks.items():
            if not mask.any():
                continue
            by_class = {
                "criba_sola_acierta": float((screen_winner[position] == winner)[mask].mean()),
                "recall": {str(limit): float((rank[position] < limit)[mask].mean()) for limit in RECALLS},
                "ajuste_de_la_cota_max": float(tightness[position][mask].max()),
                "cotas": {},
            }
            for kappa in KAPPAS:
                rows = needed[position][kappa]
                correct = winner_score[position] + kappa * winner_epsilon[position] >= reference[position]
                bytes_read = screen_bytes + np.minimum(rows * full_row, full_bytes)
                by_class["cotas"][str(kappa)] = {
                    "token_correcto": float(correct[mask].mean()),
                    "filas": summarize(rows.astype(np.float64), mask),
                    "fraccion_leida_media": float(bytes_read[mask].mean() / full_bytes),
                    "resueltos_con_menos_del_25_por_ciento": float((bytes_read <= 0.25 * full_bytes)[mask].mean()),
                }
            entry["por_clase"][key] = by_class
        result["cribas"].append(entry)
    with open(arguments.out, "w") as handle:
        json.dump(result, handle, indent=1, ensure_ascii=False)

    print(f"\nmargen exacto (mejor - segundo logit), mediana: {result['margen_exacto']['todos']['mediana']:.3f}")
    for entry in result["cribas"]:
        print(f"\n== criba {entry['criba']}: {entry['bytes_criba'] / 2**20:.0f} MiB ({entry['fraccion_criba']:.1%} de la matriz)")
        for key, by_class in entry["por_clase"].items():
            recall = by_class["recall"]
            print(f"  [{key}] sola acierta {by_class['criba_sola_acierta']:.1%}; recall@16 {recall['16']:.1%} @256 {recall['256']:.1%} "
                  f"@4096 {recall['4096']:.2%}; ajuste máx. de la cota {by_class['ajuste_de_la_cota_max']:.3f}")
            for kappa, bound in by_class["cotas"].items():
                print(f"     kappa {kappa}: correcto {bound['token_correcto']:.2%}, filas mediana {bound['filas']['mediana']:.0f} "
                      f"p99 {bound['filas']['p99']:.0f}, lee {bound['fraccion_leida_media']:.1%} de media, "
                      f"<25 %: {bound['resueltos_con_menos_del_25_por_ciento']:.1%}")
    print(f"\ntotal {time.time() - started:.0f} s; resultado en {arguments.out}")


if __name__ == "__main__":
    main()
