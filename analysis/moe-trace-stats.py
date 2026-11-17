#!/usr/bin/env python3
"""Analiza las trazas de expertos de un llama-server instrumentado (scripts/dev/llama-moe-trace.patch).

Formatos de traza (registros consecutivos, cabecera {int32 capa, int32 n_tokens, int32 n}):
  n > 0  : por token, n bytes con los ids de los expertos elegidos, en orden de probabilidad (primera versión).
  n < 0  : por token, |n| bytes de ids y |n| float32 con su probabilidad del enrutador (antes de renormalizar).
Un registro con n_tokens == 1 es un token generado; con más, es lectura de prompt.

  moe-trace-stats.py TRAZA...                                  cobertura por N (selección hecha con esas trazas)
  moe-trace-stats.py TRAZA... --select N --out sel.json        escribe la selección de N expertos por capa
  moe-trace-stats.py TRAZA... --apply sel.json                 cobertura de una selección sobre otras trazas
Opciones de selección:
  --by count|weight   contar apariciones, o sumar el peso renormalizado de cada experto (por defecto weight si la traza lo trae)
  --prompt-factor F   cuánto cuenta un token de prompt frente a uno generado (por defecto 0.25)
  --mandatory T       un experto que alguna vez lleva un peso renormalizado >= T en un token generado entra siempre (por defecto 0.5)

Columnas: «peso cubierto» = fracción del peso renormalizado de los expertos elegidos que queda dentro de la selección;
«principal» = fracción de tokens cuyo experto de mayor peso está dentro; «intactos» = tokens con todos sus expertos dentro.
Todo medido sobre los tokens generados.
"""
import argparse, collections, json, struct, sys


def read(paths, requests=None):
    """requests = (primera, última) limita a esas peticiones de la traza (0 = la primera evaluación, que suele ser el
    calentamiento del servidor). Una petición empieza con un lote de prompt (más de un token) en la capa 0."""
    events = collections.defaultdict(list)  # capa -> [(generado, ids, pesos renormalizados)]
    weighted = False
    for path in paths:
        data = open(path, 'rb').read()
        pos = 0
        request, in_prompt = -1, False
        while pos + 12 <= len(data):
            layer, tokens, used = struct.unpack_from('<iii', data, pos)
            pos += 12
            if tokens > 1 and layer == 0 and not in_prompt:
                request += 1
            if layer == 0:
                in_prompt = tokens > 1
            keep = requests is None or requests[0] <= request <= requests[1]
            for _ in range(tokens):
                if used > 0:
                    ids = data[pos:pos + used]
                    pos += used
                    weights = [1.0 / used] * used
                else:
                    count = -used
                    ids = data[pos:pos + count]
                    raw = struct.unpack_from(f'<{count}f', data, pos + count)
                    pos += count * 5
                    total = sum(raw) or 1.0
                    weights = [value / total for value in raw]
                    weighted = True
                if len(ids) == abs(used) and keep:
                    events[layer].append((tokens == 1, bytes(ids), weights))
    return events, weighted


def scores(events, by, prompt_factor):
    result = {}
    for layer, rows in events.items():
        score = collections.Counter()
        for generated, ids, weights in rows:
            factor = 1.0 if generated else prompt_factor
            for expert, weight in zip(ids, weights):
                score[expert] += factor * (weight if by == "weight" else 1.0)
        result[layer] = score
    return result


def mandatory(events, threshold):
    result = {}
    for layer, rows in events.items():
        result[layer] = {ids[0] for generated, ids, weights in rows if generated and weights[0] >= threshold}
    return result


def choose(score, forced, keep):
    # Los obligatorios no se truncan: si no caben en el presupuesto, esa selección no es viable.
    if len(forced) > keep:
        raise SystemExit(f"selección inviable: {len(forced)} expertos obligatorios en una capa y solo caben {keep}")
    ids = sorted(forced, key=lambda expert: -score[expert])
    ids += [expert for expert, _ in score.most_common() if expert not in ids][:keep - len(ids)]
    ids += [expert for expert in range(256) if expert not in ids][:keep - len(ids)]
    return sorted(ids)


def coverage(events, selection):
    mass = main = intact = total = 0
    for layer, rows in events.items():
        keep = set(selection[layer])
        for generated, ids, weights in rows:
            if not generated:
                continue
            inside = [expert in keep for expert in ids]
            mass += sum(weight for weight, hit in zip(weights, inside) if hit)
            main += inside[0]
            intact += all(inside)
            total += 1
    total = max(total, 1)
    return mass / total, main / total, intact / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+")
    parser.add_argument("--select", type=int)
    parser.add_argument("--out")
    parser.add_argument("--apply")
    parser.add_argument("--by", choices=["count", "weight"])
    parser.add_argument("--prompt-factor", type=float, default=0.25)
    parser.add_argument("--mandatory", type=float, default=0.5)
    parser.add_argument("--requests", help="solo estas peticiones de la traza, p. ej. 1-7 (para calibrar por tarea)")
    args = parser.parse_args()
    span = tuple(int(x) for x in args.requests.split("-")) if args.requests else None
    events, weighted = read(args.traces, span)
    by = args.by or ("weight" if weighted else "count")
    layers = sorted(events)
    generated = sum(1 for row in events[layers[0]] if row[0])
    print(f"capas: {len(layers)}; tokens por capa: {len(events[layers[0]])} ({generated} generados); traza con pesos: {weighted}; selección por: {by}")
    if args.apply:
        selection = {int(k): v for k, v in json.load(open(args.apply)).items()}
        mass, first, intact = coverage(events, selection)
        print(f"selección {args.apply} ({len(next(iter(selection.values())))} por capa): peso cubierto {mass:.1%}, principal {first:.1%}, intactos {intact:.1%}")
        return
    score = scores(events, by, args.prompt_factor)
    forced = mandatory(events, args.mandatory) if weighted else {layer: set() for layer in layers}
    print(f"expertos obligatorios por capa (peso >= {args.mandatory} en algún token generado): media {sum(len(v) for v in forced.values()) / len(layers):.1f}, máximo {max(len(v) for v in forced.values())}")
    print(f"{'N':>4} {'peso cubierto':>14} {'principal':>10} {'intactos':>9}")
    for keep in (8, 12, 16, 20, 24, 32, 40, 48, 64, 96):
        if keep < max(len(v) for v in forced.values()):
            print(f"{keep:>4}   inviable: no caben los obligatorios")
            continue
        selection = {layer: choose(score[layer], forced[layer], keep) for layer in layers}
        mass, first, intact = coverage(events, selection)
        print(f"{keep:>4} {mass:>14.1%} {first:>10.1%} {intact:>9.1%}")
    if args.select:
        selection = {layer: choose(score[layer], forced[layer], args.select) for layer in layers}
        json.dump({str(k): v for k, v in selection.items()}, open(args.out, 'w'))
        print(f"selección de {args.select} por capa escrita en {args.out}")


if __name__ == "__main__":
    sys.exit(main())
