#!/usr/bin/env python3
"""Atlas de expertos: ¿a qué se dedica cada experto? (plan: docs/plans/gemma4-26b-expertos-bajo-demanda.md, sección 3)

Lee una traza del servidor instrumentado (llama-moe-trace.patch) y reparte sus peticiones en grupos (dominios o tareas):
  moe-atlas.py TRAZA --groups triaje=1-7,correo=8-10,respuesta=11,herramientas=12-60 [--json atlas.json] [--keep 48]

Salida:
  1. Tokens generados por grupo.
  2. Especialización: por (capa, experto) se reparte su masa de peso (solo tokens generados) entre los grupos; un experto es «de un
     grupo» si ese grupo tiene >= 60 % de su masa. Se compara con lo que daría el azar (reparto uniforme según el tamaño de cada grupo).
  3. Transferencia: se eligen los N expertos de cada capa con un grupo y se mide cuánto cubren en los otros (cobertura del experto
     principal). La diagonal es la cobertura con calibración propia; la fila «resto» elige con los demás grupos (dejar uno fuera).
     Si la transferencia cae mucho frente a la diagonal, el conjunto depende de la tarea y merece cambiarse según el contexto.
La petición 0 de la traza es el calentamiento del servidor: se descarta si no se indica otro rango.
"""
import argparse, collections, importlib.util, json, sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("stats", Path(__file__).with_name("moe-trace-stats.py"))
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)


def parse_groups(text):
    groups = {}
    for part in text.split(","):
        name, span = part.split("=")
        first, _, last = span.partition("-")
        groups[name] = (int(first), int(last or first))
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--groups", required=True)
    parser.add_argument("--keep", type=int, default=48)
    parser.add_argument("--share", type=float, default=0.6)
    parser.add_argument("--json")
    args = parser.parse_args()
    groups = parse_groups(args.groups)
    events = {}
    for name, span in groups.items():
        events[name], weighted = stats.read([args.trace], span)
        if not weighted:
            sys.exit("la traza no trae pesos del enrutador")
    layers = sorted(events[next(iter(groups))])
    print("tokens generados por grupo (por capa):", {n: sum(1 for r in events[n][layers[0]] if r[0]) for n in groups})

    # masa de peso por (capa, experto, grupo), solo tokens generados
    mass = {n: {l: collections.Counter() for l in layers} for n in groups}
    for n in groups:
        for l in layers:
            for generated, ids, weights in events[n][l]:
                if generated:
                    for e, w in zip(ids, weights):
                        mass[n][l][e] += w
    size = {n: sum(mass[n][layers[0]].values()) for n in groups}
    total_size = sum(size.values()) or 1
    owned = collections.Counter()
    seen = 0
    for l in layers:
        experts = set().union(*(mass[n][l].keys() for n in groups))
        for e in experts:
            per = {n: mass[n][l][e] for n in groups}
            s = sum(per.values())
            if s <= 0:
                continue
            seen += 1
            top = max(per, key=per.get)
            if per[top] / s >= args.share:
                owned[top] += 1
    print(f"\nexpertos (capa, id) usados: {seen}; con >= {args.share:.0%} de su masa en un solo grupo:")
    for n in groups:
        print(f"  {n:<14} {owned[n]:>5}   (peso del grupo en la traza: {size[n] / total_size:.0%})")
    print(f"  sin dueño      {seen - sum(owned.values()):>5}")

    # transferencia
    def cover(selection, rows_by_layer):
        main_hit = total = 0
        for l in layers:
            keep = set(selection[l])
            for generated, ids, weights in rows_by_layer[l]:
                if generated:
                    main_hit += ids[0] in keep
                    total += 1
        return main_hit / max(total, 1)

    names = list(groups)
    print(f"\ntransferencia: expertos principales dentro de {args.keep} por capa (filas: dónde se elige; columnas: dónde se mide)")
    print(f"{'':<14}" + "".join(f"{n[:12]:>13}" for n in names))
    result = {}
    for a in names:
        selection = {l: stats.choose(mass[a][l], set(), args.keep) for l in layers}
        row = [cover(selection, events[b]) for b in names]
        result[a] = dict(zip(names, row))
        print(f"{a:<14}" + "".join(f"{v:>13.1%}" for v in row))
    print("\ndejando un grupo fuera (se elige con los demás y se mide en el que falta):")
    for b in names:
        rest = {l: sum((mass[n][l] for n in names if n != b), collections.Counter()) for l in layers}
        selection = {l: stats.choose(rest[l], set(), args.keep) for l in layers}
        own = {l: stats.choose(mass[b][l], set(), args.keep) for l in layers}
        print(f"  {b:<14} con los demás {cover(selection, events[b]):.1%}   con el suyo {cover(own, events[b]):.1%}")
    if args.json:
        json.dump({"transfer": result, "owned": owned}, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
