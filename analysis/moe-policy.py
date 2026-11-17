#!/usr/bin/env python3
"""¿Qué conjunto de N expertos por capa conviene tener residentes para cada petición? Simula políticas sobre una traza completa.

Usa la traza del servidor instrumentado (modelo completo, sin máscara) de moe-domains.py: las peticiones están en el orden de
`moe-domains.py --list` (12 por dominio; las 6 primeras «cal», las 6 últimas «val»). Para cada petición de validación se elige un
conjunto de N expertos por capa y se mide qué parte de los tokens GENERADOS tiene su experto principal dentro (y qué parte del peso).

Políticas:
  global     los N más pesados de las peticiones de calibración de los OTROS dominios (sin ver nada del dominio de la petición)
  prompt     los N más pesados en la lectura del prompt de esa petición (lo que se sabría antes de generar)
  mezcla a   global y prompt normalizados y mezclados: a*global + (1-a)*prompt  (a = 0,75; 0,5; 0,25)
  dominio    los N de las peticiones de calibración del MISMO dominio (el perfil de un agente que hace siempre lo mismo)
  oráculo    los N de los tokens generados de esa petición (techo; sobreajusta)
  moe-policy.py TRAZA [--n 48,72] [--per-domain]
"""
import argparse, collections, struct, sys

DOMAINS = ["triage", "mail", "notes_qa", "code", "numbers", "summarize", "knowledge", "reasoning", "spanish", "translate"]
PER = 12


def read_requests(path):
    data = open(path, "rb").read()
    pos, requests, in_prompt = 0, [], False
    while pos + 12 <= len(data):
        layer, tokens, used = struct.unpack_from("<iii", data, pos)
        pos += 12
        if tokens > 1 and layer == 0 and not in_prompt:
            requests.append(collections.defaultdict(list))
        if layer == 0:
            in_prompt = tokens > 1
        count = abs(used)
        for _ in range(tokens):
            ids = data[pos:pos + count]
            if used > 0:
                weights = [1.0 / count] * count
                pos += count
            else:
                raw = struct.unpack_from(f"<{count}f", data, pos + count)
                pos += count * 5
                total = sum(raw) or 1.0
                weights = [v / total for v in raw]
            if requests:
                requests[-1][layer].append((tokens == 1, bytes(ids), weights))
    return requests


def score(request, layer, generated):
    c = collections.Counter()
    for gen, ids, weights in request[layer]:
        if gen == generated:
            for e, w in zip(ids, weights):
                c[e] += w
    return c


def top(counter, n):
    best = [e for e, _ in collections.Counter(counter).most_common(n)]
    best += [e for e in range(128) if e not in best][: n - len(best)]
    return set(best)


def normalized(counter):
    total = sum(counter.values()) or 1.0
    return {e: v / total for e, v in counter.items()}


def cover(request, selection):
    main = mass = total = 0
    for layer, rows in request.items():
        keep = selection[layer]
        for gen, ids, weights in rows:
            if gen:
                main += ids[0] in keep
                mass += sum(w for e, w in zip(ids, weights) if e in keep)
                total += 1
    return main, mass, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--n", default="48,72")
    parser.add_argument("--per-domain", action="store_true")
    args = parser.parse_args()
    requests = read_requests(args.trace)
    if len(requests) != len(DOMAINS) * PER:
        print(f"aviso: {len(requests)} peticiones en la traza, se esperaban {len(DOMAINS) * PER}", file=sys.stderr)
    layers = sorted(requests[0])
    dom = lambda i: i // PER
    cal = lambda i: i % PER < PER // 2
    for n in [int(x) for x in args.n.split(",")]:
        policies = ["global", "prompt", "mezcla .75", "mezcla .50", "mezcla .25", "dominio", "oráculo"]
        result = {p: collections.defaultdict(lambda: [0, 0.0, 0]) for p in policies}
        for i, request in enumerate(requests):
            if cal(i) or i >= len(DOMAINS) * PER:
                continue
            sel = {p: {} for p in policies}
            for layer in layers:
                other = collections.Counter()
                same = collections.Counter()
                for j, r in enumerate(requests):
                    if j < len(DOMAINS) * PER and cal(j):
                        (same if dom(j) == dom(i) else other).update(score(r, layer, True))
                glob = normalized(other)
                prompt = normalized(score(request, layer, False))
                sel["global"][layer] = top(glob, n)
                sel["prompt"][layer] = top(prompt, n)
                for a, name in ((0.75, "mezcla .75"), (0.5, "mezcla .50"), (0.25, "mezcla .25")):
                    mix = collections.Counter({e: a * glob.get(e, 0) + (1 - a) * prompt.get(e, 0) for e in set(glob) | set(prompt)})
                    sel[name][layer] = top(mix, n)
                sel["dominio"][layer] = top(same, n)
                sel["oráculo"][layer] = top(score(request, layer, True), n)
            for p in policies:
                m, w, t = cover(request, sel[p])
                acc = result[p][DOMAINS[dom(i)]]
                acc[0] += m; acc[1] += w; acc[2] += t
        print(f"\n== N = {n} expertos por capa; peticiones de validación; cobertura del experto principal / del peso (tokens generados)")
        header = f"{'política':<12}" + "".join(f"{d[:9]:>10}" for d in DOMAINS) + f"{'TOTAL':>10}"
        print(header)
        for p in policies:
            row, tm, tw, tt = f"{p:<12}", 0, 0.0, 0
            for d in DOMAINS:
                m, w, t = result[p][d]
                row += f"{(m / t if t else 0):>10.1%}"
                tm += m; tw += w; tt += t
            print(row + f"{tm / max(tt, 1):>10.1%}   peso {tw / max(tt, 1):.1%}")


if __name__ == "__main__":
    main()
