#!/usr/bin/env python3
"""Convierte una selección de expertos (JSON de moe-trace-stats.py) en el fichero de máscara del llama-server
instrumentado (LLAMA_MOE_MASK): una línea "<capa> <id> <id> ..." por capa restringida.

  moe-mask.py sel.json --out mask.txt [--free 0-9,20-29]   las capas de --free quedan con todos sus expertos
  moe-mask.py --none --out mask.txt                        máscara vacía (modelo completo)
"""
import argparse, json, sys


def layers(text):
    result = set()
    for part in filter(None, (text or "").split(",")):
        low, _, high = part.partition("-")
        result.update(range(int(low), int(high or low) + 1))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("selection", nargs="?")
    parser.add_argument("--out", required=True)
    parser.add_argument("--free")
    parser.add_argument("--none", action="store_true")
    args = parser.parse_args()
    lines = []
    if not args.none:
        free = layers(args.free)
        for layer, ids in sorted(((int(k), v) for k, v in json.load(open(args.selection)).items())):
            if layer not in free:
                lines.append(f"{layer} " + " ".join(str(i) for i in sorted(ids)))
    open(args.out, "w").write("\n".join(lines) + ("\n" if lines else ""))
    print(f"{len(lines)} capas restringidas -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
