"""Lectura mínima de un GGUF: localizar un tensor Q8_0 y mapearlo sin copiarlo."""
import struct

import numpy as np

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
Q8_0 = 8
BLOCK = 32
# block_q8_0 de ggml: una escala en media precisión y 32 enteros de 8 bits (34 bytes).
Q8_BLOCK = np.dtype([("d", "<f2"), ("q", "i1", (BLOCK,))])


def _value(handle, kind):
    if kind == 8:
        (length,) = struct.unpack("<Q", handle.read(8))
        return handle.read(length)
    if kind == 9:
        inner, count = struct.unpack("<IQ", handle.read(12))
        if inner in _SCALARS:
            size = struct.calcsize(_SCALARS[inner])
            handle.seek(size * count, 1)
            return None
        return [_value(handle, inner) for _ in range(count)]
    fmt = _SCALARS[kind]
    return struct.unpack(fmt, handle.read(struct.calcsize(fmt)))[0]


def tensor_table(path):
    """Devuelve ({nombre: (dims, tipo, desplazamiento absoluto)}, metadatos escalares)."""
    with open(path, "rb") as handle:
        magic, _version, tensors, entries = struct.unpack("<4sIQQ", handle.read(24))
        if magic != b"GGUF":
            raise ValueError("no es un GGUF")
        metadata = {}
        for _ in range(entries):
            key = _value(handle, 8).decode()
            (kind,) = struct.unpack("<I", handle.read(4))
            value = _value(handle, kind)
            if kind not in (8, 9):
                metadata[key] = value
        infos = {}
        for _ in range(tensors):
            name = _value(handle, 8).decode()
            (rank,) = struct.unpack("<I", handle.read(4))
            dims = struct.unpack(f"<{rank}Q", handle.read(8 * rank))
            kind, offset = struct.unpack("<IQ", handle.read(12))
            infos[name] = (dims, kind, offset)
        alignment = metadata.get("general.alignment", 32)
        start = (handle.tell() + alignment - 1) // alignment * alignment
    return {name: (dims, kind, start + offset) for name, (dims, kind, offset) in infos.items()}, metadata


def output_matrix(path):
    """Matriz de salida como bloques Q8_0 mapeados: forma (vocabulario, bloques por fila)."""
    infos, _ = tensor_table(path)
    name = "output.weight" if "output.weight" in infos else "token_embd.weight"
    dims, kind, offset = infos[name]
    if kind != Q8_0:
        raise ValueError(f"{name} no es Q8_0 (tipo {kind})")
    width, vocabulary = dims
    if width % BLOCK:
        raise ValueError("la anchura no es múltiplo de 32")
    blocks = np.memmap(path, dtype=Q8_BLOCK, mode="r", offset=offset, shape=(vocabulary, width // BLOCK))
    return blocks, name


def dequantize(blocks):
    """Filas en precisión simple: (filas, anchura)."""
    rows = blocks["q"].astype(np.float32) * blocks["d"].astype(np.float32)[..., None]
    return rows.reshape(blocks.shape[0], -1)
