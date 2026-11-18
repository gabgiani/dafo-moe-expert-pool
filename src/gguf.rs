//! Minimal GGUF reader: the tensor table and scalar metadata of a memory-mapped file.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;

/// Tensor encodings of ggml, by their identifier in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    Bf16,
}

impl Encoding {
    fn from_id(id: u32) -> Result<Self> {
        Ok(match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            30 => Self::Bf16,
            other => bail!("unsupported tensor encoding {other}"),
        })
    }

    /// `(values per block, bytes per block)`.
    pub fn block(self) -> (usize, usize) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 | Self::Bf16 => (1, 2),
            Self::Q4_0 => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q2K => (256, 84),
            Self::Q3K => (256, 110),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
            Self::Q8K => (256, 292),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Tensor {
    /// Dimensions as stored: the first one is the innermost (the row width of a matrix).
    pub dims: Vec<usize>,
    pub encoding: Encoding,
    /// Position of the data in the file.
    pub offset: usize,
}

impl Tensor {
    pub fn values(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn bytes(&self) -> usize {
        let (values, bytes) = self.encoding.block();
        self.values() / values * bytes
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Scalar {
    Unsigned(u64),
    Signed(i64),
    Float(f64),
    Bool(bool),
    Text(String),
    /// A short array of scalars (per-layer head counts, the sliding-window pattern).
    List(Vec<Scalar>),
}

pub struct Gguf {
    map: Mmap,
    tensors: HashMap<String, Tensor>,
    metadata: HashMap<String, Scalar>,
    /// Long arrays of the tokenizer (`tokenizer.ggml.*`: the pieces, the merges, the token types), as where they start in the file.
    arrays: HashMap<String, Array>,
}

/// A long metadata array that is read on demand: its element type, its length and where its first element is.
#[derive(Clone, Copy)]
struct Array {
    kind: u32,
    count: usize,
    start: usize,
}

struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl Cursor<'_> {
    fn take(&mut self, length: usize) -> Result<&[u8]> {
        let end = self.position.checked_add(length).context("offset overflow")?;
        let slice = self.bytes.get(self.position..end).context("truncated GGUF header")?;
        self.position = end;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }

    fn text(&mut self) -> Result<String> {
        let length = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(length)?).into_owned())
    }

    /// Reads a metadata value; arrays are skipped (`None`).
    fn value(&mut self, kind: u32) -> Result<Option<Scalar>> {
        Ok(Some(match kind {
            0 => Scalar::Unsigned(self.take(1)?[0] as u64),
            1 => Scalar::Signed(self.take(1)?[0] as i8 as i64),
            2 => Scalar::Unsigned(u16::from_le_bytes(self.take(2)?.try_into()?) as u64),
            3 => Scalar::Signed(i16::from_le_bytes(self.take(2)?.try_into()?) as i64),
            4 => Scalar::Unsigned(self.u32()? as u64),
            5 => Scalar::Signed(self.u32()? as i32 as i64),
            6 => Scalar::Float(f32::from_le_bytes(self.take(4)?.try_into()?) as f64),
            7 => Scalar::Bool(self.take(1)?[0] != 0),
            8 => Scalar::Text(self.text()?),
            9 => {
                let inner = self.u32()?;
                let count = self.u64()? as usize;
                let fixed = match inner {
                    0 | 1 | 7 => Some(1),
                    2 | 3 => Some(2),
                    4 | 5 | 6 => Some(4),
                    10 | 11 | 12 => Some(8),
                    _ => None,
                };
                if inner != 8 && inner != 9 && count <= 4096 {
                    let mut items = Vec::with_capacity(count);
                    for _ in 0..count {
                        items.extend(self.value(inner)?);
                    }
                    return Ok(Some(Scalar::List(items)));
                }
                match fixed {
                    Some(size) => {
                        self.take(size * count)?;
                    }
                    None => {
                        for _ in 0..count {
                            self.value(inner)?;
                        }
                    }
                }
                return Ok(None);
            }
            10 => Scalar::Unsigned(self.u64()?),
            11 => Scalar::Signed(self.u64()? as i64),
            12 => Scalar::Float(f64::from_le_bytes(self.take(8)?.try_into()?)),
            other => bail!("unknown metadata type {other}"),
        }))
    }
}

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref()).with_context(|| format!("cannot open {}", path.as_ref().display()))?;
        // The file is only read; nothing in this process writes to it while it is mapped.
        let map = unsafe { Mmap::map(&file)? };
        let mut cursor = Cursor { bytes: &map, position: 0 };
        ensure!(cursor.take(4)? == b"GGUF", "not a GGUF file");
        let _version = cursor.u32()?;
        let tensor_count = cursor.u64()? as usize;
        let metadata_count = cursor.u64()? as usize;
        let mut metadata = HashMap::new();
        let mut arrays = HashMap::new();
        for _ in 0..metadata_count {
            let key = cursor.text()?;
            let kind = cursor.u32()?;
            if kind == 9 && key.starts_with("tokenizer.ggml.") {
                let mut peek = Cursor { bytes: cursor.bytes, position: cursor.position };
                let (inner, count) = (peek.u32()?, peek.u64()? as usize);
                arrays.insert(key.clone(), Array { kind: inner, count, start: peek.position });
            }
            if let Some(value) = cursor.value(kind)? {
                metadata.insert(key, value);
            }
        }
        let mut tensors = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = cursor.text()?;
            let rank = cursor.u32()? as usize;
            let dims = (0..rank).map(|_| cursor.u64().map(|dim| dim as usize)).collect::<Result<Vec<_>>>()?;
            let encoding = Encoding::from_id(cursor.u32()?)?;
            let offset = cursor.u64()? as usize;
            tensors.insert(name, Tensor { dims, encoding, offset });
        }
        let alignment = match metadata.get("general.alignment") {
            Some(Scalar::Unsigned(alignment)) => *alignment as usize,
            _ => 32,
        };
        let data = cursor.position.div_ceil(alignment) * alignment;
        for (name, tensor) in &mut tensors {
            tensor.offset += data;
            ensure!(tensor.offset + tensor.bytes() <= map.len(), "tensor {name} runs past the end of the file");
        }
        Ok(Self { map, tensors, metadata, arrays })
    }

    /// The whole mapped file.
    /// Tells the system the pages of `[offset, offset + len)` of the mapped file will be read soon (`keep`) or are not needed and may
    /// leave memory (they are read again from the file if touched). Only whole pages inside the range are affected.
    pub fn advise(&self, offset: usize, len: usize, keep: bool) {
        #[cfg(unix)]
        {
            let page = 16384usize;
            let start = offset.div_ceil(page) * page;
            let end = ((offset + len) / page * page).min(self.map.len());
            if end > start {
                if keep {
                    let _ = self.map.advise_range(memmap2::Advice::WillNeed, start, end - start);
                } else {
                    // A read-only private mapping of a file: dropped pages are read again from the file when touched.
                    let _ = unsafe { self.map.unchecked_advise_range(memmap2::UncheckedAdvice::DontNeed, start, end - start) };
                }
            }
        }
        #[cfg(not(unix))]
        let _ = (offset, len, keep);
    }

    pub fn mapped(&self) -> &[u8] {
        &self.map
    }

    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    pub fn tensors(&self) -> impl Iterator<Item = (&str, &Tensor)> {
        self.tensors.iter().map(|(name, tensor)| (name.as_str(), tensor))
    }

    pub fn metadata(&self, key: &str) -> Option<&Scalar> {
        self.metadata.get(key)
    }

    /// A long array of text (the pieces or the merges of a tokenizer).
    pub fn strings(&self, key: &str) -> Result<Vec<String>> {
        let array = self.arrays.get(key).with_context(|| format!("the model has no array `{key}`"))?;
        ensure!(array.kind == 8, "`{key}` is not an array of text");
        let mut cursor = Cursor { bytes: &self.map, position: array.start };
        (0..array.count).map(|_| cursor.text()).collect()
    }

    /// A long array of 8-bit or 32-bit integers (the token types of a tokenizer).
    pub fn integers(&self, key: &str) -> Result<Vec<i64>> {
        let array = self.arrays.get(key).with_context(|| format!("the model has no array `{key}`"))?;
        let mut cursor = Cursor { bytes: &self.map, position: array.start };
        match array.kind {
            0 | 1 => (0..array.count).map(|_| Ok(cursor.take(1)?[0] as i64)).collect(),
            4 | 5 => (0..array.count).map(|_| Ok(cursor.u32()? as i32 as i64)).collect(),
            other => bail!("`{key}` holds elements of type {other}, not integers"),
        }
    }

    /// A numeric metadata value as an unsigned integer.
    pub fn unsigned(&self, key: &str) -> Option<usize> {
        match self.metadata.get(key)? {
            Scalar::Unsigned(value) => Some(*value as usize),
            Scalar::Signed(value) if *value >= 0 => Some(*value as usize),
            _ => None,
        }
    }

    pub fn float(&self, key: &str) -> Option<f64> {
        match self.metadata.get(key)? {
            Scalar::Float(value) => Some(*value),
            Scalar::Unsigned(value) => Some(*value as f64),
            Scalar::Signed(value) => Some(*value as f64),
            _ => None,
        }
    }

    /// Element `index` of a per-layer array, or the scalar itself when the key holds one value for every layer.
    pub fn per_layer(&self, key: &str, index: usize) -> Option<Scalar> {
        match self.metadata.get(key)? {
            Scalar::List(items) => items.get(index).cloned(),
            other => Some(other.clone()),
        }
    }

    /// The stored bytes of a tensor.
    pub fn data(&self, tensor: &Tensor) -> &[u8] {
        &self.map[tensor.offset..tensor.offset + tensor.bytes()]
    }
}
