//! Reference decoding of the ggml encodings on the CPU: what the kernels must reproduce.

use anyhow::{Result, bail, ensure};

use crate::gguf::Encoding;

pub fn half_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let mantissa = (bits & 0x3ff) as f32;
    sign * match exponent {
        0 => mantissa * 2f32.powi(-24),
        31 if mantissa == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15),
    }
}

fn half_at(bytes: &[u8], offset: usize) -> f32 {
    half_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
}

/// Decodes whole blocks of `bytes` into `values`.
pub fn dequantize(encoding: Encoding, bytes: &[u8], values: &mut [f32]) -> Result<()> {
    let (per_block, block_bytes) = encoding.block();
    ensure!(bytes.len() % block_bytes == 0, "the data is not a whole number of blocks");
    ensure!(values.len() == bytes.len() / block_bytes * per_block, "the output does not match the data");
    for (block, values) in bytes.chunks_exact(block_bytes).zip(values.chunks_exact_mut(per_block)) {
        match encoding {
            Encoding::F32 => values[0] = f32::from_le_bytes(block.try_into()?),
            Encoding::F16 => values[0] = half_at(block, 0),
            Encoding::Q8_0 => {
                let scale = half_at(block, 0);
                for (value, &quant) in values.iter_mut().zip(&block[2..]) {
                    *value = quant as i8 as f32 * scale;
                }
            }
            Encoding::Q5_1 => {
                let (scale, minimum) = (half_at(block, 0), half_at(block, 2));
                let high = u32::from_le_bytes(block[4..8].try_into()?);
                let quants = &block[8..];
                for index in 0..16 {
                    let low = (quants[index] & 0x0f) as u32 | (((high >> index) << 4) & 0x10);
                    let upper = (quants[index] >> 4) as u32 | ((high >> (index + 12)) & 0x10);
                    values[index] = low as f32 * scale + minimum;
                    values[index + 16] = upper as f32 * scale + minimum;
                }
            }
            Encoding::Q4K => {
                let (scale, minimum) = (half_at(block, 0), half_at(block, 2));
                let scales = &block[4..16];
                let quants = &block[16..];
                // Eight groups of 32 values, each with a 6-bit scale and a 6-bit minimum.
                let group = |index: usize| -> (f32, f32) {
                    let (factor, offset) = if index < 4 {
                        (scales[index] & 63, scales[index + 4] & 63)
                    } else {
                        (
                            (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4),
                            (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
                        )
                    };
                    (scale * factor as f32, minimum * offset as f32)
                };
                for pair in 0..4 {
                    let (first_scale, first_minimum) = group(pair * 2);
                    let (second_scale, second_minimum) = group(pair * 2 + 1);
                    for index in 0..32 {
                        let quant = quants[pair * 32 + index];
                        values[pair * 64 + index] = first_scale * (quant & 0x0f) as f32 - first_minimum;
                        values[pair * 64 + 32 + index] = second_scale * (quant >> 4) as f32 - second_minimum;
                    }
                }
            }
            other => bail!("no reference decoding for {other:?}"),
        }
    }
    Ok(())
}
