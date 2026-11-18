//! What this machine can do with a model, decided without flags: how many experts to keep in memory, how long a context, how many
//! conversations at once, whether images fit. `dafo-compiler serve|chat` and `dafo-serve` use it for everything the user did not set,
//! and print what they chose and why. The rules are the ones measured in `docs/`: weights may take about 46 % of a Mac's memory (what a
//! 16 GB Mac runs a 7.4 GiB model with); on a discrete GPU the experts that fit in video memory go there, a share of the free RAM
//! holds the next ones, and the rest stay on disk.

use std::path::Path;

use crate::gguf::Gguf;

const GIB: u64 = 1 << 30;

/// What the machine offers.
#[derive(Clone, Debug)]
pub struct Machine {
    pub gpu: String,
    pub ram: u64,
    pub ram_available: u64,
    /// `(free, total)` video memory of a discrete GPU (CUDA); `None` where the GPU shares the system memory (Apple silicon).
    pub video: Option<(u64, u64)>,
    /// Memory the GPU may use on a unified-memory machine.
    pub unified_gpu_budget: u64,
    pub disk_free: u64,
}

/// What a model needs.
#[derive(Clone, Debug)]
pub struct ModelSize {
    pub file_bytes: u64,
    /// Bytes of the expert tensors of the whole model, and the experts per layer (0 for a dense model).
    pub expert_bytes: u64,
    pub experts: usize,
    pub layers: usize,
    /// Keys and values of one conversation: a fixed part and bytes per token of context.
    pub cache_fixed: usize,
    pub cache_per_token: usize,
}

impl ModelSize {
    pub fn of(gguf: &Gguf, file_bytes: u64, cache: (usize, usize)) -> Self {
        let experts = gguf.tensor("blk.0.ffn_gate_inp.weight").map_or(0, |router| router.dims[1] as usize);
        let expert_bytes = gguf.tensors().filter(|(name, _)| name.contains("ffn_gate_up_exps") || name.contains("ffn_down_exps")).map(|(_, tensor)| tensor.bytes() as u64).sum();
        let layers = (0..).take_while(|layer| gguf.tensor(&format!("blk.{layer}.attn_q.weight")).is_some()).count();
        Self { file_bytes, expert_bytes, experts, layers, cache_fixed: cache.0, cache_per_token: cache.1 }
    }

    /// Bytes of one expert of every layer.
    pub fn expert_across_layers(&self) -> u64 {
        self.expert_bytes / self.experts.max(1) as u64
    }

    pub fn dense_bytes(&self) -> u64 {
        self.file_bytes.saturating_sub(self.expert_bytes)
    }
}

/// What to do with a model on a machine.
#[derive(Clone, Debug)]
pub struct Plan {
    /// Experts per layer kept resident (`None`: the whole model is mapped).
    pub pool: Option<usize>,
    /// Of those, how many in video memory (discrete GPUs with a pool).
    pub device_experts: Option<usize>,
    pub context: usize,
    pub slots: usize,
    pub vision: bool,
    pub visual_tokens: usize,
    /// One line on the machine and what was chosen, and notes on what to expect.
    pub summary: String,
    pub notes: Vec<String>,
}

/// The machine this program runs on. On a discrete NVIDIA GPU this opens the device for a moment to ask its free memory.
pub fn machine(models_folder: &Path) -> Machine {
    let system = sysinfo::System::new_with_specifics(sysinfo::RefreshKind::nothing().with_memory(sysinfo::MemoryRefreshKind::nothing().with_ram()));
    let ram = system.total_memory();
    let ram_available = system.available_memory();
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let target = models_folder.ancestors().find(|path| path.exists()).unwrap_or(models_folder).canonicalize().unwrap_or_else(|_| models_folder.to_path_buf());
    let disk_free = disks.list().iter().filter(|disk| target.starts_with(disk.mount_point())).max_by_key(|disk| disk.mount_point().as_os_str().len()).map_or(0, |disk| disk.available_space());
    #[cfg(target_os = "macos")]
    {
        match crate::metal::Gpu::new() {
            Ok(gpu) => Machine { gpu: gpu.name(), ram, ram_available, video: None, unified_gpu_budget: gpu.working_set_bytes(), disk_free },
            Err(_) => Machine { gpu: "no GPU".into(), ram, ram_available, video: None, unified_gpu_budget: 0, disk_free },
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        match crate::cuda::Gpu::new() {
            Ok(gpu) => {
                let (free, total) = gpu.memory();
                Machine { gpu: gpu.name(), ram, ram_available, video: Some((free as u64, total as u64)), unified_gpu_budget: 0, disk_free }
            }
            Err(_) => Machine { gpu: "no usable GPU".into(), ram, ram_available, video: Some((0, 0)), unified_gpu_budget: 0, disk_free },
        }
    }
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / GIB as f64
}

/// The plan for `model` on `machine`; `projector` says whether the model has a vision projector beside it.
pub fn plan(machine: &Machine, model: &ModelSize, projector: bool) -> Plan {
    let mut notes = Vec::new();
    let expert_all = model.expert_across_layers().max(1);
    let dense = model.dense_bytes();
    let slot_cost = |context: usize| (model.cache_fixed + model.cache_per_token * context) as u64;
    let (pool, device_experts, context, slots, vision, visual_tokens);
    match machine.video {
        // Apple silicon: one memory for everything.
        None => {
            let experts = model.experts;
            let size = if experts > 0 { crate::pool::automatic_pool_size(machine.ram, dense, expert_all, experts) } else { 0 };
            pool = (experts > 0 && size < experts).then_some(size);
            let resident = if pool.is_some() { dense + size as u64 * expert_all } else { model.file_bytes };
            let vision_bytes = if projector { GIB * 14 / 10 } else { 0 };
            let kv_budget = (machine.ram as f64 * 0.80) as u64;
            let kv_budget = kv_budget.saturating_sub(resident + vision_bytes + GIB * 3 / 2);
            let mut wanted = if machine.ram >= 48 * GIB { 32768 } else { 16384 };
            let mut count = (kv_budget / slot_cost(wanted).max(1)).clamp(0, 4) as usize;
            if count == 0 {
                count = 1;
                wanted = ((kv_budget.saturating_sub(model.cache_fixed as u64)) / model.cache_per_token.max(1) as u64).clamp(4096, 16384) as usize;
                notes.push(format!("little memory is left for the conversation: the context is {wanted} tokens"));
            }
            context = wanted;
            slots = count;
            device_experts = None;
            vision = projector;
            visual_tokens = 560;
            if resident > (machine.ram as f64 * 0.60) as u64 {
                notes.push("this model is large for this machine's memory: close other applications".into());
            }
        }
        // A discrete GPU: video memory, then pinned RAM, then disk.
        Some((free, _)) => {
            let margin = GIB / 2;
            let big = free >= 12 * GIB;
            context = if big { 16384 } else { 8192 };
            slots = if big { 3 } else { 2 };
            let kv = slot_cost(context) * slots as u64;
            let mut with_vision = projector;
            let vision_bytes = |tokens: usize| GIB * 12 / 10 + if tokens > 280 { GIB * 45 / 100 } else { GIB / 4 };
            let tokens = if free >= 10 * GIB { 560 } else { 280 };
            let mut left = free as i64 - margin as i64 - dense as i64 - kv as i64 - if with_vision { vision_bytes(tokens) as i64 } else { 0 };
            if left < 0 && with_vision {
                with_vision = false;
                left = free as i64 - margin as i64 - dense as i64 - kv as i64;
                notes.push("images are off: there is not enough video memory for the vision encoder next to the model".into());
            }
            vision = with_vision;
            visual_tokens = tokens;
            if model.experts == 0 {
                pool = None;
                device_experts = None;
                if left < 0 {
                    notes.push("this model does not fit in this GPU's memory".into());
                }
            } else {
                let in_video = (left.max(0) as u64 / expert_all) as usize;
                if in_video >= model.experts {
                    pool = None;
                    device_experts = None;
                } else {
                    let pinned = ((machine.ram_available as f64 * 0.30) as u64).min(6 * GIB);
                    let in_ram = ((pinned / expert_all) as usize).min(model.experts - in_video);
                    let total = in_video + in_ram;
                    pool = Some(total.max(8).min(model.experts));
                    device_experts = Some(in_video);
                    if total < model.experts / 3 {
                        notes.push(format!("only {total} of {} experts per layer fit in video memory and RAM: the others are read from disk when needed", model.experts));
                    }
                    if in_video * 2 < model.experts {
                        notes.push("most experts are outside video memory and are read over PCIe: expect slow generation".into());
                    }
                }
            }
        }
    }
    let summary = {
        let place = match (pool, device_experts) {
            (Some(total), Some(video)) => format!("{video} experts per layer in video memory and {} in RAM, {} reachable on disk", total - video, model.experts - total),
            (Some(total), None) => format!("{total} of {} experts per layer in memory, the rest reachable on disk", model.experts),
            (None, _) => "the whole model in memory".to_owned(),
        };
        let memory = match machine.video {
            Some((free, total)) => format!("{} with {:.0} of {:.0} GB of video memory free, {:.0} GB of RAM", machine.gpu, gib(free), gib(total), gib(machine.ram)),
            None => format!("{}, {:.0} GB of memory", machine.gpu, gib(machine.ram)),
        };
        format!("{memory}: {place}; context {context} tokens; up to {slots} request{} at once; images {}", if slots == 1 { "" } else { "s" }, if vision { "on" } else { "off" })
    };
    Plan { pool, device_experts, context, slots, vision, visual_tokens, summary, notes }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gemma_26b() -> ModelSize {
        // Gemma 4 26B-A4B complete: 15.8 GiB, experts 13.4 GiB, 128 per layer, 30 layers; caches 325 MiB + 20 KiB per token.
        ModelSize { file_bytes: 16_947_541_728, expert_bytes: 14_400_000_000, experts: 128, layers: 30, cache_fixed: 325 << 20, cache_per_token: 20 << 10 }
    }

    fn mac(ram_gib: u64) -> Machine {
        Machine { gpu: "Apple M".into(), ram: ram_gib * GIB, ram_available: ram_gib * GIB / 2, video: None, unified_gpu_budget: ram_gib * GIB * 7 / 10, disk_free: 100 * GIB }
    }

    #[test]
    fn a_16_gb_mac_keeps_a_pool_and_a_few_conversations() {
        let plan = plan(&mac(16), &gemma_26b(), true);
        assert!(plan.pool.is_some_and(|size| (40..=56).contains(&size)), "{:?}", plan.pool);
        assert_eq!(plan.device_experts, None);
        assert!(plan.vision && plan.slots >= 1 && plan.context >= 8192);
    }

    #[test]
    fn a_big_mac_keeps_the_whole_model() {
        let plan = plan(&mac(96), &gemma_26b(), true);
        assert_eq!(plan.pool, None);
        assert_eq!(plan.slots, 4);
    }

    #[test]
    fn a_small_nvidia_card_splits_the_experts_between_video_memory_and_ram() {
        let machine = Machine { gpu: "RTX".into(), ram: 24 * GIB, ram_available: 12 * GIB, video: Some((5 * GIB, 6 * GIB)), unified_gpu_budget: 0, disk_free: 100 * GIB };
        let plan = plan(&machine, &gemma_26b(), true);
        assert!(plan.device_experts.is_some() && plan.pool.is_some());
        assert!(plan.notes.iter().any(|note| note.contains("slow") || note.contains("off")), "{:?}", plan.notes);
    }

    #[test]
    fn a_big_card_holds_everything() {
        let machine = Machine { gpu: "RTX".into(), ram: 64 * GIB, ram_available: 40 * GIB, video: Some((22 * GIB, 24 * GIB)), unified_gpu_budget: 0, disk_free: 100 * GIB };
        let plan = plan(&machine, &gemma_26b(), true);
        assert_eq!(plan.pool, None);
        assert!(plan.vision);
    }
}

/// The experts a Gemma 4 26B-A4B starts from (the ones real business-task work used most, hottest first, calibrated on a trace of the full
/// model): the first `size` of each layer. `None` for a model it was not measured on.
pub fn default_ranking(model: &ModelSize, size: usize) -> Option<Vec<(usize, Vec<usize>)>> {
    if model.experts != 128 || model.layers != 30 {
        return None;
    }
    let value: serde_json::Value = serde_json::from_str(include_str!("../expert-ranking-gemma4-26b-a4b.json")).ok()?;
    value
        .as_object()?
        .iter()
        .map(|(layer, ids)| Some((layer.parse().ok()?, ids.as_array()?.iter().take(size).map(|id| id.as_u64().map(|id| id as usize)).collect::<Option<Vec<_>>>()?)))
        .collect()
}
