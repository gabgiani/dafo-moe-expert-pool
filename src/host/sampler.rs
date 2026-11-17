//! Temperature, top-k, top-p and repetition penalty over the logits, on the CPU; greedy at temperature 0.

#[derive(Clone, Debug)]
pub struct Sampling {
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub repeat_penalty: f32,
    /// How many of the latest tokens the penalty looks at (0: none).
    pub repeat_last_n: usize,
    pub seed: u64,
}

impl Default for Sampling {
    fn default() -> Self {
        Self { temperature: 0.7, top_k: 20, top_p: 0.8, repeat_penalty: 1.15, repeat_last_n: 64, seed: 299_792_458 }
    }
}

pub struct Sampler {
    settings: Sampling,
    state: u64,
}

impl Sampler {
    pub fn new(settings: Sampling) -> Self {
        let state = settings.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        Self { settings, state }
    }

    pub fn window(&self) -> usize {
        self.settings.repeat_last_n
    }

    fn random(&mut self) -> f64 {
        // xorshift64*
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        (self.state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn sample(&mut self, mut logits: Vec<f32>, recent: &[u32]) -> u32 {
        let penalty = self.settings.repeat_penalty;
        if penalty != 1.0 && penalty > 0.0 {
            // Each distinct token of the window is penalised once, as llama.cpp does: penalising every occurrence compounds.
            let mut seen = std::collections::HashSet::with_capacity(recent.len());
            for &token in recent {
                if seen.insert(token)
                    && let Some(logit) = logits.get_mut(token as usize)
                {
                    *logit = if *logit > 0.0 { *logit / penalty } else { *logit * penalty };
                }
            }
        }
        if self.settings.temperature <= 0.0 {
            return logits.iter().enumerate().fold((0u32, f32::NEG_INFINITY), |best, (index, &value)| if value > best.1 { (index as u32, value) } else { best }).0;
        }
        let mut candidates: Vec<(u32, f32)> = logits.iter().enumerate().map(|(index, &value)| (index as u32, value)).collect();
        let keep = if self.settings.top_k > 0 { self.settings.top_k.min(candidates.len()) } else { candidates.len().min(2048) };
        candidates.select_nth_unstable_by(keep - 1, |a, b| b.1.total_cmp(&a.1));
        candidates.truncate(keep);
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        let largest = candidates[0].1 as f64;
        let mut weights: Vec<f64> = candidates.iter().map(|(_, value)| ((*value as f64 - largest) / self.settings.temperature).exp()).collect();
        let total: f64 = weights.iter().sum();
        weights.iter_mut().for_each(|weight| *weight /= total);
        if self.settings.top_p > 0.0 && self.settings.top_p < 1.0 {
            let mut cumulative = 0.0;
            let mut cut = weights.len();
            for (index, weight) in weights.iter().enumerate() {
                cumulative += weight;
                if cumulative >= self.settings.top_p {
                    cut = index + 1;
                    break;
                }
            }
            weights.truncate(cut);
            candidates.truncate(cut);
            let total: f64 = weights.iter().sum();
            weights.iter_mut().for_each(|weight| *weight /= total);
        }
        let mut draw = self.random();
        for (candidate, weight) in candidates.iter().zip(&weights) {
            draw -= weight;
            if draw <= 0.0 {
                return candidate.0;
            }
        }
        candidates[0].0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_takes_the_largest_and_a_repeated_token_is_penalised_once() {
        let mut sampler = Sampler::new(Sampling { temperature: 0.0, repeat_penalty: 2.0, ..Sampling::default() });
        assert_eq!(sampler.sample(vec![1.0, 3.0, 2.0], &[]), 1);
        // Token 1 seen five times: once penalised it is 3.0 / 2 = 1.5, still above 1.4; compounding would drop it to 0.09.
        assert_eq!(sampler.sample(vec![1.0, 3.0, 1.4], &[1, 1, 1, 1, 1]), 1);
    }
}
