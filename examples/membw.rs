//! Memory bandwidth of this machine: how fast the CPU can read a large buffer with 1, 2, 4 and 8 threads (a sum of every 64-byte line,
//! so the memory system is the limit), and copy it. What a CPU computing mixture-of-experts weights from RAM can reach.
use std::time::Instant;

fn main() {
    let megabytes: usize = std::env::args().nth(1).and_then(|value| value.parse().ok()).unwrap_or(1024);
    let words = megabytes * (1 << 20) / 8;
    let data: Vec<u64> = (0..words as u64).collect();
    println!("buffer of {megabytes} MiB; reads in GB/s (best of 5):");
    for threads in [1usize, 2, 4, 8] {
        let chunk = words / threads;
        let mut best = 0f64;
        for _ in 0..5 {
            let started = Instant::now();
            let total: u64 = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..threads)
                    .map(|index| {
                        let slice = &data[index * chunk..(index + 1) * chunk];
                        scope.spawn(move || {
                            let mut sum = 0u64;
                            for line in slice.chunks_exact(8) {
                                sum = sum.wrapping_add(line[0]).wrapping_add(line[1]).wrapping_add(line[2]).wrapping_add(line[3]).wrapping_add(line[4]).wrapping_add(line[5]).wrapping_add(line[6]).wrapping_add(line[7]);
                            }
                            sum
                        })
                    })
                    .collect();
                handles.into_iter().map(|handle| handle.join().unwrap()).fold(0u64, u64::wrapping_add)
            });
            std::hint::black_box(total);
            best = best.max((chunk * threads * 8) as f64 / started.elapsed().as_secs_f64() / 1e9);
        }
        println!("  {threads} thread(s): {best:.1} GB/s");
    }
    let mut copy = vec![0u64; words];
    let started = Instant::now();
    for _ in 0..3 {
        copy.copy_from_slice(&data);
        std::hint::black_box(&copy);
    }
    println!("  copy (1 thread, reads + writes counted once): {:.1} GB/s", (3 * words * 8) as f64 / started.elapsed().as_secs_f64() / 1e9);
}
