//! Thin layer over Metal: a device with its queue, compiled kernels, buffers and timed command buffers.

use std::ffi::c_void;
use std::ptr::NonNull;

use anyhow::{Context, Result, anyhow};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLDispatchType, MTLLibrary, MTLResourceOptions, MTLSize,
};

pub struct Gpu {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

pub struct Library(Retained<ProtocolObject<dyn MTLLibrary>>);

pub struct Kernel(Retained<ProtocolObject<dyn MTLComputePipelineState>>);

/// A buffer the CPU and the GPU share (unified memory: no copy between them).
pub struct Buffer(Retained<ProtocolObject<dyn MTLBuffer>>);

pub struct Encoder<'a>(&'a ProtocolObject<dyn MTLComputeCommandEncoder>);

/// Seconds a command buffer took: on the GPU, and from commit until it completed.
#[derive(Clone, Copy, Debug)]
pub struct Timing {
    pub gpu: f64,
    pub wall: f64,
}

impl Gpu {
    pub fn new() -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice().context("no Metal device")?;
        let queue = device.newCommandQueue().context("cannot create a command queue")?;
        Ok(Self { device, queue })
    }

    pub fn name(&self) -> String {
        self.device.name().to_string()
    }

    pub fn library(&self, source: &str) -> Result<Library> {
        self.device
            .newLibraryWithSource_options_error(&NSString::from_str(source), None)
            .map(Library)
            .map_err(|error| anyhow!("kernel source does not compile: {error}"))
    }

    pub fn kernel(&self, library: &Library, name: &str) -> Result<Kernel> {
        let function = library.0.newFunctionWithName(&NSString::from_str(name)).with_context(|| format!("no kernel named {name}"))?;
        self.device
            .newComputePipelineStateWithFunction_error(&function)
            .map(Kernel)
            .map_err(|error| anyhow!("cannot build the pipeline of {name}: {error}"))
    }

    pub fn buffer(&self, bytes: usize) -> Result<Buffer> {
        self.device
            .newBufferWithLength_options(bytes.max(1), MTLResourceOptions::StorageModeShared)
            .map(Buffer)
            .with_context(|| format!("cannot allocate a buffer of {bytes} bytes"))
    }

    /// A buffer the host reads and writes too (on Metal every buffer is: memory is shared).
    pub fn buffer_shared(&self, bytes: usize) -> Result<Buffer> {
        self.buffer(bytes)
    }

    pub fn buffer_from<T: Copy>(&self, values: &[T]) -> Result<Buffer> {
        let mut buffer = self.buffer(size_of_val(values))?;
        buffer.as_mut_slice::<T>()[..values.len()].copy_from_slice(values);
        Ok(buffer)
    }

    /// A buffer over memory the caller owns (a mapped file): no copy. `pointer` and `bytes` must be multiples of the page size,
    /// and the memory must outlive the buffer.
    ///
    /// # Safety
    /// The memory behind `pointer` must stay mapped and unchanged while the buffer is in use.
    pub unsafe fn buffer_no_copy(&self, pointer: *const u8, bytes: usize) -> Result<Buffer> {
        let pointer = NonNull::new(pointer as *mut c_void).context("null pointer")?;
        unsafe {
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(pointer, bytes, MTLResourceOptions::StorageModeShared, None)
                .map(Buffer)
                .with_context(|| format!("cannot wrap {bytes} bytes of mapped memory in a buffer"))
        }
    }

    /// Largest buffer the device can create.
    pub fn max_buffer_bytes(&self) -> usize {
        self.device.maxBufferLength()
    }

    /// Encodes one command buffer, runs it and waits for it.
    pub fn run(&self, encode: impl FnOnce(&Encoder)) -> Result<Timing> {
        self.run_with(false, encode)
    }

    /// Like [`Gpu::run`] on an encoder whose dispatches may overlap: the caller puts a [`Encoder::barrier`] wherever one needs what
    /// an earlier one wrote.
    pub fn run_concurrent(&self, encode: impl FnOnce(&Encoder)) -> Result<Timing> {
        self.run_with(true, encode)
    }

    fn run_with(&self, concurrent: bool, encode: impl FnOnce(&Encoder)) -> Result<Timing> {
        let commands = self.queue.commandBuffer().context("cannot create a command buffer")?;
        let encoder = if concurrent {
            commands.computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
        } else {
            commands.computeCommandEncoder()
        }
        .context("cannot create an encoder")?;
        encode(&Encoder(&encoder));
        encoder.endEncoding();
        let started = std::time::Instant::now();
        commands.commit();
        commands.waitUntilCompleted();
        let wall = started.elapsed().as_secs_f64();
        if let Some(error) = commands.error() {
            return Err(anyhow!("the command buffer failed: {error}"));
        }
        Ok(Timing { gpu: commands.GPUEndTime() - commands.GPUStartTime(), wall })
    }
}

impl Buffer {
    pub fn bytes(&self) -> usize {
        self.0.length()
    }

    pub fn as_slice<T: Copy>(&self) -> &[T] {
        // The buffer is shared storage owned by `self`; its contents live as long as it does.
        unsafe { std::slice::from_raw_parts(self.0.contents().as_ptr().cast::<T>(), self.bytes() / size_of::<T>()) }
    }

    /// A copy of the contents (the GPU must be idle).
    pub fn to_vec<T: Copy>(&self) -> Vec<T> {
        self.as_slice::<T>().to_vec()
    }

    /// A copy of `length` bytes from `offset` (the GPU must be idle).
    pub fn read_bytes(&self, offset: usize, length: usize) -> Vec<u8> {
        self.as_slice::<u8>()[offset..offset + length].to_vec()
    }

    /// Writes `data` at `offset` bytes into the buffer (the GPU must be idle).
    pub fn write_bytes(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        anyhow::ensure!(offset + data.len() <= self.bytes(), "a write of {} bytes at {offset} does not fit a buffer of {}", data.len(), self.bytes());
        self.as_mut_slice::<u8>()[offset..offset + data.len()].copy_from_slice(data);
        Ok(())
    }

    pub fn as_mut_slice<T: Copy>(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.0.contents().as_ptr().cast::<T>(), self.bytes() / size_of::<T>()) }
    }
}

impl Encoder<'_> {
    /// Everything dispatched before this point has finished writing its buffers when what comes after starts.
    pub fn barrier(&self) {
        self.0.memoryBarrierWithScope(MTLBarrierScope::Buffers);
    }

    pub fn kernel(&self, kernel: &Kernel) {
        self.0.setComputePipelineState(&kernel.0);
    }

    pub fn buffer(&self, index: usize, buffer: &Buffer, offset: usize) {
        // The buffer outlives the command buffer: `Gpu::run` waits for completion before returning.
        unsafe { self.0.setBuffer_offset_atIndex(Some(&buffer.0), offset, index) }
    }

    /// A small value passed by copy (a `constant T &` argument of the kernel).
    pub fn value<T: Copy>(&self, index: usize, value: &T) {
        let pointer = NonNull::from(value).cast::<c_void>();
        unsafe { self.0.setBytes_length_atIndex(pointer, size_of::<T>(), index) }
    }

    /// Threadgroup memory of `bytes` for the kernel argument at `index`.
    pub fn shared_memory(&self, index: usize, bytes: usize) {
        unsafe { self.0.setThreadgroupMemoryLength_atIndex(bytes, index) }
    }

    pub fn dispatch(&self, groups: [usize; 3], threads: [usize; 3]) {
        let size = |[width, height, depth]: [usize; 3]| MTLSize { width, height, depth };
        self.0.dispatchThreadgroups_threadsPerThreadgroup(size(groups), size(threads));
    }
}
