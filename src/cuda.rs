//! Thin layer over CUDA for the NVIDIA backend: a device with its context and stream, kernels compiled at run time, buffers, and timed
//! launches. It has the shape of `metal.rs` (the runtime is written against both). Nothing here is linked at build time: the driver
//! (`nvcuda.dll`, `libcuda.so.1`) and the run-time compiler (`nvrtc64_*.dll`, `libnvrtc.so`) are loaded when the first device is made, so a
//! machine without NVIDIA software can still build the crate, and one with only the driver and the compiler DLL can run a model.
//!
//! Differences from Metal that the runtime knows about: device memory is not shared with the CPU, so a [`Buffer`] made by [`Gpu::buffer`]
//! is read and written with [`Buffer::to_vec`] and [`Buffer::write_bytes`]; a buffer the host also touches (the state of the runtime, the
//! demand counters) is made with [`Gpu::buffer_shared`], pinned host memory the device reads and writes over PCIe, and is used through
//! `as_slice`/`as_mut_slice` as on Metal. Kernels launched on one stream run in order, so there are no barriers.

use std::cell::{Cell, RefCell};
use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use libloading::Library as DynamicLibrary;

type CuResult = c_int;
type CuDevicePtr = u64;
type Handle = *mut c_void;

macro_rules! api {
    ($($name:ident : fn($($arg:ty),*) -> $ret:ty = $symbol:literal;)*) => {
        struct Driver {
            _library: DynamicLibrary,
            $($name: unsafe extern "C" fn($($arg),*) -> $ret,)*
        }
        impl Driver {
            fn load(library: DynamicLibrary) -> Result<Self> {
                unsafe {
                    Ok(Self {
                        $($name: *library.get::<unsafe extern "C" fn($($arg),*) -> $ret>($symbol).with_context(|| format!("the CUDA driver has no {}", String::from_utf8_lossy($symbol)))?,)*
                        _library: library,
                    })
                }
            }
        }
    };
}

api! {
    cu_init: fn(c_uint) -> CuResult = b"cuInit\0";
    cu_device_get: fn(*mut c_int, c_int) -> CuResult = b"cuDeviceGet\0";
    cu_device_get_name: fn(*mut c_char, c_int, c_int) -> CuResult = b"cuDeviceGetName\0";
    cu_device_get_attribute: fn(*mut c_int, c_int, c_int) -> CuResult = b"cuDeviceGetAttribute\0";
    cu_device_total_mem: fn(*mut usize, c_int) -> CuResult = b"cuDeviceTotalMem_v2\0";
    cu_ctx_create: fn(*mut Handle, c_uint, c_int) -> CuResult = b"cuCtxCreate_v2\0";
    cu_ctx_set_current: fn(Handle) -> CuResult = b"cuCtxSetCurrent\0";
    cu_stream_create: fn(*mut Handle, c_uint) -> CuResult = b"cuStreamCreate\0";
    cu_stream_synchronize: fn(Handle) -> CuResult = b"cuStreamSynchronize\0";
    cu_module_load_data: fn(*mut Handle, *const c_void) -> CuResult = b"cuModuleLoadData\0";
    cu_module_get_function: fn(*mut Handle, Handle, *const c_char) -> CuResult = b"cuModuleGetFunction\0";
    cu_mem_alloc: fn(*mut CuDevicePtr, usize) -> CuResult = b"cuMemAlloc_v2\0";
    cu_mem_free: fn(CuDevicePtr) -> CuResult = b"cuMemFree_v2\0";
    cu_memcpy_htod: fn(CuDevicePtr, *const c_void, usize) -> CuResult = b"cuMemcpyHtoD_v2\0";
    cu_memcpy_dtoh: fn(*mut c_void, CuDevicePtr, usize) -> CuResult = b"cuMemcpyDtoH_v2\0";
    cu_memset_d8: fn(CuDevicePtr, u8, usize) -> CuResult = b"cuMemsetD8_v2\0";
    cu_mem_host_alloc: fn(*mut *mut c_void, usize, c_uint) -> CuResult = b"cuMemHostAlloc\0";
    cu_mem_free_host: fn(*mut c_void) -> CuResult = b"cuMemFreeHost\0";
    cu_mem_host_get_device_pointer: fn(*mut CuDevicePtr, *mut c_void, c_uint) -> CuResult = b"cuMemHostGetDevicePointer_v2\0";
    cu_mem_get_info: fn(*mut usize, *mut usize) -> CuResult = b"cuMemGetInfo_v2\0";
    cu_launch_kernel: fn(Handle, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, Handle, *mut *mut c_void, *mut *mut c_void) -> CuResult = b"cuLaunchKernel\0";
    cu_func_set_attribute: fn(Handle, c_int, c_int) -> CuResult = b"cuFuncSetAttribute\0";
    cu_event_create: fn(*mut Handle, c_uint) -> CuResult = b"cuEventCreate\0";
    cu_event_record: fn(Handle, Handle) -> CuResult = b"cuEventRecord\0";
    cu_event_synchronize: fn(Handle) -> CuResult = b"cuEventSynchronize\0";
    cu_event_elapsed_time: fn(*mut f32, Handle, Handle) -> CuResult = b"cuEventElapsedTime\0";
    cu_get_error_string: fn(CuResult, *mut *const c_char) -> CuResult = b"cuGetErrorString\0";
}

struct Nvrtc {
    _library: DynamicLibrary,
    create_program: unsafe extern "C" fn(*mut Handle, *const c_char, *const c_char, c_int, *const *const c_char, *const *const c_char) -> c_int,
    compile_program: unsafe extern "C" fn(Handle, c_int, *const *const c_char) -> c_int,
    get_log_size: unsafe extern "C" fn(Handle, *mut usize) -> c_int,
    get_log: unsafe extern "C" fn(Handle, *mut c_char) -> c_int,
    get_cubin_size: unsafe extern "C" fn(Handle, *mut usize) -> c_int,
    get_cubin: unsafe extern "C" fn(Handle, *mut c_char) -> c_int,
    destroy_program: unsafe extern "C" fn(*mut Handle) -> c_int,
}

impl Nvrtc {
    fn load(library: DynamicLibrary) -> Result<Self> {
        unsafe {
            macro_rules! symbol {
                ($name:literal) => {
                    *library.get($name).with_context(|| format!("NVRTC has no {}", String::from_utf8_lossy($name)))?
                };
            }
            Ok(Self {
                create_program: symbol!(b"nvrtcCreateProgram\0"),
                compile_program: symbol!(b"nvrtcCompileProgram\0"),
                get_log_size: symbol!(b"nvrtcGetProgramLogSize\0"),
                get_log: symbol!(b"nvrtcGetProgramLog\0"),
                get_cubin_size: symbol!(b"nvrtcGetCUBINSize\0"),
                get_cubin: symbol!(b"nvrtcGetCUBIN\0"),
                destroy_program: symbol!(b"nvrtcDestroyProgram\0"),
                _library: library,
            })
        }
    }
}

/// Where the NVRTC library may be: the environment, the `bin` of every installed toolkit, then the usual search path.
fn nvrtc_candidates() -> Vec<PathBuf> {
    let mut found = Vec::new();
    if let Ok(path) = std::env::var("DAFO_NVRTC") {
        found.push(PathBuf::from(path));
    }
    let mut directories: Vec<PathBuf> = Vec::new();
    if let Ok(path) = std::env::var("CUDA_PATH") {
        directories.push(PathBuf::from(&path).join("bin"));
        directories.push(PathBuf::from(path).join("lib64"));
    }
    if cfg!(windows) {
        if let Ok(entries) = std::fs::read_dir(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA") {
            let mut versions: Vec<PathBuf> = entries.filter_map(|entry| entry.ok().map(|entry| entry.path())).collect();
            versions.sort();
            versions.reverse();
            directories.extend(versions.into_iter().map(|version| version.join("bin")));
        }
        // An application can ship the library beside its executable.
        if let Ok(executable) = std::env::current_exe() {
            if let Some(parent) = executable.parent() {
                directories.insert(0, parent.to_path_buf());
            }
        }
    } else {
        directories.extend(["/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib64"].map(PathBuf::from));
    }
    for directory in directories {
        let Ok(entries) = std::fs::read_dir(&directory) else { continue };
        let mut names: Vec<String> = entries
            .filter_map(|entry| entry.ok().map(|entry| entry.file_name().to_string_lossy().into_owned()))
            .filter(|name| {
                let lower = name.to_lowercase();
                (lower.starts_with("nvrtc64_") && lower.ends_with(".dll") && !lower.contains("builtins") && !lower.contains(".alt"))
                    || (lower.starts_with("libnvrtc.so") && !lower.contains("builtins"))
            })
            .collect();
        names.sort();
        names.reverse();
        found.extend(names.into_iter().map(|name| directory.join(name)));
    }
    found.push(PathBuf::from(if cfg!(windows) { "nvrtc64_120_0.dll" } else { "libnvrtc.so" }));
    found
}

struct Api {
    driver: Driver,
    nvrtc: Nvrtc,
}

fn check(api: &Api, code: CuResult, what: &str) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    let mut text: *const c_char = std::ptr::null();
    unsafe { (api.driver.cu_get_error_string)(code, &mut text) };
    let message = if text.is_null() { String::new() } else { unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned() };
    Err(anyhow!("CUDA {what} failed with error {code} {message}"))
}

pub struct Gpu {
    api: Arc<Api>,
    context: Handle,
    stream: Handle,
    name: String,
    architecture: (c_int, c_int),
    total_memory: usize,
}

pub struct Library(Handle);

pub struct Kernel {
    function: Handle,
    name: &'static str,
}

/// Memory on the device, or (made by [`Gpu::buffer_shared`]) pinned host memory both sides use.
pub struct Buffer {
    api: Arc<Api>,
    device: CuDevicePtr,
    host: *mut c_void,
    bytes: usize,
}

unsafe impl Send for Buffer {}

pub struct Encoder<'a> {
    gpu: &'a Gpu,
    function: Cell<Handle>,
    parameters: RefCell<Vec<Vec<u8>>>,
    shared: Cell<usize>,
    error: RefCell<Option<anyhow::Error>>,
}

/// Seconds a run took: on the GPU, and from submission until it completed.
#[derive(Clone, Copy, Debug)]
pub struct Timing {
    pub gpu: f64,
    pub wall: f64,
}

impl Gpu {
    pub fn new() -> Result<Self> {
        let driver_names: &[&str] = if cfg!(windows) { &["nvcuda.dll"] } else { &["libcuda.so.1", "libcuda.so"] };
        let driver_library = driver_names
            .iter()
            .find_map(|name| unsafe { DynamicLibrary::new(name).ok() })
            .context("cannot load the NVIDIA driver (nvcuda.dll / libcuda.so): is an NVIDIA GPU with its driver installed?")?;
        let driver = Driver::load(driver_library)?;
        let nvrtc_library = nvrtc_candidates()
            .into_iter()
            .find_map(|path| {
                // NVRTC loads its `nvrtc-builtins` library by name when it compiles: the directory it lives in must be searched.
                if let Some(directory) = path.parent().filter(|directory| !directory.as_os_str().is_empty()) {
                    let separator = if cfg!(windows) { ";" } else { ":" };
                    let current = std::env::var("PATH").unwrap_or_default();
                    if !current.split(separator).any(|entry| std::path::Path::new(entry) == directory) {
                        // SAFETY: done while the device is made, before the engine thread runs kernels.
                        unsafe { std::env::set_var("PATH", format!("{}{separator}{current}", directory.display())) };
                    }
                }
                unsafe { DynamicLibrary::new(&path).ok() }
            })
            .context("cannot load NVRTC (nvrtc64_*.dll / libnvrtc.so): install the CUDA toolkit or put the library beside the program (DAFO_NVRTC names it)")?;
        let api = Arc::new(Api { driver, nvrtc: Nvrtc::load(nvrtc_library)? });
        unsafe {
            check(&api, (api.driver.cu_init)(0), "cuInit")?;
            let mut device = 0;
            check(&api, (api.driver.cu_device_get)(&mut device, 0), "cuDeviceGet")?;
            let mut name = [0 as c_char; 256];
            check(&api, (api.driver.cu_device_get_name)(name.as_mut_ptr(), 256, device), "cuDeviceGetName")?;
            let (mut major, mut minor) = (0, 0);
            check(&api, (api.driver.cu_device_get_attribute)(&mut major, 75, device), "cuDeviceGetAttribute")?;
            check(&api, (api.driver.cu_device_get_attribute)(&mut minor, 76, device), "cuDeviceGetAttribute")?;
            let mut total_memory = 0usize;
            check(&api, (api.driver.cu_device_total_mem)(&mut total_memory, device), "cuDeviceTotalMem")?;
            let mut context = std::ptr::null_mut();
            check(&api, (api.driver.cu_ctx_create)(&mut context, 0, device), "cuCtxCreate")?;
            let mut stream = std::ptr::null_mut();
            check(&api, (api.driver.cu_stream_create)(&mut stream, 0), "cuStreamCreate")?;
            let name = CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned();
            Ok(Self { api, context, stream, name, architecture: (major, minor), total_memory })
        }
    }

    fn current(&self) {
        unsafe { (self.api.driver.cu_ctx_set_current)(self.context) };
    }

    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// Memory of the device, and what is free now (bytes).
    pub fn memory(&self) -> (usize, usize) {
        self.current();
        let (mut free, mut total) = (0usize, 0usize);
        unsafe { (self.api.driver.cu_mem_get_info)(&mut free, &mut total) };
        (free, total.max(self.total_memory))
    }

    /// Compiles `source` (CUDA C++) for this device.
    pub fn library(&self, source: &str) -> Result<Library> {
        self.current();
        let api = &*self.api;
        let architecture = format!("--gpu-architecture=sm_{}{}", self.architecture.0, self.architecture.1);
        let options = [architecture, "--std=c++17".to_owned(), "--use_fast_math=false".to_owned()];
        let options: Vec<CString> = options[..2].iter().map(|option| CString::new(option.as_str()).unwrap()).collect();
        let pointers: Vec<*const c_char> = options.iter().map(|option| option.as_ptr()).collect();
        let source = CString::new(source).context("the kernel source contains a NUL")?;
        let name = CString::new("dafo.cu").unwrap();
        unsafe {
            let mut program = std::ptr::null_mut();
            let code = (api.nvrtc.create_program)(&mut program, source.as_ptr(), name.as_ptr(), 0, std::ptr::null(), std::ptr::null());
            if code != 0 {
                bail!("nvrtcCreateProgram failed with error {code}");
            }
            let compiled = (api.nvrtc.compile_program)(program, pointers.len() as c_int, pointers.as_ptr());
            let mut size = 0usize;
            (api.nvrtc.get_log_size)(program, &mut size);
            let mut log = vec![0u8; size.max(1)];
            (api.nvrtc.get_log)(program, log.as_mut_ptr() as *mut c_char);
            let log = String::from_utf8_lossy(&log).trim_end_matches('\0').to_owned();
            if compiled != 0 {
                (api.nvrtc.destroy_program)(&mut program);
                bail!("the kernel source does not compile (NVRTC error {compiled}):\n{log}");
            }
            let mut cubin_size = 0usize;
            (api.nvrtc.get_cubin_size)(program, &mut cubin_size);
            let mut cubin = vec![0u8; cubin_size];
            (api.nvrtc.get_cubin)(program, cubin.as_mut_ptr() as *mut c_char);
            (api.nvrtc.destroy_program)(&mut program);
            let mut module = std::ptr::null_mut();
            check(api, (api.driver.cu_module_load_data)(&mut module, cubin.as_ptr() as *const c_void), "cuModuleLoadData")?;
            Ok(Library(module))
        }
    }

    pub fn kernel(&self, library: &Library, name: &'static str) -> Result<Kernel> {
        self.current();
        let cname = CString::new(name).unwrap();
        let mut function = std::ptr::null_mut();
        check(&self.api, unsafe { (self.api.driver.cu_module_get_function)(&mut function, library.0, cname.as_ptr()) }, &format!("cuModuleGetFunction({name})"))?;
        Ok(Kernel { function, name })
    }

    /// Zeroed device memory.
    pub fn buffer(&self, bytes: usize) -> Result<Buffer> {
        self.current();
        let bytes = bytes.max(1);
        let mut device = 0;
        check(&self.api, unsafe { (self.api.driver.cu_mem_alloc)(&mut device, bytes) }, &format!("cuMemAlloc of {bytes} bytes"))?;
        check(&self.api, unsafe { (self.api.driver.cu_memset_d8)(device, 0, bytes) }, "cuMemsetD8")?;
        Ok(Buffer { api: self.api.clone(), device, host: std::ptr::null_mut(), bytes })
    }

    /// Zeroed pinned host memory the device reads and writes over PCIe; the host reads and writes it with `as_slice` and `as_mut_slice`.
    pub fn buffer_shared(&self, bytes: usize) -> Result<Buffer> {
        self.current();
        let bytes = bytes.max(1);
        let mut host = std::ptr::null_mut();
        // DEVICEMAP (2): the device can address it.
        check(&self.api, unsafe { (self.api.driver.cu_mem_host_alloc)(&mut host, bytes, 2) }, &format!("cuMemHostAlloc of {bytes} bytes"))?;
        unsafe { std::ptr::write_bytes(host as *mut u8, 0, bytes) };
        let mut device = 0;
        check(&self.api, unsafe { (self.api.driver.cu_mem_host_get_device_pointer)(&mut device, host, 0) }, "cuMemHostGetDevicePointer")?;
        Ok(Buffer { api: self.api.clone(), device, host, bytes })
    }

    pub fn buffer_from<T: Copy>(&self, values: &[T]) -> Result<Buffer> {
        let mut buffer = self.buffer(size_of_val(values))?;
        buffer.write_bytes(0, unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, size_of_val(values)) })?;
        Ok(buffer)
    }

    /// Device memory holding a copy of `bytes` bytes at `pointer` (there is no memory shared with the CPU to wrap).
    ///
    /// # Safety
    /// `pointer` must be readable for `bytes` bytes.
    pub unsafe fn buffer_no_copy(&self, pointer: *const u8, bytes: usize) -> Result<Buffer> {
        let mut buffer = self.buffer(bytes)?;
        buffer.write_bytes(0, unsafe { std::slice::from_raw_parts(pointer, bytes) })?;
        Ok(buffer)
    }

    /// Largest buffer to make in one piece.
    pub fn max_buffer_bytes(&self) -> usize {
        self.total_memory
    }

    pub fn run(&self, encode: impl FnOnce(&Encoder)) -> Result<Timing> {
        self.current();
        let api = &*self.api;
        let encoder = Encoder { gpu: self, function: Cell::new(std::ptr::null_mut()), parameters: RefCell::new(Vec::new()), shared: Cell::new(0), error: RefCell::new(None) };
        let (mut start, mut end) = (std::ptr::null_mut(), std::ptr::null_mut());
        unsafe {
            check(api, (api.driver.cu_event_create)(&mut start, 0), "cuEventCreate")?;
            check(api, (api.driver.cu_event_create)(&mut end, 0), "cuEventCreate")?;
            check(api, (api.driver.cu_event_record)(start, self.stream), "cuEventRecord")?;
        }
        let wall = std::time::Instant::now();
        encode(&encoder);
        unsafe {
            check(api, (api.driver.cu_event_record)(end, self.stream), "cuEventRecord")?;
            check(api, (api.driver.cu_stream_synchronize)(self.stream), "the kernels failed (cuStreamSynchronize)")?;
        }
        if let Some(error) = encoder.error.borrow_mut().take() {
            return Err(error);
        }
        let mut milliseconds = 0f32;
        unsafe { (api.driver.cu_event_elapsed_time)(&mut milliseconds, start, end) };
        Ok(Timing { gpu: milliseconds as f64 / 1000.0, wall: wall.elapsed().as_secs_f64() })
    }

    /// Kernels on one stream run in order, so there is nothing to overlap: the same as [`Gpu::run`].
    pub fn run_concurrent(&self, encode: impl FnOnce(&Encoder)) -> Result<Timing> {
        self.run(encode)
    }
}

impl Buffer {
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The contents, for a buffer made by [`Gpu::buffer_shared`] (host memory). Call it only when the GPU is idle.
    pub fn as_slice<T: Copy>(&self) -> &[T] {
        assert!(!self.host.is_null(), "this buffer lives on the device: read it with to_vec");
        unsafe { std::slice::from_raw_parts(self.host as *const T, self.bytes / size_of::<T>()) }
    }

    pub fn as_mut_slice<T: Copy>(&mut self) -> &mut [T] {
        assert!(!self.host.is_null(), "this buffer lives on the device: write it with write_bytes");
        unsafe { std::slice::from_raw_parts_mut(self.host as *mut T, self.bytes / size_of::<T>()) }
    }

    /// A copy of the contents (the GPU must be idle).
    pub fn to_vec<T: Copy>(&self) -> Vec<T> {
        let count = self.bytes / size_of::<T>();
        if !self.host.is_null() {
            return self.as_slice::<T>().to_vec();
        }
        let mut out: Vec<T> = Vec::with_capacity(count);
        unsafe {
            (self.api.driver.cu_memcpy_dtoh)(out.as_mut_ptr() as *mut c_void, self.device, count * size_of::<T>());
            out.set_len(count);
        }
        out
    }

    /// A copy of `length` bytes from `offset` (the GPU must be idle).
    pub fn read_bytes(&self, offset: usize, length: usize) -> Vec<u8> {
        if !self.host.is_null() {
            return self.as_slice::<u8>()[offset..offset + length].to_vec();
        }
        let mut out = vec![0u8; length];
        unsafe { (self.api.driver.cu_memcpy_dtoh)(out.as_mut_ptr() as *mut c_void, self.device + offset as u64, length) };
        out
    }

    /// Writes `data` at `offset` bytes into the buffer (the GPU must be idle).
    pub fn write_bytes(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        anyhow::ensure!(offset + data.len() <= self.bytes, "a write of {} bytes at {offset} does not fit a buffer of {}", data.len(), self.bytes);
        if !self.host.is_null() {
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), (self.host as *mut u8).add(offset), data.len()) };
            return Ok(());
        }
        check(&self.api, unsafe { (self.api.driver.cu_memcpy_htod)(self.device + offset as u64, data.as_ptr() as *const c_void, data.len()) }, "cuMemcpyHtoD")
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            if self.host.is_null() {
                (self.api.driver.cu_mem_free)(self.device);
            } else {
                (self.api.driver.cu_mem_free_host)(self.host);
            }
        }
    }
}

impl Encoder<'_> {
    /// No-op: a stream runs its kernels in order.
    pub fn barrier(&self) {}

    pub fn kernel(&self, kernel: &Kernel) {
        self.function.set(kernel.function);
        self.parameters.borrow_mut().clear();
        let _ = kernel.name;
    }

    fn set(&self, index: usize, bytes: Vec<u8>) {
        let mut parameters = self.parameters.borrow_mut();
        if parameters.len() <= index {
            parameters.resize(index + 1, Vec::new());
        }
        parameters[index] = bytes;
    }

    pub fn buffer(&self, index: usize, buffer: &Buffer, offset: usize) {
        self.set(index, (buffer.device + offset as u64).to_le_bytes().to_vec());
    }

    /// A small value passed by copy.
    pub fn value<T: Copy>(&self, index: usize, value: &T) {
        let bytes = unsafe { std::slice::from_raw_parts(value as *const T as *const u8, size_of::<T>()) };
        self.set(index, bytes.to_vec());
    }

    /// The CUDA kernels declare their shared memory themselves: nothing to reserve.
    pub fn shared_memory(&self, _index: usize, _bytes: usize) {}

    pub fn dispatch(&self, groups: [usize; 3], threads: [usize; 3]) {
        if self.error.borrow().is_some() {
            return;
        }
        let api = &self.gpu.api;
        let parameters = self.parameters.borrow();
        let mut pointers: Vec<*mut c_void> = parameters.iter().map(|bytes| bytes.as_ptr() as *mut c_void).collect();
        let code = unsafe {
            (api.driver.cu_launch_kernel)(
                self.function.get(),
                groups[0] as c_uint,
                groups[1] as c_uint,
                groups[2] as c_uint,
                threads[0] as c_uint,
                threads[1] as c_uint,
                threads[2] as c_uint,
                self.shared.get() as c_uint,
                self.gpu.stream,
                pointers.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        self.shared.set(0);
        if let Err(error) = check(api, code, &format!("cuLaunchKernel (groups {groups:?}, threads {threads:?})")) {
            *self.error.borrow_mut() = Some(error);
        }
    }
}
