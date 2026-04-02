use std::{
    collections::HashSet,
    ffi::c_void,
    io::{Write, stdout},
    mem::MaybeUninit,
    ops::DerefMut,
    ptr::NonNull,
    sync::Arc,
    time::Instant,
};

use anyhow::anyhow;
use bytes::{BufMut, BytesMut};
use cuda_lib::{
    CudaDeviceId, CudaDeviceMemory, Device,
    cudart_sys::{cudaMemcpy, cudaMemcpyDeviceToHost, cudaMemcpyHostToDevice},
    rt::cudaSetDevice,
};
use fabric_lib::{
    FabricEngine, RdmaDomainInfo, Worker,
    api::{
        DomainAddress, DomainGroupRouting, ImmTransferRequest, MemoryRegionDescriptor,
        MemoryRegionHandle, PagedTransferRequest, SingleTransferRequest,
        TransferCompletionEntry, TransferId, TransferRequest,
    },
    detect_topology,
};
use serde::{Deserialize, Serialize};

fn addr_to_string(addr: DomainAddress) -> String {
    let mut s = String::new();
    for b in &addr.0 {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn parse_addr(s: &str) -> anyhow::Result<DomainAddress> {
    if !s.len().is_multiple_of(2) {
        return Err(anyhow!("Expected even number of characters"));
    }
    let mut buf = BytesMut::with_capacity(s.len() / 2);
    for i in 0..s.len() / 2 {
        let byte = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)?;
        buf.put_u8(byte);
    }
    Ok(DomainAddress(buf.freeze()))
}

fn fill_random_u64(vec: &mut [u64], seed: u64) {
    let mut state = seed;
    for v in vec.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *v = state;
    }
}

fn fill_random_bytes(vec: &mut [u8], seed: u64) {
    assert!(vec.len().is_multiple_of(size_of::<u64>()));
    let vec64 = unsafe {
        std::slice::from_raw_parts_mut(
            vec.as_mut_ptr() as *mut u64,
            vec.len() / size_of::<u64>(),
        )
    };
    fill_random_u64(vec64, seed);
}

fn memcpy_h2d(dst: NonNull<c_void>, src: &[u8]) {
    assert_eq!(
        unsafe {
            cudaMemcpy(
                dst.as_ptr(),
                src.as_ptr() as *const c_void,
                src.len(),
                cudaMemcpyHostToDevice,
            )
        },
        0
    );
}

fn memcpy_d2h(dst: &mut [u8], src: NonNull<c_void>) {
    assert_eq!(
        unsafe {
            cudaMemcpy(
                dst.as_mut_ptr() as *mut c_void,
                src.as_ptr(),
                dst.len(),
                cudaMemcpyDeviceToHost,
            )
        },
        0
    );
}

macro_rules! wait_completion {
    ($engine:expr, $pattern:pat => $result:expr) => {{
        loop {
            std::hint::spin_loop();
            if let Some(comp) = $engine.poll_transfer_completion() {
                match comp {
                    $pattern => break Ok($result),
                    _ => break Err(anyhow!("Unexpected completion entry. {:?}", comp)),
                }
            }
        }
    }};
}

fn avg_std(list: &[f64]) -> (usize, f64, f64) {
    let n = list.len();
    let sum: f64 = list.iter().sum();
    let mean = sum / n as f64;
    let variance: f64 = list.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let stddev = variance.sqrt();
    (n, mean, stddev)
}

const IMM_PING: u32 = 0x50494E47; // "PING"
const IMM_PONG: u32 = 0x504F4E47; // "PONG"

#[derive(Serialize, Deserialize)]
struct PingPongRequest {
    pub client_addr: DomainAddress,
    pub client_mr_desc: MemoryRegionDescriptor,
    pub write_bytes: usize,
    pub warmups: i32,
    pub repeats: i32,
}

#[derive(Serialize, Deserialize)]
struct PingPongResponse {
    n: usize,
    elapsed_avg_us: f64,
    elapsed_std_us: f64,
    bw_avg_gbps: f64,
}

const MESSAGE_BUF_SIZE: usize = 64 << 20;
const CUDA_BUF_SIZE: usize = 256 << 20;

fn build_engine(args: &[String]) -> anyhow::Result<(Vec<u8>, FabricEngine)> {
    let topo_groups = detect_topology()?;
    let selected_gpus: Vec<_>;
    let nets_per_gpu;
    let all_gpus: Vec<_> = topo_groups.iter().map(|g| g.cuda_device).collect();
    if args.len() == 1 {
        selected_gpus = all_gpus.clone();
        nets_per_gpu = topo_groups[0].domains.len();
    } else {
        selected_gpus = args[1].split(',').map(|s| s.parse::<u8>().unwrap()).collect();
        nets_per_gpu = args[2].parse().unwrap();
    }

    // Prepare workers
    let mut workers = Vec::new();
    println!("Topology:");
    for &cuda_device in &selected_gpus {
        let topo_group = topo_groups
            .iter()
            .find(|g| g.cuda_device == cuda_device)
            .ok_or(anyhow!("Cannot find cuda:{} in topology", cuda_device))?;
        let domain_list: Vec<_> =
            topo_group.domains.iter().take(nets_per_gpu).cloned().collect();
        if domain_list.len() < nets_per_gpu {
            eprintln!("Not enough domains for GPU:{}", cuda_device);
            std::process::exit(1);
        }
        let worker_cpu = topo_group.cpus[0];
        let uvm_cpu = topo_group.cpus[1];

        print!(
            "  GPU: {}, NUMA: {}, Worker CPU: {:2}, UVM CPU: {:2}, NICs:",
            cuda_device, topo_group.numa, worker_cpu, uvm_cpu
        );
        for info in &domain_list {
            print!(" {:14}", info.name());
        }
        println!();
        let worker = Worker {
            domain_list,
            pin_worker_cpu: Some(worker_cpu),
            pin_uvm_cpu: Some(uvm_cpu),
        };
        workers.push((cuda_device, worker));
    }

    // Set up the fabric engine and its workers.
    let engine = FabricEngine::new(workers)?;

    Ok((selected_gpus, engine))
}

struct MemoryResource {
    _host_box: Box<MaybeUninit<[u8; MESSAGE_BUF_SIZE]>>,
    host_buf: NonNull<c_void>,
    host_mr_handle: MemoryRegionHandle,
    cuda_res: Vec<CudaResource>,
}

struct CudaResource {
    cuda_device: u8,
    cuda_buf: CudaDeviceMemory,
    cuda_mr_handle: MemoryRegionHandle,
    cuda_mr_desc: MemoryRegionDescriptor,
}

fn alloc_and_register_memory(
    selected_gpus: &[u8],
    engine: &FabricEngine,
) -> anyhow::Result<MemoryResource> {
    print!("Registering memory...");
    stdout().flush()?;
    let mut _host_box = Box::<[u8; MESSAGE_BUF_SIZE]>::new_uninit();
    let host_buf = unsafe {
        NonNull::new_unchecked(_host_box.deref_mut().as_mut_ptr() as *mut c_void)
    };
    let host_mr_handle =
        engine.register_memory_local(host_buf, MESSAGE_BUF_SIZE, Device::Host)?;
    print!(" cpu");
    stdout().flush()?;

    let mut cuda_res = Vec::new();
    for &cuda_device in selected_gpus {
        cudaSetDevice(cuda_device as i32)?;
        let cuda_buf = CudaDeviceMemory::device(CUDA_BUF_SIZE)?;
        let (cuda_mr_handle, cuda_mr_desc) = engine.register_memory_allow_remote(
            cuda_buf.ptr(),
            cuda_buf.size(),
            Device::Cuda(CudaDeviceId(cuda_device)),
        )?;
        cuda_res.push(CudaResource {
            cuda_device,
            cuda_buf,
            cuda_mr_handle,
            cuda_mr_desc,
        });
        print!(" cuda:{}", cuda_device);
        stdout().flush()?;
    }
    println!();

    Ok(MemoryResource { _host_box, host_buf, host_mr_handle, cuda_res })
}

fn unregister_memory(
    engine: &FabricEngine,
    host_buf: NonNull<c_void>,
    cuda_res: &[CudaResource],
) -> anyhow::Result<()> {
    engine.unregister_memory(host_buf)?;
    for res in cuda_res {
        engine.unregister_memory(res.cuda_buf.ptr())?;
    }
    Ok(())
}

fn server_main(args: Vec<String>) -> anyhow::Result<()> {
    if args.len() != 1 && args.len() != 3 {
        eprintln!("Server Usage:");
        eprintln!("Default runs with all GPUs and all NICs:");
        eprintln!("  {}", args[0]);
        eprintln!("  {}   selected_gpus     nets_per_gpu", args[0]);
        eprintln!("Example:");
        eprintln!("  {}   0,1,2,3,4,5,6,7   4", args[0]);
        std::process::exit(1);
    }

    let (selected_gpus, engine) = build_engine(&args)?;

    println!(
        "Initialized TransferEngine. num_nets={}, nets_per_gpu={}, link_speed={:.0}Gbps",
        engine.num_domains(),
        engine.num_domains() / selected_gpus.len(),
        engine.aggregated_link_speed() as f64 / 1e9,
    );
    println!("Main Address: {}", addr_to_string(engine.main_address()));

    let MemoryResource { _host_box, host_buf, host_mr_handle, cuda_res } =
        alloc_and_register_memory(&selected_gpus, &engine)?;

    // Use first GPU for ping-pong
    let server_cuda_res = &cuda_res[0];

    loop {
        // 1. Receive ping-pong request from client
        print!("Waiting for client to connect...");
        engine.submit_recv(
            TransferId(100),
            host_mr_handle,
            host_buf,
            MESSAGE_BUF_SIZE,
        )?;
        stdout().flush()?;
        let (_transfer_id, data_len) = wait_completion!(
            engine,
            TransferCompletionEntry::Recv { transfer_id, data_len } => (transfer_id, data_len)
        )?;

        let data = unsafe {
            std::slice::from_raw_parts(host_buf.as_ptr() as *const u8, data_len)
        };
        let request: PingPongRequest = postcard::from_bytes(data)?;
        println!(" Received request: bytes={}, repeats={}", request.write_bytes, request.repeats);

        // 2. Send server's mr_desc to client (so client can write back)
        let server_mr_desc = server_cuda_res.cuda_mr_desc.clone();
        let setup_data = postcard::to_slice(&server_mr_desc, unsafe {
            std::slice::from_raw_parts_mut(host_buf.as_ptr() as *mut u8, MESSAGE_BUF_SIZE)
        })?;

        engine.submit_send(
            TransferId(200),
            request.client_addr.clone(),
            host_mr_handle,
            host_buf,
            setup_data.len(),
        )?;
        wait_completion!(engine, TransferCompletionEntry::Send(_) => ())?;
        println!("Sent server mr_desc to client");

        // 3. Prepare data (fill with pattern)
        {
            let mut tmp = vec![0u8; request.write_bytes];
            fill_random_bytes(&mut tmp, 0xABCD1234u64);
            memcpy_h2d(server_cuda_res.cuda_buf.ptr(), &tmp);
        }

        // 4. Ping-Pong Benchmark
        let mut elapsed_list = Vec::new();

        for repeat in -request.warmups..request.repeats {
            // --- PING: Server -> Client ---
            let t0 = Instant::now();

            // Submit Write to Client (with IMM to notify client)
            engine.submit_transfer(
                TransferId(300 + repeat as u64),
                TransferRequest::Single(SingleTransferRequest {
                    src_mr: server_cuda_res.cuda_mr_handle,
                    src_offset: 0,
                    length: request.write_bytes as u64,
                    imm_data: Some(IMM_PING),
                    dst_mr: request.client_mr_desc.clone(),
                    dst_offset: 0,
                    domain: DomainGroupRouting::RoundRobinSharded {
                        num_shards: engine.nets_per_gpu(),
                    },
                }),
                None,
            )?;

            // Wait for local Write completion (CQ)
            wait_completion!(
                engine,
                TransferCompletionEntry::Transfer(TransferId(id)) if id == 300 + repeat as u64 => ()
            )?;

            // --- PONG: Wait for Client to write back ---
            // Wait for ImmData completion event from client
            let imm_data = wait_completion!(
                engine,
                TransferCompletionEntry::ImmData(imm) => imm
            )?;
            if imm_data != IMM_PONG {
                return Err(anyhow!("Expected PONG imm data but got {:#X}", imm_data));
            }

            let t1 = Instant::now();

            if repeat >= 0 {
                elapsed_list.push(t1.duration_since(t0).as_secs_f64() * 1e6); // microseconds
            }
        }

        // 5. Calculate statistics
        let (n, elapsed_avg_us, elapsed_std_us) = avg_std(&elapsed_list);
        // RTT bandwidth = 2 * bytes / time (round trip, two ways)
        let bw_avg_gbps = 2.0 * request.write_bytes as f64 / (elapsed_avg_us * 1e-6) * 8.0 / 1e9;

        let response = PingPongResponse {
            n,
            elapsed_avg_us,
            elapsed_std_us,
            bw_avg_gbps,
        };

        // 6. Send result to client
        let resp_data = postcard::to_slice(&response, unsafe {
            std::slice::from_raw_parts_mut(host_buf.as_ptr() as *mut u8, MESSAGE_BUF_SIZE)
        })?;
        engine.submit_send(
            TransferId(400),
            request.client_addr,
            host_mr_handle,
            host_buf,
            resp_data.len(),
        )?;
        wait_completion!(engine, TransferCompletionEntry::Send(_) => ())?;

        println!(
            "Ping-pong done: RTT={:.2}±{:.2} us, BW={:.2} Gbps",
            elapsed_avg_us, elapsed_std_us, bw_avg_gbps
        );
    }
}

fn verify_data(cuda_res: &CudaResource, expected_seed: u64, len: usize) -> anyhow::Result<()> {
    let mut tmp = vec![0u8; len];
    memcpy_d2h(&mut tmp, cuda_res.cuda_buf.ptr());
    let mut gold = vec![0u8; len];
    fill_random_bytes(&mut gold, expected_seed);
    if tmp != gold {
        return Err(anyhow!("Data verification failed"));
    }
    Ok(())
}

fn client_main(args: Vec<String>) -> anyhow::Result<()> {
    if args.len() != 4 {
        eprintln!("Client Usage:");
        eprintln!("  {} selected_gpus nets_per_gpu server_address", args[0]);
        std::process::exit(1);
    }
    let server_addr = parse_addr(&args[3])?;

    let (selected_gpus, engine) = build_engine(&args)?;
    println!(
        "Initialized TransferEngine. num_nets={}, nets_per_gpu={}, link_speed={:.0}Gbps",
        engine.num_domains(),
        engine.num_domains() / selected_gpus.len(),
        engine.aggregated_link_speed() as f64 / 1e9,
    );

    let MemoryResource { _host_box, host_buf, host_mr_handle, cuda_res } =
        alloc_and_register_memory(&selected_gpus, &engine)?;

    let client_cuda_res = &cuda_res[0];

    // Ping-pong test with different sizes
    let test_sizes = vec![
        64,
        256,
        1024,
        4096,
        16384,
        65536,
        256 * 1024,
        1024 * 1024,
        4 * 1024 * 1024,
        16 * 1024 * 1024,
    ];

    for write_bytes in test_sizes {
        if write_bytes > CUDA_BUF_SIZE {
            println!("Skipping {} bytes (exceeds buffer size)", write_bytes);
            continue;
        }

        print!("Ping-pong bytes: {:10} ... ", write_bytes);
        stdout().flush()?;

        let warmups = 50i32;
        let repeats = 100i32;

        // 1. Send Request (contains client mr_desc)
        let request = PingPongRequest {
            client_addr: engine.main_address(),
            client_mr_desc: client_cuda_res.cuda_mr_desc.clone(),
            write_bytes,
            warmups,
            repeats,
        };
        let data = postcard::to_slice(&request, unsafe {
            std::slice::from_raw_parts_mut(host_buf.as_ptr() as *mut u8, MESSAGE_BUF_SIZE)
        })?;
        engine.submit_send(
            TransferId(100),
            server_addr.clone(),
            host_mr_handle,
            host_buf,
            data.len(),
        )?;
        wait_completion!(engine, TransferCompletionEntry::Send(_) => ())?;

        // 2. Receive server's mr_desc
        engine.submit_recv(TransferId(200), host_mr_handle, host_buf, MESSAGE_BUF_SIZE)?;
        let (_, data_len) = wait_completion!(
            engine,
            TransferCompletionEntry::Recv { transfer_id, data_len } => (transfer_id, data_len)
        )?;
        let server_mr_desc: MemoryRegionDescriptor = postcard::from_bytes(unsafe {
            std::slice::from_raw_parts(host_buf.as_ptr() as *const u8, data_len)
        })?;

        // 3. Ping-pong loop
        for repeat in -warmups..repeats {
            // --- Wait for Server's PING (Write + IMM) ---
            let imm_data = wait_completion!(
                engine,
                TransferCompletionEntry::ImmData(imm) => imm
            )?;
            if imm_data != IMM_PING {
                return Err(anyhow!("Expected PING imm data but got {:#X}", imm_data));
            }

            // Optional: Verify data from server (first iteration only)
            if repeat == 0 && write_bytes <= 1024 * 1024 {
                if let Err(e) = verify_data(client_cuda_res, 0xABCD1234u64, write_bytes) {
                    println!("Warning: Data verification failed: {}", e);
                }
            }

            // --- PONG: Write back to Server ---
            engine.submit_transfer(
                TransferId(500 + repeat as u64),
                TransferRequest::Single(SingleTransferRequest {
                    src_mr: client_cuda_res.cuda_mr_handle,
                    src_offset: 0,
                    length: write_bytes as u64,
                    imm_data: Some(IMM_PONG),
                    dst_mr: server_mr_desc.clone(),
                    dst_offset: 0,
                    domain: DomainGroupRouting::RoundRobinSharded {
                        num_shards: engine.nets_per_gpu(),
                    },
                }),
                None,
            )?;

            // Wait for local write completion
            wait_completion!(
                engine,
                TransferCompletionEntry::Transfer(TransferId(id)) if id == 500 + repeat as u64 => ()
            )?;
        }

        // 4. Receive benchmark result
        engine.submit_recv(TransferId(400), host_mr_handle, host_buf, MESSAGE_BUF_SIZE)?;
        let (_, data_len) = wait_completion!(
            engine,
            TransferCompletionEntry::Recv { transfer_id, data_len } => (transfer_id, data_len)
        )?;
        let response: PingPongResponse = postcard::from_bytes(unsafe {
            std::slice::from_raw_parts(host_buf.as_ptr() as *const u8, data_len)
        })?;

        println!(
            "RTT: {:8.2} ± {:6.2} us, BW: {:6.2} Gbps",
            response.elapsed_avg_us, response.elapsed_std_us, response.bw_avg_gbps
        );
    }

    println!("Done!");
    unregister_memory(&engine, host_buf, &cuda_res)?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    logging_lib::init(&logging_lib::LoggingOpts {
        log_format: logging_lib::LogFormat::Text,
        log_color: logging_lib::LogColor::Auto,
        log_directives: None,
    })?;
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() <= 3 { server_main(args) } else { client_main(args) }
}
