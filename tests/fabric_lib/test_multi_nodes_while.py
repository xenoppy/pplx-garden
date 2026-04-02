#!/usr/bin/env python3
"""
Multi-node single_write test for TransferEngine.
Reference: tests/fabric_lib/test_transfer_engine.py::test_single_write

Uses torch.distributed (TCP backend) to exchange DomainAddress / MR descriptors
across nodes, then performs RDMA write over the real network.
"""

import dataclasses
import os
import pickle
import queue
import sys
import threading
import time
from typing import Any

import torch
import torch.distributed as dist

from pplx_garden.fabric_lib import (
    DomainAddress,
    MemoryRegionDescriptor,
    MemoryRegionHandle,
    TransferEngine,
)
from pplx_garden.utils import logging_utils

logging_utils.setup(level=logging_utils.logging.INFO)
logger = logging_utils.get_logger(__name__)

CUDA_BUF_SIZE = 256 << 20
MESSAGE_BUF_SIZE = 64 << 20
NUM_WARMUP_ITERS = 50
NUM_LATENCY_ITERS = 200


@dataclasses.dataclass(slots=True)
class SingleWriteRequest:
    """Same structure as the original test's Request.content == Single()."""

    addr: DomainAddress
    mr_desc: MemoryRegionDescriptor
    seed: int
    offset: int
    length: int


def on_error_panic(error: str) -> None:
    raise RuntimeError("fabric-lib error:" + error)


def build_engine(cuda_device: int) -> TransferEngine:
    """Build a TransferEngine for the selected GPU."""
    print(f"[build_engine] cuda_device={cuda_device}: calling detect_topology()...", flush=True)
    system_topo = TransferEngine.detect_topology()
    print(f"[build_engine] detected {len(system_topo)} topo groups", flush=True)
    builder = TransferEngine.builder()

    found = False
    for group in system_topo:
        if group.cuda_device != cuda_device:
            continue
        worker_cpu = group.cpus[0]
        uvm_cpu = group.cpus[1]
        builder.add_gpu_domains(group.cuda_device, group.domains, worker_cpu, uvm_cpu)
        found = True
        logger.info(
            "Registered CUDA device %d, CPU #%d, UVM CPU #%d",
            group.cuda_device,
            worker_cpu,
            uvm_cpu,
        )

    if not found:
        raise RuntimeError(f"CUDA device {cuda_device} not found in topology")
    print(f"[build_engine] cuda_device={cuda_device}: building engine...", flush=True)
    engine = builder.build()
    print(f"[build_engine] cuda_device={cuda_device}: engine built OK, addr={engine.main_address}", flush=True)
    return engine


def generate_random_bytes(seed: int, size: int) -> torch.Tensor:
    rng = torch.Generator("cpu")
    rng.manual_seed(seed)
    return torch.randint(0, 256, (size,), dtype=torch.uint8, generator=rng)


def run_server(rank: int, world_size: int, cuda_device: int) -> None:
    """Rank 0: waits for client requests and writes data to their MRs."""
    print(f"[Rank {rank}] run_server start", flush=True)
    engine = build_engine(cuda_device)

    # Gather all DomainAddresses so clients know how to reach us.
    my_addr = engine.main_address
    print(f"[Rank {rank}] all_gather_object start", flush=True)
    addr_list: list[Any] = [None] * world_size
    dist.all_gather_object(addr_list, my_addr)
    print(f"[Rank {rank}] all_gather_object done: {addr_list}", flush=True)
    logger.info("Server address: %s", my_addr)

    # Register one local CUDA buffer.
    cuda_buf = torch.empty(
        CUDA_BUF_SIZE, dtype=torch.uint8, device=f"cuda:{cuda_device}"
    )
    cuda_mr_handle, _ = engine.register_tensor(cuda_buf)

    # Setup bouncing RECVs for incoming client requests.
    recv_queue: queue.Queue[bytes] = queue.Queue()
    recv_flag = [False]  # to work around Python's late binding in closures
    def on_recv(msg):
        recv_flag[0] = True
        recv_queue.put(msg)

    engine.submit_bouncing_recvs(
        count=world_size - 1,
        len=MESSAGE_BUF_SIZE,
        on_recv=on_recv,
        on_error=on_error_panic,
    )

    cond = threading.Condition()
    completions = 0

    def transfer_callback() -> None:
        nonlocal completions
        with cond:
            completions += 1
            cond.notify_all()

    # Wait for each client request and RDMA write back to it.
    for _ in range(world_size - 1):
        recv_flag[0] = False
        msg = recv_queue.get()
        request: SingleWriteRequest = pickle.loads(msg)
        logger.info("Received request from client %s", request.addr)

        # Generate the same golden data the client expects.
        gold = generate_random_bytes(request.seed - 1, request.length)
        cuda_buf[request.offset : request.offset + request.length].copy_(gold)

        engine.submit_write(
            src_mr=cuda_mr_handle,
            offset=request.offset,
            length=request.length,
            imm_data=None,
            dst_mr=request.mr_desc,
            dst_offset=request.offset,
            on_done=transfer_callback,
            on_error=on_error_panic,
        )

    with cond:
        while completions < world_size - 1:
            cond.wait()

    # Send a small ACK back to each client so they can exit their RECV loop cleanly.
    # In the original local test the client does `recv_queue.get()` to match this.
    ack_threads = []
    for client_addr in addr_list[1:]:
        done = threading.Event()

        def ack(_done: threading.Event = done, _addr: DomainAddress = client_addr) -> None:
            engine.submit_send(_addr, pickle.dumps(None), _done.set, on_error_panic)
            _done.wait()

        t = threading.Thread(target=ack)
        t.start()
        ack_threads.append(t)

    for t in ack_threads:
        t.join()

    # --- latency ping-pong ---
    print(f"[Rank {rank}] starting ping-pong latency test...", flush=True)
    ping_iters = (world_size - 1) * (NUM_LATENCY_ITERS + NUM_WARMUP_ITERS)

    max_num_token, dim = 128, 7168

    for num_token in range(8, max_num_token + 1, 8):
        send_msg = pickle.dumps({"addr": my_addr, "tensor": torch.rand([num_token, dim], dtype=torch.bfloat16, device=f'cuda:{cuda_device}')})
        print(
            f"[Rank {rank}] ping-pong config: token_num={num_token}, msg_size={len(send_msg)} bytes, iters={ping_iters}",
            flush=True,
        )
        for _ in range(ping_iters):
            for client_addr in addr_list[1:]:
                while not recv_flag[0]:
                    time.sleep(0)  # wait for the server's ping back

                recv_flag[0] = False
                msg = recv_queue.get()
                done = threading.Event()
                engine.submit_send(client_addr, msg, done.set, on_error_panic)
                done.wait()
    print(f"[Rank {rank}] ping-pong done", flush=True)

    dist.barrier()
    logger.info("Server done.")


def run_client(rank: int, world_size: int, cuda_device: int) -> None:
    """Rank > 0: sends its MR info to the server and verifies the RDMA write."""
    print(f"[Rank {rank}] run_client start", flush=True)
    engine = build_engine(cuda_device)

    # Exchange addresses.
    my_addr = engine.main_address
    print(f"[Rank {rank}] all_gather_object start", flush=True)
    addr_list: list[Any] = [None] * world_size
    dist.all_gather_object(addr_list, my_addr)
    print(f"[Rank {rank}] all_gather_object done: {addr_list}", flush=True)
    server_addr = addr_list[0]
    logger.info("Server address: %s", server_addr)

    # Register receive buffer.
    cuda_buf = torch.zeros(
        CUDA_BUF_SIZE, dtype=torch.uint8, device=f"cuda:{cuda_device}"
    )
    _, cuda_mr_desc = engine.register_tensor(cuda_buf)

    # Setup bouncing RECV to catch the server's ACK.
    recv_queue: queue.Queue[bytes] = queue.Queue()
    recv_flag = [False]  # to work around Python's late binding in closures
    def on_recv(msg):
        recv_flag[0] = True
        recv_queue.put(msg)
    engine.submit_bouncing_recvs(
        count=1,
        len=MESSAGE_BUF_SIZE,
        on_recv=on_recv,
        on_error=on_error_panic,
    )

    seed = 0xABCDABCD987
    offset = 1024
    length = 1000

    request = SingleWriteRequest(
        addr=my_addr,
        mr_desc=cuda_mr_desc,
        seed=seed,
        offset=offset,
        length=length,
    )

    send_done = threading.Event()
    engine.submit_send(
        server_addr,
        pickle.dumps(request),
        send_done.set,
        on_error_panic,
    )
    send_done.wait()
    logger.info("Rank %d: sent request to server", rank)

    # Wait for server ACK (so we know the RDMA write is complete).
    print(f"[Rank {rank}] waiting for server ACK...", flush=True)
    recv_queue.get()
    recv_flag[0] = False
    print(f"[Rank {rank}] received ACK", flush=True)

    # Verify the data written by the server.
    gold = generate_random_bytes(seed - 1, length)
    buf = cuda_buf[offset : offset + length].to("cpu")
    assert torch.equal(gold, buf), f"Data mismatch on rank {rank}"
    logger.info("Rank %d: verified successfully", rank)

    # --- latency ping-pong ---
    max_num_token, dim = 128, 7168
    total_results = []
    print(f"[Rank {rank}] starting ping-pong latency test...", flush=True)
    for num_token in range(8, max_num_token + 1, 8):
        tensor_to_send = torch.rand([num_token, dim], dtype=torch.bfloat16, device=f'cuda:{cuda_device}')

        print(f"[Rank {rank}] ping-pong iteration with num_token={num_token}...", flush=True)
        ping_data = pickle.dumps({"addr": my_addr, "tensor": tensor_to_send})
        ping_msg_size = len(ping_data)
        ping_iters = NUM_LATENCY_ITERS + NUM_WARMUP_ITERS
        print(
            f"[Rank {rank}] ping-pong config: msg_size={ping_msg_size} bytes, iters={ping_iters}",
            flush=True,
        )
        latencies: list[float] = []
        for _ in range(ping_iters):
            t0 = time.perf_counter_ns()
            send_done = threading.Event()
            engine.submit_send(server_addr, ping_data, send_done.set, on_error_panic)
            # send_done.wait()
            while not recv_flag[0]:
                time.sleep(0)  # wait for the server's ping back
            recv_buffer = recv_queue.get()
            t1 = time.perf_counter_ns()
            recv_data = pickle.loads(recv_buffer)
            recv_flag[0] = False
            if not torch.allclose(recv_data["tensor"], tensor_to_send, atol=1e-3, rtol=1e-3):  # sanity check
                print("❌Received tensor does not match sent tensor")
            else:
                print("✅Received tensor matches sent tensor")
            if _ >= NUM_WARMUP_ITERS:  # skip warmup iters
                latencies.append((t1 - t0) / 1000.0)  # us

        avg_us = sum(latencies) / len(latencies)
        min_us = min(latencies)
        max_us = max(latencies)
        total_results.append((num_token, ping_msg_size, avg_us, min_us, max_us))
        print("=" * 60, flush=True)
    for num_token, msg_size, avg_us, min_us, max_us in total_results:
        print(f"num_token={num_token:3d} | msg_size={msg_size:4d} | avg={avg_us:8.2f} us | min={min_us:8.2f} us | max={max_us:8.2f} us", flush=True)
    dist.barrier()
    logger.info("Rank %d: done.", rank)


def main() -> None:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    print(f"[Rank {rank}/{world_size}] Starting... local_rank={local_rank}", flush=True)

    try:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        print(f"[Rank {rank}] Gloo init OK", flush=True)
    except Exception as e:
        print(f"[Rank {rank}] Gloo init FAILED: {e}", flush=True)
        raise

    try:
        if rank == 0:
            run_server(rank, world_size, local_rank)
        else:
            run_client(rank, world_size, local_rank)
    except Exception as e:
        print(f"[Rank {rank}] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        print(f"[Rank {rank}] Destroying process group", flush=True)
        dist.destroy_process_group()

    print(f"[Rank {rank}] Done.", flush=True)


if __name__ == "__main__":
    main()
