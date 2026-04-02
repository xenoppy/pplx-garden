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
    offset: int
    max_length: int


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
    client_addr = addr_list[1] #only one to one test for now.
    logger.info("Client address: %s", client_addr)



    # Register one local CUDA buffer.
    cuda_buf = torch.empty(
        CUDA_BUF_SIZE, dtype=torch.uint8, device=f"cuda:{cuda_device}"
    )
    cuda_mr_handle, cuda_mr_desc = engine.register_tensor(cuda_buf)
    logger.info("Registered local CUDA buffer as MR: handle=%s, desc=%s", cuda_mr_handle, cuda_mr_desc)
    mr_list: list[Any] = [None] * world_size
    dist.all_gather_object(mr_list, cuda_mr_desc)
    logger.info("All MR descs: %s", mr_list)
    client_mr_desc = mr_list[1]

    # Wait for each client request and RDMA write back to it.
    for _ in range(world_size - 1):
        recv_imm = threading.Event()

        # def on_imm(imm: int) -> None:
        #     print(f"Received imm: {imm}, expected: {num_token}", flush=True)
        #     assert imm == num_token, f"Expected imm {num_token} but got {imm}"
        #     recv_imm.set()

        # engine.set_imm_callback(on_imm)
        imm_queue: queue.Queue[int] = queue.Queue()
        engine.set_imm_callback(imm_queue.put)

        max_num_token, dim = 128, 7168
        offset = 0
        logger.info("Ready to submit_write to client %s with imm", client_addr)
        for num_token in range(8, max_num_token + 1, 8):
            ping_iters = NUM_LATENCY_ITERS + NUM_WARMUP_ITERS
            tensor_length = num_token * dim * 2
            for _ in range(ping_iters):     
                logger.info("Waiting for imm of num_token=%d", num_token)           
                imm = imm_queue.get()
                logger.info("Received imm, submitting write with imm=%d", imm)
                engine.submit_write(
                    src_mr=cuda_mr_handle,
                    offset=offset,
                    length=tensor_length,
                    imm_data=num_token,
                    dst_mr=client_mr_desc,
                    dst_offset=offset,
                    on_done=None,
                    on_error=on_error_panic,
                )
            offset = offset + tensor_length



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
    logger.info("Client address: %s", my_addr)

    # Register receive buffer.
    cuda_buf = torch.zeros(
        CUDA_BUF_SIZE, dtype=torch.uint8, device=f"cuda:{cuda_device}"
    )
    cuda_mr_handle, cuda_mr_desc = engine.register_tensor(cuda_buf)
    logger.info("Registered local CUDA buffer as MR: handle=%s, desc=%s", cuda_mr_handle, cuda_mr_desc)
    mr_list: list[Any] = [None] * world_size
    dist.all_gather_object(mr_list, cuda_mr_desc)
    logger.info("All MR descs: %s", mr_list)
    server_mr_desc = mr_list[0]

    recv_imm = threading.Event()

    def on_imm(imm: int) -> None:
        print(f"Received imm: {imm}, expected: {num_token}", flush=True)
        assert imm == num_token, f"Expected imm {num_token} but got {imm}"
        recv_imm.set()

    engine.set_imm_callback(on_imm)


    logger.info("Ready to submit_write to server %s with imm", server_addr)
    max_num_token, dim = 128, 7168
    total_results = []
    offset = 0
    for num_token in range(8, max_num_token + 1, 8):
        ping_iters = NUM_LATENCY_ITERS + NUM_WARMUP_ITERS
        tensor_length = num_token * dim * 2
        latencies: list[float] = []
        for _ in range(ping_iters):
            t0 = time.perf_counter_ns()
            write_done = threading.Event()
            engine.submit_write(
                src_mr=cuda_mr_handle,
                offset=0,
                length=tensor_length,
                imm_data=num_token, #just for notifcation
                dst_mr=server_mr_desc,
                dst_offset=0,
                on_done=write_done.set,
                on_error=on_error_panic,
            )
            write_done.wait()  # wait for the write to complete (optional, can be None)
            logger.info("Write Done with imm=%d, waiting for imm", num_token)
            recv_imm.wait()
            t1 = time.perf_counter_ns()
            recv_imm.clear()
            if _ >= NUM_WARMUP_ITERS:  # skip warmup iters
                latencies.append((t1 - t0) / 1000.0)  # us  

        offset = offset + tensor_length

        avg_us = sum(latencies) / len(latencies)
        min_us = min(latencies)
        max_us = max(latencies)
        total_results.append((num_token, tensor_length, avg_us, min_us, max_us))
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
