#!/usr/bin/env python3
"""Python port of fabric-debug/src/main.rs ping-pong benchmark.

Uses torch.distributed (gloo backend) to exchange DomainAddress / MR descriptors,
then measures RDMA write latency with immediate data over the fabric library.
"""

import argparse
import os
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
IMM_PING = 0x50494E47
IMM_PONG = 0x504F4E47
NUM_WARMUPS = 50
NUM_REPEATS = 100

TEST_SIZES = [
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
]


def on_error_panic(error: str) -> None:
    raise RuntimeError("fabric-lib error:" + error)


def build_engine(cuda_device: int, nets_per_gpu: int) -> TransferEngine:
    system_topo = TransferEngine.detect_topology()
    builder = TransferEngine.builder()

    found = False
    for group in system_topo:
        if group.cuda_device != cuda_device:
            continue
        if len(group.domains) < nets_per_gpu:
            raise RuntimeError(
                f"Not enough NICs for cuda:{cuda_device}: "
                f"expected {nets_per_gpu}, got {len(group.domains)}"
            )
        if len(group.cpus) < 2:
            raise RuntimeError(
                f"Not enough CPUs for cuda:{cuda_device}: "
                f"expected at least 2, got {len(group.cpus)}"
            )
        worker_cpu = group.cpus[0]
        uvm_cpu = group.cpus[1]
        builder.add_gpu_domains(
            group.cuda_device,
            group.domains[:nets_per_gpu],
            worker_cpu,
            uvm_cpu,
        )
        found = True
        logger.info(
            "Registered CUDA device %d, CPU #%d, UVM CPU #%d",
            group.cuda_device,
            worker_cpu,
            uvm_cpu,
        )

    if not found:
        raise RuntimeError(f"CUDA device {cuda_device} not found in topology")

    engine = builder.build()
    logger.info(
        "Initialized TransferEngine. num_nets=%d, nets_per_gpu=%d, link_speed=%.0fGbps",
        engine.num_domains,
        engine.nets_per_gpu,
        engine.aggregated_link_speed / 1e9,
    )
    logger.info("Main Address: %s", engine.main_address)
    return engine


def generate_random_bytes(seed: int, size: int) -> torch.Tensor:
    rng = torch.Generator("cpu")
    rng.manual_seed(seed)
    return torch.randint(0, 256, (size,), dtype=torch.uint8, generator=rng)


def avg_std(values: list[float]) -> tuple[int, float, float]:
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    stddev = variance**0.5
    return n, mean, stddev


def verify_data(
    cuda_buf: torch.Tensor, expected_seed: int, length: int, offset: int = 0
) -> None:
    gold = generate_random_bytes(expected_seed, length)
    buf = cuda_buf[offset : offset + length].to("cpu")
    if not torch.equal(gold, buf):
        raise RuntimeError("Data verification failed")


def run_server(
    rank: int,
    world_size: int,
    cuda_device: int,
    nets_per_gpu: int,
) -> None:
    engine = build_engine(cuda_device, nets_per_gpu)

    # Exchange addresses via torch.distributed.
    my_addr = engine.main_address
    addr_list: list[Any] = [None] * world_size
    dist.all_gather_object(addr_list, my_addr)
    client_addr = addr_list[1] if world_size > 1 else None

    # Register one local CUDA buffer.
    cuda_buf = torch.empty(
        CUDA_BUF_SIZE, dtype=torch.uint8, device=f"cuda:{cuda_device}"
    )
    cuda_mr_handle, cuda_mr_desc = engine.register_tensor(cuda_buf)

    # Exchange MR descriptors.
    mr_list: list[Any] = [None] * world_size
    dist.all_gather_object(mr_list, cuda_mr_desc)
    client_mr_desc = mr_list[1] if world_size > 1 else None

    if world_size < 2:
        logger.warning("World size < 2, no client to benchmark against")
        return

    pong_event = threading.Event()

    def on_imm(imm: int) -> None:
        if imm == IMM_PONG:
            pong_event.set()
        else:
            logger.warning("Unexpected IMM data: 0x%08X", imm)

    engine.set_imm_callback(on_imm)

    for write_bytes in TEST_SIZES:
        if write_bytes > CUDA_BUF_SIZE:
            print(f"Skipping {write_bytes} bytes (exceeds buffer size)")
            continue

        # Prepare data with the same seed as the Rust binary.
        tmp = generate_random_bytes(0xABCD1234, write_bytes)
        cuda_buf[:write_bytes].copy_(tmp)

        elapsed_list: list[float] = []

        for repeat in range(-NUM_WARMUPS, NUM_REPEATS):
            t0 = time.perf_counter_ns()

            write_done = threading.Event()
            engine.submit_write(
                src_mr=cuda_mr_handle,
                offset=0,
                length=write_bytes,
                imm_data=IMM_PING,
                dst_mr=client_mr_desc,
                dst_offset=0,
                on_done=write_done.set,
                on_error=on_error_panic,
                num_shards=None,
            )
            write_done.wait()
            pong_event.wait()
            pong_event.clear()

            t1 = time.perf_counter_ns()

            if repeat >= 0:
                elapsed_list.append((t1 - t0) / 1000.0)  # microseconds

        n, elapsed_avg_us, elapsed_std_us = avg_std(elapsed_list)
        bw_avg_gbps = 2.0 * write_bytes / (elapsed_avg_us * 1e-6) * 8.0 / 1e9

        print(
            f"Ping-pong done: bytes={write_bytes} "
            f"RTT={elapsed_avg_us:.2f}±{elapsed_std_us:.2f} us, "
            f"BW={bw_avg_gbps:.2f} Gbps"
        )

        # Broadcast stats to client so it can print the result table too.
        stats: list[Any] = [n, elapsed_avg_us, elapsed_std_us, bw_avg_gbps]
        dist.broadcast_object_list(stats, src=0)

    dist.barrier()
    logger.info("Server done.")


def run_client(
    rank: int,
    world_size: int,
    cuda_device: int,
    nets_per_gpu: int,
) -> None:
    engine = build_engine(cuda_device, nets_per_gpu)

    # Exchange addresses via torch.distributed.
    my_addr = engine.main_address
    addr_list: list[Any] = [None] * world_size
    dist.all_gather_object(addr_list, my_addr)
    server_addr = addr_list[0]

    # Register one local CUDA buffer.
    cuda_buf = torch.empty(
        CUDA_BUF_SIZE, dtype=torch.uint8, device=f"cuda:{cuda_device}"
    )
    cuda_mr_handle, cuda_mr_desc = engine.register_tensor(cuda_buf)

    # Exchange MR descriptors.
    mr_list: list[Any] = [None] * world_size
    dist.all_gather_object(mr_list, cuda_mr_desc)
    server_mr_desc = mr_list[0]

    ping_event = threading.Event()

    def on_imm(imm: int) -> None:
        if imm == IMM_PING:
            ping_event.set()
        else:
            logger.warning("Unexpected IMM data: 0x%08X", imm)

    engine.set_imm_callback(on_imm)

    for write_bytes in TEST_SIZES:
        if write_bytes > CUDA_BUF_SIZE:
            continue

        print(f"Ping-pong bytes: {write_bytes:10} ... ", end="", flush=True)

        for repeat in range(-NUM_WARMUPS, NUM_REPEATS):
            ping_event.wait()
            ping_event.clear()

            if repeat == 0 and write_bytes <= 1024 * 1024:
                try:
                    verify_data(cuda_buf, 0xABCD1234, write_bytes, offset=0)
                except RuntimeError as e:
                    print(f"Warning: Data verification failed: {e}")

            write_done = threading.Event()
            engine.submit_write(
                src_mr=cuda_mr_handle,
                offset=0,
                length=write_bytes,
                imm_data=IMM_PONG,
                dst_mr=server_mr_desc,
                dst_offset=0,
                on_done=write_done.set,
                on_error=on_error_panic,
                num_shards=None,
            )
            write_done.wait()

        # Receive benchmark result from server.
        stats: list[Any] = [None, None, None, None]
        dist.broadcast_object_list(stats, src=0)
        n, elapsed_avg_us, elapsed_std_us, bw_avg_gbps = stats
        print(
            f"RTT: {elapsed_avg_us:8.2f} ± {elapsed_std_us:6.2f} us, "
            f"BW: {bw_avg_gbps:6.2f} Gbps"
        )

    dist.barrier()
    logger.info("Rank %d: done.", rank)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Python port of fabric-debug ping-pong benchmark"
    )
    parser.add_argument(
        "--nets-per-gpu",
        type=int,
        default=int(os.environ.get("NETS_PER_GPU", "1")),
        help="Number of NICs per GPU",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", "0")),
        help="CUDA device to use",
    )
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    print(
        f"[Rank {rank}/{world_size}] Starting... cuda_device={args.cuda_device}",
        flush=True,
    )

    try:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        print(f"[Rank {rank}] Gloo init OK", flush=True)
    except Exception as e:
        print(f"[Rank {rank}] Gloo init FAILED: {e}", flush=True)
        raise

    try:
        if rank == 0:
            run_server(rank, world_size, args.cuda_device, args.nets_per_gpu)
        else:
            run_client(rank, world_size, args.cuda_device, args.nets_per_gpu)
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
