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

logger = logging_utils.get_logger(__name__)

CUDA_BUF_SIZE = 256 << 20
MESSAGE_BUF_SIZE = 64 << 20


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
    system_topo = TransferEngine.detect_topology()
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
    return builder.build()


def generate_random_bytes(seed: int, size: int) -> torch.Tensor:
    rng = torch.Generator("cpu")
    rng.manual_seed(seed)
    return torch.randint(0, 256, (size,), dtype=torch.uint8, generator=rng)


def run_server(rank: int, world_size: int, cuda_device: int) -> None:
    """Rank 0: waits for client requests and writes data to their MRs."""
    engine = build_engine(cuda_device)

    # Gather all DomainAddresses so clients know how to reach us.
    my_addr = engine.main_address
    addr_list: list[Any] = [None] * world_size
    dist.all_gather_object(addr_list, my_addr)
    logger.info("Server address: %s", my_addr)

    # Register one local CUDA buffer.
    cuda_buf = torch.empty(
        CUDA_BUF_SIZE, dtype=torch.uint8, device=f"cuda:{cuda_device}"
    )
    cuda_mr_handle, _ = engine.register_tensor(cuda_buf)

    # Setup bouncing RECVs for incoming client requests.
    recv_queue: queue.Queue[bytes] = queue.Queue()
    engine.submit_bouncing_recvs(
        count=world_size - 1,
        len=MESSAGE_BUF_SIZE,
        on_recv=recv_queue.put,
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

    dist.barrier()
    logger.info("Server done.")


def run_client(rank: int, world_size: int, cuda_device: int) -> None:
    """Rank > 0: sends its MR info to the server and verifies the RDMA write."""
    engine = build_engine(cuda_device)

    # Exchange addresses.
    my_addr = engine.main_address
    addr_list: list[Any] = [None] * world_size
    dist.all_gather_object(addr_list, my_addr)
    server_addr = addr_list[0]
    logger.info("Server address: %s", server_addr)

    # Register receive buffer.
    cuda_buf = torch.zeros(
        CUDA_BUF_SIZE, dtype=torch.uint8, device=f"cuda:{cuda_device}"
    )
    _, cuda_mr_desc = engine.register_tensor(cuda_buf)

    # Setup bouncing RECV to catch the server's ACK.
    recv_queue: queue.Queue[bytes] = queue.Queue()
    engine.submit_bouncing_recvs(
        count=1,
        len=MESSAGE_BUF_SIZE,
        on_recv=recv_queue.put,
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
    recv_queue.get()

    # Verify the data written by the server.
    gold = generate_random_bytes(seed - 1, length)
    buf = cuda_buf[offset : offset + length].to("cpu")
    assert torch.equal(gold, buf), f"Data mismatch on rank {rank}"
    logger.info("Rank %d: verified successfully", rank)

    dist.barrier()
    logger.info("Rank %d: done.", rank)


def main() -> None:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    try:
        if rank == 0:
            run_server(rank, world_size, local_rank)
        else:
            run_client(rank, world_size, local_rank)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
