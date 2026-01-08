import math
from loguru import logger
import datetime
import torch
import torch.distributed as dist

def get_parallel_degree(world_size, num_heads):
    # ulysses_degree is faster, and must be a divisor of num_heads
    ulysses_degree = math.gcd(world_size, num_heads)
    ring_degree = world_size // ulysses_degree
    return ulysses_degree, ring_degree

def get_device(ulysses_degree, ring_degree):
    if ulysses_degree > 1 or ring_degree > 1:
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
            get_world_group,
        )

        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=24*7))
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=ring_degree, 
            ulysses_degree=ulysses_degree
        )

        device = torch.device(f"cuda:{get_world_group().rank}")
        torch.cuda.set_device(get_world_group().rank)

        logger.info(f'rank={get_world_group().rank} device={str(device)}')
    else:
        device = "cuda"
    return device