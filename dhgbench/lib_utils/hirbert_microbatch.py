"""Bounded packet batches and CUDA memory feedback for shared H-IR training."""

import math
from dataclasses import asdict, dataclass

import torch


def packet_cost(batch):
    chunks = batch.view.chunks + batch.view.relation_chunks
    return (sum(len(chunk.tokens) for chunk in chunks) + batch.membership_pairs[:, 0].unique().numel(),
            len(batch.view.x), len(chunks))


@dataclass
class BatchState:
    packets: int
    chunk_batch_size: int
    forward_tokens: int
    successful_batches: int = 0
    cooldown: int = 0
    oom_retries: int = 0


class MicroBatchPolicy:
    def __init__(self, args):
        self.args = args
        self.cuda = torch.device(args.device).type == 'cuda'
        self.state = BatchState(args.micro_batch_size, args.hir_chunk_batch_size,
                                min(args.bert_max_batch_tokens, args.hir_chunk_batch_size * args.hir_max_seq_length))
        self.budget = args.gpu_memory_budget_gb * 1024 ** 3
        self.target = args.gpu_target_memory_gb * 1024 ** 3
        if self.cuda:
            # Bound the process allocator, in addition to packet/token limits.
            # CUDA context/other processes are accounted for by live free memory.
            total = torch.cuda.get_device_properties(args.device).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, self.budget / total), args.device)

    def state_dict(self):
        return asdict(self.state)

    def load_state_dict(self, state):
        self.state = BatchState(**state)

    def fits(self, costs, cost):
        return (len(costs) < self.state.packets and
                sum(value[0] for value in costs) + cost[0] <= self.args.micro_batch_max_tokens and
                sum(value[1] for value in costs) + cost[1] <= self.args.micro_batch_max_nodes and
                sum(value[2] for value in costs) + cost[2] <= self.args.micro_batch_max_chunks)

    def reduce(self):
        """Keep local structure intact before resorting to smaller neighborhoods."""
        state = self.state
        state.oom_retries += 1
        state.cooldown = self.args.micro_batch_tune_interval * 4
        if state.packets > 1:
            state.packets = max(1, state.packets // 2)
        elif state.chunk_batch_size > 1:
            state.chunk_batch_size = max(1, state.chunk_batch_size // 2)
        elif state.forward_tokens > self.args.hir_max_seq_length:
            state.forward_tokens = max(self.args.hir_max_seq_length, state.forward_tokens // 2)
        else:
            return False
        return True

    def observe(self, allocated_peak, reserved_peak, packets):
        state = self.state
        state.successful_batches += 1
        if not self.cuda or not self.args.auto_micro_batch:
            return
        free, _ = torch.cuda.mem_get_info(self.args.device)
        reserved = torch.cuda.memory_reserved(self.args.device)
        ceiling = min(self.budget, max(0, free + reserved - 512 * 1024 ** 2))
        target = min(self.target, ceiling * .9)
        if allocated_peak > target * 1.08 or reserved_peak > ceiling:
            if state.packets > 1:
                state.packets = max(1, state.packets // 2)
            elif state.chunk_batch_size > 1:
                state.chunk_batch_size = max(1, state.chunk_batch_size // 2)
            state.cooldown = self.args.micro_batch_tune_interval
            if reserved_peak > ceiling:
                torch.cuda.empty_cache()
            return
        if state.cooldown:
            state.cooldown -= 1
            return
        if state.successful_batches % self.args.micro_batch_tune_interval or packets < min(2, state.packets):
            return  # A small final batch is not a memory calibration batch.
        if allocated_peak < target * .9:
            if packets >= state.packets:
                state.packets = min(self.args.micro_batch_max_size, max(state.packets + 1, math.ceil(state.packets * 1.5)))
            # Grow actual BERT parallelism as well as retained packet graphs.
            # This increases useful tokens/call, rather than adding padding.
            state.chunk_batch_size = min(self.args.bert_max_chunk_batch_size, max(1, state.chunk_batch_size * 2))
            state.forward_tokens = min(self.args.bert_max_batch_tokens,
                                       state.chunk_batch_size * self.args.hir_max_seq_length)


def synchronize(args):
    if torch.device(args.device).type == 'cuda':
        torch.cuda.synchronize(args.device)


def memory_peaks(args):
    if torch.device(args.device).type != 'cuda':
        return 0.0, 0.0
    return torch.cuda.max_memory_allocated(args.device), torch.cuda.max_memory_reserved(args.device)
