"""Degree-preserving double-edge swaps on sparse node--hyperedge incidences."""

import random

import torch
from torch import Tensor


def rewire_incidence(hyperedge_index: Tensor, num_nodes: int, num_hyperedges: int,
                     seed: int = 42, multiplier: int = 5) -> tuple[Tensor, dict]:
    """Target multiplier * |I| successful swaps, with at most 100 trials/target.

    Incidences are a set, as in the H-IR compiler. IDs and empty node/edge slots
    are preserved. changed_incidence_ratio is |I_original - I_final| / |I_original|,
    i.e. the fraction of original memberships absent after all swaps.
    Only a local Python RNG is used; features and labels are never inputs.
    """
    if hyperedge_index.ndim != 2 or hyperedge_index.shape[0] != 2:
        raise ValueError("hyperedge_index must have shape [2, num_incidences]")
    if multiplier < 0 or num_nodes < 0 or num_hyperedges < 0:
        raise ValueError("multiplier and node/edge counts must be nonnegative")
    original = set(map(tuple, hyperedge_index.detach().cpu().t().tolist()))
    if any(not (0 <= node < num_nodes and 0 <= edge < num_hyperedges)
           for node, edge in original):
        raise ValueError("incidence IDs must lie within the declared node/edge counts")
    pairs = sorted(original)
    present = set(original)
    count = len(pairs)
    target = multiplier * count
    max_attempts = max(1000, 100 * target)
    successful = attempts = 0
    rng = random.Random(seed)
    while successful < target and attempts < max_attempts and count >= 2:
        attempts += 1
        i, j = rng.sample(range(count), 2)
        u, e1 = pairs[i]
        v, e2 = pairs[j]
        if u == v or e1 == e2 or (u, e2) in present or (v, e1) in present:
            continue
        present.remove((u, e1))
        present.remove((v, e2))
        pairs[i], pairs[j] = (u, e2), (v, e1)
        present.add(pairs[i])
        present.add(pairs[j])
        successful += 1

    if successful == target:
        reason = "target reached"
    elif count < 2:
        reason = "fewer than two distinct incidences; no swap is possible"
    else:
        reason = "attempt limit reached; candidates share an endpoint or would duplicate an incidence"
    stats = dict(num_nodes=num_nodes, num_hyperedges=num_hyperedges,
                 num_incidences=count, target_swaps=target, successful_swaps=successful,
                 attempts=attempts, max_attempts=max_attempts, stop_reason=reason,
                 changed_incidence_ratio=len(original - present) / count if count else 0.0)
    rewired = torch.tensor(sorted(present), dtype=torch.long,
                           device=hyperedge_index.device).reshape(-1, 2).t().contiguous()
    return rewired, stats
