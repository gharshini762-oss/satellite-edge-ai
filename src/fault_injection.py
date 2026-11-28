import os
import sys
import csv
import json
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models import build_resnet18
from src.policy import softmax_p, policy
from src.verify_weights import verify_phase4_weights


def load_test_bundle(device="cuda"):
    """Load normalized test tensors onto GPU and aligned manifest arrays."""
    with open("results/norm_stats.json", encoding="utf-8") as f:
        stats = json.load(f)
    mean = stats.get("mean") or stats.get("train_mean")
    std = stats.get("std") or stats.get("train_std")

    X_raw = np.load("data/processed/test_X.npy")
    y_np = np.load("data/processed/test_y.npy").astype(np.int64)

    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    X_norm = (torch.from_numpy(X_raw.astype(np.float32)) / 10000.0).clamp(0.0, 1.0)
    X_norm = (X_norm - mean_t) / std_t
    X_gpu = X_norm.to(device)

    with open("results/manifest_export.csv", encoding="utf-8") as f:
        test_rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    assert len(test_rows) == len(y_np), f"Manifest mismatch: {len(test_rows)} vs {len(y_np)}"

    man_y = np.array([int(r["label"]) for r in test_rows], dtype=np.int64)
    assert np.array_equal(man_y, y_np), "Manifest test labels do not align with test_y.npy!"

    cloud_pct = np.array([float(r["cloud_pct"]) for r in test_rows], dtype=np.float64)
    clear_pct = np.array([float(r["clear_pct"]) for r in test_rows], dtype=np.float64)
    roi_ids = np.array([r["roi_id"] for r in test_rows])
    return X_gpu, y_np, cloud_pct, clear_pct, roi_ids


def get_target_weight_names(model):
    """Return weight parameter names for Conv2d and Linear layers only."""
    names = []
    for mod_name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            names.append(f"{mod_name}.weight" if mod_name else "weight")
    return names


def sample_unique_positions(total_slots, n_flips, rng):
    """Sample n_flips unique integer positions in [0, total_slots) with O(n) memory."""
    if n_flips <= 0:
        return np.empty(0, dtype=np.int64)
    if n_flips > total_slots:
        raise ValueError(f"Requested {n_flips} flips > {total_slots} total slots")
    chosen = np.unique(rng.integers(0, total_slots, size=n_flips, dtype=np.int64))
    while len(chosen) < n_flips:
        need = n_flips - len(chosen)
        extra = rng.integers(0, total_slots, size=max(need * 2, 16), dtype=np.int64)
        chosen = np.unique(np.concatenate([chosen, extra]))
    if len(chosen) > n_flips:
        chosen = rng.choice(chosen, size=n_flips, replace=False)
    return chosen


def precompute_int8_cache(clean_state, target_names):
    """Per-channel symmetric INT8 quantization along output channel dim=0."""
    cache = {}
    for name in target_names:
        w = clean_state[name].float().cpu()
        reduce_dims = tuple(range(1, w.ndim))
        scale = w.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8) / 127.0
        q = torch.round(w / scale).clamp(-128, 127).to(torch.int8).numpy().copy()
        cache[name] = (q, scale)
    return cache


def _build_layout(clean_state, target_names):
    counts = np.array([clean_state[n].numel() for n in target_names], dtype=np.int64)
    offsets = np.cumsum(counts)
    return counts, offsets, int(offsets[-1])


def inject_fp32_flips(clean_state, target_names, rng, ber=None, n_flips=None, bit_positions=None):
    """Flip bits in FP32 Conv2d/Linear weights without mutating clean_state."""
    bits_allowed = np.array(list(range(32)) if bit_positions is None else bit_positions, dtype=np.int64)
    n_b = len(bits_allowed)
    _, offsets, total_elems = _build_layout(clean_state, target_names)
    total_slots = total_elems * n_b

    if n_flips is None:
        assert ber is not None, "Specify either ber or n_flips"
        n_flips = int(rng.binomial(total_slots, ber))

    out_state = dict(clean_state)
    if n_flips == 0:
        return out_state, 0

    slots = sample_unique_positions(total_slots, n_flips, rng)
    flat_elems = slots // n_b
    bit_idx = bits_allowed[slots % n_b]

    tensor_ids = np.searchsorted(offsets, flat_elems, side="right")
    for tid in np.unique(tensor_ids):
        name = target_names[tid]
        base = 0 if tid == 0 else int(offsets[tid - 1])
        mask = tensor_ids == tid
        local_elems = flat_elems[mask] - base
        local_bits = bit_idx[mask]

        arr = clean_state[name].detach().cpu().numpy().astype(np.float32, copy=True)
        u32 = arr.view(np.uint32).reshape(-1)
        for ei, bi in zip(local_elems, local_bits):
            u32[ei] ^= np.uint32(1) << np.uint32(bi)
        out_state[name] = torch.from_numpy(arr)

    return out_state, int(n_flips)


def inject_int8_flips(clean_state, int8_cache, target_names, rng, ber=None, n_flips=None, bit_positions=None):
    """Flip bits in simulated per-channel INT8 weights, then dequantize to float32."""
    bits_allowed = np.array(list(range(8)) if bit_positions is None else bit_positions, dtype=np.int64)
    n_b = len(bits_allowed)
    _, offsets, total_elems = _build_layout(clean_state, target_names)
    total_slots = total_elems * n_b

    if n_flips is None:
        assert ber is not None, "Specify either ber or n_flips"
        n_flips = int(rng.binomial(total_slots, ber))

    out_state = dict(clean_state)
    # Always load dequantized INT8 weights for every target layer (even untouched ones)
    q_copies = {}
    for name in target_names:
        q, _ = int8_cache[name]
        q_copies[name] = q.copy()

    if n_flips > 0:
        slots = sample_unique_positions(total_slots, n_flips, rng)
        flat_elems = slots // n_b
        bit_idx = bits_allowed[slots % n_b]
        tensor_ids = np.searchsorted(offsets, flat_elems, side="right")

        for tid in np.unique(tensor_ids):
            name = target_names[tid]
            base = 0 if tid == 0 else int(offsets[tid - 1])
            mask = tensor_ids == tid
            local_elems = flat_elems[mask] - base
            local_bits = bit_idx[mask]

            u8 = q_copies[name].view(np.uint8).reshape(-1)
            for ei, bi in zip(local_elems, local_bits):
                u8[ei] ^= np.uint8(1 << int(bi))

    for name in target_names:
        _, scale = int8_cache[name]
        out_state[name] = torch.from_numpy(q_copies[name]).float() * scale

    return out_state, int(n_flips)


@torch.no_grad()
def evaluate_state(model, state_dict, X_gpu, y_np, cloud_pct, clear_pct, batch_size=64):
    """Evaluate a state_dict on GPU-resident test set with explicit NaN/Inf protection."""
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    chunks = []
    n = X_gpu.shape[0]
    for i in range(0, n, batch_size):
        chunks.append(model(X_gpu[i : i + batch_size]).float().cpu().numpy())
    logits = np.concatenate(chunks, axis=0)

    finite = np.isfinite(logits).all(axis=1)
    non_finite_frac = float((~finite).mean())

    # Non-finite predictions count strictly as incorrect (never class 0)
    preds = np.full(n, -1, dtype=np.int64)
    if finite.any():
        preds[finite] = logits[finite].argmax(axis=1)
    acc = float((preds == y_np).mean())

    # Safe probabilities: non-finite tiles become NaN (not discarded by p >= tau)
    p = np.full(n, np.nan, dtype=np.float64)
    if finite.any():
        p[finite] = softmax_p(logits[finite])

    pol_08 = policy(p, cloud_pct, clear_pct, tau=0.8)
    pol_05 = policy(p, cloud_pct, clear_pct, tau=0.5)

    return {
        "acc": acc,
        "non_finite_frac": non_finite_frac,
        "fd_08": pol_08["false_deletion"],
        "bw_08": pol_08["bandwidth_saved"],
        "fd_05": pol_05["false_deletion"],
        "bw_05": pol_05["bandwidth_saved"],
        "logits": logits,
    }


def run_unit_tests():
    verify_phase4_weights(["results/ckpt_seed0.pt"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()

    X_gpu, y_np, cloud_pct, clear_pct, _ = load_test_bundle("cuda")
    model = build_resnet18().cuda().eval()
    clean_state = torch.load("results/ckpt_seed0.pt", map_location="cpu")
    target_names = get_target_weight_names(model)
    int8_cache = precompute_int8_cache(clean_state, target_names)

    total_w = sum(clean_state[n].numel() for n in target_names)
    print(f"Target layers: {len(target_names)} Conv2d/Linear weights | {total_w:,} weights")
    print(f"Total target bits: FP32 = {total_w * 32:,} | INT8 = {total_w * 8:,} (4.00x ratio)")

    # Test 1: BER=0 FP32 parity vs saved Phase 4 Seed-0 logits
    t0 = time.perf_counter()
    res_fp32 = evaluate_state(model, clean_state, X_gpu, y_np, cloud_pct, clear_pct)
    dt_fp32 = time.perf_counter() - t0
    ref_logits = np.load("results/test_logits_fp32_seed0.npy")
    max_diff = float(np.abs(res_fp32["logits"] - ref_logits).max())
    assert max_diff < 1e-4, f"FP32 BER=0 mismatch vs Phase 4 logits! max_diff={max_diff}"
    print(
        f"[PASS] FP32 BER=0 parity: max_diff={max_diff:.2e} | "
        f"acc={res_fp32['acc']:.4f} | fd@0.8={res_fp32['fd_08']:.4f} | fd@0.5={res_fp32['fd_05']:.4f} ({dt_fp32:.2f}s)"
    )

    # Test 2: Simulated per-channel INT8 BER=0 baseline
    rng = np.random.default_rng(0)
    int8_clean_state, n0 = inject_int8_flips(clean_state, int8_cache, target_names, rng, n_flips=0)
    assert n0 == 0
    res_int8 = evaluate_state(model, int8_clean_state, X_gpu, y_np, cloud_pct, clear_pct)
    agree = float((res_int8["logits"].argmax(1) == ref_logits.argmax(1)).mean())
    print(
        f"[PASS] Sim-INT8 BER=0 clean: acc={res_int8['acc']:.4f} (agree vs FP32={agree:.4f}) | "
        f"fd@0.8={res_int8['fd_08']:.4f} | fd@0.5={res_int8['fd_05']:.4f}"
    )

    # Test 3: Exact Hamming distance & clean_state immutability
    before_sum = float(clean_state[target_names[0]].sum())
    st_f, nf = inject_fp32_flips(clean_state, target_names, np.random.default_rng(42), n_flips=100)
    assert nf == 100
    hamming_fp32 = sum(
        int(np.bitwise_count(
            clean_state[n].numpy().view(np.uint32) ^ st_f[n].numpy().view(np.uint32)
        ).sum())
        for n in target_names
    )
    assert hamming_fp32 == 100, f"Expected Hamming distance 100, got {hamming_fp32}"
    assert float(clean_state[target_names[0]].sum()) == before_sum, "clean_state was mutated!"
    print(f"[PASS] FP32 exact Hamming distance = {hamming_fp32} bits; clean_state untouched")

    # Test 4: Smoke test single exponent flip vs INT8 MSB flip
    st_exp, _ = inject_fp32_flips(clean_state, target_names, np.random.default_rng(7), n_flips=10, bit_positions=[30])
    res_exp = evaluate_state(model, st_exp, X_gpu, y_np, cloud_pct, clear_pct)
    st_i8, _ = inject_int8_flips(clean_state, int8_cache, target_names, np.random.default_rng(7), n_flips=10, bit_positions=[7])
    res_i8 = evaluate_state(model, st_i8, X_gpu, y_np, cloud_pct, clear_pct)
    print(
        f"[PASS] 10-flip MSB smoke test | "
        f"FP32 bit-30: acc={res_exp['acc']:.4f}, non_finite={res_exp['non_finite_frac']:.4f} | "
        f"INT8 bit-7: acc={res_i8['acc']:.4f}, non_finite={res_i8['non_finite_frac']:.4f}"
    )
    print(f"Estimated Day 31 runtime (~300 GPU evals): ~{dt_fp32 * 300 / 60:.1f} minutes")


if __name__ == "__main__":
    run_unit_tests()