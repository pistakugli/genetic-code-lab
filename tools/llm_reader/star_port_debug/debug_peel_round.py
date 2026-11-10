"""Debug driver: per-round trace of the peel loop on the small teacher."""
import sys
sys.path.insert(0, "/home/z/my-project/gcl-landing/tools/llm_reader")
import numpy as np
from swiglu_head_to_head import (
    Oracle, make_teacher, sample_inputs, estimate_odd_map, active_spans,
    oddsvd_candidates, refine_single_atom, silu, psi,
)

atom_count = 16
dim, gates, ups = make_teacher(atom_count, 260621)
oracle = Oracle(gates, ups)
rng = np.random.default_rng(260622)

x_span = sample_inputs(rng, 1000, dim)
y_plus = oracle(x_span); y_minus = oracle(-x_span)
odd = 0.5 * (y_plus - y_minus)
L, *_ = np.linalg.lstsq(x_span, odd, rcond=None)
L = L.T
print("odd map relative Frobenius error:", np.linalg.norm(L - 0.5 * ups.T @ gates) / np.linalg.norm(0.5 * ups.T @ gates))

U, V = active_spans(L, dim)
# ground truth odd map svd for comparison
Lt = 0.5 * ups.T @ gates
sv_true = np.linalg.svd(Lt, compute_uv=False)
sv_est = np.linalg.svd(L, compute_uv=False)
print("top-6 true odd svd:", np.round(sv_true[:6], 3))
print("top-6 est  odd svd:", np.round(sv_est[:6], 3))

x_fit = sample_inputs(rng, 6000, dim)
r_fit = oracle(x_fit)
base_energy = float(np.mean(r_fit * r_fit))
print("base residual energy:", round(base_energy, 4))

pairs = oddsvd_candidates(L, 6)
for k, (up_dir, gate_dir, sigma) in enumerate(pairs):
    a_init = up_dir  # U.T @ (U @ up_dir)
    b_init = gate_dir
    a, b, loss = refine_single_atom(x_fit, r_fit, U, V, a_init, b_init)
    up = U @ a; gate = V @ b
    # what fraction of residual energy does this fitted atom explain?
    pred = silu(x_fit @ gate)[:, None] * up[None, :]
    err = r_fit - pred
    drop = 1 - float(np.mean(err * err)) / base_energy
    # true-atom best match
    cos_g = np.abs(gates @ gate) / np.linalg.norm(gates, axis=1)
    cos_u = np.abs(ups @ up) / np.linalg.norm(ups, axis=1)
    joint = np.minimum(cos_g, cos_u)
    best = int(np.argmax(joint))
    print(f"pair {k}: sigma={sigma:.3f} refined_drop={drop:.4f} best_joint_cos={joint[best]:.4f} (true atom {best})")
