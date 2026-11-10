"""Per-round trace of the starport peel, mirroring peel() exactly."""
import sys
sys.path.insert(0, "/home/z/my-project/gcl-landing/tools/llm_reader")
import numpy as np
from swiglu_head_to_head import (
    Oracle, make_teacher, sample_inputs, active_spans,
    quartic_gate, refine_single_atom, silu,
)

atom_count = 16
dim, gates, ups = make_teacher(atom_count, 260621)
oracle = Oracle(gates, ups)
rng = np.random.default_rng(260622)
x_span = sample_inputs(rng, 1000, dim)
odd = 0.5 * (oracle(x_span) - oracle(-x_span))
L, *_ = np.linalg.lstsq(x_span, odd, rcond=None)
U, V = active_spans(L.T, dim)

atoms = []

def model_eval(x):
    if not atoms:
        return np.zeros((x.shape[0], dim))
    out = np.zeros((x.shape[0], dim))
    for up, gate in atoms:
        out += silu(x @ gate)[:, None] * up[None, :]
    return out

def residual_fn(x):
    return oracle(x) - model_eval(x)

x_verify = sample_inputs(rng, 4000, dim)
y_verify = residual_fn(x_verify)
energy = float(np.mean(y_verify * y_verify))
total = energy

for rnd in range(atom_count):
    x_fit = sample_inputs(rng, 8000, dim)
    r_fit = residual_fn(x_fit)
    d = quartic_gate(residual_fn, V, s0=0.8, rng=rng, steps=10)
    gate_w = V @ d
    phi = silu(x_fit @ gate_w)
    up_w = (phi @ r_fit) / max(float(phi @ phi), 1e-30)
    a, b, loss = refine_single_atom(x_fit, r_fit, U, V, U.T @ up_w, V.T @ gate_w)
    up, gate = U @ a, V @ b
    phi2 = silu(x_fit @ gate)
    up = (phi2 @ r_fit) / max(float(phi2 @ phi2), 1e-30)
    cosg0 = np.abs(gates @ gate_w)
    cosg1 = np.abs(gates @ gate)
    cosu1 = np.abs(ups @ up) / np.linalg.norm(ups, axis=1)
    i0 = int(np.argmax(cosg0)); i1 = int(np.argmax(cosg1))
    atom_pred = silu(x_verify @ gate)[:, None] * up[None, :]
    pe = float(np.mean(atom_pred * atom_pred))
    ra = y_verify - atom_pred
    ea = float(np.mean(ra * ra))
    explained = (energy - ea) / max(pe, 1e-30)
    amp = pe / total
    joint = np.minimum(cosg1, cosu1)[i1]
    print(f"r{rnd:02d}: quartic {cosg0[i0]:.3f}(a{i0}) refined {cosg1[i1]:.3f}(a{i1}) "
          f"| explained {explained:.3f} | amp {amp:.4f} | |gate| {np.linalg.norm(gate):.3f} | bestJoint {joint:.3f}")
    atoms.append((up, gate))
    y_verify = ra
    energy = ea

# final grade
rec_g = np.stack([g for _, g in atoms]); rec_u = np.stack([u for u, _ in atoms])
paired = 0
for i in range(atom_count):
    cg = np.abs(rec_g @ gates[i]); cu = np.abs(rec_u @ ups[i]) / np.linalg.norm(rec_u, axis=1)
    j = np.minimum(cg, cu)
    if j.max() >= 0.99: paired += 1
print("paired:", paired, "/", atom_count)
