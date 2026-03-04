"""Ablation study and weight sensitivity analysis for the enhanced PINN loss.

Responds to reviewer comment 7:
  "The design of the enhanced loss function (gradient-norm, mean-guidance)
   lacks theoretical justification, and no sensitivity analysis is provided."

Experiments
-----------
1. Ablation:   4 configs that isolate the contribution of each new term.
2. Sensitivity: grid search over w_sob and w_mean to test robustness.

Results are printed as tables and saved to ablation_results.csv.
"""

import csv
import time

import numpy as np
import scipy.stats as si
import tensorflow as tf
import matplotlib.pyplot as plt

from pinn_wd import physics_informed_nn_wd


# ── Problem constants (must match main.ipynb) ────────────────────────────────
R                        = 0.05
Q1, Q2, Q3              = 0.01, 0.02, 0.03
SIG1, SIG2, SIG3        = 0.1, 0.2, 0.3
ALPHA1, ALPHA2, ALPHA3  = 0.2, 0.3, 0.5
T, K                     = 1.0, 17.5
S_RANGE                  = (15.0, 20.0)
T_RANGE                  = (0.01, 0.99)

A11, A22, A33 = SIG1**2, SIG2**2, SIG3**2
A12, A13, A23 = SIG1*SIG2, SIG1*SIG3, SIG2*SIG3

SIG2_HAT = (A11*ALPHA1**2 + A22*ALPHA2**2 + A33*ALPHA3**2
            + 2*A12*ALPHA1*ALPHA2 + 2*A13*ALPHA1*ALPHA3
            + 2*A23*ALPHA2*ALPHA3)
Q_HAT = (ALPHA1*(Q1 + A11/2) + ALPHA2*(Q2 + A22/2)
         + ALPHA3*(Q3 + A33/2) - SIG2_HAT/2)


# ── Data helpers ─────────────────────────────────────────────────────────────

def normalize(X, lb, ub):
    return 2.0 * (X - lb) / (ub - lb) - 1.0


def bs_exact(X):
    S1, S2, S3, t = X[:, 0:1], X[:, 1:2], X[:, 2:3], X[:, 3:4]
    tau = T - t
    d1 = (np.log(S1**ALPHA1 * S2**ALPHA2 * S3**ALPHA3 / K)
          + (R - Q_HAT + SIG2_HAT/2) * tau) / np.sqrt(SIG2_HAT * tau)
    d2 = d1 - np.sqrt(SIG2_HAT * tau)
    from scipy.stats import norm
    return (np.exp(-Q_HAT * tau) * S1**ALPHA1 * S2**ALPHA2 * S3**ALPHA3 * norm.cdf(d1)
            - np.exp(-R * tau) * K * norm.cdf(d2))


def make_data(n, m, seed):
    rng = np.random.RandomState(seed)

    X_col = np.column_stack([rng.uniform(*S_RANGE, n) for _ in range(3)]
                             + [rng.uniform(*T_RANGE, n)])
    u_col = bs_exact(X_col)

    X_term = np.column_stack([rng.uniform(*S_RANGE, m) for _ in range(3)]
                              + [np.ones(m)])
    u_term = np.maximum(
        X_term[:, 0:1]**ALPHA1 * X_term[:, 1:2]**ALPHA2 * X_term[:, 2:3]**ALPHA3 - K, 0
    )

    t_bc = rng.uniform(*T_RANGE, m)
    X_bc1 = np.column_stack([np.full(m, S_RANGE[0]), np.full(m, S_RANGE[1]),
                               np.full(m, S_RANGE[1]), t_bc])
    X_bc2 = np.column_stack([np.full(m, S_RANGE[1]), np.full(m, S_RANGE[0]),
                               np.full(m, S_RANGE[0]),
                               rng.uniform(*T_RANGE, m)])

    X_all = np.vstack([X_col, X_term, X_bc1, X_bc2])
    lb, ub = X_all.min(axis=0), X_all.max(axis=0)

    return (normalize(X_col,  lb, ub), u_col,
            normalize(X_term, lb, ub), u_term,
            normalize(X_bc1,  lb, ub), bs_exact(X_bc1),
            normalize(X_bc2,  lb, ub), bs_exact(X_bc2),
            lb, ub, X_col, u_col)


# ── One training run ─────────────────────────────────────────────────────────

def run_one(w_sob, w_mean, n_iter, seed, n=3000, m=1000):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    (X_col, u_col, X_term, u_term,
     X_bc1, u_bc1, X_bc2, u_bc2,
     lb, ub, X_col_raw, u_col_raw) = make_data(n, m, seed)

    model = physics_informed_nn_wd(
        X_col, u_col, lb, ub,
        X_term, u_term,
        X_bc1, u_bc1,
        X_bc2, u_bc2,
        w_sob=w_sob, w_mean=w_mean
    )
    model.train(Eq_name="ablation", n_iter=n_iter, print_every=n_iter)

    u_pred = model.predict(X_col)
    l2 = float(np.linalg.norm(u_col - u_pred, 2) / np.linalg.norm(u_col, 2))
    return l2


# ── Experiment settings ───────────────────────────────────────────────────────

N_ITER = 10000   # iterations per run (use 40001 for final paper results)
SEED   = 42
N_COL  = 3000
N_BC   = 1000

# ── 1. Ablation ───────────────────────────────────────────────────────────────

ABLATION_CONFIGS = [
    ("B: +Sobolev",  0.01,  0.0 ),   # gradient-norm regularisation only
    ("C: +Mean",     0.0,   0.1 ),   # mean-guidance only
]

print("=" * 60)
print(f"ABLATION STUDY  (n_iter={N_ITER}, seed={SEED})")
print(f"{'Config':<22} {'w_sob':>8} {'w_mean':>8} {'L2 Error':>12}")
print("-" * 60)

ablation_rows = []
for name, w_sob, w_mean in ABLATION_CONFIGS:
    t0 = time.time()
    l2 = run_one(w_sob, w_mean, N_ITER, SEED, N_COL, N_BC)
    elapsed = time.time() - t0
    ablation_rows.append((name, w_sob, w_mean, l2))
    print(f"{name:<22} {w_sob:>8.3f} {w_mean:>8.3f} {l2:>12.4e}  ({elapsed:.0f}s)")

print("=" * 60)

# ── 2. Weight sensitivity: w_sob ─────────────────────────────────────────────

WSOB_GRID  = [0.001, 0.01, 0.1]
WMEAN_FIX  = 0.1

print(f"\nSENSITIVITY: w_sob  (w_mean = {WMEAN_FIX} fixed)")
print(f"{'w_sob':>10} {'L2 Error':>12}")
print("-" * 25)

wsob_rows = []
for ws in WSOB_GRID:
    l2 = run_one(ws, WMEAN_FIX, N_ITER, SEED, N_COL, N_BC)
    wsob_rows.append((ws, l2))
    print(f"{ws:>10.3f} {l2:>12.4e}")

# ── 3. Weight sensitivity: w_mean ────────────────────────────────────────────

WMEAN_GRID = [0.01, 0.1, 1.0]
WSOB_FIX   = 0.01

print(f"\nSENSITIVITY: w_mean  (w_sob = {WSOB_FIX} fixed)")
print(f"{'w_mean':>10} {'L2 Error':>12}")
print("-" * 25)

wmean_rows = []
for wm in WMEAN_GRID:
    l2 = run_one(WSOB_FIX, wm, N_ITER, SEED, N_COL, N_BC)
    wmean_rows.append((wm, l2))
    print(f"{wm:>10.3f} {l2:>12.4e}")

# ── Save CSV ──────────────────────────────────────────────────────────────────

with open("ablation_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["experiment", "config", "w_sob", "w_mean", "L2_error"])
    for name, ws, wm, l2 in ablation_rows:
        w.writerow(["ablation", name, ws, wm, l2])
    for ws, l2 in wsob_rows:
        w.writerow(["sensitivity_wsob", f"w_sob={ws}", ws, WMEAN_FIX, l2])
    for wm, l2 in wmean_rows:
        w.writerow(["sensitivity_wmean", f"w_mean={wm}", WSOB_FIX, wm, l2])

print("\nResults saved → ablation_results.csv")

# ── Sensitivity plots ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.semilogx([r[0] for r in wsob_rows], [r[1] for r in wsob_rows],
            'bo-', linewidth=1.5, markersize=7)
ax.set_xlabel(r'$w_\mathrm{sob}$')
ax.set_ylabel('Relative L2 Error')
ax.set_title(r'Sensitivity to $w_\mathrm{sob}$  ($w_\mathrm{mean}=0.1$)')
ax.grid(True)

ax = axes[1]
ax.semilogx([r[0] for r in wmean_rows], [r[1] for r in wmean_rows],
            'rs-', linewidth=1.5, markersize=7)
ax.set_xlabel(r'$w_\mathrm{mean}$')
ax.set_ylabel('Relative L2 Error')
ax.set_title(r'Sensitivity to $w_\mathrm{mean}$  ($w_\mathrm{sob}=0.01$)')
ax.grid(True)

plt.tight_layout()
plt.savefig("sensitivity_analysis.png", dpi=300, bbox_inches='tight')
print("Sensitivity plot saved → sensitivity_analysis.png")
plt.show()
