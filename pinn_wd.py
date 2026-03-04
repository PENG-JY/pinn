import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

from FNN_original import Network_FNN


class physics_informed_nn_wd:
    """PINN for 3-asset geometric basket call option (Black-Scholes PDE).

    Loss = L_pde + L_term + L_bc1 + L_bc2
         + w_sob  * L_sobolev   (gradient-norm regularisation)
         + w_mean * L_mean      (first-moment consistency)

    Parameters
    ----------
    w_sob  : float, weight for Sobolev gradient-norm term (default 0.01)
    w_mean : float, weight for mean-guidance term         (default 0.10)
    """

    # PDE parameters
    r                   = 0.05
    q1, q2, q3         = 0.01, 0.02, 0.03
    sigma1, sigma2, sigma3 = 0.1, 0.2, 0.3

    def __init__(self, X_col, u_col, lb, ub,
                 X_term, u_term,
                 X_bc1, u_bc1,
                 X_bc2, u_bc2,
                 w_sob=0.01, w_mean=0.10, w_bc=0.1):

        self.lb, self.ub   = lb, ub
        self.lb_tf = tf.constant(lb, dtype=tf.float32)
        self.ub_tf = tf.constant(ub, dtype=tf.float32)
        self.w_sob         = w_sob
        self.w_mean        = w_mean
        self.w_bc          = w_bc

        # Output clamping: shifts initial prediction to mid-range so the
        # network cannot trivially collapse to u=0 (original approach).
        self.upper_bound = float(np.max(u_col)) + 1.0
        self.u_base      = self.upper_bound * 0.5

        def t(arr): return tf.cast(tf.convert_to_tensor(arr), tf.float32)

        # Interior collocation points (PDE residual + L2 monitoring)
        self.X    = t(X_col)
        self.s1   = t(X_col[:, 0:1])
        self.s2   = t(X_col[:, 1:2])
        self.s3   = t(X_col[:, 2:3])
        self.t_   = t(X_col[:, 3:4])
        self.u    = t(u_col)

        # Terminal condition (t = T)
        self.X2   = t(X_term)
        self.l1   = t(X_term[:, 0:1])
        self.l2   = t(X_term[:, 1:2])
        self.l3   = t(X_term[:, 2:3])
        self.p    = t(X_term[:, 3:4])
        self.u2   = t(u_term)

        # Boundary conditions (store full tensors for direct Neural_Net calls)
        self.X_bc1  = t(X_bc1)
        self.bc1_s1 = t(X_bc1[:, 0:1]);  self.bc1_s2 = t(X_bc1[:, 1:2])
        self.bc1_s3 = t(X_bc1[:, 2:3]);  self.bc1_t  = t(X_bc1[:, 3:4])
        self.u_bc1  = t(u_bc1)

        self.X_bc2  = t(X_bc2)
        self.bc2_s1 = t(X_bc2[:, 0:1]);  self.bc2_s2 = t(X_bc2[:, 1:2])
        self.bc2_s3 = t(X_bc2[:, 2:3]);  self.bc2_t  = t(X_bc2[:, 3:4])
        self.u_bc2  = t(u_bc2)

        self.Neural_Net = Network_FNN()

    # ── Forward pass ──────────────────────────────────────────────────────────

    def _call_net(self, X):
        """Evaluate network with output shifted to mid-range and clamped.

        u = clip(u_raw + u_base, 0, upper_bound)

        The u_base shift means the network starts at ~upper_bound/2,
        making it impossible to trivially satisfy the PDE with u≈0.
        """
        u_raw = self.Neural_Net(X)
        return tf.clip_by_value(u_raw + self.u_base, 0.0, self.upper_bound)

    def net_Eq(self, s1, s2, s3, t):
        """Black-Scholes PDE residual in raw (unnormalized) coordinates.

        Network inputs s1, s2, s3, t are normalised to [-1, 1], but the PDE
        coefficients (sigma*S_i) use raw asset prices S_i ∈ [lb, ub].
        Chain-rule scaling converts tape-computed derivatives from normalised
        to raw coordinates before substituting into the PDE.

            ∂u/∂S_i = (∂u/∂s_i) * c_i,   c_i = 2 / (ub_i - lb_i)
            ∂²u/∂S_i² = (∂²u/∂s_i²) * c_i²
            ∂²u/∂S_i∂S_j = (∂²u/∂s_i∂s_j) * c_i * c_j
        """
        lb = self.lb_tf
        ub = self.ub_tf
        # Scale factors: derivative w.r.t. raw coord = deriv w.r.t. norm * c
        c1 = 2.0 / (ub[0] - lb[0])
        c2 = 2.0 / (ub[1] - lb[1])
        c3 = 2.0 / (ub[2] - lb[2])
        ct = 2.0 / (ub[3] - lb[3])

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([s1, s2, s3, t])
            X = tf.concat([s1, s2, s3, t], axis=1)
            u = self._call_net(X)
            # Computed inside context so tape records these ops for 2nd-order
            grads1 = tape.gradient(u, [s1, s2, s3, t])
            u_s1, u_s2, u_s3, u_t = grads1

        u_s1s1 = tape.gradient(u_s1, s1)
        u_s2s2 = tape.gradient(u_s2, s2)
        u_s3s3 = tape.gradient(u_s3, s3)
        u_s1s2 = tape.gradient(u_s1, s2)
        u_s1s3 = tape.gradient(u_s1, s3)
        u_s2s3 = tape.gradient(u_s2, s3)
        del tape

        # Raw asset prices from normalised inputs
        S1 = (s1 + 1.0) / c1 + lb[0]
        S2 = (s2 + 1.0) / c2 + lb[1]
        S3 = (s3 + 1.0) / c3 + lb[2]

        # Chain-rule-corrected derivatives w.r.t. raw coordinates
        u_S1 = u_s1 * c1;   u_S2 = u_s2 * c2;   u_S3 = u_s3 * c3
        u_T  = u_t  * ct
        u_S1S1 = u_s1s1 * c1**2
        u_S2S2 = u_s2s2 * c2**2
        u_S3S3 = u_s3s3 * c3**2
        u_S1S2 = u_s1s2 * c1 * c2
        u_S1S3 = u_s1s3 * c1 * c3
        u_S2S3 = u_s2s3 * c2 * c3

        sig1, sig2, sig3 = self.sigma1, self.sigma2, self.sigma3
        return (
            u_T
            + 0.5 * (sig1**2 * S1**2 * u_S1S1
                   + sig2**2 * S2**2 * u_S2S2
                   + sig3**2 * S3**2 * u_S3S3
                   + 2 * sig1 * sig2 * S1 * S2 * u_S1S2
                   + 2 * sig1 * sig3 * S1 * S3 * u_S1S3
                   + 2 * sig2 * sig3 * S2 * S3 * u_S2S3)
            + (self.r - self.q1) * S1 * u_S1
            + (self.r - self.q2) * S2 * u_S2
            + (self.r - self.q3) * S3 * u_S3
            - self.r * u
        )

    # ── Loss ──────────────────────────────────────────────────────────────────

    def loss(self):
        """Compute total loss and its components."""
        # PDE residual
        eq = self.net_Eq(self.s1, self.s2, self.s3, self.t_)
        loss_pde = tf.reduce_mean(tf.square(eq))

        # Terminal condition
        u_term = self._call_net(self.X2)
        loss_term = tf.reduce_mean(tf.square(self.u2 - u_term))

        # Boundary conditions
        u_bc1 = self._call_net(self.X_bc1)
        u_bc2 = self._call_net(self.X_bc2)
        loss_bc1 = tf.reduce_mean(tf.square(self.u_bc1 - u_bc1))
        loss_bc2 = tf.reduce_mean(tf.square(self.u_bc2 - u_bc2))

        # Sobolev regularisation: gradient-norm penalty at terminal points.
        # Theoretical basis: the exact solution u ∈ H^1(Ω); penalising ||∇u||^2
        # at the terminal surface enforces smoothness (Czarnecki et al., 2017).
        with tf.GradientTape() as tape:
            tape.watch(self.X2)
            u_sob = self._call_net(self.X2)
        g = tape.gradient(u_sob, self.X2)
        loss_sob = tf.reduce_mean(tf.square(g)) if g is not None else 0.0

        # Mean-guidance: first-moment consistency on the terminal condition.
        # Penalises systematic bias in the predicted mean, improving calibration.
        loss_mean = tf.square(tf.reduce_mean(u_term) - tf.reduce_mean(self.u2))

        total = (loss_pde + loss_term
                 + self.w_bc   * (loss_bc1 + loss_bc2)
                 + self.w_sob  * loss_sob
                 + self.w_mean * loss_mean)
        return total, loss_pde, loss_term, loss_bc1, loss_bc2

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, Eq_name, n_iter, lr=1e-3, print_every=100):
        """Adam training loop.

        Returns
        -------
        MSE_history : list of total loss per iteration
        L2error_u   : list of relative L2 error (logged every print_every iters)
        """
        optimizer   = tf.keras.optimizers.Adam(learning_rate=lr)
        MSE_history = []
        L2error_u   = []

        for i in range(n_iter):
            with tf.GradientTape() as tape:
                total, l_pde, l_term, l_bc1, l_bc2 = self.loss()
            grads = tape.gradient(total, self.Neural_Net.variables)
            grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in grads]
            optimizer.apply_gradients(zip(grads, self.Neural_Net.variables))

            MSE_history.append(float(total))

            if i % print_every == 0 or i == n_iter - 1:
                u_pred = self.predict(self.X)
                rel_l2 = float(np.linalg.norm(self.u.numpy() - u_pred, 2)
                                / np.linalg.norm(self.u.numpy(), 2))
                L2error_u.append(rel_l2)
                print(f"[{i:05d}] total={float(total):.3e}  pde={float(l_pde):.3e}"
                      f"  term={float(l_term):.3e}  bc1={float(l_bc1):.3e}"
                      f"  bc2={float(l_bc2):.3e}  L2={rel_l2:.3e}")

        return MSE_history, L2error_u

    # ── L-BFGS fine-tuning ────────────────────────────────────────────────────

    def _get_flat(self):
        return tf.concat([tf.reshape(v, [-1]) for v in self.Neural_Net.variables], axis=0)

    def _set_flat(self, flat):
        idx = 0
        for v in self.Neural_Net.variables:
            n = tf.size(v)
            v.assign(tf.reshape(flat[idx:idx + n], v.shape))
            idx += n

    def _loss_and_grad_lbfgs(self, flat):
        self._set_flat(tf.cast(flat, tf.float32))
        with tf.GradientTape() as tape:
            total, *_ = self.loss()
        grads    = tape.gradient(total, self.Neural_Net.variables)
        flat_g   = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
        return float(total), flat_g.numpy().astype(np.float64)

    def train_lbfgs(self, maxiter=50000):
        """L-BFGS-B fine-tuning after Adam pre-training."""
        print("L-BFGS-B fine-tuning ...")
        x0 = self._get_flat().numpy().astype(np.float64)
        minimize(self._loss_and_grad_lbfgs, x0, jac=True, method='L-BFGS-B',
                 options={'maxiter': maxiter, 'maxcor': 50,
                          'ftol': np.finfo(float).eps})
        print("Done.")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X_star):
        """Return network prediction as a numpy array."""
        X = tf.cast(X_star, tf.float32)
        return self._call_net(X).numpy()

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def analyze_cross_section(self, fixed_s2=0.0, fixed_s3=0.0, fixed_t=-0.5):
        """Plot a 1-D cross-section u vs s1 at fixed s2, s3, t (normalised coords)."""
        import matplotlib.pyplot as plt
        s1 = np.linspace(-1.0, 1.0, 100).reshape(-1, 1)
        X_test = np.concatenate([s1,
                                  np.full_like(s1, fixed_s2),
                                  np.full_like(s1, fixed_s3),
                                  np.full_like(s1, fixed_t)], axis=1)
        u_pred = self.predict(X_test)

        fig, ax = plt.subplots()
        ax.plot(s1, u_pred, label='u_pred')
        ax.set_xlabel('s1 (normalised)'); ax.set_ylabel('u')
        ax.set_title('Cross-section: u vs s1')
        ax.legend(); ax.grid(True)
        fig.savefig('cross_section.png', dpi=300, bbox_inches='tight')
        plt.show()
