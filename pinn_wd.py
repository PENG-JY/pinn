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
                 w_sob=0.01, w_mean=0.10):

        self.lb, self.ub   = lb, ub
        self.w_sob         = w_sob
        self.w_mean        = w_mean

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

        # Boundary conditions
        self.bc1_s1 = t(X_bc1[:, 0:1]);  self.bc1_s2 = t(X_bc1[:, 1:2])
        self.bc1_s3 = t(X_bc1[:, 2:3]);  self.bc1_t  = t(X_bc1[:, 3:4])
        self.u_bc1  = t(u_bc1)

        self.bc2_s1 = t(X_bc2[:, 0:1]);  self.bc2_s2 = t(X_bc2[:, 1:2])
        self.bc2_s3 = t(X_bc2[:, 2:3]);  self.bc2_t  = t(X_bc2[:, 3:4])
        self.u_bc2  = t(u_bc2)

        self.Neural_Net = Network_FNN()

    # ── Forward pass ──────────────────────────────────────────────────────────

    def net_u(self, s1, s2, s3, t):
        """Evaluate u and its first-order partial derivatives."""
        X = tf.concat([s1, s2, s3, t], axis=1)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X)
            u = self.Neural_Net(X)
        dX = tape.gradient(u, X)
        del tape
        return u, dX[:, 0:1], dX[:, 1:2], dX[:, 2:3], dX[:, 3:4]

    def net_Eq(self, s1, s2, s3, t):
        """Black-Scholes PDE residual at given collocation points."""
        with tf.GradientTape(persistent=True) as D2:
            D2.watch([s1, s2, s3])
            u, u_s1, u_s2, u_s3, u_t = self.net_u(s1, s2, s3, t)
        u_s1s1 = D2.gradient(u_s1, s1)
        u_s2s2 = D2.gradient(u_s2, s2)
        u_s3s3 = D2.gradient(u_s3, s3)
        u_s1s2 = D2.gradient(u_s1, s2)
        u_s1s3 = D2.gradient(u_s1, s3)
        u_s2s3 = D2.gradient(u_s2, s3)
        del D2

        sig1, sig2, sig3 = self.sigma1, self.sigma2, self.sigma3
        return (
            u_t
            + 0.5 * (sig1**2 * s1**2 * u_s1s1
                   + sig2**2 * s2**2 * u_s2s2
                   + sig3**2 * s3**2 * u_s3s3
                   + 2 * sig1 * sig2 * s1 * s2 * u_s1s2
                   + 2 * sig1 * sig3 * s1 * s3 * u_s1s3
                   + 2 * sig2 * sig3 * s2 * s3 * u_s2s3)
            + (self.r - self.q1) * s1 * u_s1
            + (self.r - self.q2) * s2 * u_s2
            + (self.r - self.q3) * s3 * u_s3
            - self.r * u
        )

    # ── Loss ──────────────────────────────────────────────────────────────────

    def loss(self):
        """Compute total loss and its components."""
        # PDE residual
        eq = self.net_Eq(self.s1, self.s2, self.s3, self.t_)
        loss_pde = tf.reduce_mean(tf.square(eq))

        # Terminal condition
        u_term, *_ = self.net_u(self.l1, self.l2, self.l3, self.p)
        loss_term  = tf.reduce_mean(tf.square(self.u2 - u_term))

        # Boundary conditions
        u_bc1, *_ = self.net_u(self.bc1_s1, self.bc1_s2, self.bc1_s3, self.bc1_t)
        u_bc2, *_ = self.net_u(self.bc2_s1, self.bc2_s2, self.bc2_s3, self.bc2_t)
        loss_bc1  = tf.reduce_mean(tf.square(self.u_bc1 - u_bc1))
        loss_bc2  = tf.reduce_mean(tf.square(self.u_bc2 - u_bc2))

        # Sobolev regularisation: gradient-norm penalty at terminal points.
        # Theoretical basis: the exact solution u ∈ H^1(Ω); penalising ||∇u||^2
        # at the terminal surface enforces smoothness (Czarnecki et al., 2017).
        with tf.GradientTape() as tape:
            tape.watch(self.X2)
            u_sob = self.Neural_Net(self.X2)
        g = tape.gradient(u_sob, self.X2)
        loss_sob = tf.reduce_mean(tf.square(g)) if g is not None else 0.0

        # Mean-guidance: first-moment consistency on the terminal condition.
        # Penalises systematic bias in the predicted mean, improving calibration.
        loss_mean = tf.square(tf.reduce_mean(u_term) - tf.reduce_mean(self.u2))

        total = (loss_pde + loss_term + loss_bc1 + loss_bc2
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
        return self.Neural_Net(X).numpy()

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
