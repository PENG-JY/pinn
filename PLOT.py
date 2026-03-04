import numpy as np
import matplotlib.pyplot as plt


class PLOT:
    """Plotting utilities for PINN option pricing results."""

    def __init__(self, Eq_name, n, X, Exact_u, u_pred,
                 Xzhongzhi, uzhongzhi, zhongzhi_u_pred, n_iter):
        n_show = min(200, len(X))
        self.s1       = X[:n_show, 0:1]
        self.t        = X[:n_show, 3:4]
        self.Exact_u  = Exact_u[:n_show]
        self.u_pred   = u_pred[:n_show]

        n_show2 = min(50, len(Xzhongzhi))
        self.zhongzhi_t        = Xzhongzhi[:n_show2, 3:4]
        self.uzhongzhi         = uzhongzhi[:n_show2]
        self.zhongzhi_u_pred   = zhongzhi_u_pred[:n_show2]

        self.Eq_name = Eq_name
        self.n_iter  = n_iter

    def u_pred_exact_t(self):
        fig, ax = plt.subplots()
        ax.scatter(self.t, self.Exact_u,        color='red',    s=10, label='Exact (interior)')
        ax.scatter(self.t, self.u_pred,          color='blue',   s=10, label='PINN  (interior)')
        ax.scatter(self.zhongzhi_t, self.uzhongzhi,       color='green',  s=10, label='Exact (terminal)')
        ax.scatter(self.zhongzhi_t, self.zhongzhi_u_pred, color='orange', s=10, label='PINN  (terminal)')
        ax.set_xlabel('t'); ax.set_ylabel('u')
        ax.set_title('Exact vs PINN  (vs t)')
        ax.legend(); ax.grid(True)
        fig.savefig('scatter_u_t.png', dpi=300, bbox_inches='tight')
        plt.show()

    def u_pred_exact_s1(self):
        fig, ax = plt.subplots()
        ax.scatter(self.s1, self.Exact_u, color='red',  s=10, label='Exact')
        ax.scatter(self.s1, self.u_pred,  color='blue', s=10, label='PINN')
        ax.set_xlabel('S1'); ax.set_ylabel('u')
        ax.set_title('Exact vs PINN  (vs S1)')
        ax.legend(); ax.grid(True)
        fig.savefig('scatter_u_s1.png', dpi=300, bbox_inches='tight')
        plt.show()

    def u_pred_s1_s2(self, s1, s2, u):
        self._contour(s1, s2, u, 'S1', 'S2', 'u_pred (S1-S2)', 'u_pred_s1_s2.png')

    def u_exact_s1_s2(self, s1, s2, u):
        self._contour(s1, s2, u, 'S1', 'S2', 'u_exact (S1-S2)', 'u_exact_s1_s2.png')

    def error_s1_s2(self, s1, s2, error):
        self._contour(s1, s2, error, 'S1', 'S2', 'Error (S1-S2)', 'error_s1_s2.png')

    def u_pred_s1_t(self, s1, t, u):
        self._contour(s1, t, u, 'S1', 't', 'u_pred (S1-t)', 'u_pred_s1_t.png')

    def u_exact_s1_t(self, s1, t, u):
        self._contour(s1, t, u, 'S1', 't', 'u_exact (S1-t)', 'u_exact_s1_t.png')

    def error_s1_t(self, s1, t, error):
        self._contour(s1, t, error, 'S1', 't', 'Error (S1-t)', 'error_s1_t.png')

    def _contour(self, X, Y, Z, xlabel, ylabel, title, fname):
        fig, ax = plt.subplots()
        cf = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.9)
        fig.colorbar(cf, ax=ax)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.show()

    def MSE(self, n_iter, MSE_history):
        fig, ax = plt.subplots()
        ax.plot(MSE_history, 'r-', linewidth=0.8, label='Total loss')
        ax.set_xlabel('Iteration'); ax.set_ylabel('Loss')
        ax.set_yscale('log'); ax.legend(); ax.grid(True)
        fig.savefig('loss.png', dpi=300, bbox_inches='tight')
        plt.show()

    def L2_error(self, n_iter, L2error_u):
        if not L2error_u:
            return
        fig, ax = plt.subplots()
        ax.plot(L2error_u, 'b-', linewidth=0.8, label='Rel. L2 error')
        ax.set_xlabel('Checkpoint'); ax.set_ylabel('Relative L2 error')
        ax.set_yscale('log'); ax.legend(); ax.grid(True)
        fig.savefig('L2error_u.png', dpi=300, bbox_inches='tight')
        plt.show()
