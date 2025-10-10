#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

# ---------------- KF (position-only) ----------------
def wrap3(x):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    assert x.shape == (3,1)
    return x

class PositionKF:
    def __init__(self, x0=None, P0=None, q=0.05, r=1.0):
        self.x = wrap3([0,0,0] if x0 is None else x0)
        self.P = np.eye(3, dtype=float) * (1.0 if P0 is None else float(P0))
        self.Q = np.eye(3, dtype=float) * float(q)
        self.R_default = np.eye(3, dtype=float) * float(r)
        self.F = np.eye(3, dtype=float)     # x_{k+1} = F x_k + w  (identity)
        self.H = np.eye(3, dtype=float)     # z_k = H x_k + v     (direct position)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update_pos(self, z, R=None, gate_chi2=1e9):
        z  = wrap3(z)
        H  = self.H
        Rm = self.R_default if R is None else np.asarray(R, dtype=float)

        y = z - (H @ self.x)
        S = H @ self.P @ H.T + Rm

        # Optional Mahalanobis gate (off by default with huge threshold)
        m2 = float(y.T @ np.linalg.solve(S, y))
        if m2 > gate_chi2:
            return

        PHt = self.P @ H.T
        # Use solve instead of explicit inverse for stability
        K   = PHt @ np.linalg.solve(S, np.eye(3))

        self.x = self.x + K @ y
        I = np.eye(3, dtype=float)
        self.P = (I - K @ H) @ self.P

# ---------------- Synthetic data gen ----------------
def synthetic_truth(N, dt):
    """
    Make a smooth 3D path:
      - horizontal circle (radius R) with slow climb & gentle vertical oscillation
    """
    t = np.arange(N)*dt
    R = 50.0
    w = 2*np.pi/40.0
    x = R*np.cos(w*t)
    y = R*np.sin(w*t)
    z = 0.5*t + 5.0*np.sin(0.2*t)
    return np.vstack([x,y,z])  # shape (3,N)

def simulate_measurements(truth_xyz, gps_std_xyz=(2.0, 2.0, 3.0), seed=7):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, np.array(gps_std_xyz).reshape(3,1), size=truth_xyz.shape)
    return truth_xyz + noise

# ---------------- Main experiment ----------------
def main():
    # Params
    N  = 250
    dt = 0.2

    # Truth + noisy GPS-like measurements
    truth = synthetic_truth(N, dt)                   # (3,N)
    meas  = simulate_measurements(truth, (2.0, 2.0, 3.0), seed=42)

    # KF setup
    kf = PositionKF(q=0.05, r=1.0)
    # Use per-sensor R for GPS (variances, not std)
    R_gps = np.diag([2.0**2, 2.0**2, 3.0**2])

    # Storage
    est = np.zeros_like(truth)
    Pdiag = np.zeros((3, N))

    # Run filter
    for k in range(N):
        kf.predict()
        kf.update_pos(meas[:,k], R=R_gps)  # position-only update
        est[:,k] = kf.x.ravel()
        Pdiag[:,k] = np.diag(kf.P)

    # ---------------- Visualization ----------------
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(truth[0], truth[1], truth[2], 'k-', lw=2, label='Truth')
    ax.scatter(meas[0], meas[1], meas[2], c='tab:red', s=12, alpha=0.4, label='Measurements')
    ax.plot(est[0], est[1], est[2], color='tab:blue', lw=2, label='Kalman estimate')

    ax.set_title('3D Position: Truth vs Noisy Measurements vs Kalman')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend(loc='upper left')
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()

    # Error plot (optional but handy)
    fig2, axs = plt.subplots(3,1, figsize=(10,7), sharex=True)
    t = np.arange(N)*dt
    err = est - truth
    labels = ['x','y','z']
    for i in range(3):
        axs[i].plot(t, err[i], color='tab:blue', lw=1.5, label='error')
        axs[i].plot(t, 2*np.sqrt(Pdiag[i]), 'r--', lw=1, label='±2σ' if i==0 else None)
        axs[i].plot(t,-2*np.sqrt(Pdiag[i]), 'r--', lw=1)
        axs[i].set_ylabel(f'{labels[i]} err (m)')
        axs[i].grid(True, alpha=0.3)
    axs[-1].set_xlabel('time (s)')
    axs[0].legend(loc='upper right')
    fig2.suptitle('Per-axis error and ±2σ from P')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
