#!/usr/bin/env python
# CONSTANT ACCELERATION MODEL 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

# ---------------- KF (position-only) ----------------
def wrap(x, n):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    assert x.shape == (n,1)
    return x

class PositionKF:  # now CA model: 9-state [p, v, a]
    def __init__(self, x0=None, P0=None, q=0.05, r=1.0):
        self.I3 = np.eye(3, dtype=float)
        self.I9 = np.eye(9, dtype=float)
        self.q_j = float(q)         # interpret q as jerk PSD for CA model

        # x = [px,py,pz, vx,vy,vz, ax,ay,az]^T
        self.x = wrap(([0,0,0, 0,0,0, 0,0,0] if x0 is None else x0), 9)
        self.P = np.eye(9, dtype=float) * (1.0 if P0 is None else float(P0))
        self.R_default = np.eye(3, dtype=float) * float(r)

        # F and H will be set in predict; H maps position from 9-state
        self.F = np.eye(9, dtype=float)
        self.H = np.hstack([self.I3, np.zeros((3,6))])  # z = [I3 0 0] x

    def predict(self, dt: float):
        dt2 = dt*dt
        # CA state transition
        F = np.block([
            [self.I3,          dt*self.I3,   0.5*dt2*self.I3],
            [np.zeros((3,3)),  self.I3,      dt*self.I3     ],
            [np.zeros((3,3)),  np.zeros((3,3)), self.I3     ]
        ])
        self.F = F

        # Discrete process noise for white jerk PSD q_j
        dt3, dt4, dt5 = dt**3, dt**4, dt**5
        Qpp = (dt5/20.0) * self.I3
        Qpv = (dt4/8.0)  * self.I3
        Qpa = (dt3/6.0)  * self.I3
        Qvv = (dt3/3.0)  * self.I3
        Qva = (dt2/2.0)  * self.I3
        Qaa = (dt)       * self.I3
        Qd  = self.q_j * np.block([
            [Qpp, Qpv, Qpa],
            [Qpv, Qvv, Qva],
            [Qpa, Qva, Qaa]
        ])

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Qd

    def update_pos(self, z, R=None, gate_chi2=1e9):
        z  = wrap(z, 3)
        H  = self.H
        Rm = self.R_default if R is None else np.asarray(R, dtype=float)

        y = z - (H @ self.x)
        S = H @ self.P @ H.T + Rm

        PHt = self.P @ H.T
        # Efficient solve: K = PHt @ S^{-1}
        K = np.linalg.solve(S.T, PHt.T).T

        # Joseph-form covariance update for numerical robustness
        self.x = self.x + K @ y
        I = self.I9
        KH = K @ H
        self.P = (I - KH) @ self.P @ (I - KH).T + K @ Rm @ K.T


# ---------------- Synthetic data gen ----------------
def synthetic_truth(N, dt):
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
    N  = 500
    dt = 0.2

    # Truth + noisy GPS-like measurements
    truth = synthetic_truth(N, dt)                   # (3,N)
    meas  = simulate_measurements(truth, (2.0, 2.0, 3.0), seed=42)

    # Init: estimate v0 from first two meas; a0 = 0
    v0 = (meas[:,1] - meas[:,0]) / dt
    a0 = np.zeros(3)
    x0 = np.hstack([meas[:,0], v0, a0]).astype(float)   # [p v a] (9,)

    # Priors: large pos/vel/acc uncertainty (vel/acc larger)
    P0 = np.block([
        [np.eye(3)*1e2,  np.zeros((3,3)), np.zeros((3,3))],
        [np.zeros((3,3)), np.eye(3)*1e3,  np.zeros((3,3))],
        [np.zeros((3,3)), np.zeros((3,3)), np.eye(3)*1e4]
    ])

    # CA filter (q = jerk PSD). Start with 0.05–0.2; bump up if it still lags on curves
    kf = PositionKF(x0=x0, P0=None, q=0.1, r=1.0)
    kf.x = wrap(x0, 9)
    kf.P = P0

    # Measurement covariance (variances)
    R_gps = np.diag([2.0**2, 2.0**2, 3.0**2])

    # Storage
    est = np.zeros_like(truth)
    Pdiag = np.zeros((3, N))

    # Run filter
    for k in range(N):
        kf.predict(dt)
        kf.update_pos(meas[:,k], R=R_gps)
        est[:,k] = kf.x[:3].ravel()               # position only
        Pdiag[:, k] = np.diag(kf.P[:3, :3])       # position variances
        # print(np.diag(kf.P))  # optional debug

    # ---------------- Visualization ----------------
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(truth[0], truth[1], truth[2], 'k-', lw=2, label='Truth')
    ax.scatter(meas[0], meas[1], meas[2], c='tab:red', s=12, alpha=0.4, label='Measurements')
    ax.plot(est[0], est[1], est[2], color='tab:blue', lw=2, label='Kalman estimate')

    ax.set_title('3D Position (CA Model): Truth vs Measurements vs Kalman')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend(loc='upper left')
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()

    # Error plot
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
    fig2.suptitle('Per-axis error and ±2σ from P (CA)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
