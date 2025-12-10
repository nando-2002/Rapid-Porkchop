import numpy as np
import matplotlib.pyplot as plt

def plot_orbit(a, e, inc, raan, argp, mu, n_points=500, show=True, ax=None):
    nu = np.linspace(0, 2*np.pi, n_points)
    p = a * (1 - e**2)
    r_pf = p / (1 + e * np.cos(nu))
    x_pf = r_pf * np.cos(nu)
    y_pf = r_pf * np.sin(nu)
    z_pf = np.zeros_like(x_pf)
    r_pf_vec = np.vstack((x_pf, y_pf, z_pf))

    # Rotation matrices (unchanged)
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)
    
    R3_O = np.array([[ cO, -sO, 0], [ sO,  cO, 0], [  0,   0, 1]])
    R1_i = np.array([[1,  0,   0], [0, ci, -si], [0, si,  ci]])
    R3_w = np.array([[ cw, -sw, 0], [ sw,  cw, 0], [  0,   0, 1]])
    
    Q_pqw_to_eci = R3_O @ R1_i @ R3_w
    r_eci = Q_pqw_to_eci @ r_pf_vec
    x, y, z = r_eci

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    ax.plot(x, y, z, label='Orbit')
    ax.scatter([0], [0], [0], color='y', label='Central body')

    # Fixed NumPy 2.0 ptp usage
    max_range = np.max([np.ptp(x), np.ptp(y), np.ptp(z)]) / 2.0
    mid_x = (x.max() + x.min()) / 2.0
    mid_y = (y.max() + y.min()) / 2.0
    mid_z = (z.max() + z.min()) / 2.0
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    
    if show:
        plt.show()
    return ax

def _orbit_points(a, e, inc, raan, argp, n_points=500):
    # True anomaly grid
    nu = np.linspace(0, 2*np.pi, n_points)

    # Radius in perifocal frame
    p = a * (1 - e**2)
    r_pf = p / (1 + e * np.cos(nu))

    # Perifocal coordinates (PQW)
    x_pf = r_pf * np.cos(nu)
    y_pf = r_pf * np.sin(nu)
    z_pf = np.zeros_like(x_pf)
    r_pf_vec = np.vstack((x_pf, y_pf, z_pf))

    # Rotation matrix PQW -> ECI (3-1-3: Ω, i, ω)
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)

    R3_O = np.array([[ cO, -sO, 0],
                     [ sO,  cO, 0],
                     [  0,   0, 1]])

    R1_i = np.array([[1,  0,   0],
                     [0, ci, -si],
                     [0, si,  ci]])

    R3_w = np.array([[ cw, -sw, 0],
                     [ sw,  cw, 0],
                     [  0,   0, 1]])

    Q_pqw_to_eci = R3_O @ R1_i @ R3_w
    r_eci = Q_pqw_to_eci @ r_pf_vec
    return r_eci  # shape (3, n_points)


def plot_two_orbits(orb1, orb2, labels=("Orbit 1", "Orbit 2"),
                    n_points=500, show=True, ax=None):
    """
    Plot two Keplerian orbits in 3D from orbital elements.

    orbX = (a, e, inc, raan, argp) with angles in radians.
    """
    a1, e1, i1, O1, w1 = orb1
    a2, e2, i2, O2, w2 = orb2

    r1 = _orbit_points(a1, e1, i1, O1, w1, n_points)
    r2 = _orbit_points(a2, e2, i2, O2, w2, n_points)

    x1, y1, z1 = r1
    x2, y2, z2 = r2

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    ax.plot(x1, y1, z1, label=labels[0])
    ax.plot(x2, y2, z2, label=labels[1])

    # Central body at origin
    ax.scatter([0], [0], [0], color='y', label='Central body')

    # Equal aspect for both orbits
    xs = np.concatenate((x1, x2))
    ys = np.concatenate((y1, y2))
    zs = np.concatenate((z1, z2))

    max_range = np.max([np.ptp(xs), np.ptp(ys), np.ptp(zs)]) / 2.0
    mid_x = (xs.max() + xs.min()) / 2.0
    mid_y = (ys.max() + ys.min()) / 2.0
    mid_z = (zs.max() + zs.min()) / 2.0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()

    if show:
        plt.show()

    return ax

def plot_orbit_from_state(r0, v0, mu, n_points=500, show=True, ax=None):
    """
    Plot a Keplerian orbit in 3D from a Cartesian state (r0, v0).

    Parameters
    ----------
    r0 : array-like, shape (3,)
        Position vector at epoch.
    v0 : array-like, shape (3,)
        Velocity vector at epoch.
    mu : float
        Gravitational parameter of central body.
    n_points : int
        Number of points along the orbit.
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)

    # Specific angular momentum and its unit vector
    h_vec = np.cross(r0, v0)
    h = np.linalg.norm(h_vec)
    k_hat = h_vec / h

    # Eccentricity vector and scalar
    r_norm = np.linalg.norm(r0)
    v_norm = np.linalg.norm(v0)
    e_vec = (np.cross(v0, h_vec) / mu) - (r0 / r_norm)
    e = np.linalg.norm(e_vec)

    # Semi-latus rectum p from h and mu
    p = h**2 / mu

    # Build an orthonormal basis in the orbital plane
    # x_hat along eccentricity (toward periapsis) if non-circular
    if e > 1e-10:
        x_hat = e_vec / e
    else:
        # For nearly circular orbits, choose any direction in plane
        x_hat = r0 / r_norm
    y_hat = np.cross(k_hat, x_hat)

    # True anomaly grid
    nu = np.linspace(0, 2*np.pi, n_points)

    # Radius as function of true anomaly
    r = p / (1 + e * np.cos(nu))

    # Positions in orbital plane basis
    x = r * np.cos(nu)
    y = r * np.sin(nu)

    # Map to inertial coordinates
    R = np.outer(x, x_hat) + np.outer(y, y_hat)   # shape (n_points, 3)
    X, Y, Z = R[:, 0], R[:, 1], R[:, 2]

    # Create axis if needed
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Plot full orbit and current position
    ax.plot(X, Y, Z, label='Orbit')
    ax.scatter([0], [0], [0], color='y', label='Central body')
    ax.scatter([r0[0]], [r0[1]], [r0[2]], color='r', label='Current position')

    # Equal aspect
    max_range = np.max([np.ptp(X), np.ptp(Y), np.ptp(Z)]) / 2.0
    mid_x = (X.max() + X.min()) / 2.0
    mid_y = (Y.max() + Y.min()) / 2.0
    mid_z = (Z.max() + Z.min()) / 2.0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()

    if show:
        plt.show()

    return ax

def _orbit_from_state(r0, v0, mu, n_points=500):
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)

    # Specific angular momentum and eccentricity
    h_vec = np.cross(r0, v0)
    h = np.linalg.norm(h_vec)
    k_hat = h_vec / h

    r_norm = np.linalg.norm(r0)
    e_vec = (np.cross(v0, h_vec) / mu) - (r0 / r_norm)
    e = np.linalg.norm(e_vec)

    # Semi-latus rectum
    p = h**2 / mu

    # In-plane basis
    if e > 1e-10:
        x_hat = e_vec / e
    else:
        x_hat = r0 / r_norm
    y_hat = np.cross(k_hat, x_hat)

    # True anomaly and radius
    nu = np.linspace(0, 2*np.pi, n_points)
    r = p / (1 + e * np.cos(nu))

    # Coordinates in plane and then inertial
    x = r * np.cos(nu)
    y = r * np.sin(nu)
    R = np.outer(x, x_hat) + np.outer(y, y_hat)  # (n_points, 3)

    return R  # positions along orbit


def plot_two_orbits_from_states(r1, v1, r2, v2, mu,
                                labels=("Orbit 1", "Orbit 2"),
                                n_points=500, show=True, ax=None):
    """
    Plot two Keplerian orbits in 3D from Cartesian states (r, v).

    rX, vX : array-like shape (3,)
    mu     : gravitational parameter
    """
    R1 = _orbit_from_state(r1, v1, mu, n_points)
    R2 = _orbit_from_state(r2, v2, mu, n_points)

    X1, Y1, Z1 = R1[:, 0], R1[:, 1], R1[:, 2]
    X2, Y2, Z2 = R2[:, 0], R2[:, 1], R2[:, 2]

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    ax.plot(X1, Y1, Z1, label=labels[0])
    ax.plot(X2, Y2, Z2, label=labels[1])

    # Mark central body and current positions
    ax.scatter([0], [0], [0], color='y', label='Central body')
    r1 = np.asarray(r1, float)
    r2 = np.asarray(r2, float)
    ax.scatter([r1[0]], [r1[1]], [r1[2]], color='r', label=f'{labels[0]} now')
    ax.scatter([r2[0]], [r2[1]], [r2[2]], color='g', label=f'{labels[1]} now')

    # Equal aspect for both orbits
    xs = np.concatenate((X1, X2))
    ys = np.concatenate((Y1, Y2))
    zs = np.concatenate((Z1, Z2))

    max_range = np.max([np.ptp(xs), np.ptp(ys), np.ptp(zs)]) / 2.0
    mid_x = (xs.max() + xs.min()) / 2.0
    mid_y = (ys.max() + ys.min()) / 2.0
    mid_z = (zs.max() + zs.min()) / 2.0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()

    if show:
        plt.show()

    return ax

def plot_three_orbits_from_states(r1, v1, r2, v2, r3, v3, mu,
                                  labels=("Orbit 1", "Orbit 2", "Orbit 3"),
                                  n_points=500, show=True, ax=None):
    R1 = _orbit_from_state(r1, v1, mu, n_points)
    R2 = _orbit_from_state(r2, v2, mu, n_points)
    R3 = _orbit_from_state(r3, v3, mu, n_points)

    X1, Y1, Z1 = R1[:, 0], R1[:, 1], R1[:, 2]
    X2, Y2, Z2 = R2[:, 0], R2[:, 1], R2[:, 2]
    X3, Y3, Z3 = R3[:, 0], R3[:, 1], R3[:, 2]

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    ax.plot(X1, Y1, Z1, label=labels[0])
    ax.plot(X2, Y2, Z2, label=labels[1])
    ax.plot(X3, Y3, Z3, label=labels[2])

    ax.scatter([0], [0], [0], color='y', label='Central body')

    r1 = np.asarray(r1, float)
    r2 = np.asarray(r2, float)
    r3 = np.asarray(r3, float)
    ax.scatter([r1[0]], [r1[1]], [r1[2]], color='r', label=f'{labels[0]} now')
    ax.scatter([r2[0]], [r2[1]], [r2[2]], color='g', label=f'{labels[1]} now')
    ax.scatter([r3[0]], [r3[1]], [r3[2]], color='b', label=f'{labels[2]} now')

    xs = np.concatenate((X1, X2, X3))
    ys = np.concatenate((Y1, Y2, Y3))
    zs = np.concatenate((Z1, Z2, Z3))

    max_range = np.max([np.ptp(xs), np.ptp(ys), np.ptp(zs)]) / 2.0
    mid_x = (xs.max() + xs.min()) / 2.0
    mid_y = (ys.max() + ys.min()) / 2.0
    mid_z = (zs.max() + zs.min()) / 2.0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()

    if show:
        plt.show()

    return ax
