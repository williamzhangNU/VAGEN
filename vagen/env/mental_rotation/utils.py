import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple
import math

def _quat_normalize(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    w, x, y, z = q
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n == 0:
        raise ValueError("Zero-norm quaternion.")
    return (w/n, x/n, y/n, z/n)

def quat_multiply(q1, q2):
    """
    Hamilton convention quaternion multiplication.
    q1, q2: (w, x, y, z)  # unit quaternions recommended for rotations
    
    Returns:
        (w, x, y, z) : q = q1 ⊗ q2
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return (w, x, y, z)

def euler_xyz_to_quat(x_angle, y_angle, z_angle, degrees=False):
    """
    Convert extrinsic X-Y-Z Euler angles to quaternion (w, x, y, z) using SciPy convention.
    Parameters
    ----------
    x_angle, y_angle, z_angle : float
        Rotation angles around fixed world X, Y, Z axes, in radians by default.
    degrees : bool
        If True, input angles are in degrees.

    Returns
    -------
    tuple : (w, x, y, z)  quaternion in Hamilton convention.
    """
    # SciPy expects string 'XYZ' for extrinsic, upper-case means extrinsic (fixed axes)
    r = R.from_euler('XYZ', [x_angle, y_angle, z_angle], degrees=degrees)
    q_xyzw = r.as_quat()  # SciPy returns (x, y, z, w)
    # Rearrange to (w, x, y, z)
    return (q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])

def quat_to_euler_xyz_scipy(q_wxyz, degrees=True):
    """
    Quaternion (w, x, y, z) -> extrinsic X-Y-Z Euler angles (x, y, z).
    Uses SciPy's convention: uppercase 'XYZ' = extrinsic fixed-axes rotations.
    """
    w, x, y, z = q_wxyz
    # SciPy expects (x, y, z, w)
    r = R.from_quat([x, y, z, w])
    euler = r.as_euler('XYZ', degrees=degrees)  # extrinsic X-Y-Z
    # Returns np.array([x_angle, y_angle, z_angle])
    return tuple(euler)

def quat_equal(q1: Tuple[float, float, float, float],
               q2: Tuple[float, float, float, float],
               angle_tol: float = 1e-2) -> bool:
    a = _quat_normalize(q1)
    b = _quat_normalize(q2)
    dot = abs(a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]) 
    dot = min(1.0, max(-1.0, dot))
    delta = 2.0 * math.acos(dot)
    return delta <= angle_tol

def euler_equal_xyz_extrinsic(e1: Tuple[float, float, float],
                              e2: Tuple[float, float, float],
                              degrees: bool = True,
                              angle_tol: float = 1e-2) -> bool:
    q1 = euler_xyz_to_quat(*e1, degrees=degrees)
    q2 = euler_xyz_to_quat(*e2, degrees=degrees)
    return quat_equal(q1, q2, angle_tol=angle_tol)

if __name__ == "__main__":
    init_quat = (-0.2625, 0.7497, -0.5614, 0.2321)
    rotate_1 = (90, 270, 90)
    quat_1 = euler_xyz_to_quat(rotate_1[0], rotate_1[1], rotate_1[2], degrees=True)

    rotate_2_y = (0, -90, 0)
    quat_2_y = euler_xyz_to_quat(rotate_2_y[0], rotate_2_y[1], rotate_2_y[2], degrees=True)

    rotate_2_z = (0, 0, 90)
    quat_2_z = euler_xyz_to_quat(rotate_2_z[0], rotate_2_z[1], rotate_2_z[2], degrees=True)

    rotate_2_x = (90, 0, 0)
    quat_2_x = euler_xyz_to_quat(rotate_2_x[0], rotate_2_x[1], rotate_2_x[2], degrees=True)
    
    quat_2 = quat_multiply(quat_2_z, quat_2_y)
    quat_2 = quat_multiply(quat_2_x, quat_2)
    
    print("quat_1_rot: ", quat_1)
    print("quat_1_target: ", quat_multiply(quat_1, init_quat))
    print("quat_2_rot: ", quat_2)
    print("quat_2_target: ", quat_multiply(quat_2, init_quat))