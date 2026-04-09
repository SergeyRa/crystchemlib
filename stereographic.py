"""Routines for stereographic projections

Functions
---------
rodrigues : numpy.ndarray
    Rodrigues rotation matrix
orient : numpy.ndarray
    Standard orientation transform
SGP : tuple
    Stereographic projection
"""


def rodrigues(a, b):
    """Rodrigues rotation matrix

    Parameters
    ----------
    a : numpy.ndarray, list or tuple
        original direction (Cartesian)
    b : numpy.ndarray, list or tuple
        final direction (Cartesian)

    Returns
    -------
    numpy.ndarray
        3*3 rotation matrix which aligns a with b
    """

    from numpy import array, cross, dot, eye
    from numpy.linalg import norm

    an = array(a) / norm(array(a))
    bn = array(b) / norm(array(b))

    if norm(an + bn) == 0:
        for i in range(3):
            if abs(an[i]) != norm(an):
                R = -eye(3)
                R[i, i] = 1
                return R

    v = cross(an, bn)
    sinth = norm(v)
    kx, ky, kz = v / norm(v)
    K = array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    costh = dot(an, bn)
    return eye(3) + sinth*K + (1-costh)*(K@K)


def orient(UB):
    """Standard orientation transform

    Parameters
    ----------
    UB : np.ndarray
        UB matrix

    Returns
    -------
    np.ndarray
        3*3 rotation matrix which aligns
        c with z and a* with x
    """

    from numpy.linalg import inv

    A = inv(UB).T  # direct basis
    R1 = rodrigues(A.T[2], [0, 0, 1])
    UBnew = R1 @ UB
    R2 = rodrigues(UBnew.T[0], [1, 0, 0])
    return R2 @ R1


def SGP(x, y, z, upper='o', lower='x'):
    """Stereographic projection

    Parameters
    ----------
    x, y, z : float
        Cartesian coordinates
    upper, lower : str
        marks for upper and lower hemishperes

    Returns
    -------
    tuple
        (theta, rho, mark)
    """

    from numpy import arctan2, tan, acos

    theta = arctan2(y, x)
    rho = tan(
        acos(abs(z) / (x**2 + y**2 + z**2)**0.5) / 2
    )
    return (theta, rho, upper if (z >= 0) else lower)
