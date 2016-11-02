import numpy as np


def keplerian(param, time, epoch, modeltype='k1'):
    """
    Keplerian curve.

    :param array-like param: Keplerian parameters (K, P, sqrt(e)*cos(w),
    sqrt(e)*sin(w), L0, v0, epoch)
    """
    # prepare to broadcast time
    t = np.atleast_1d(time)[:, np.newaxis]

    if modeltype == 'k1':
        K_ms = param[:, 0]
        P_day = param[:, 1]
        secos = param[:, 2]
        sesin = param[:, 3]
        ma0 = param[:, 4]
        # tp = param[4]
        v0 = param[:, 5]
        acc = 0  # param[6] # Secular acceleration

        # Compute mean anomaly
        omega_rad = np.arctan2(sesin, secos)
        # ma0 = ml0 - omega_rad
        ma = 2*np.pi/P_day * (t - epoch) + ma0
        # ma = 2*np.pi/P_day * (time - tp)

        # Compute true anomaly
        ecc = secos**2 + sesin**2
        nu = trueanomaly(ma, ecc)

        # Add secular acceleration
        rvdrift = acc * (t - epoch)/365.25

        return v0 + rvdrift + K_ms * (np.cos(nu + omega_rad) +
                                      ecc * np.cos(omega_rad))

    elif modeltype == 'k0':
        # No planet. Return only secular acceleration.
        v0 = param[:, 0]
        # acc = param[1]
        acc = 0

        rvdrift = acc * (t - epoch)/365.25

        return v0 + rvdrift

    else:
        raise NameError('Unknown Modeltype.')


def trueanomaly(ma, ecc, method='Newton', niterationmax=1e4):
    """
    Computes mean anomaly.

    :param np.array or float ma: Mean anomaly.

    :param np.array ecc: orbital eccentricity.

    :param str method: optimisation method. Either 'Newton' or 'Halley'.

    :param int niterationmax: maximum number of iterations.
    """

    # Initialise eccentric anomaly at mean anomaly
    ma = np.atleast_1d(ma)
    ea0 = ma.copy()

    # Tweak very large eccentricities
    ecc = np.atleast_1d(ecc)
    ecc = np.where(ecc > 0.99, ecc*0.0 + 0.99, ecc)

    niteration = 0
    while niteration <= niterationmax:
        ff = ea0 - ecc*np.sin(ea0) - ma
        dff = 1 - ecc*np.cos(ea0)

        if method == 'Newton':
            # Use Newton method
            ea = ea0 - ff / dff

        elif method == 'Halley':
            # Use Halley's parabolic method
            d2ff = ecc*np.sin(ea0)

            discr = dff**2 - 2 * ff * d2ff

            ea = np.where((discr < 0), ea0 - dff / d2ff,
                          ea0 - 2 * ff / (dff + np.sign(dff) * np.sqrt(discr)))

        # Check if convergence is reached
        if np.linalg.norm(ea - ea0, ord=1) <= 1e-5:
            # Compute true anomaly
            nu = 2. * np.arctan2(np.sqrt(1. + ecc) * np.sin(ea / 2.),
                                 np.sqrt(1. - ecc) * np.cos(ea / 2.))
            return nu

        # Increase iteration number;
        niteration += 1

        # Update eccentric anomaly value
        ea0 = ea

    else:
        raise RuntimeError('Maximum number of iterations reched. '
                           'Eccentric anomaly comoputation not converged.')
