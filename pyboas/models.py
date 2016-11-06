import warnings
import numpy as np

TWOPI = 2 * np.pi


def compute_ma_ic(omega, ecc):
    ta_ic = np.pi/2. - omega

    # eccentric anomaly at inferior conjunction
    ea = 2 * np.arctan2(np.sin(ta_ic / 2.) * np.sqrt(1 - ecc),
                        np.cos(ta_ic / 2.) * np.sqrt(1 + ecc))

    # mean anomaly at velocity minimum
    return ea - ecc * np.sin(ea)


def compute_ma_vmax(omega, ecc):
    ta_max = -omega

    # eccentric anomaly at inferior conjunction
    ea = 2 * np.arctan2(np.sin(ta_max / 2.) * np.sqrt(1 - ecc),
                        np.cos(ta_max / 2.) * np.sqrt(1 + ecc))

    # mean anomaly at velocity maximum
    return ea - ecc * np.sin(ea)


def compute_ma_vmin(omega, ecc):
    ta_min = np.pi - omega

    # eccentric anomaly at inferior conjunction
    ea = 2 * np.arctan2(np.sin(ta_min / 2.) * np.sqrt(1 - ecc),
                        np.cos(ta_min / 2.) * np.sqrt(1 + ecc))

    # mean anomaly at velocity minimum
    return ea - ecc * np.sin(ea)


class Keplerian(object):
    """
    Implements Keplerian curves using flexible parametrization.
    """

    def __init__(self, **paramdict):
        """
        """
        # To avoid problems, first set all sensitive core parameters
        # (what an awful hack)
        for param in ('ecc', 'omega', 'ma0', 'node'):
            self.__setattr__(param, 0.0)

        if 'epoch' not in paramdict and (
                        'ma0' in paramdict or 'ml0' in paramdict):
            raise NameError('Epoch must be specified if ma0 or ml0 are '
                            'given.')

        for par in paramdict:
            self.__setattr__(par, paramdict[par])

        """
        # Define things deferently, according to which parameter is given
        if 'ma0' in paramdict:
            # Destroy ma0 property
            self.ma0 = paramdict.pop('ma0')

        elif 'ml0' in paramdict:
            print('Mean longitude given')
            # Mean longitude at epoch is main param, define properties for
            # the remaining "phase" parameters
            self.ml0 = paramdict.pop('ml0')
        """
        return

    @property
    def logk1(self):
        if self.k1 is not None:
            return np.log(self.k1)
        else:
            return None

    @logk1.setter
    def logk1(self, value):
        self.k1 = np.exp(value)

    @logk1.deleter
    def logk1(self):
        self.k1 = None

    @property
    def logper(self):
        if self.per is not None:
            return np.log(self.per)
        else:
            return None

    @logper.setter
    def logper(self, value):
        self.per = np.exp(value)

    @logper.deleter
    def logper(self):
        self.per = None

    @property
    def ml0(self):
        return self.omega + self.node + self.ma0

    @ml0.setter
    def ml0(self, value):
        self.ma0 = value - (self.omega + self.node)

    @property
    def tp(self):
        return self.epoch - self.ma0 * self.per / TWOPI

    @tp.setter
    def tp(self, value):
        self.ma0 = TWOPI * (self.epoch - value) / self.per

    @property
    def tc(self):
        # mean anomaly at inferior conjunction
        ma_ic = compute_ma_ic(self.omega, self.ecc)

        return (ma_ic - self.ma0) * self.per / TWOPI + self.epoch

    @tc.setter
    def tc(self, value):
        # mean anomaly at inferior conjunction
        ma_ic = compute_ma_ic(self.omega, self.ecc)

        self.ma0 = ma_ic - TWOPI * (value - self.epoch) / self.per

    @property
    def tvmax(self):
        # mean anomaly at time of velocity maximum
        ma_max = compute_ma_vmax(self.omega, self.ecc)
        return (ma_max - self.ma0) * self.per / TWOPI + self.epoch

    @tvmax.setter
    def tvmax(self, value):
        # mean anomaly at time of velocity maximum
        ma_max = compute_ma_vmax(self.omega, self.ecc)

        self.ma0 = ma_max - TWOPI * (value - self.epoch) / self.per

    @property
    def tvmin(self):
        # true anomaly at time of velocity minimum
        ma_min = compute_ma_vmin(self.omega, self.ecc)
        return (ma_min - self.ma0) * self.per / TWOPI + self.epoch

    @tvmin.setter
    def tvmin(self, value):
        # true anomaly at time of velocity minimum
        ma_min = compute_ma_vmin(self.omega, self.ecc)

        self.ma0 = ma_min - TWOPI * (value - self.epoch) / self.per

    # TODO test esin ecos are correct. Include sqrt(e)sin, etc.
    @property
    def esin(self):
        if self.ecc is not None and self.omega is not None:
            return self.ecc * np.sin(self.omega)
        else:
            return None

    @esin.setter
    def esin(self, value):
        self._esin = value
        if self.ecos is not None:
            self.ecc = np.sqrt(value**2 + self.ecos**2)
            self.omega = np.arctan2(value, self.ecos)

    @property
    def ecos(self):
        if self.ecc is not None and self.omega is not None:
            return self.ecc * np.cos(self.omega)
        else:
            return None

    @ecos.setter
    def ecos(self, value):
        self._ecos = value
        if self._esin is not None:
            self.ecc = np.sqrt(value ** 2 + self._esin ** 2)
            self.omega = np.arctan2(self._esin, value)

    def get_rv(self, time):
        """
        Radial velocity at time.

        :param np.array time: Time at which RV is computed.
        """
        time = np.atleast_1d(time)[:, np.newaxis]

        ma = self.ma0 + TWOPI * (time - self.epoch) / self.per

        ta = trueanomaly(ma, self.ecc)

        return self.k1 * (np.cos(ta + self.omega) +
                          self.ecc * np.cos(self.omega))


def keplerian_fixed(param, time, epoch, modeltype='k1'):
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
