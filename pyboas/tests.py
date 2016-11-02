from math import pi, sqrt
import numpy as np
import numpy.random

from pyboas.predictor import computepredictive


class Norm(object):
    """
    Class implementing normal distribution pdf.
    Faster than using scipy.stats classes.
    """
    def __init__(self, loc, scale=1.):
        self.loc = np.atleast_2d(loc)
        self.scale = scale

    def pdf(self, x):
        # Brodcast x with location
        return (np.sqrt(2*pi)*self.scale)**(-1) * \
            np.exp(-(x - np.atleast_3d(self.loc))**2/(2 * self.scale**2))


def model(posterior, t):
    amp, per, t = np.atleast_2d(posterior[:, 0], posterior[:, 1], t)
    return amp * np.sin(2*pi*t.T/per)


def test_posteriorpredictive(tpred=10, sigma=1, nposterior=1000,
                             npoints=1000, scale=1):
    """
    Script to demonstrate construction of predictive distribution based on a
    sample from the posterior.
    """

    # Create posterior samples for model

    # Posterior amplitude is centred is N(10, 1)
    post_amplitude = np.random.randn(nposterior) + 10
    # Posterior period is N(1, 0.1)
    post_per = np.random.randn(nposterior) * 0.01 + 1

    # Compute model at these values, likelihood function evaluate in x and
    # compute mean over all posterior elements.
    posterior = np.array([post_amplitude, post_per]).T
    locarray = model(posterior, tpred)
    likefunc = Norm(loc=locarray, scale=sigma)

    # Construct array to evaluate sum of likelihoods
    x = np.linspace(locarray.min() - 3 * sigma,
                    locarray.max() + 3 * sigma, npoints)

    postpred = likefunc.pdf(x).mean(axis=1)

    # Compute same thing using predictor module function
    postpred2 = computepredictive(posterior, tpred, model, Norm, scale,
                                  npoints=npoints)

    assert np.allclose(postpred, postpred2), "Different results obtained."


if __name__ == '__main__':
    test_posteriorpredictive()