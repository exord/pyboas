"""
Module containing functions and classes to produce posterior predictive
distribution.
"""
import numpy as np


class Predictor(object):
    """
    Global class to implement different predictor based on different
    distribution functions.
    """
    def __init__(self, tpred):
        self.tpred = tpred


class GaussPredictor(Predictor):
    """
    Class implementing predictor with Gaussian likelihood.
    """


def computepredictive(posterior, tpred, model, likefun, scale, *modelargs,
                      **kwargs):
    """
    Estimate predictive distribution for a future datum at time tpred, based
    on a posterior sample.

    The algorithm uses the fact that the posterior predictive can be written
    as the expectation value over the posterior distribution of the probability
    distribution for a new observation (likelihood).

    :param np.array posterior: posterior sample with dimensions (n, k), where
    n is the sample size and k is the dimension of space.

    :param float tpred: time at which to estimate the predictive distribution.

    :param callable model: function taking posterior sample as first element and
    a time as second element and producing the model that will be used as
    location array for the likelihood function.

    :param class or instance likefun: a Class or an instance, containing the
    method pdf that evaluates the likelihood at a given point x.

    :param float or np.array scale: scale of the probability distribution for
    new datum. If an array is passed, it must have same length than posterior
    sample (n). This is used to allow for jitter terms and similar.

    Other params
    ------------
    :param optional modelargs: parameters passed to the model function.

    :keyword int npoints: number of points used to evaluate likelihood.

    """
    npoints = kwargs.pop('npoints', 500)

    # Compute the value of the model at time tpred for each posterior sample
    # element. This array will act as the location parameter for the likelihood
    # of the new datum
    loc = model(posterior, tpred, *modelargs)

    # Compute likelihood using this location and a given scale parameter.
    # If scale is an array, it should have the same length than posterior.
    like = likefun(loc, scale)

    # Create auxiliary array where to evaluate posterior predictive
    x = np.linspace(loc.min() - 3*scale, loc.max() + 3*scale, npoints)

    return x, like.pdf(x).mean(axis=1)


def samplepredictive(posterior, tpred, model, likefun, scale, *modelargs,
                     **kwargs):
    """
    Produce samples from the posterior predictive distribution based on
    samples from the posterior of the model parameters.


    """
    # Equivalently, we can draw samples from this distribution.
    # The predicted data value for each posterior sample is simply
    # y = m(theta) + N(0, sigma**2).

    return model(posterior, tpred) + np.random.randn(nposterior, 1) * sigma