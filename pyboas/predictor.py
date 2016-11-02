"""
Module containing functions and classes to produce posterior predictive
distribution.
"""
from math import pi
import numpy as np


class Predictor(object):
    """
    Global class implementing predictors.

    Predictors using different distribution functions should be subclassed
    from this one.
    """

    def __init__(self, posterior, model, extramodelargs=()):
        """

        :param np.array posterior: posterior sample with dimensions (n, k), where
        n is the sample size and k is the dimension of space.

        :param callable model: function taking posterior sample as first
        argument and a time as second element and producing the model that will
        be used as location array for the likelihood function.

        :param list extramodelargs: additional arguments passed to model
        function.
        """

        self.posterior = np.atleast_2d(posterior)

        # Define model and its extra arguments
        self.model = model
        self.modelargs = extramodelargs

    @staticmethod
    def likefunc(v, loc, **kwargs):
        """
        Abstract method for likelihood functions
        """
        raise NotImplementedError('To be implemented on a sub-class basis.')

    def make_prediction(self, timepred, npoints=500, **kwargs):
        """
        Estimate posterior predictive distribution for a single future datum.

        This method estimates the posterior predictive distribution for a
        future datum at time tpred, based on a posterior sample.

        The algorithm uses the fact that the posterior predictive can be written
        as the expectation value over the posterior distribution of the
        probability distribution for a new observation (likelihood).

        First, it computes the value of the model at time tpred for each
        posterior sample element. Then uses this array
        as the location parameter for the likelihood of the new datum at time t.

        :param iterable or float timepred: times on which posterior predictive
        distributions are computed.

        :param int npoints: number of points used to evaluate likelihood.

        :keyword bool verbose: if True will print number of steps performed and
        remaing.

        :return x, predictives: the x values where predictive distributions are
         evaluated and the distributions.
        """
        verbose = kwargs.pop('verbose', False)

        times = np.atleast_1d(timepred)

        # Prepare output arrays
        predictives = np.zeros((len(times), npoints))
        x = np.zeros_like(predictives)

        for i, t in enumerate(times):
            if verbose:
                print('Step {} of {}'.format(i+1, len(times)))
            # Compute location parameter at time t
            loc = self.model(self.posterior, t, *self.modelargs)

            # Create auxiliary array where to evaluate posterior predictive
            v = np.linspace(loc.min(), loc.max(), npoints)
            # TODO extend this array on both sides in a generic manner.
            # v = np.linspace(loc.min() - 3 * scale, loc.max() + 3 * scale,
            #                npoints)

            # If scale is an array, it should have the same length than
            # posterior.
            like = self.likefunc(v, loc, **kwargs)

            x[i] = v
            predictives[i] = like.mean(axis=1)

        return x, predictives


class GaussPredictor(Predictor):
    """
    Class implementing predictor with Gaussian likelihood.
    """

    def __init__(self, posterior, model, extramodelargs=(),
                 likescale=1):
        """
        :param np.array posterior: posterior sample with dimensions (n, k),
        where n is the sample size and k is the dimension of space.

        :param callable model: function taking posterior sample as first
        argument and a time as second element and producing the model that will
        be used as location array for the likelihood function.

        :param float likescale: scale of the likelihood function of the future
        datum.

        """

        super(GaussPredictor, self).__init__(posterior, model,
                                             extramodelargs=extramodelargs)
        self.scale = likescale

    @staticmethod
    def likefunc(v, loc, scale=1):
        """
        Gaussian likelihood function.

        The Gaussian is located in the position loc and has with variance
        scale**2.

        :param float or np.array v: new datum value where .

        :param float or np.array loc: location parameter of Gaussian function.

        :param float or np.array scale: scale parameter for Gaussian function.

        :return np.array: value of likelihood function for different values
        of the future datum value and for all location and scales values. The
        shape is (nv, nloc).
        """
        v, loc, scale = np.atleast_1d(v, loc, scale)

        # Preparing to broadcast
        v = v[:, np.newaxis]

        return np.exp(-0.5 * (v - loc) ** 2 / scale ** 2) / \
            np.sqrt(2 * pi * scale ** 2)


# TODO complete this function and make it a method.
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


"""
:param class or instance likefun: a Class or an instance, containing the
method pdf that evaluates the likelihood at a given point x.

:param float or np.array scale: scale of the probability distribution for
new datum. If an array is passed, it must have same length than posterior
sample (n). This is used to allow for jitter terms and similar.
"""
