"""
Module containing functions and classes to produce posterior predictive
distribution.
"""
from math import pi
# import warnings
import numpy as np


class Predictor(object):
    """
    Global class implementing predictors.

    Predictors using different distribution functions should be subclassed
    from this one.
    """

    def __init__(self, posterior, model, extramodelargs=()):
        """

        :param np.array posterior: posterior sample with dimensions (n, k),
        where n is the sample size and k is the dimension of space.

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

        # Initialise time for prediction.
        self._times = None
        self.newtimes = None

        # Initialise
        self.predictives = np.zeros((0, 0))
        self.x = np.zeros((0, 0))

    def reset(self):
        """Deletes previous predictions done with the Predictor."""
        self.times = None
        self.predictives = np.zeros((0, 0))
        self.x = np.zeros((0, 0))

    @property
    def times(self):
        return self._times

    @times.setter
    def times(self, timepred=None):
        """Set new times for prediction. Concatenates if already exist."""
        if timepred is None and self._times is None:
            raise TypeError('Prediction time(s) must be given.')

        elif self._times is None:
            self.newtimes = np.atleast_1d(timepred)
            self._times = self.newtimes

        elif timepred is None:
            self.newtimes = self.times

        else:
            self.newtimes = np.atleast_1d(timepred)
            self._times = np.concatenate((self.times, self.newtimes))

    @staticmethod
    def likefunc(v, loc, **kwargs):
        """
        Abstract method for likelihood functions
        """
        raise NotImplementedError('To be implemented on a sub-class basis.')

    @staticmethod
    def likedraw(loc, **kwargs):
        """
        Abstract method to draw samples from likelihood function.
        """
        raise NotImplementedError('To be implemented on a sub-class basis.')

    def make_prediction(self, timepred=None, npoints=500, **kwargs):
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
        distributions are computed. If None, it default to current times
        attribute, if not None.

        :param int npoints: number of points used to evaluate likelihood.

        :keyword bool verbose: if True will print number of steps performed and
        remaing.

        :return x, predictives: the x values where predictive distributions are
         evaluated and the distributions.
        """
        verbose = kwargs.pop('verbose', False)

        # Sets new prediction times. Note that this concatenates if previous
        # prediction times were defined.
        self.times = timepred

        # Prepare output arrays
        predictives = np.zeros((len(self.newtimes), npoints))
        x = np.zeros_like(predictives)

        # Compute location parameter at all newtimes at the same time.
        # Up to x10 faster than doing it for each time, but need to be sure
        # memory is not overflown.

        loc = np.zeros((len(self.newtimes), len(self.posterior)))
        # loc = self.model(self.posterior, self.newtimes, *self.modelargs)

        for i, t in enumerate(self.newtimes):
            if verbose:
                print('Step {} of {}'.format(i+1, len(self.newtimes)))

            loc[i] = self.model(self.posterior, t, *self.modelargs)

            # Create auxiliary array where to evaluate posterior predictive
            deltax = loc[i].max() - loc[i].min()
            # TODO check if this vector limits is always enough. Results
            # change a lot.
            v = np.linspace(loc[i].min() - deltax*2,
                            loc[i].max() + deltax*2, npoints)

            # If scale is an array, it should have the same length than
            # posterior.
            like = self.likefunc(v, loc[i], **kwargs)

            x[i] = v
            predictives[i] = like.mean(axis=1)

        # Update predictives and x attributes.
        if len(self.predictives) == 0:
            self.predictives = predictives
            self.x = x
        else:
            self.predictives = np.vstack((self.predictives, predictives))
            self.x = np.vstack((self.x, x))

        return x, predictives

    def samplepredictive(self, timesample, samplesize=None, **kwargs):
        """
        Produce samples from the posterior predictive distribution.

        """
        # Set new times for prediction. Note that this concatenates previous
        # prediction times.
        t = np.atleast_1d(timesample)

        # The predicted data value for each posterior sample is simply
        # y = m(theta) + N(0, sigma**2).
        if samplesize != self.posterior.shape[0]:
            post = self.posterior.copy()
            np.random.shuffle(post)
            psample = post[:samplesize]
        else:
            psample = self.posterior.copy()

        loc = self.model(psample, t, *self.modelargs)

        return self.likedraw(loc, **kwargs)


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
        self.scale = likescale  # TODO: where should like scale be defined?

    @staticmethod
    def likefunc(v, loc, scale=None):
        """
        Gaussian likelihood function.

        The Gaussian is located in the position loc and has with variance
        scale**2.

        :param float or np.array v: new datum value.

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

    @staticmethod
    def likedraw(loc, scale=1):
        """
        Draw sample from normal distribution

        :param float or np.array loc: location parameter of Gaussian function.

        :param float or np.array scale: scale parameter for Gaussian function.
        """
        return loc + np.random.randn(len(loc), 1) * scale


"""
:param class or instance likefun: a Class or an instance, containing the
method pdf that evaluates the likelihood at a given point x.

:param float or np.array scale: scale of the probability distribution for
new datum. If an array is passed, it must have same length than posterior
sample (n). This is used to allow for jitter terms and similar.
"""
