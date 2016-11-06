#! /usr/env/bin python
import os
import pickle
import numpy as np
from scipy.integrate import trapz
import pylab as plt

from pyboas import predictor, models

# Reconstruct location of directory containing data.
# TODO if possible make this work when script is called using execfile.
print(__file__)
basedir = os.path.dirname(__file__)
datadir = os.path.abspath(os.path.join(basedir, '..', 'data'))


def read_data():
    # Read data
    t, rv, erv = np.loadtxt(os.path.join(datadir,
                                         'HD222582_keck.rdb'),
                            skiprows=2, unpack=True)
    t -= 10000
    return t[:24], rv[:24], erv[:24]


def run():
    # np.random.seed(010605)

    # Read posterior samples from data directory.
    print('Reading data...')
    t, rv, erv = read_data()

    print('Reading posterior samples....')
    f = open(os.path.join(datadir, 'HD222582_emcee_chain.dat'))
    chain = pickle.load(f)
    f.close()

    # Burn-in and flatten chain
    s = chain.shape
    posterior = chain[:, -500:, :].reshape(s[0] * 500, s[2])
    # posterior = chain.reshape(s[0]*s[1], s[2])

    # Define array with times where to predict future measurement.
    timepred = np.linspace(1500, 2200, 50)
    epoch = t.mean()

    # The scale of the Gaussian likelihood is taken as the mean error + jitter.
    likescale = np.sqrt(erv.mean() ** 2 + posterior[:, -1] ** 2)

    # Instantianate GaussPredictor
    gpredictor = predictor.GaussPredictor(posterior, models.keplerian,
                                          extramodelargs=(epoch, 'k1')
                                          )

    print('Making predictions....')
    # Make predicions
    x, post_preds = gpredictor.make_prediction(timepred, verbose=True,
                                               scale=likescale)

    # Compute information gain # TODO replace with 'informer' methods.
    infos = np.zeros(len(x))
    for i in range(len(x)):
        y = post_preds[i][post_preds[i] > 0]
        infos[i] = trapz(y * np.log(y), x[i][post_preds[i] > 0])

    print('Drawing sample....')
    # Draw samples
    tsample = np.array([1600, 1700, 1800, 1870, 1920, 2020])

    postpred_sample = gpredictor.samplepredictive(tsample,
                                                  scale=likescale.mean())

    # Plot results
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot data.
    ax.errorbar(t, rv, erv, fmt='or', ms=8, zorder=4)

    # time array for plots
    tt = np.linspace(700, 2200, 1000)

    # Pick 20 random curves from posterior
    post = posterior.copy()
    np.random.shuffle(post)
    rvall = models.keplerian(post[:20], tt, epoch, 'k1')

    # Plot sample posterior curves
    ax.plot(tt, rvall, '-', color='0.45', alpha=0.3)

    # Plot samples from posterior predictive
    for i, t in enumerate(tsample):
        ax.plot(t + np.random.rand(100)*5-2.5, postpred_sample[i, :100], ',m')

    # Define axis for expected information gain
    ax2 = ax.twinx()
    ax2.plot(timepred, -infos, '-g', lw=2)

    # Labels
    ax.set_xlabel('BJD - 2450000', fontsize=16)
    ax.set_ylabel('RV [m/s]', fontsize=16)
    ax2.set_ylabel('E[$\Delta$I]', fontsize=16)
    plt.show()

    return gpredictor


"""
if __name__ == '__main__':
    run()
"""