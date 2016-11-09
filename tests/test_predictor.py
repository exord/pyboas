#! /usr/env/bin python
import numpy as np

from pyboas import predictor, models

# Build random 3-parameter normal posterior.
posterior = np.random.randn(100, 3)


def toy_model(param, time):
    time = np.atleast_1d(time)[:, np.newaxis]

    a = param[:, 0]
    b = param[:, 1]
    c = param[:, 2]

    return a*time**2 + b*time + c


def test_basic_shape():
    """Test basic shape conditions on output of predictions."""
    time = np.random.rand(4, )

    pred1 = predictor.GaussPredictor(posterior, toy_model)

    pred1.make_prediction(time, scale=1)

    # Test shape of predictive distributions and x
    assert pred1.x.shape == pred1.predictives.shape
    # Test len of time array and predictives
    assert len(time) == len(pred1.predictives)

    return


def test_time_concatenation():
    """
    Test feature to concatenate prediction times over make_prediction calls.
    """

    # Built random time array
    time = np.random.rand(4,)

    pred1 = predictor.GaussPredictor(posterior, toy_model)
    pred2 = predictor.GaussPredictor(posterior, toy_model)

    # Run first predictor with full time array
    pred1.make_prediction(time, scale=1)

    # Run second predictor twice
    pred2.make_prediction(time[:2], scale=1)
    pred2.make_prediction(time[2:], scale=1)

    assert np.allclose(pred1.predictives, pred2.predictives)
    assert np.allclose(pred1.x, pred2.x)

    return


def test_sample_draw():
    # Built random time array
    time = np.random.rand(4, )

    pred1 = predictor.GaussPredictor(posterior, toy_model, likescale=1)

    pred1.samplepredictive(time, 100)


def ok():
    print('\033[92mOK\033[0m')


def failed():
    print('\033[91mFAILED\033[0m')


def test_all():
    print('Testing basic functioning....\t'),
    try:
        test_basic_shape()
        ok()
    except AssertionError:
        failed()

    print('Testing time concatenation....\t'),
    try:
        test_time_concatenation()
        ok()
    except AssertionError:
        failed()
    return

if __name__ == '__main__':
    test_all()
