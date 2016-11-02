from distutils.core import setup

setup(name='pyboas',
      description='Bayesian Observation predictor and Adaptive Scheduler.',
      version='0.1.0',
      author='Rodrigo F. Diaz',
      author_email='rodrigo.diaz@unige.ch',
      #url='http://obswww.unige.ch/~diazr/pygpr',
      packages=['pyboas'],
      requires=['numpy', 'scipy']
      )
