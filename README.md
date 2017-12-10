Searches for the best TensorForce hyperparameter combination using Bayesian Optimization (Gaussian Processes). Currently only searching over PPOAgent hypers. WIP!

Requirements: gym, GPy, GPyOpt, psycopg2, python-box

1. Setup Postgres, `createdb hypersearch`
1. $ python hsearch.py

## Bayesian Optimization Code
`gp.py` comes from [thuijskens/bayesian-optimization](https://github.com/thuijskens/bayesian-optimization), blog post [here](https://thuijskens.github.io/2016/12/29/bayesian-optimisation/). This uses sklearn's built-in Gaussian Process feature with minimal overhead, and maximal flexibility. [GPyOpt](https://github.com/SheffieldML/GPyOpt) is a great alternative which wraps sklearn's GP and provides many configurations & sane defaults. I'm still comparing the two, and might bake both in with an argument to choose. LMK if y'all have thoughts.

I considered various alternatives besides: [optunity](http://optunity.readthedocs.io/en/latest/), [hyperopt](https://github.com/hyperopt/hyperopt), and more (see this [Reddit post](https://www.reddit.com/r/MachineLearning/comments/4g2rnu/bayesian_optimization_for_python/)). They didn't meet my needs for various reasons: they don't use Bayesian Opt, or they're too strict / inflexible to work easily with TensorForce. IM me on gitter/tensorforce for more details.