#!/usr/bin/env python
# 
# Copyright (c) 2012 Kyle Gorman
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# 
# em2gaus.py: Expectation maximization for mixture of two univariate Gaussians
# Kyle Gorman <kgorman@ling.upenn.edu>
# 
# This code depends on numpy
# 
# See USAGE below for user instructions.    
# 
# The EM procedure was introduced in: 
# 
# A. Dempster, N. Laird, and D. Rubin. 1977. Maximum likelihood from incomplete
# data via the EM algorithm. Journal of the Royal Statistical Society, Series 
# B 39(1): 1-38.
# 
# The update rules for mixtures of two univariate Gaussians are taken from 
# course notes for COGS 502 at the University of Pennsylvania, taught by Mark 
# Liberman and Stephen Isard. This particular code was written for my 2012 
# University of Pennsylvania dissertation, though wasn't ultimately used.
#
# THIS IS EXPERIMENTAL CODE: PLEASE USE AT YOUR OWN RISK.

from random import uniform
from math import sqrt, log, exp, pi

from stats import chisqprob
# requires:
# 
# http://www.nmr.mgh.harvard.edu/Neural_Systems_Group/gary/python/stats.py
# http://www.nmr.mgh.harvard.edu/Neural_Systems_Group/gary/python/pstat.py

## default parameters
_mix = .5
_mu_min = 0.
_mu_max = 1.
_sigma_min = .1
_sigma_max = 1.
## number of random restarts
_n_iterations = 10
_rand_restarts = 1000

class Gaussian(object):
    """ 
    Class representing a single univariate Gaussian
    """ 

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __repr__(self):
        return 'Gaussian({0:4.6}, {1:4.6})'.format(self.mu, self.sigma)

    def pdf(self, datum):
        """
        Returns the probability of a datum given the current parameters
        
        Write this in pure Python or scipy later
        """
        u = (datum - self.mu) / abs(self.sigma)
        y = (1 / (sqrt(2 * pi) * abs(self.sigma))) * exp(-u * u / 2)
        return y


class GaussianMixture(object):
    """
    Class representing mixture of two univariate Gaussians and their EM 
    estimation
    
    data:      iterable of numerical values
    mu_min:    minimum start value for mean
    mu_max:    maximum start value for mean
    sigma_min: minimum start value for sigma
    sigma_max: maximum start value for sigma
    mix:       optional mixing parameter (default: .5)
    """

    def __init__(self, data, mu_min, mu_max, sigma_min, sigma_max, mix=_mix):
        self.data = data
        self.one = Gaussian(uniform(mu_min, mu_max), 
                            uniform(sigma_min, sigma_max))
        self.two = Gaussian(uniform(mu_min, mu_max), 
                            uniform(sigma_min, sigma_max))
        self.mix = mix

    def Estep(self):
        """
        Perform an E(stimation)-step, freshening up self.loglike in the process
        and yielding tuples of weights (for whatever purpose)
        """
        # compute weights
        self.loglike = 0. # = log(p = 1)
        for datum in self.data:
            # unnormalized weights
            wp1 = self.one.pdf(datum) * self.mix
            wp2 = self.two.pdf(datum) * (1. - self.mix)
            # compute denominator
            den = wp1 + wp2
            # normalize
            wp1 /= den
            wp2 /= den
            # add into loglike
            self.loglike += log(wp1 + wp2)
            # yield weight tuple
            yield (wp1, wp2)

    def Mstep(self, weights):
        """
        Perform an M(aximization)-step
        """
        # compute denominators
        (left, rigt) = zip(*weights)
        one_den = sum(left)
        two_den = sum(rigt)
        # compute new means
        self.one.mu = sum(w * d / one_den for (w, d) in zip(left, data))
        self.two.mu = sum(w * d / two_den for (w, d) in zip(rigt, data))
        # compute new sigmas
        self.one.sigma = sqrt(sum(w * ((d - self.one.mu) ** 2) \
                                      for (w, d) in zip(left, data)) / one_den)
        self.two.sigma = sqrt(sum(w * ((d - self.two.mu) ** 2) \
                                      for (w, d) in zip(rigt, data)) / two_den)
        # compute new mix
        self.mix = one_den / len(data)

    def iterate(self, N=1, verbose=False):
        """
        Perform N iterations, then compute log-likelihood
        """
        for i in xrange(1, N + 1):
            self.Mstep(self.Estep())
            if verbose:
                print '{0:2} {1}'.format(i, self)
        self.Estep() # to freshen up self.loglike

    def __repr__(self):
        return 'GaussianMixture({0}, {1}, mix={2.03})'.format(self.one, 
                                                        self.two, self.mix)

    def __str__(self):
        return 'Mixture: {0}, {1}, mix={2:.03})'.format(self.one, 
                                                        self.two, self.mix)


if __name__ == '__main__':

    from numpy import mean, std
    from sys import argv, stderr

    if len(argv) != 2: 
        exit('USAGE: ./em2gaus.py DATA')

    ## read in data
    data = [float(d) for d in open(argv[1], 'r')]

    ## compute unimodal model
    uni = Gaussian(mean(data), std(data))
    uni_loglike = sum(log(uni.pdf(d)) for d in data)

    print 'Best singleton: {0}'.format(uni)
    print 'Null LL: {0:4.6}'.format(uni_loglike)

    ## find best one
    # set defaults
    best_gaus = None
    best_loglike = float('-inf')
    stderr.write('Computing best model with random restarts...\n')
    for i in xrange(_rand_restarts):
        mix = GaussianMixture(data, _mu_min, _mu_max, _sigma_min, _sigma_max)
        # I catch division errors from bad starts, and just throw them out...
        for i in xrange(_n_iterations):
            try:
                mix.iterate()
                if mix.loglike > best_loglike:
                    best_loglike = mix.loglike
                    best_gaus = mix
            except (ZeroDivisionError, ValueError):
                pass
    print 'Best {0}'.format(best_gaus)
    print 'Alternative LL: {0:4.6}'.format(best_gaus.loglike)
    test_stat = -2 * uni_loglike + 2 * best_gaus.loglike
    print 'Test statistic for LLR (Chi-sq, df=3): {0:4.6}'.format(test_stat)
    print 'P = {0:4.6}'.format(chisqprob(test_stat, 3))