#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import run_module_suite, assert_allclose

import pywt


def test_upcoef_docstring():
    data = [1, 2, 3, 4, 5, 6]
    (cA, cD) = pywt.dwt(data, 'db2', 'sp1')
    rec = pywt.upcoef('a', cA, 'db2') + pywt.upcoef('d', cD, 'db2')
    expect = [-0.25, -0.4330127, 1., 2., 3., 4., 5.,
              6., 1.78589838, -1.03108891]
    assert_allclose(rec, expect)
    n = len(data)
    rec = (pywt.upcoef('a', cA, 'db2', take=n) +
           pywt.upcoef('d', cD, 'db2', take=n))
    assert_allclose(rec, data)


def test_upcoef_reconstruct():
    data = np.arange(3)
    a = pywt.downcoef('a', data, 'haar')
    d = pywt.downcoef('d', data, 'haar')

    rec = (pywt.upcoef('a', a, 'haar', take=3) +
           pywt.upcoef('d', d, 'haar', take=3))
    assert_allclose(rec, data)


def test_downcoef_multilevel():
    r = np.random.randn(16)
    nlevels = 3
    # calling with level=1 nlevels times
    a1 = r.copy()
    for i in range(nlevels):
        a1 = pywt.downcoef('a', a1, 'haar', level=1)
    # call with level=nlevels once
    a3 = pywt.downcoef('a', r, 'haar', level=3)
    assert_allclose(a1, a3)


def test_upcoef_multilevel():
    r = np.random.randn(4)
    nlevels = 3
    # calling with level=1 nlevels times
    a1 = r.copy()
    for i in range(nlevels):
        a1 = pywt.upcoef('a', a1, 'haar', level=1)
    # call with level=nlevels once
    a3 = pywt.upcoef('a', r, 'haar', level=3)
    assert_allclose(a1, a3)


if __name__ == '__main__':
    run_module_suite()
