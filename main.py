#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2025 Bence Göblyös

import numpy as np
import scipy as sci
import json
from uncertainties import ufloat

# %% Load data

with open("./measurement1.json") as f:
    raw = json.load(f)

i = 3
freqs = np.array(list(map(lambda x: x["value"], raw[i]["freqs"])))
sigmas = np.array(list(map(lambda x: x["uncertainty"], raw[i]["freqs"])))

D = raw[i]["D"]["value"]
E = raw[i]["E"]["value"]

multifreq = np.reshape(list(map(lambda i : np.array(list(map(lambda x: x["value"], raw[i]["freqs"]))), range(4))), (32))
multisigma = np.reshape(list(map(lambda i : np.array(list(map(lambda x: x["uncertainty"], raw[i]["freqs"]))), range(4))), (32))

# %% Constants

MHzToK = 4.7991939324e-5
GHzToK = 4.7991939324e-2
gToK = 6.71714e-5
g = 2.0026

spinX = np.matrix('0 1 0; 1 0 1; 0 1 0') / np.sqrt(2)
spinY = np.matrix('0 0-1j 0; 0+1j 0 0-1j; 0 0+1j 0') / np.sqrt(2)
spinZ = np.matrix('1 0 0; 0 0 0; 0 0 -1')

spinX2 = np.matmul(spinX, spinX)
spinY2 = np.matmul(spinY, spinY)
spinZ2 = np.matmul(spinZ, spinZ)

dirs = np.array([
    [1, 1, 1],
    [1, -1, -1],
    [-1, 1, -1],
    [-1, -1, 1]
]) / np.sqrt(3)

HNV = MHzToK * (D * (spinZ2 - (spinX2 + spinY2 + spinZ2) / 3) + E * (spinX2 - spinY2))

# %% Functions

# Generate rotation matrix to bring v1 to v2
def R(v1, v2):
    u = np.cross(v1, v2)
    c = np.dot(v1, v2)
    s = np.linalg.norm(u)
    u = u / s
    
    asym = np.array([
        [    0, -u[2],  u[1]],
        [ u[2],     0, -u[0]],
        [-u[1],  u[0],     0]        
    ])
    
    return c * np.identity(3) + s * asym + (c - 1) * np.outer(u, u)

def sphericalToCartesian(r, theta, phi):
    return np.array([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta)
    ])

# Return Zeeman splitting Hamiltonian without any magnitude scaling
def unscaledZeeman(dir):
    return gToK * 10 * g * (dir[0] * spinX + dir[1] * spinY + dir[2] * spinZ)
    
# Generate Hamiltonians for all 4 orientations
def hamiltonians(Bvec):
    B = np.linalg.norm(Bvec)
    rot = R(Bvec/B, np.array([0, 0, 1]))
    return np.array([
        B * unscaledZeeman(np.matmul(rot, dirs[0])) + HNV,
        B * unscaledZeeman(np.matmul(rot, dirs[1])) + HNV,
        B * unscaledZeeman(np.matmul(rot, dirs[2])) + HNV,
        B * unscaledZeeman(np.matmul(rot, dirs[3])) + HNV,
    ])

def expectedPeaks(x, Bx, By, Bz):
    B = np.array([Bx, By, Bz])
    Hs = hamiltonians(B)
    
    vals0 = np.sort(np.linalg.eigvals(Hs[0]) / GHzToK)
    vals1 = np.sort(np.linalg.eigvals(Hs[1]) / GHzToK)
    vals2 = np.sort(np.linalg.eigvals(Hs[2]) / GHzToK)
    vals3 = np.sort(np.linalg.eigvals(Hs[3]) / GHzToK)
    
    return np.sort(np.array([
        abs(vals0[0] - vals0[2]),
        abs(vals0[0] - vals0[1]),
        abs(vals1[0] - vals1[2]),
        abs(vals1[0] - vals1[1]),
        abs(vals2[0] - vals2[2]),
        abs(vals2[0] - vals2[1]),
        abs(vals3[0] - vals3[2]),
        abs(vals3[0] - vals3[1]),
    ]))

def multiSim(x, theta, phi, B1, B2, B3, B4):
    return np.hstack((
        expectedPeaks(x, *sphericalToCartesian(B1, theta, phi)),
        expectedPeaks(x, *sphericalToCartesian(B2, theta, phi)),
        expectedPeaks(x, *sphericalToCartesian(B3, theta, phi)),
        expectedPeaks(x, *sphericalToCartesian(B4, theta, phi))
    ))
    
# %% Fit single measurement

res, cov = sci.optimize.curve_fit(
    expectedPeaks,
    [0],
    freqs,
    sigma=sigmas,
    absolute_sigma=True,
    maxfev=100000
)


unc = np.sqrt(np.diag(cov))
errs = unc / res

print(f'Norm: {np.linalg.norm(res)} mT\n')
print(f'Max error: {np.max(errs):e}\n')
print('{:+.2uS}'.format(ufloat(res[0], unc[0])))
print('{:+.2uS}'.format(ufloat(res[1], unc[1])))
print('{:+.2uS}'.format(ufloat(res[2], unc[2])))

# %% Fit all measurements at once, knowing that they share the orientation but has different norms

res, cov = sci.optimize.curve_fit(
    multiSim,
    [0],
    multifreq,
    sigma=multisigma,
    absolute_sigma=True,
    maxfev=100000
)

unc = np.sqrt(np.diag(cov))

print('theta: {:+.2uS}'.format(ufloat(res[0], unc[0])))
print('phi:   {:+.2uS}'.format(ufloat(res[1], unc[1])))
print('B1:    {:+.2uS}'.format(ufloat(res[2], unc[2])))
print('B2:    {:+.2uS}'.format(ufloat(res[3], unc[3])))
print('B3:    {:+.2uS}'.format(ufloat(res[4], unc[4])))
print('B4:    {:+.2uS}'.format(ufloat(res[5], unc[5])))
