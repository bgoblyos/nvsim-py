#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2025 Bence Göblyös

import numpy as np
import pandas as pd
import scipy as sci
import matplotlib.pyplot as plt
import json
from uncertainties import ufloat
from itertools import combinations, chain

# %% Load data

tmp = []

for i in [1, 2]:
    with open(f"./measurement{i}.json") as f:
        raw = json.load(f)
        for entry in raw:
            tmp.append({
                "D": entry["D"]["value"],
                "E": entry["E"]["value"],
                "freqs": np.array(list(map(lambda x: x["value"], entry["freqs"]))),
                "sigmas": np.array(list(map(lambda x: x["uncertainty"], entry["freqs"])))
        })
    
data = pd.DataFrame.from_dict(tmp)

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

def getHNV(D, E):
    return MHzToK * (D * (spinZ2 - (spinX2 + spinY2 + spinZ2) / 3) + E * (spinX2 - spinY2))


# Return Zeeman splitting Hamiltonian without any magnitude scaling
def unscaledZeeman(dir):
    return gToK * 10 * g * (dir[0] * spinX + dir[1] * spinY + dir[2] * spinZ)
    
# Generate Hamiltonians for all 4 orientations
def hamiltonians(Bvec, D, E):
    B = np.linalg.norm(Bvec)
    rot = R(Bvec/B, np.array([0, 0, 1]))
    HNV = getHNV(D, E)
    return np.array([
        B * unscaledZeeman(np.matmul(rot, dirs[0])) + HNV,
        B * unscaledZeeman(np.matmul(rot, dirs[1])) + HNV,
        B * unscaledZeeman(np.matmul(rot, dirs[2])) + HNV,
        B * unscaledZeeman(np.matmul(rot, dirs[3])) + HNV,
    ])

def expectedPeaks(B, D, E):
    Hs = hamiltonians(B, D, E)
    
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

def sphericalSim(x, theta, phi, B):
    return expectedPeaks(x, *sphericalToCartesian(B, theta, phi))
    

def singleCartesianFit(freqs, sigmas, D, E): 
    def wrapper(x, Bx, By, Bz):
        return expectedPeaks(np.array([Bx, By, Bz]), D, E)
    
    
    res, cov = sci.optimize.curve_fit(
        wrapper,
        [0],
        freqs,
        sigma=sigmas,
        absolute_sigma=True,
        maxfev=100000
    )


    unc = np.sqrt(np.diag(cov))
    errs = unc / res
    
    SStot = np.sum((freqs - np.mean(freqs)) ** 2)
    SSres = np.sum((freqs - expectedPeaks(res, D, E)) ** 2)
    
    R2 = 1 - SSres/SStot
    
    print(f'Norm: {np.linalg.norm(res)} mT\n')
    print(f'Max error: {np.max(errs):e}\n')
    print('{:+.2uS}'.format(ufloat(res[0], unc[0])))
    print('{:+.2uS}'.format(ufloat(res[1], unc[1])))
    print('{:+.2uS}'.format(ufloat(res[2], unc[2])))
    
    return {
        "B": res,
        "Uncertainty": unc,
        "R2": R2
    }

def multiFit(freqs_arr, sigmas_arr, D, E):
    n = len(freqs_arr)
    p = n + 2
    freqs = np.hstack(freqs_arr)
    sigmas = np.hstack(sigmas_arr)
    Ds = np.array(D)
    Es = np.array(E)
    
    
    
    def wrapper(x, theta, phi, *B):
        acc = []
        for i in range(n):
            acc.append(
                expectedPeaks(
                    sphericalToCartesian(B[i], theta, phi),
                    Ds[i],
                    Es[i]
                )
            )
        return np.hstack(acc)
    
    
    res, cov = sci.optimize.curve_fit(
        wrapper,
        [0],
        freqs,
        p0 = np.ones(p),
        sigma=sigmas,
        absolute_sigma=True,
        maxfev=100000
    )


    unc = np.sqrt(np.diag(cov))
    errs = unc / res
    
    SStot = np.sum((freqs - np.mean(freqs)) ** 2)
    SSres = np.sum((freqs - wrapper(0, *res)) ** 2)
    
    dftot = len(freqs) - 1
    dfres = len(freqs) - p - 1
    
    R2 = 1 - SSres/SStot
    R2A = 1 - (SSres / dfres)/(SStot / dftot)
    
    Chi2 = np.sum((freqs - wrapper(0, *res)) ** 2 / (sigmas ** 2))
    Chi2R = Chi2 / p
    
    print(f'Max error: {np.max(errs):e}\n')
    print('theta: {:+.2uS}'.format(ufloat(res[0], unc[0])))
    print('phi:   {:+.2uS}'.format(ufloat(res[1], unc[1])))
    for i in range(n):
        print('B{}:    {:+.2uS}'.format(i+1, ufloat(res[i+2], unc[i+2])))
    
    return {
        "B": res,
        "Uncertainty": unc,
        "R2": R2,
        "R2A": R2A
    }

# %% Perform single fit

i = 0
singleCartesianFit(data.freqs[i], data.sigmas[i], data.D[i], data.E[i])

# %% Fit multiple

# First 4 samples corresponding to one orientation
i = range(0,4)

multiFit(data.freqs[i], data.sigmas[i], data.D[i], data.E[i])

# %% Estimate impact on R2

tmp = []

for i in range(4):
    j = range(0, i+1)
    res =  multiFit(data.freqs[j], data.sigmas[j], data.D[j], data.E[j])
    tmp.append(res)
   
series1 = pd.DataFrame.from_dict(tmp)

tmp = []

for i in range(4):
    j = range(4, i+5)
    res =  multiFit(data.freqs[j], data.sigmas[j], data.D[j], data.E[j])
    tmp.append(res)
   
series2 = pd.DataFrame.from_dict(tmp)

plt.plot([1, 2, 3, 4], series1.R2)
plt.plot([1, 2, 3, 4], series2.R2)
plt.show()

plt.plot([1, 2, 3, 4], series1.R2A, 'o-')
plt.plot([1, 2, 3, 4], series2.R2A, 'o-')
plt.xlabel("Spektrumok száma")
plt.xticks([1, 2, 3, 4])
plt.ylabel("Csökkentett R²")
plt.show()

# %%

results1 = []
s = {0, 1, 2, 3}

for n in range(1,5):
    tmp = []
    for i in combinations(s, n):
        j = list(i)
        res =  multiFit(data.freqs[j], data.sigmas[j], data.D[j], data.E[j])
        tmp.append(res["R2A"])
    results1.append(np.array(tmp))
    
Rs1 = list(chain(*results1))

lengths = list(map(len, results1))

params1 = []
for i, l in enumerate(lengths):
    params1 += ([i+1] * l)
  

plt.scatter(params, Rs)

results2 = []
s = {4, 5, 6, 7}

for n in range(1,5):
    tmp = []
    for i in combinations(s, n):
        j = list(i)
        res =  multiFit(data.freqs[j], data.sigmas[j], data.D[j], data.E[j])
        tmp.append(res["R2A"])
    results2.append(np.array(tmp))
    
Rs2 = list(chain(*results2))

lengths = list(map(len, results))

params2 = []
for i, l in enumerate(lengths):
    params2 += ([i+1] * l)
  

plt.scatter(params1, Rs1)
plt.show()

# %%

means = list(map(np.mean, results1))
devs = list(map(np.std, results1))
plt.errorbar(range(1,5), means, yerr=devs, fmt="o")


means = list(map(np.mean, results2))
devs = list(map(np.std, results2))
plt.errorbar(range(1,5), means, yerr=devs, fmt="o")

plt.show()