from math import *
import numpy as np
import numpy.linalg as la
from inflfun import *
import matplotlib.pyplot  as plt
import math as m



def create_discretized_horizon(horizon=0.003, steps=10):
    """
    input:
    ------
        horion: max length of horizon
        steps: number of sub divisions desired in horizon 
    returns:
    -------
        a horizon with 'step' numbers of discrete subdivisions
    """
    step_size = horizon/float(steps)
    return np.arange(0., horizon+step_size, step_size)

def compute_weighted_mass(delta_divs, omega):

    horizon = delta_divs[-1]
    step_size = delta_divs[1] - delta_divs[0]
    
    mw = 0.0
    for i in range(len(delta_divs)):
        curr_bnd = step_size*i
        mw += omega(curr_bnd, horizon)*(curr_bnd)**2*step_size

    return 2*mw

def dispersion_function(E, rho, delta_divs, ki, omega):
    """
    input:
    ------
        E : youngs modulus of elasticity
        rho: material density
        mw : weighted volume/mass
        delta_divs : array of discritized horion
        ki: integral wave nubmer
    output:
    ------
        angular frequncey as a function of ki 
    """

    step_size = delta_divs[1] - delta_divs[0]
    horizon = delta_divs[-1]
    N = len(delta_divs)
    mw = compute_weighted_mass(delta_divs, omega) 
    sum1=0.0; sum2=0.0;

    for i in range(N):
        sum2 += omega(i*step_size, horizon)*(1.0 - m.cos(ki*i*step_size))*step_size
        sum1 += omega(i*step_size, horizon)*m.sin(ki*i*step_size)*i*step_size**2

    sum1 = sum1**2/mw
    omega_sq = 2*E/(rho*mw)*(sum1 + sum2)
    return math.sqrt(omega_sq)

def evaluate_wave_frq(kappa_max, E, rho, delta_divs, omega):

    wav_freq = np.zeros(kappa_max, dtype=float)
    for ki in range(kappa_max):
        wav_freq[ki] =  dispersion_function(E, rho, delta_divs, ki, omega)

    return wav_freq


def plot_wav_disp(wave_freqs, horizons):
    """
    plots the wave frequency vs the wave number
    """

    plt.figure()
    x_axis = np.arange(0, len(wave_freqs[0]), 1)

    figure = plt.plot(x_axis, x_axis, label='wave equn')

    for i in range(len(horizons)):
        figure = plt.plot(x_axis, wave_freqs[i],label=str(horizons[i]*1000)+' mm')
    
    plt.legend(loc=2)
    plt.xlim(0, len(x_axis))
    plt.ylim(0, len(x_axis))
    return figure

def evaluate_varying_horizons():

    steps = 3
    #horizons = np.zeros(num_horizons, dtype=float)
    horizons = np.array([3, 2.325, 1.65, 0.975, 0.3])/1000
    horizon_divs = np.zeros((len(horizons),steps+1), dtype=float)
    for i in range(len(horizons)):
        horizon_divs[i] = create_discretized_horizon(horizons[i], steps)
    

    #simplification for material properties
    E = 1.0; rho = 1.0; 
    wave_speed = sqrt(E/rho) #1 here
    kappa_max = 3000 #max wave number we need
    
    #establish which influence functions we'd like
    #choices are:
    # 1. unit_infl_fun
    # 2. gaussian_infl_fun
    # 3. parabolic_decay
    omega = unit_infl_fun
    
    wave_freqs = np.zeros((len(horizons), kappa_max), dtype=float)

    for i in range(len(horizons)):
        wave_freqs[i][:] = evaluate_wave_frq(kappa_max, E, rho, horizon_divs[i], omega)


    fig = plot_wav_disp(wave_freqs, horizons)
    plt.show(block=False)

    return wave_freqs, horizons
