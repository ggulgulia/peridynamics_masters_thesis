"""
this file is for managing the calculation of influence function 
for peridynamic function
"""
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from fenics_mesh_tools import get_colors

def gaussian_infl_fun1(zeta, horizon):
    """
    returns a guassian influence function 
    with horizon multiplied by 0.5

    """
    if len(np.shape(zeta))>1: #if we get array of bond vectors
        bnd_len = la.norm(zeta, 2, axis=1)
        return np.exp(-(bnd_len**2/(0.5*horizon)**2))*(bnd_len<horizon).astype(float)
    else: #if only one bond vector
        bnd_len = la.norm(zeta,2, axis=0)
        return np.exp(-(bnd_len**2/(0.5*horizon)**2))*float(bnd_len<horizon)


def gaussian_infl_fun2(zeta, horizon):
    """
    returns a guassian influence function

    """
    if len(np.shape(zeta))>1: #if we get array of bond vectors
        bnd_len = la.norm(zeta, 2, axis=1)
        return np.exp(-(bnd_len**2/(horizon)**2))*(bnd_len<horizon).astype(float)
    else: #if only one bond vector
        bnd_len = la.norm(zeta,2, axis=0)
        return np.exp(-(bnd_len**2/(horizon)**2))*float(bnd_len<horizon)



def parabolic_infl_fun1(zeta, horizon):
    """
    returns a parabolically decaying influence function 
    : borrowed from peridigm++ code
    """
    if len(np.shape(zeta))>1:
        bnd_len = la.norm(zeta, 2, axis=1)
        scaled_bnd_len = bnd_len/horizon

        val = (-4.0*scaled_bnd_len**2 + 4.0*scaled_bnd_len)*(bnd_len<horizon).astype(float)
        for i in np.where(scaled_bnd_len<0.5):
            val[i] = 1.0
        return val 
    else:
        bnd_len = la.norm(zeta, 2, axis=0)
        scaled_bnd_len = bnd_len/horizon
        if scaled_bnd_len<0.5:
            return 1.0
        else:
            return (-4.0*scaled_bnd_len**2 + 4.0*scaled_bnd_len)*(bnd_len<horizon)


def parabolic_infl_fun2(zeta, horizon):
    """
    returns a pure parabolic function giving higher
    weights to nearby points
    """
    if len(np.shape(zeta))>1:
        bnd_len = la.norm(zeta, 2, axis=1)
        val = -(bnd_len/horizon)**2*(bnd_len<horizon).astype(float)
        for i in np.where(val<0.0):
            val[i] += 1.0
        return val
    else:
        bnd_len = la.norm(zeta,2, axis=0)
        val = (bnd_len/horizon)**2*float(bnd_len<horizon)
        if val<0.0:
            val += 1
        return val



def inverted_parabolic_infl_fun(zeta, horizon):
    """
    returns an inverted parabolic influence function 
    to give more weight to distant peridynamic points
    compared to nearby points
    """
    if len(np.shape(zeta))>1:
        bnd_len = la.norm(zeta,2, axis=1)
        return (bnd_len/horizon)**2*(bnd_len<horizon).astype(float)
    else:
        bnd_len = la.norm(zeta,2, axis=0)
        return (bnd_len/horizon)**2*(float(bnd_len<horizon))



def cubic_infl_func(bnd_len, horizon):
    """
    TODO: complete implementation
    """
    pass


def unit_infl_fun(zeta, horizon):
    """
    
    """
    if len(np.shape(zeta))>1:
        bnd_len = la.norm(zeta, 2, axis=1)
        return np.ones(len(bnd_len))*(bnd_len<horizon).astype(float)
    else:
        bnd_len = la.norm(zeta, 2, axis=0)
        return 1.0*float(abs(bnd_len) < horizon)


def plot_1D_influence_functions():

    """
    this method simply plots the different influence
    functions that have been defined above, but only 
    in 1D. The purpose here is to only give an intution
    behind the choice of influence functions used

    usage: simply call the function and observe the plots
    """
    
    def omega1(domain, horizon, p=1):
        """
        1d gaussian influence function 
        """
        bnd_len = np.abs(domain)

        return np.exp(-(bnd_len/(p*horizon))**2)*(np.abs(domain)<horizon).astype(float)
        
    def omega2(domain, horizon, q=0.5):
        """
        1d narrow gaussian influence function 
        """
        bnd_len = np.abs(domain)

        return np.exp(-(bnd_len/(q*horizon))**2)*(np.abs(domain)<horizon).astype(float)
    

    def omega3(domain, horizon):
        """
        parabolic influence function 
        """

        val = -(abs(domain)/horizon)**2*(abs(domain)<horizon).astype(float)
        for i in np.where(val<0.0):
            val[i] += 1.0
        return val

    def omega4(domain, horizon):
        """
        peridigm++ like parabolic influence function 
        """
        scaled_bnd_len = np.abs(domain)/horizon

        val = (-4.0*scaled_bnd_len**2 + 4.0*scaled_bnd_len)*(np.abs(domain)<horizon).astype(float)
        for i in np.where(scaled_bnd_len<0.5):
            val[i] = 1.0
        return val

    def omega5(domain, horizon):
        """
        1d unit unit_influence function 
        """
        return np.ones(len(domain),dtype=float)*(abs(domain)<horizon).astype(float)
    
    #def omega3(domain, horizon):
    #    """
    #    1d inverted parabolic influence function 
    #    """

    #    return (abs(domain)/horizon)**2*(abs(domain)<horizon).astype(float)
    deltaX  = 0.001
    domain = np.arange(-1.0, 1.0+deltaX, deltaX)
    horizon = 0.5

    ### to make plots pretty ###
    dpi = 3
    axis_font = {'size': str(int(3*dpi))}
    title_font = {'size': str(18*dpi)}
    legend_size = {'size': str(8*dpi)}
    tick_size = 12*dpi
    marker_size=100*3.5*dpi
    
    colors = get_colors(6)
    plt.figure()
    plt.plot(domain, omega1(domain, horizon), color=colors[1], linewidth=3.0, label=r'$\mathbf{\omega}_{1}\langle\xi\rangle$')
    plt.plot(domain, omega2(domain, horizon), color=colors[2], linewidth=3.0, label=r'$\mathbf{\omega}_{2}\langle\xi\rangle$')
    plt.plot(domain, omega3(domain, horizon), color=colors[3], linewidth=3.0, label=r'$\mathbf{\omega}_{3}\langle\xi\rangle$')
    plt.plot(domain, omega4(domain, horizon), color=colors[4], linewidth=3.0, label=r'$\mathbf{\omega}_{4}\langle\xi\rangle$')
    plt.plot(domain, omega5(domain, horizon), color=colors[5], linewidth=3.0, label=r'$\mathbf{\omega}_{5}\langle\xi\rangle$')

    plt.legend(loc=1,prop = legend_size)
    plt.xlim(-1.0,1.0); plt.ylim(-0.5, 1.5)
    plt.xticks(fontsize=16); plt.yticks(fontsize=16) 
    plt.xlabel(r'bond length $\xi$', fontsize=10*dpi, **axis_font)
    plt.ylabel(r'influence function $\mathbf{\omega}_{i}\langle\xi\rangle$', fontsize=10*dpi, **axis_font)
    plt.title("Various peridynamic influence functions")
    plt.show(block=False)

    pass


def get_influence_function_symbol():
    """
    returns latex like math symbol for influence function
    refer Perdiynamic literature for symbol convention
    :returns: TODO

    """

    omega1 = r'$\omega_1\langle\xi\rangle$'
    omega2 = r'$\omega_2\langle\xi\rangle$'
    omega3 = r'$\omega_3\langle\xi\rangle$'
    omega4 = r'$\omega_4\langle\xi\rangle$'
    omega5 = r'$\omega_5\langle\xi\rangle$'
    omega6 = r'$\omega_6\langle\xi\rangle$'
    infl_fun_symbols = {'omega1':omega1, 'omega2':omega2, 'omega3':omega3, 'omega4':omega4, 'omega5':omega5, 'omega6':omega6}
    return infl_fun_symbols

