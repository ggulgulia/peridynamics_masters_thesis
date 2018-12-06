"""
this file is for managing the calculation of influence function 
for peridynamic function
"""
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def gaussian_infl_fun1(zeta, horizon):
    """
    returns a guassian influence function 
    with horizon multiplied by 0.5

    """
    if len(np.shape(zeta))>1: #if we get array of bond vectors
        bnd_len = la.norm(zeta, 2, axis=1)
        return -np.exp(-(bnd_len**2/(0.5*horizon)**2))*(bnd_len<horizon).astype(float)
    else: #if only one bond vector
        bnd_len = la.norm(zeta,2, axis=0)
        return -np.exp(-(bnd_len**2/(0.5*horizon)**2))*float(bnd_len<horizon)


def gaussian_infl_fun2(zeta, horizon):
    """
    returns a guassian influence function

    """
    if len(np.shape(zeta))>1: #if we get array of bond vectors
        bnd_len = la.norm(zeta, 2, axis=1)
        return -np.exp(-(bnd_len**2/(horizon)**2))*(bnd_len<horizon).astype(float)
    else: #if only one bond vector
        bnd_len = la.norm(zeta,2, axis=0)
        return -np.exp(-(bnd_len**2/(horizon)**2))*float(bnd_len<horizon)



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
        return -val 
    else:
        bnd_len = la.norm(zeta, 2, axis=0)
        scaled_bnd_len = bnd_len/horizon
        if scaled_bnd_len<0.5:
            return -1.0
        else:
            return -(-4.0*scaled_bnd_len**2 + 4.0*scaled_bnd_len)*(bnd_len<horizon)


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
        return -val
    else:
        bnd_len = la.norm(zeta,2, axis=0)
        val = (bnd_len/horizon)**2*(bnd_len<horizon)*float(bnd_len<horizon)
        if val<0.0:
            val += 1
        return -val



def inverted_parabolic_infl_fun(zeta, horizon):
    """
    returns an inverted parabolic influence function 
    to give more weight to distant peridynamic points
    compared to nearby points
    """
    if len(np.shape(zeta))>1:
        bnd_len = la.norm(zeta,2, axis=1)
        return -(bnd_len/horizon)**2*(bnd_len<horizon).astype(float)
    else:
        bnd_len = la.norm(zeta,2, axis=0)
        return -(bnd_len/horizon)**2*(float(bnd_len<horizon))



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
        return 1.0*float(bnd_len < horizon)


def plot_1D_influence_functions():

    """
    this method simply plots the different influence
    functions that have been defined above, but only 
    in 1D. The purpose here is to only give an intution
    behind the choice of influence functions used

    usage: simply call the function and observe the plots
    """

    def omega1(domain, horizon):
        """
        1d unit unit_influence function 
        """
        return np.ones(len(domain),dtype=float)*(abs(domain)<horizon).astype(float)
    
    def omega2(domain, horizon):
        """
        1d gaussian influence function 
        """
        bnd_len = np.abs(domain)

        return np.exp(-(bnd_len/horizon)**2)*(np.abs(domain)<horizon).astype(float)

    def omega3(domain, horizon):
        """
        1d inverted parabolic influence function 
        """

        return (abs(domain)/horizon)**2*(abs(domain)<horizon).astype(float)

    def omega4(domain, horizon):
        """
        parabolic influence function 
        """

        val = -(abs(domain)/horizon)**2*(abs(domain)<horizon).astype(float)
        for i in np.where(val<0.0):
            val[i] += 1.0
        return val

    def omega5(domain, horizon):
        """
        peridigm++ like parabolic influence function 
        """
        scaled_bnd_len = np.abs(domain)/horizon

        val = (-4.0*scaled_bnd_len**2 + 4.0*scaled_bnd_len)*(np.abs(domain)<horizon).astype(float)
        for i in np.where(scaled_bnd_len<0.5):
            val[i] = 1.0
        return val

    deltaX  = 0.001
    domain = np.arange(-1.0, 1.0+deltaX, deltaX)
    horizon = 0.4
    plt.figure()
    plt.plot(domain, omega1(domain, horizon), label='unit')
    plt.plot(domain, omega2(domain, horizon), label='exponential')
    plt.plot(domain, omega3(domain, horizon), label='invert parabola')
    plt.plot(domain, omega4(domain, horizon), label='parabola')
    plt.plot(domain, omega5(domain, horizon), label='peridigm parabola')

    plt.legend(loc=2)
    plt.xlim(-1.0,1.0)
    plt.ylim(-0.5, 1.5)
    plt.xlabel("bond length")
    plt.ylabel("value of influene function")
    plt.title("Various peridynamic influence functions(tested with horizon="+str(horizon)+")")
    plt.show(block=False)

    pass
