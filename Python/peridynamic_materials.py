"""this file is inteneded to take advantage of various material properties and compute other mechanical properties of materials needed for peridynamic computation

formulas for computation of properties: 
# constant needed in plane stress (compare Handbook Section 6.3.1.1)
nu=0.3
E=200e9
bulk = E/(3*(1 - (2*nu)))
mu=E/(2*(1 + nu))
gamma(for 2d) = 4*mu / (3*bulk + 4*mu)
"""

def compute_mu(E, nu):
    """ 
    returns the shear modulus of elasticity
    input:
    ------
        E : youngs modulus of elasticity
        nu: poissons ratio 
    """
    return 0.5*E/(1+nu)

def compute_bulk(E, nu, dim):
    """
    returns the bulk modulus of elasticity (needed only for 2D problems)
    """
    if dim == 2:
        return E/(3*(1 - (2*nu))) 
    if dim == 3:
        return E*nu/(1+nu)/(1-2*nu)


def compute_gamma(mu, bulk, dim):
    """
    returns the gamma (needed only for 2D problems)
    """
    if dim == 2:
        return 4*mu/(3*bulk + 4*mu)
    else:
        return 1.0


def get_steel_properties(dim=2):
    """
    returns the mechanical properties of steel
    in SI Units
    (source: google.com)
    input:
    ------
        dim : geometric dimension of the problem
    output:
    -------
        E : youngs modulus
        nu: poissons ratio
        rho: density
        mu:
        bulk:
        gamma:
    """
    E = 200e10 
    nu = 0.3
    rho = 8050 #density kg/m3
    mu    = compute_mu(E, nu)
    bulk  = compute_bulk(E, nu, dim)
    gamma = compute_gamma(mu, bulk, dim)

    return E, nu, rho, mu, bulk, gamma

def get_aluminum_properties(dim=2):
    """
    returns the mechanical properties of steel
    in SI Units
    (source: google.com)
    input:
    ------
        dim : geometric dimension of the problem
    output:
    -------
        E : youngs modulus
        nu: poissons ratio
        rho: density
        mu:
        bulk:
        gamma:
    """
    E = 69e10 
    nu = 0.334
    rho = 2700 #density in kg/m3
    mu    = compute_mu(E, nu)
    bulk  = compute_bulk(E, nu, dim)
    gamma = compute_gamma(bulk, mu, dim)

    return E, nu, rho, mu, bulk, gamma

def get_concrete_properties(dim=2):
    """
    TODO: implement
    """
    pass 
