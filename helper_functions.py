"""
Monte-Carlo tools for checking out particles living on a spherical surface.

TODO List:
1. muVT ensemble
2. Visualization
"""

#!/anaconda/bin/python
import numpy as np

def get_random_point():
    """Produce a random point on the unit sphere, uniformly distributed."""
    
    z_rand = 2*np.random.random()-1
    phi_rand = 2*np.pi*np.random.random()
    x_rand = np.sqrt(1 - z_rand**2)*np.cos(phi_rand)
    y_rand = np.sqrt(1 - z_rand**2)*np.sin(phi_rand)

    return np.array([x_rand, y_rand, z_rand])

def init_config(n_particles, R):
    """Returns an initial state with particles distributed uniformly over
    a sphere of radius R."""
    
    initial_config = np.zeros((n_particles, 3))
    
    for i in range(0, n_particles):
        initial_config[i] = R*get_random_point()
        
    return initial_config

def get_dists(config):
    """From a given configuration, compute the distances between every particle and
    every other particle without double-counting. Returns an array of all of
    these distances (scalars.) Since the potential energy only depends on the
    distance between two particles, this is useful for computing the total
    energy."""
    
    n_particles = len(config)
    dist_matrix = np.sqrt(np.sum(((config[:, np.newaxis] - config)**2),axis=2))
    return dist_matrix[np.triu_indices(n_particles,1)]

def get_cosines(config, R):
    """From a given configuration, compute the cosines between every particle and
    every other particle without double-counting. Returns an array of all of
    these cosines (scalars.) This is useful for computing the radial
    distribution function."""

    n_particles = len(config)
    cosines_matrix =  np.sum((config[:, np.newaxis]*config),axis=2)/R**2
    return cosines_matrix[np.triu_indices(n_particles,1)]

def get_energy(dists, LJ_depth, LJ_loc):
    """Returns Lennard-Jones energy of a given configuration."""
    
    return np.sum(LJ_depth*((LJ_loc/dists)**12 - 2*(LJ_loc/dists)**6))

normalize_vector = lambda x: x/np.sqrt(x.dot(x))

def displace_particle(vec, sigma, R):
    """Takes a single particle and displaces it by a small amount.
    How it works:
    1. Get small number delta
    2. Get random point on unit sphere
    3. Rescale random point by delta
    4. Add new point 'dvec' to input vector 'vec'
    5. Renormalize 'vec + dvec' to keep it on the original sphere
    """
    
    delta = np.random.normal(0,sigma)
    dvec = delta*get_random_point()
    vec += dvec
    vec = R*normalize_vector(vec)

    return vec
