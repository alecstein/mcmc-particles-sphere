from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from helper_functions import get_random_point, init_config, get_dists, get_cosines, displace_particle

class BaseEnsemble(object):

    def __init__(self, n_particles = 10, radius = 1, beta = 1):
        # Configuration parameters (particles, volume (area), temperature)        
        self.n_particles = n_particles
        self.radius = radius
        self.beta = beta

        # Initialize the configuration
        self.config = init_config(self.n_particles, self.radius)
        
        # Energetic parameters
        self.LJ_depth = 1 # depth of LJ-minimum
        self.LJ_loc = 1 # location of LJ-minimum

        # Radial distribution function
        self.RDF = None

        # Other parameters
        self.sigma = 0.1

    def get_energy(self, config):
        # Lennard-Jones energy of a configuration
        dists = get_dists(config)
        return np.sum(self.LJ_depth*((self.LJ_loc/dists)**12 - 2*(self.LJ_loc/dists)**6))

    def monte_carlo_move(self):
        pass    

    def equilibrate(self, n_iters = 10**3):
        for i in tqdm(range(n_iters)):
            self.monte_carlo_move()
            
    def get_RDF(self, n_iters = 10**4, n_bins = 10**2, plot = True):
        """RDF = radial distribution function. 
        After equilibrating, do some number of Monte-Carlo moves and estimate
        the radial distribution function. If plot = True, plot it."""
        
        bin_edges = np.linspace(-1, 1, n_bins + 1) 
        hist = np.zeros(n_bins)
    
        for i in tqdm(range(n_iters)):
            self.monte_carlo_move()
            cosines = get_cosines(self.config, self.radius)
            hist = hist + np.histogram(cosines, bin_edges)[0]

        self.RDF = hist*(self.n_particles-1)/(np.sum(hist)*(bin_edges[1]-bin_edges[0]))

        if plot == True:
            plt.figure(figsize=(10,6), dpi = 80)
            plt.bar(bin_edges[:-1], self.RDF, width = np.diff(bin_edges), edgecolor = 'none', color = 'red')
            plt.title('Radial distribution function', fontsize = 18)
            plt.ylabel('$\\rho g(\cos\\theta)$', fontsize = 18)
            plt.xlabel('$\cos\\theta$', fontsize = 18)
            plt.show()

class NVTEnsemble(BaseEnsemble):
    """Fixed particle number, volume (area), and temperature."""

    def __init__(self, n_particles = 10, radius = 1, beta = 1):
        BaseEnsemble.__init__(self, n_particles, radius, beta)
                 
    def try_displacement_move(self):
        """Take one particle and do one 'Monte-Carlo move.'  
        How it works
        1. Choose and displace random particle 
        2. Check if change in
        energy is favorable 
        3. Accept/reject move
        """

        # Choose and displace random particle
        trial_config = np.copy(self.config)
        particle_id = np.random.randint(0,self.n_particles)
        trial_config[particle_id] = displace_particle(trial_config[particle_id], self.sigma, self.radius)

        # Compute change in energy dE
        dE = self.get_energy(trial_config) - self.get_energy(self.config)

        # Accept/reject move (main part of Metropolis algorithm)
        exp_argument = -self.beta*dE
        q = np.float64(np.exp(min(0,exp_argument)))
        p = np.random.random()
        if p <= q:
            self.config = trial_config

    def monte_carlo_move(self):
        self.try_displacement_move()

class muVTEnsemble(NVTEnsemble):
    """Fixed chemical potential, volume (area), and temperature."""

    def __init__(self, n_particles = 10, radius = 1, beta = 1):
        NVTEnsemble.__init__(self, n_particles, radius, beta)
        self.mu = -10000
        self.area = 4*np.pi*self.radius**2

    def try_insert_move(self):
        trial_config = np.copy(self.config)
        new_particle = self.radius*get_random_point()[np.newaxis,:]
        trial_config = np.vstack((trial_config, new_particle))
        
        dE = self.get_energy(trial_config) - self.get_energy(self.config)

        exp_argument = -self.beta*dE + self.beta*self.mu + np.log(self.area/(self.n_particles+1))
        q = np.float64(np.exp(min(0,exp_argument)))
        p = np.random.random()
        if p <= q:
            self.config = trial_config
    
    def try_delete_move(self):
        trial_config = np.copy(self.config)
        # Delete a particle at random
        trial_config = np.delete(trial_config, np.random.randint(self.n_particles), axis = 0)

        dE = self.get_energy(trial_config) - self.get_energy(self.config)

        exp_argument = -self.beta*dE + self.beta*self.mu + np.log(self.area/self.n_particles)
        
        q = np.float64(np.exp(min(0,exp_argument)))
        p = np.random.random()
        if p <= q:
            self.config = trial_config
    
    def monte_carlo_move(self):

        self.n_particles = len(self.config)

        r = np.random.random()

        if r <= 0.25:
            self.try_delete_move()
            
        elif r > 0.25 and r <= 0.5:
            self.try_insert_move()

        else:
            self.try_displacement_move()

        print(self.n_particles)
