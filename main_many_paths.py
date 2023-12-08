import numpy as np
import math
import os
import sys
import csv
import pandas as pd
import time
from numba import jit
from numba import prange

np.set_printoptions(precision=12, suppress = True)

path = os.path.realpath(__file__)
path = path.strip(os.path.basename(sys.argv[0]))

'''
A complete realization of a horizontal dynamic (self.horizontal_iter going from 0 to N_horizontal - 1) initializes self.X
Each iteration of the vertical dynamic (self.vertical_iter += 1) updates self.X and self.PI 
'''
class Paths:
    def __init__(self):
        # Generalized coordinates each of dimensions (2, N_horizontal, N_atoms, 3) 
        self.X = None # Generalized coordinates
        self.PI = None # Generalized momenta

        self.dS = None # (N_horizontal, N_atoms, 3)
        self.force = None # (N_horizontal, N_atoms, 3)
        self.force_jacobian = None # (N_horizontal, N_atoms, 3, 3)
        self.force_divergence = None # (N_horizontal, N_atoms, 3)

        # Iterators
        self.initial_horizontal_iter = None
        self.horizontal_iter = None # Goes from 0 to N_horizontal
        self.vertical_iter = None # Goes from 0 to N_vertical
        self.restart_N_vertical = None

        # Parameters
        self.box_length = None # Returned by initialize_fcc_lattice() using the user-given density and number of cells 

        # Observables 
        self.temperature_vector = None # contains the temperature averaged over atoms (of shape (N_horizontal))
        self.temperature_mean = None # contains the temperature averaged over atoms and horizontal iterations (of shape 1) 

        self.potential_energy_vector = None
        self.potential_energy_mean = None

        self.kinentic_energy_vector = None
        self.kinentic_energy_mean = None

        self.total_energy_vector = None
        self.total_energy_mean = None

    def initialize_horizontal_dynamic(self):
 
      global N_atoms, box_length, N_horizontal


      if restart_horizontal_from_file:
        # Read the csv file into a DataFrame
        df = pd.read_csv('horizontal_trajectories.csv')

        # Get the maximum horizontal_iter value, which corresponds to the last trajectories
        last_horizontal_iter = df['horizontal_iter'].max()
        N_atoms = df['atom_index'].max()

        # Filter the DataFrame to only include the last trajectories
        last_traj_df = df[df['horizontal_iter'] == last_horizontal_iter]

        # Initialize self.X as a zero array
        self.X = np.zeros((2, N_horizontal, N_atoms, 3))

        self.horizontal_iter = 0
        self.vertical_iter = 0

        # Fill self.X with the data from the last trajectory
        for idx, row in last_traj_df.iterrows():
          atom_idx = int(row['atom_index']) - 1  # Adjust atom_index to 0-indexing
          self.X[0, 0, atom_idx, :] = [row['x'], row['y'], row['z']]
          self.X[1, 0, atom_idx, :] = [row['vx'], row['vy'], row['vz']]

          if type_of_potential == 'no_potential' and lennard_jones_on:  
            if init_FCC:
              _, _, N_atoms, box_length = initialize_fcc_lattice2(N_cells, density)
              self.box_length = box_length
              print(f"N_atoms = {N_atoms}")
            else: 
              N_atoms = natoms
              box_length = 100*lj_sigma
              self.box_length = box_length
          elif type_of_potential == 'no_potential' and spring_on:  
            self.box_length = 100
            box_length = 100
            N_atoms = natoms
          elif type_of_potential == 'HO':
            self.box_length = 10*np.sqrt((1./(m*omega**2))*k_T)
            box_length = 10*np.sqrt((1./(m*omega**2))*k_T)
            N_atoms = natoms
          else:
            raise ValueError("No potential and no Lennard Jones")

        self.kinetic_energy = np.zeros(N_horizontal)
        self.potential_energy_vector = np.zeros(N_horizontal)
        self.total_energy = np.zeros(N_horizontal)
        print(f"Restarting horizontal dynamics from initial configuration with horizontal_iter = {last_horizontal_iter}\nNew N_horizontal = {N_horizontal}")
      else: 
        if type_of_potential == 'HO':
          if spring_on:
            box_length = 10*np.sqrt((1./(m*omega**2))*k_T)
            N_atoms = natoms
            r = np.zeros((natoms, 3))
            if natoms == 2:
              r[0, :] = np.array([0.5,0.1,-0.1])
              r[1, :] = np.array([-0.5,-0.1,0.1])
            elif natoms == 3:
              r[0, :] = np.array([0.5,0.0,0.0])
              r[1, :] = np.array([-0.5,0.0,0.0])
              r[2, :] = np.array([0.0,0.5,0.0])
            else:
              ValueError("spring_on only works with 2 or 3 atoms")
            v = np.random.normal(loc = 0.0, scale = 0, size = 3*N_atoms).reshape(N_atoms, 3)
          else:
            box_length = 10*np.sqrt((1./(m*omega**2))*k_T)
            N_atoms = natoms
            r = np.random.normal(loc = 0.0, scale = np.sqrt((1./(m*omega**2))*k_T), size = 3*N_atoms).reshape(N_atoms,3)
            v = np.random.normal(loc = 0.0, scale = np.sqrt(k_T/m), size = 3*N_atoms).reshape(N_atoms, 3)
        elif type_of_potential == 'no_potential':
          if lennard_jones_on:
            ''' 
            Initialize with the number of cells and the size of the box
            In this case, you can choose the density but make sure the LJ eq. distance
            is the lattice spacing aka L / N_cells = r_LJ
            '''
            r_eq = (2**(1. /6 )) * lj_sigma 
            if init_FCC:
              r, lattice_spacing, N_atoms, box_length = initialize_fcc_lattice2(N_cells, density)
            else: 
              N_atoms = natoms
              r = np.zeros((natoms, 3))
              #_, _, _, box_length = initialize_fcc_lattice2(N_cells, density)
              box_length = 100*lj_sigma
              if natoms == 2:
                #r[0, :] = np.array([0.5*r_eq,0.0,0.0])
                #r[1, :] = np.array([-0.5*r_eq,0.0,0.0])
                r[0, :] = np.array([0.5*r_eq,0.1*r_eq,0.0])
                r[1, :] = np.array([-0.5*r_eq,0.0,-0.1*r_eq])
              elif natoms == 3:
                r[0, :] = np.array([0.5*r_eq,0.0,0.0])
                r[1, :] = np.array([-0.5*r_eq,0.0,0.0])
                r[2, :] = np.array([0.0,0.5*r_eq,0.0])
              else:
                ValueError("spring_on only works with 2 or 3 atoms")
              
            if init_FCC:
              print(f"You asked for N_cells = {N_cells} and density = {density}. \nInitializing {N_atoms} in {N_cells} FCC cells in a box of dimension {box_length}.")
              print(f"LJ equilibrium distance is {r_eq} and nearest FCC distance is {lattice_spacing / np.sqrt(2)}")
            else:
              print(f"Initializing {N_atoms} atoms without FCC")

            ''' Velocities drawn from Boltzmann dist '''
            #np.random.seed(0)
            #v = np.random.normal(loc = 0.0, scale = 0., size = 3*N_atoms).reshape(N_atoms, 3)
            vx = np.random.normal(loc = 0.0, scale = np.sqrt(k_T/m), size = N_atoms)
            vy = np.random.normal(loc = 0.0, scale = np.sqrt(k_T/m), size = N_atoms)
            vz = np.random.normal(loc = 0.0, scale = np.sqrt(k_T/m), size = N_atoms)
            v = np.transpose([vx, vy, vz])
            v -= np.mean(v, axis = 0)
          elif spring_on:
            N_atoms = natoms
            r = np.zeros((natoms, 3))
            box_length = 10*spring_l0
            if natoms == 2:
              #r[0, :] = np.array([0.5,0.1,0.1])
              #r[1, :] = np.array([-0.5,-0.1,-0.1])
              r[0, :] = np.array([0.5,0.0,0.0])
              r[1, :] = np.array([-0.5,0.0,0.0])
            elif natoms == 3:
              r[0, :] = np.array([0.5,0.0,0.0])
              r[1, :] = np.array([-0.5,0.0,0.0])
              r[2, :] = np.array([0.0,0.5,0.0])
            else:
              ValueError("spring_on only works with 2 or 3 atoms")
            vx = np.random.normal(loc = 0.0, scale = np.sqrt(k_T/m), size = N_atoms)
            vy = np.random.normal(loc = 0.0, scale = np.sqrt(k_T/m), size = N_atoms)
            vz = np.random.normal(loc = 0.0, scale = np.sqrt(k_T/m), size = N_atoms)
            v = np.transpose([vx, vy, vz])
            v -= np.mean(v, axis = 0)
          else:
            raise ValueError("No potential and no pair interaction: free particle???")

        self.initial_horizontal_iter = 0
        self.horizontal_iter = 0
        self.vertical_iter = 0
        self.X = np.zeros((2, N_horizontal, N_atoms, 3))
        self.kinetic_energy = np.zeros(N_horizontal)
        self.potential_energy_vector = np.zeros(N_horizontal)
        self.total_energy = np.zeros(N_horizontal)

        self.X[0, 0, :, :] = r
        self.X[1, 0, :, :] = v

        self.box_length = box_length
    
        print('Done with the initialization')

    def OVRVO(self):
      
      '''
      Carries out OVRVO from initial positions self.X[0, 0, ...] and initial velocities self.X[1, 0, ...] 
      Each timestep evolves all atoms 
      Force is calculated from get_force() method once per timestep
      '''
      c1 = math.exp(-gamma*horizontal_dt) # c1 = a in the Crooks article
      if gamma == 0:
        c2 = 1 #c2 = b (the time rescaling fator) in the Crooks article
      else:
        c2 = np.sqrt(2/(gamma*horizontal_dt)*math.tanh(gamma*horizontal_dt/2))

      r = self.X[0, 0, ...]
      v = self.X[1, 0, ...]

      print(f"Starting OVRVO algorithm from horizontal_iter = {self.horizontal_iter} to {N_horizontal}")

      force_value = self.get_force_OVRVO() # (N_horizontal, N_atoms, 3)

      while self.horizontal_iter < N_horizontal - 1:

        self.horizontal_iter += 1
        if (self.horizontal_iter % freq_output_horizontal == 0): print(self.horizontal_iter)

        # O-block
        v = np.sqrt(c1)*v + np.sqrt((1-c1)*k_T/m) * np.random.normal(0,1,3*N_atoms).reshape((N_atoms, 3))
  
        # V-block
        v = v + 0.5*c2*horizontal_dt*force_value/m

        # R-block
        r = r + c2*horizontal_dt*v

        self.X[0, self.horizontal_iter, ...] = r
        self.X[1, self.horizontal_iter, ...] = v
 
        force_value = self.get_force_OVRVO()
 
        # V-block

        v = v + 0.5*c2*horizontal_dt*force_value/m

        # O-block
        v = np.sqrt(c1)*v + np.sqrt((1-c1)*k_T/m) * np.random.normal(0,1,3*N_atoms).reshape((N_atoms, 3))

        self.X[0, self.horizontal_iter, :, :] = r
        self.X[1, self.horizontal_iter, :, :] = v

      return 0


    def write_horizontal(self):

      write_mode = 'a' if self.vertical_iter > 0 else 'w' # Overwrite if files already exists
      first_line = not(os.path.exists("horizontal_trajectories.csv") and self.vertical_iter > 0)
      
      if restart_horizontal_from_file:
        filename = "new_horizontal_trajectories.csv"
      else:
        filename = "horizontal_trajectories.csv"

      with open(filename, write_mode) as outfile_traj:
        if first_line:
          outfile_traj.write(",".join(["vertical_iter", "horizontal_iter", "atom_index", "x", "y", "z", "vx", "vy", "vz"]) + "\n")

          for hor in range(N_horizontal):  
            if hor % freq_output_horizontal == 0: 
              for atom_idx in range(1, N_atoms+1): 
                vx, vy, vz = [f'{self.X[1, hor, atom_idx -1, i]:12.8g}' for i in range(3)]
                
                dx = self.X[0, hor, atom_idx - 1, 0] 
                dy = self.X[0, hor, atom_idx - 1, 1] 
                dz = self.X[0, hor, atom_idx - 1, 2] 

                #dx = dx - self.box_length * np.round( dx / self.box_length)
                #dy = dy - self.box_length * np.round( dy / self.box_length)
                #dz = dz - self.box_length * np.round( dz / self.box_length)

                # Write folded positions
                x, y, z = [f"{alpha:12.8g}" for alpha in [dx, dy, dz]]

                row = [str(self.vertical_iter), str(hor), str(atom_idx), x, y, z, vx, vy, vz]
                outfile_traj.write(",".join(row) + "\n")


      if restart_horizontal_from_file:
        filename = "new_horizontal_observables.csv"
      else:
        filename = "horizontal_observables.csv"

      self.get_kinetic_energy_vector()
      self.get_potential_energy_vector()
      self.get_total_energy_vector()
      temperatures = self.get_temperature_vector()

      first_line = not(os.path.exists(filename) and self.vertical_iter > 0)
      with open(filename, write_mode) as outfile_traj:
        if first_line:
          outfile_traj.write(",".join(["vertical_iter", "horizontal_iter", "temperature", "kinetic_energy", "potential_energy", "total_energy"]) + "\n")
        rows = zip([str(self.vertical_iter)]*N_horizontal,
               [f'{int(alpha)}' for alpha in range(N_horizontal) if alpha % freq_output_horizontal == 0],
               [f'{temperatures[int(alpha)]}' for alpha in range(N_horizontal) if alpha % freq_output_horizontal == 0],
               [f'{self.kinetic_energy_vector[alpha]}' for alpha in range(N_horizontal) if alpha % freq_output_horizontal == 0],
               [f'{self.potential_energy_vector[alpha]}' for alpha in range(N_horizontal) if alpha % freq_output_horizontal == 0],
               [f'{self.total_energy_vector[alpha]}' for alpha in range(N_horizontal) if alpha % freq_output_horizontal == 0])
        for row in rows:
          outfile_traj.write(",".join(row) + "\n")
 

    def write_vertical(self):

      if restart_vertical_from_file == 0: write_mode = 'a' if self.vertical_iter > 0 else 'w'
      else: write_mode = 'a' 

      with open("vertical_trajectories.csv", write_mode) as outfile_traj:

        if self.vertical_iter == 0: 
          outfile_traj.write(",".join(["vertical_iter", "horizontal_iter", "atom_index", "x", "y", "z", "vx", "vy", "vz", "pix", "piy", "piz", "pivx", "pivy", "pivz"]) + "\n")
        for hor in range(N_horizontal):
          if hor % freq_output_horizontal == 0:  
            for atom_idx in range(1, N_atoms+1): 
              x, y, z = [f'{self.X[0, hor, atom_idx -1, i]:.5f}' for i in range(3)]  
              vx, vy, vz = [f'{self.X[1, hor, atom_idx -1, i]:.5f}' for i in range(3)] 
                        
              pix, piy, piz = [f'{self.PI[0, hor, atom_idx -1, i]:.5f}' for i in range(3)]  
              pivx, pivy, pivz = [f'{self.PI[1, hor, atom_idx -1, i]:.5f}' for i in range(3)] 

              row = [str(self.vertical_iter), str(hor), str(atom_idx), x, y, z, vx, vy, vz, pix, piy, piz, pivx, pivy, pivz]
              outfile_traj.write(",".join(row) + "\n")


      self.get_kinetic_energy_vector()
      self.get_potential_energy_vector()
      self.get_total_energy_vector()
      temperatures = self.get_temperature_vector()

      with open("vertical_observables.csv", write_mode) as outfile_traj:

        if self.vertical_iter == 0: 
          outfile_traj.write(",".join(["vertical_iter", "horizontal_iter", "temperature", "kinetic_energy", "potential_energy", "total_energy"]) + "\n")
        rows = zip([str(self.vertical_iter) for _ in range(N_horizontal)], 
                   [f'{alpha}' for alpha in range(int(N_horizontal))],
                   [f'{temperatures[int(alpha)]}' for alpha in range(N_horizontal)],
                   [f'{self.kinetic_energy_vector[alpha]}' for alpha in range(N_horizontal)],
                   [f'{self.potential_energy_vector[alpha]}' for alpha in range(N_horizontal)],
                   [f'{self.total_energy_vector[alpha]}' for alpha in range(N_horizontal)])
 
        for row in rows:
          outfile_traj.write(",".join(row) + "\n")


      self.get_kinetic_energy_mean()
      self.get_potential_energy_mean()
      self.get_total_energy_mean()
      temperature = self.get_temperature_mean()
      fict_temperature = self.get_fictitious_temperature()
      vertical_hamiltonian = self.get_vertical_hamiltonian()

      with open("mean_vertical_observables.csv", write_mode) as outfile_traj:
        if self.vertical_iter == 0:
          outfile_traj.write(",".join(["vertical_iter", "temperature", "fict_temperature", "kinetic_energy", "potential_energy", "total_energy", "vertical_hamiltonian"]) + "\n")
        rows = zip([str(self.vertical_iter)],
                   [f'{temperature}'], 
                   [f'{fict_temperature}'], 
                   [f'{self.kinetic_energy_mean}'], 
                   [f'{self.potential_energy_mean}'],
                   [f'{self.total_energy_mean}'],
                   [f'{self.vertical_hamiltonian}'])
        for row in rows:
          outfile_traj.write(",".join(row) + "\n")

      print(f"{'Parameter':<25} | {'Value'}")
      print(f"{'-'*25} | {'-'*20}")
      print(f"{'Vertical Iter':<25} | {self.vertical_iter}")
      print(f"{'Kinetic Energy Mean':<25} | {self.kinetic_energy_mean}")
      print(f"{'Potential Energy Mean':<25} | {self.potential_energy_mean}")
      print(f"{'Total Energy Mean':<25} | {self.total_energy_mean}")
      print(f"{'Temperature Mean':<25} | {self.temperature_mean}")
      print(f"{'Fictitious Temperature':<25} | {fict_temperature}")
      print(f"{'Vertical Hamiltonian':<25} | {vertical_hamiltonian}")

      return 0

    def get_force_OVRVO(self):
        if type_of_potential == 'HO':
          external_force = -m*omega**2*self.X[0, self.horizontal_iter, ...]
        elif type_of_potential == 'no_potential':
          external_force = np.zeros_like(self.X[0, self.horizontal_iter, ...])
        else:
          raise ValueError("Unknown potential type")

        if lennard_jones_on:
          pair_force = self.LJ_force_OVRVO_image()
        elif spring_on:
          pair_force = self.spring_force_OVRVO() 
        else:
          pair_force = np.zeros_like(self.X[0, self.horizontal_iter, ...])

        self.force = external_force + pair_force
        return self.force


    def LJ_force_OVRVO_image(self):
      '''
      At a given horizontal_iter, calculate the force and store the potential_energy
      '''
      force, self.potential_energy_vector[self.horizontal_iter] = LJ_compute_force_helper(self.X[0, self.horizontal_iter, ...], box_length)
      return force

    def spring_force_OVRVO(self):
      '''
      At a given horizontal_iter, calculate the force and store the potential_energy
      '''
      force, self.potential_energy_vector[self.horizontal_iter] = spring_compute_force_helper(self.X[0, self.horizontal_iter, ...], self.box_length)
      return force



    def initialize_vertical_dynamic(self):

      global N_horizontal, N_atoms, box_length

      if restart_vertical_from_file:
        if not(os.path.exists("vertical_trajectories.csv")):
          raise ValueError("Vertical dynamic cannot start without previous initial trajectories")

        df = pd.read_csv('vertical_trajectories.csv')
        N_vertical = df['vertical_iter'].max() + 1
        N_horizontal = df['horizontal_iter'].max() + 1
        N_atoms = df['atom_index'].max()  
        self.X = np.zeros((2, N_horizontal, N_atoms, 3))
        self.PI = np.zeros((2, N_horizontal, N_atoms, 3))
        self.vertical_iter = N_vertical 
        self.restart_N_vertical = N_vertical
            
        self.kinetic_energy = np.zeros(N_horizontal)
        self.potential_energy_vector = np.zeros(N_horizontal)
        self.total_energy = np.zeros(N_horizontal)

        _, _, _, box_length = initialize_fcc_lattice(N_cells, density)
        self.box_length = box_length

        # Fill self.X and self.PI with the data from the last trajectory
        for idx, row in df.iterrows():
          atom_idx = int(row['atom_index']) - 1  # Adjust atom_index to 0-indexing
          horizontal_iter = int(row['horizontal_iter'])  
        
          self.X[0, horizontal_iter, atom_idx, :] = [row['x'], row['y'], row['z']]
          self.X[1, horizontal_iter, atom_idx, :] = [row['vx'], row['vy'], row['vz']]

          self.PI[0, horizontal_iter, atom_idx, :] = [row['pix'], row['piy'], row['piz']]
          self.PI[1, horizontal_iter, atom_idx, :] = [row['pivx'], row['pivy'], row['pivz']]
      
        self.get_dS()
      else:
        if not(os.path.exists("new_horizontal_trajectories.csv")):
          raise ValueError("Vertical dynamic cannot start without initial horizontal path")

        # Read the csv file into a DataFrame
        df = pd.read_csv('new_horizontal_trajectories.csv')

        if df['horizontal_iter'].max() + 1 !=  N_horizontal:
          raise ValueError("new_horizontal_trajectories.csv has a different number of trajectories than N_horizontal")
        N_atoms = df['atom_index'].max()  

        # Initialize self.X as a zero array
        self.X = np.zeros((2, N_horizontal, N_atoms, 3))


        if type_of_potential == 'no_potential' and lennard_jones_on:  
            if init_FCC: 
              _, _, N_atoms, box_length = initialize_fcc_lattice2(N_cells, density)
              self.box_length = box_length
            else:
              box_length = 100*lj_sigma
              self.box_length = box_length
        elif type_of_potential == 'no_potential' and spring_on:  
            self.box_length = 100
            box_length = 100
            N_atoms = natoms
        elif type_of_potential == 'HO':
            self.box_length = 10*np.sqrt((1./(m*omega**2))*k_T)
            box_length = 10*np.sqrt((1./(m*omega**2))*k_T)
            N_atoms = natoms
        else:
            raise ValueError("No potential and no Lennard Jones")

        # Fill self.X with the data from the last trajectory
        for idx, row in df.iterrows():
          atom_idx = int(row['atom_index']) - 1  # Adjust to 0-indexing
          horizontal_iter = int(row['horizontal_iter'])  
        
          self.X[0, horizontal_iter, atom_idx, :] = [row['x'], row['y'], row['z']]
          self.X[1, horizontal_iter, atom_idx, :] = [row['vx'], row['vy'], row['vz']]


        self.PI = np.random.normal(size = 2*N_horizontal*N_atoms*3, scale = np.sqrt(M*fict_k_T)).reshape((2, N_horizontal, N_atoms, 3))

        self.vertical_iter = 0
        self.restart_N_vertical = 0

        self.kinetic_energy = np.zeros(N_horizontal)
        self.potential_energy_vector = np.zeros(N_horizontal)
        self.total_energy = np.zeros(N_horizontal)

        # Calculate dS
        self.get_dS()

      return 0

    def BAOAB(self):

      ''' 
      dS is always available here
      either calculated in the previous iteration of BAOAB or in the vertical initialization
      '''
      start_block = time.time()
      # B block
      self.PI -= self.dS*(vertical_dt/2)
      end_block = time.time()
      if time_check: print(f"b block1: {end_block - start_block}")

      # A block
      start_block = time.time()
      self.X += self.PI*(vertical_dt/(2*M))
      end_block = time.time()
      if time_check: print(f"a block1: {end_block - start_block}")

      # O block
      start_block = time.time()
      random_vector = np.random.multivariate_normal(mean = (0.0, 0.0, 0.0),\
      cov = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], size = (2, N_horizontal, N_atoms))

      if fict_gamma != 0.0:
        self.PI *= math.exp(-fict_gamma*vertical_dt) 
        self.PI += math.sqrt(fict_k_T * M * (1-math.exp(-2*fict_gamma*vertical_dt)))*random_vector
      end_block = time.time()
      if time_check: print(f"o block: {end_block - start_block}")

      # A block
      start_block = time.time()
      self.X += self.PI*(vertical_dt/(2*M))
      self.get_dS()
      end_block = time.time()
      if time_check: print(f"a2 block: {end_block - start_block}")

      # B block
      start_block = time.time()
      self.PI -= self.dS*(vertical_dt/2)
      end_block = time.time()
      if time_check: print(f"b2 block: {end_block - start_block}")

      return 0

    def get_temperature_vector(self): 
      '''
      X.shape = (2, N_horizontal, N_atoms, 3). Mean over number of atoms and number of components. No constraints  
      Returns a 1-dimensional array of size N_horizontal
      '''
      self.temperature_vector = (1./3)*np.sum(m*self.X[1]**2, axis = (1, 2))/self.X.shape[2] # size (2, N_horizontal, N_atoms, 3)
      return self.temperature_vector 
  
    def get_temperature_mean(self): 
      '''
      X.shape = (2, N_horizontal, N_atoms, 3). Mean over path (real time, number of atoms and number of components). No constraints  
      Returns a float
      '''
      self.temperature_mean = (1./3)*np.sum(m*self.X[1]**2, axis = (0, 1, 2))/(self.X.shape[1]*self.X.shape[2]) 
      return self.temperature_mean

    def get_fictitious_temperature(self): 
      ''' 
      PI.shape = (2, N_vertical, 2*N_atoms, 3)
      Mean over real time, number of fictious atoms, and components
      '''
      return (1./3)*np.sum(self.PI[1]**2/M)/(self.PI.shape[1]*self.PI.shape[2]) # PI.shape = (2, N_vertical, 2*N_atoms, 3)

    def get_kinetic_energy_vector(self):
      self.kinetic_energy_vector = 0.5*m*np.sum(self.X[1, ...]**2, axis = (1,2))
      return self.kinetic_energy_vector

    def get_kinetic_energy_mean(self):
      self.kinetic_energy_mean = 0.5*m*np.sum(self.X[1, ...]**2, axis = (0,1,2))/N_horizontal
      return self.kinetic_energy_mean
 
    def get_total_energy_vector(self):
      self.total_energy_vector = self.kinetic_energy_vector + self.potential_energy_vector
      return self.total_energy_vector

    def get_total_energy_mean(self):
      self.total_energy_mean = self.kinetic_energy_mean + self.potential_energy_mean
      return self.total_energy_mean

    def get_potential_energy_vector(self):
      self.potential_energy_vector = compute_potential_energy_helper(self.X, self.box_length)
      return self.potential_energy_vector

    def get_potential_energy_mean(self):
      #if self.potential_energy_vector == None:
      #  raise ValueError("Cannot calculate potential_energy_vector_mean before potential_energy_vector")
      self.potential_energy_mean = np.mean(self.potential_energy_vector)
      return self.potential_energy_mean
 
    def get_force_divergence(self):    
      if type_of_potential == 'HO': 
        external_force_divergence = -3*m*omega**2
      elif type_of_potential == 'no_potential':
        external_force_divergence = 0

      if lennard_jones_on == 1:
        lj_force_divergence = LJ_force_divergence(self)
      
      return external_force_divergence + lj_force_divergence

    def LJ_force_divergence(self):

      traces_over_time = []
      for jacobian_at_t in self.jacobians:
        traces = [np.trace(jacobian) for jacobian in jacobian_at_t]
        traces_over_time.append(traces)
      return np.array(traces_over_time).reshape(N_horizontal, N_atoms)

    def get_vertical_hamiltonian(self): # X is of size (2, N_horizontal, N_atoms, 3) 
      self.S = get_S_helper(self.X, self.potential_energy_vector[0])
      print(f"potential energy 0 = {self.potential_energy_vector[0]}")
      print(f"self.S = {self.S}")
      print(f"kinetic part = {np.sum(self.PI**2/M)}")
      self.vertical_hamiltonian = self.S + np.sum(self.PI**2/(2*M))
      return self.vertical_hamiltonian

    def get_dS(self): # X is of size (2, N, N_atoms, 3) 
      self.dS = get_dS_helper_manually2(self.X)
      #self.dS = get_dS_helper_manually(self.X)
      #quit()
      return self.dS


'''
Below are all the functions which use numba for acceleration, and some utility functions 
They rely on numba and so are outside the Paths class because numba works better with functions than with methods
'''
@jit(nopython=True)
def BAOAB_force(X): # X[0] is of size (N_horizontal, N_atoms, 3)
    force = np.zeros_like(X[0])
    box_lengths = np.array([box_length, box_length, box_length])
    if lennard_jones_on or spring_on:
      for t in range(N_horizontal):
        for i in range(N_atoms):
            for j in range(i+1, N_atoms):
                r_ij = X[0, t, j] - X[0, t, i]
                
                # Apply the minimum image convention
                r_ij -= box_length * np.round(r_ij / box_length)
                r_ij_norm = np.sqrt(np.sum(r_ij**2))  # Replaced np.linalg.norm

                if r_ij_norm == 0:
                    raise ValueError("r_ij_norm = 0")

                if lennard_jones_on:
                    if r_ij_norm > lj_cutoff:
                        continue

                    r_ij_inv = 1. / r_ij_norm
                    lj_cutoff_inv = 1. / lj_cutoff

                    # Compute the force magnitude according to the Lennard-Jones potential
                    if r_ij_norm < lj_cap:
                      lj_cap_inv = 1. / lj_cap
                      # Capping the potential
                      force_mag = 24*lj_epsilon*(2*lj_sigma**12 * lj_cap_inv**13 - lj_sigma**6 * lj_cap_inv**7)
                      force_mag -= 24 * lj_epsilon * (2*lj_sigma**12*lj_cutoff_inv**13 - lj_sigma**6*lj_cutoff_inv**7)
                    else:
                      force_mag = 24 * lj_epsilon * (2*lj_sigma**12*r_ij_inv**13 - lj_sigma**6*r_ij_inv**7)
                      force_mag -= 24 * lj_epsilon * (2*lj_sigma**12*lj_cutoff_inv**13 - lj_sigma**6*lj_cutoff_inv**7)

                    # Add this force to the total force on both particles i and j
                    # Note: the force on particle j is the negative of the force on particle i
                    force_ij = force_mag * r_ij * r_ij_inv

                    force[t, i] -= force_ij
                    force[t, j] += force_ij

                if spring_on:
                  # Compute the force magnitude according to the spring potential
                  force_mag = - spring_k*(r_ij_norm - spring_l0) 

                  # Add this force to the total force on both particles i and j
                  # Note: the force on particle j is the negative of the force on particle i
                  force_ij = force_mag * r_ij

                  force[i, :] -= force_ij
                  force[j, :] += force_ij
    return force

@jit(nopython=True)
def BAOAB_force_jacobian(X):

    jacobians_over_time = np.zeros((N_horizontal, N_atoms, 3, 3))
    box_lengths = np.array([box_length, box_length, box_length])
    if lennard_jones_on or spring_on:

        for t in range(N_horizontal):
            for i in range(N_atoms):
                # Calculate the Jacobian of the force F_i acting on atom i
                for j in range(N_atoms):
                    if j != i:
                        r_ij = X[0, t, j] - X[0, t, i]
                        # Apply the minimum image convention
                        r_ij -= box_length * np.round(r_ij / box_length)
                        r_ij_norm = np.sqrt(np.sum(r_ij**2))  # Replaced np.linalg.norm

                        if r_ij_norm == 0:
                          raise ValueError("r_ij = 0")

                        if lennard_jones_on:
                            if r_ij_norm > lj_cutoff:
                              continue

                            if r_ij_norm < lj_cap:
                              A = 24*lj_epsilon*(lj_sigma**6 / lj_cap**7 - 2*lj_sigma**12 / lj_cap**13)

                              for alpha in range(3):
                                  delta_alpha = r_ij[alpha]
                                  for beta in range(alpha, 3):
                                      if alpha == beta:
                                          result = A*r_ij_norm**(-1) - delta_alpha**2 * r_ij_norm**(-3)
                                      else:
                                          delta_beta = r_ij[beta]
                                          result = - A * delta_alpha * delta_beta * r_ij_norm**(-3)

                                      jacobians_over_time[t, i, alpha, beta] -= result
                                      jacobians_over_time[t, i, beta, alpha] -= result
                            else:
                              lj_factor1 = 24 * lj_epsilon
                              lj_factor2 = 2 * lj_sigma ** 12
                              lj_factor3 = lj_sigma ** 6
                              r_ij_inv16 = r_ij_norm ** (-16)
                              r_ij_inv14 = r_ij_norm ** (-14)
                              r_ij_inv10 = r_ij_norm ** (-10)
                              r_ij_inv8 = r_ij_norm ** (-8)

                              for alpha in range(3):
                                  delta_alpha = r_ij[alpha]
                                  for beta in range(alpha, 3):
                                      if alpha == beta:
                                        # Diagonal terms
                                          result = lj_factor1 * delta_alpha * (lj_factor2 * (14 * delta_alpha ** 2 * r_ij_inv16 - r_ij_inv14) - lj_factor3 * (8 * delta_alpha ** 2 * r_ij_inv10 - r_ij_inv8))
                                      else:
                                          # Off-diagonal elements (symmetric)
                                          delta_beta = r_ij[beta]
                                          result = lj_factor1 * delta_alpha * delta_beta * (28 * lj_factor2 * r_ij_inv16 - 8 * lj_factor3 * r_ij_inv10)

                                      jacobians_over_time[t, i, alpha, beta] -= result
                                      jacobians_over_time[t, i, beta, alpha] -= result

                        if spring_on:
                              for alpha in range(3):
                                  delta_alpha = r_ij[alpha]
                                  for beta in range(alpha, 3):
                                      if alpha == beta:
                                          result = spring_k*(delta_alpha**2/r_ij_norm - r_ij_norm - spring_l0)
                                      else:
                                          delta_beta = r_ij[beta]
                                          result = spring_k * delta_alpha * delta_beta / r_ij_norm

                                      jacobians_over_time[t, i, alpha, beta] -= result
                                      jacobians_over_time[t, i, beta, alpha] -= result
    return jacobians_over_time


@jit(nopython=True)
def BAOAB_force_jacobian2(X):

    jacobians_over_time = np.zeros((N_horizontal, N_atoms, N_atoms, 3, 3))
    box_lengths = np.array([box_length, box_length, box_length])
    if lennard_jones_on:
        for t in range(N_horizontal):
            for i in range(N_atoms):
                for j in range(N_atoms):
                    if j > i:
                        r_ij = X[0, t, j] - X[0, t, i]
                        # Apply the minimum image convention
                        r_ij -= box_length * np.round(r_ij / box_length)
                        r_ij_norm = np.sqrt(np.sum(r_ij**2))  # Replaced np.linalg.norm

                        if r_ij_norm == 0:
                          raise ValueError("r_ij = 0")

                        if lennard_jones_on:
                            if r_ij_norm > lj_cutoff:
                              continue

                            lj_factor1 = 24 * lj_epsilon
                            lj_factor2 = 2 * lj_sigma ** 12
                            lj_factor3 = lj_sigma ** 6
                            r_ij_inv16 = r_ij_norm ** (-16)
                            r_ij_inv14 = r_ij_norm ** (-14)
                            r_ij_inv10 = r_ij_norm ** (-10)
                            r_ij_inv8 = r_ij_norm ** (-8)

                            for alpha in range(3):
                                delta_alpha = r_ij[alpha]
                                for beta in range(alpha, 3):
                                    if alpha == beta:
                                        # Diagonal terms
                                        result = lj_factor1 * (lj_factor3 * (8 * delta_alpha ** 2 * r_ij_inv10 - r_ij_inv8)\
                                                             - lj_factor2 * (14 * delta_alpha ** 2 * r_ij_inv16 - r_ij_inv14))

                                        jacobians_over_time[t, i, i, alpha, alpha] += result

                                        jacobians_over_time[t, i, j, alpha, alpha] = - result

                                        jacobians_over_time[t, j, i, alpha, alpha] = - result
                                    else:
                                        # Off-diagonal elements (symmetric in i j and alpha beta)
                                        delta_beta = r_ij[beta]
                                        result = lj_factor1 * delta_alpha * delta_beta * (8 * lj_factor3 * r_ij_inv10 - 14 * lj_factor2 * r_ij_inv16)

                                        jacobians_over_time[t, i, i, alpha, beta] += result
                                        jacobians_over_time[t, i, i, beta, alpha] += result

                                        jacobians_over_time[t, i, j, alpha, beta] = - result
                                        jacobians_over_time[t, i, j, beta, alpha] = - result

                                        jacobians_over_time[t, j, i, alpha, beta] = - result
                                        jacobians_over_time[t, j, i, beta, alpha] = - result

    if spring_on:
        for t in range(N_horizontal):
            for i in range(N_atoms):
                for j in range(N_atoms):
                    if j > i:
                        r_ij = X[0, t, j] - X[0, t, i]
                        # Apply the minimum image convention
                        r_ij -= box_length * np.round(r_ij / box_length)
                        r_ij_norm = np.sqrt(np.sum(r_ij**2))  # Replaced np.linalg.norm

                        if r_ij_norm == 0:
                          raise ValueError("r_ij = 0")

                        if lennard_jones_on:
                            if r_ij_norm > lj_cutoff:
                              continue

                            for alpha in range(3):
                                delta_alpha = r_ij[alpha]
                                for beta in range(alpha, 3):
                                    if alpha == beta:
                                        # Diagonal terms
                                        result = spring_k * ( 1 - spring_l0 * (1. / r_ij_norm + delta_alpha**2 / r_ij_norm**3) )
                                        jacobians_over_time[t, i, i, alpha, alpha] += result

                                        jacobians_over_time[t, i, j, alpha, alpha] = - result

                                        jacobians_over_time[t, j, i, alpha, alpha] = - result
                                    else:
                                        # Off-diagonal elements (symmetric in i j and alpha beta)
                                        delta_beta = r_ij[beta]
                                        result = spring_k * spring_l0 * delta_alpha * delta_beta / r_ij_norm**2
                                        jacobians_over_time[t, i, i, alpha, beta] += result
                                        jacobians_over_time[t, i, i, beta, alpha] += result

                                        jacobians_over_time[t, i, j, alpha, beta] = - result
                                        jacobians_over_time[t, i, j, beta, alpha] = - result

                                        jacobians_over_time[t, j, i, alpha, beta] = - result
                                        jacobians_over_time[t, j, i, beta, alpha] = - result
 
    return jacobians_over_time



@jit(nopython=True)
def get_force(X):
  if type_of_potential == 'HO':
    external_force = -m*omega**2*X[0]
  elif type_of_potential == 'no_potential':
    external_force = np.zeros_like(X[0])
  else:
    raise ValueError("Unknown potential type")

  pair_force = BAOAB_force(X)
    
  force = external_force + pair_force
  return force

#@jit(nopython=True)
def get_force_jacobian(X): # of size (N_horizontal, N_atoms, 3 , 3)
  # Take the Jacobian of the potential
  external_force_jacobian = np.zeros((N_horizontal, N_atoms, 3, 3))
  pair_force_jacobian = np.zeros((N_horizontal, N_atoms, 3, 3))
  for t in range(N_horizontal):
    for i in range(N_atoms):
      for j in range(N_atoms):
          # Take the Jacobian of the potential
          if type_of_potential == 'HO':
              external_force_jacobian[t, i, :, :] = -m * omega**2 * np.eye(3)
          elif type_of_potential == 'no_potential':
              external_force_jacobian[t, i, :, :] = np.zeros((3, 3))

  #if type_of_potential == 'HO':
  #  external_force_jacobian = np.tile(-m*omega**2 * np.eye(3), (N_horizontal, N_atoms, 1, 1))
  #elif type_of_potential == 'no_potential':
  #  external_force_jacobian = np.zeros((N_horizontal, N_atoms, 3, 3))

  # Add the Jacobian of the pair interaction
  pair_force_jacobian = BAOAB_force_jacobian(X)

  total_force_jacobian = external_force_jacobian + pair_force_jacobian
      
  return total_force_jacobian


@jit(nopython=True)
def get_force_jacobian2(X): # of size (N_horizontal, N_atoms, 3 , 3)
  # Take the Jacobian of the potential
  external_force_jacobian = np.zeros((N_horizontal, N_atoms, N_atoms, 3, 3))
  pair_force_jacobian = np.zeros((N_horizontal, N_atoms, N_atoms, 3, 3))
  for t in range(N_horizontal):
    for i in range(N_atoms):
      for j in range(N_atoms):
          # Take the Jacobian of the potential
          if type_of_potential == 'HO':
              external_force_jacobian[t, i, i, :, :] = -m * omega**2 * np.eye(3)
          elif type_of_potential == 'no_potential':
              external_force_jacobian[t, i, j, :, :] = np.zeros((3, 3))

  #if type_of_potential == 'HO':
  #  external_force_jacobian = np.tile(-m*omega**2 * np.eye(3), (N_horizontal, N_atoms, 1, 1))
  #elif type_of_potential == 'no_potential':
  #  external_force_jacobian = np.zeros((N_horizontal, N_atoms, 3, 3))

  # Add the Jacobian of the pair interaction
  pair_force_jacobian = BAOAB_force_jacobian2(X)

  total_force_jacobian = external_force_jacobian + pair_force_jacobian
      
  return total_force_jacobian

#@jit(nopython=True)
def get_dS_helper(X):
  
  start_block_time = time.time()
  grad_forces = get_force_jacobian(X)
  end_block_time = time.time()
  if time_check: print(f"grad_forces: {end_block_time - start_block_time}")

  start_block_time = time.time()
  forces = get_force(X)
  end_block_time = time.time()
  if time_check: print(f"forces: {end_block_time - start_block_time}")
  
  a = np.exp(-gamma*horizontal_dt)
  b = np.sqrt(2/(gamma*horizontal_dt)*math.tanh(gamma*horizontal_dt/2))

  pos = X[0, ...]
  vel = X[1, ...]

  '''
  dS of size (N, 3) in Jurij's code (1 atom) => now dS of size (N_horizontal, N_atoms, 3) for N_atoms atoms. 
  Operations are vectorized on the first axis (keep in mind the numpy convention: axis = 0, 1, 2 etc...)
  numba does not like np.einsum so only the get_force and get_force_jacobian are accelerated
  '''

  start_block_time = time.time()
  dS_n = [beta*m/((1-a)*fict_beta) * (\
    (1+a)/(b*horizontal_dt)**2 * (2*pos[1:-1] - pos[2:] - pos[:-2])\
    + (1/(2*m))*((1+a)*forces[1:-1]-forces[:-2]-a*forces[2:])\
    + np.einsum('...ij,...j', grad_forces[1:-1], (1/(2*m)) * ((1+a)*pos[1:-1] - a*pos[:-2]-pos[2:]))\
    + np.einsum('...ij,...j', grad_forces[1:-1], ((b*horizontal_dt)/(2*m))**2 * (1+a) * forces[1:-1])\
    + np.sqrt(a)/(b*horizontal_dt)*(vel[2:]-vel[:-2])),\
    beta*m/((1-a)*fict_beta) * ((1+a) * vel[1:-1] + np.sqrt(a)/(b*horizontal_dt) * (pos[:-2]-pos[2:]))]
  end_block_time = time.time()
  if time_check: print(f"dS_n: {end_block_time - start_block_time}")

  
  start_block_time = time.time()
  dS_r_0 = -beta * forces[0]/fict_beta + (beta*m/((1-a)*fict_beta)) * (\
    -(1+a)/(b*horizontal_dt)**2 *(pos[1]-pos[0])\
    - (1/(2*m))*(a*forces[1]-forces[0]) \
    + np.sqrt(a)/(b*horizontal_dt) * (vel[1] + vel[0]) \
    - 1/(2*m) * np.einsum('...ij,...j', grad_forces[0], pos[1] - pos[0])\
    + np.sqrt(a)*b*horizontal_dt/(2*m) * np.einsum('...ij,...j', grad_forces[0], vel[0])\
    + (b*horizontal_dt)**2/(2*m)**2 * np.einsum('...ij,...j', grad_forces[0], forces[0]))
  end_block_time = time.time()
  if time_check: print(f"dS_r_0: {end_block_time - start_block_time}")

  start_block_time = time.time()
  dS_v_0 = beta*m*vel[0]/fict_beta + beta*m/((1-a)*fict_beta) * np.sqrt(a) * (-(pos[1]-pos[0]) / (b*horizontal_dt) + b*horizontal_dt/2 * forces[0]/m + np.sqrt(a)*vel[0])
  end_block_time = time.time()
  if time_check: print(f"dS_v_0: {end_block_time - start_block_time}")

  start_block_time = time.time()
  dS_r_N =  beta*m/((1-a)*fict_beta) * (\
    (1+a)/(b*horizontal_dt)**2 * (pos[-1]-pos[-2])\
    + (1/(2*m))*(a*forces[-1]-forces[-2]) \
    - np.sqrt(a)/(b*horizontal_dt) * (vel[-1] + vel[-2]) \
    + (a/(2*m)) * np.einsum('...ij,...j', grad_forces[-1], pos[-1] - pos[-2])\
    + (b*horizontal_dt)**2/(2*m)**2 * a * np.einsum('...ij,...j', grad_forces[-1], forces[-1])\
    - np.sqrt(a)*b*horizontal_dt/(2*m)*np.einsum('...ij,...j', grad_forces[-1], vel[-1]))
  end_block_time = time.time()
  if time_check: print(f"dS_r_N: {end_block_time - start_block_time}")

  start_block_time = time.time()
  dS_v_N = beta*m/((1-a)*fict_beta) * (-np.sqrt(a) * ((pos[-1]-pos[-2]) / (b*horizontal_dt) + b*horizontal_dt/2 * forces[-1]/m) + vel[-1])
  end_block_time = time.time()
  if time_check: print(f"dS_v_N: {end_block_time - start_block_time}")

  start_block_time = time.time()
  dS_r_array = np.concatenate(([dS_r_0], dS_n[0], [dS_r_N]), axis=0)
  dS_v_array = np.concatenate(([dS_v_0], dS_n[1], [dS_v_N]), axis=0)
  end_block_time = time.time()
  if time_check: print(f"concatenate: {end_block_time - start_block_time}")

  return np.array([dS_r_array, dS_v_array])

#@jit(nopython=True)
def get_dS_helper_manually(X):
  '''
  Same as get_dS_helper() but not vectorized for checking (it yields same results)
  '''

  grad_forces = get_force_jacobian(X)
  forces = get_force(X)

  a = np.exp(-gamma*horizontal_dt)
  b = np.sqrt(2/(gamma*horizontal_dt)*math.tanh(gamma*horizontal_dt/2))

  pos = X[0, ...]
  vel = X[1, ...]

  dS_n, dS_r_0, dS_v_0, dS_r_N, dS_v_N, dS_r_array, dS_v_array, result = [], [], [], [], [], [], [], []
  for i in range(N_atoms):
    #print(np.einsum('...ij,...j', grad_forces[0, i], pos[1, i] - pos[0, i]))
    #print(np.einsum('...ij,...j', grad_forces[0, i], forces[0, i]))
    #print(np.einsum('...ij,...j', grad_forces[0, i], vel[0, i]))
    #print(np.einsum('...ij,...j', grad_forces[-1, i], vel[-1, i]))
    #print(np.einsum('...ij,...j', grad_forces[1:-1, i], ((1+a)*pos[1:-1, i] - a*pos[:-2, i]-pos[2:, i])))
    #print(np.einsum('...ij,...j', grad_forces[1:-1, i], forces[1:-1, i]))

    dS_n.append([beta*m/((1-a)*fict_beta) * (\
    (1+a)/(b*horizontal_dt)**2 * (2*pos[1:-1, i] - pos[2:, i] - pos[:-2, i])\
    + (1/(2*m))*((1+a)*forces[1:-1, i]-forces[:-2, i]-a*forces[2:, i])\
    + np.einsum('...ij,...j', grad_forces[1:-1, i], (1/(2*m)) * ((1+a)*pos[1:-1, i] - a*pos[:-2, i]-pos[2:, i]))\
    + np.einsum('...ij,...j', grad_forces[1:-1, i], ((b*horizontal_dt)/(2*m))**2 * (1+a) * forces[1:-1, i])\
    + np.sqrt(a)/(b*horizontal_dt)*(vel[2:, i]-vel[:-2, i])),\
    beta*m/((1-a)*fict_beta) * ((1+a) * vel[1:-1, i] + np.sqrt(a)/(b*horizontal_dt) * (pos[:-2, i]-pos[2:, i]))])

  
    dS_r_0.append(-beta * forces[0, i]/fict_beta + (beta*m/((1-a)*fict_beta)) * (\
    -(1+a)/(b*horizontal_dt)**2 *(pos[1, i]-pos[0, i])\
    - (1/(2*m))*(a*forces[1, i]-forces[0, i]) \
    + np.sqrt(a)/(b*horizontal_dt) * (vel[1, i] + vel[0, i]) \
    - 1/(2*m) * np.einsum('...ij,...j', grad_forces[0, i], pos[1, i] - pos[0, i])\
    + np.sqrt(a)*b*horizontal_dt/(2*m) * np.einsum('...ij,...j', grad_forces[0, i], vel[0, i])\
    + (b*horizontal_dt)**2/(2*m)**2 * np.einsum('...ij,...j', grad_forces[0, i], forces[0, i])))

    dS_v_0.append(beta*m*vel[0, i]/fict_beta + beta*m/((1-a)*fict_beta) * np.sqrt(a) * (-(pos[1, i]-pos[0, i]) / (b*horizontal_dt) + b*horizontal_dt/2 * forces[0, i]/m + np.sqrt(a)*vel[0, i]))

    dS_r_N.append(beta*m/((1-a)*fict_beta) * (\
    (1+a)/(b*horizontal_dt)**2 * (pos[-1, i]-pos[-2, i])\
    + (1/(2*m))*(a*forces[-1, i]-forces[-2, i]) \
    - np.sqrt(a)/(b*horizontal_dt) * (vel[-1, i] + vel[-2, i]) \
    + (a/(2*m)) * np.einsum('...ij,...j', grad_forces[-1, i], pos[-1, i] - pos[-2, i])\
    + (b*horizontal_dt)**2/(2*m)**2 * a * np.einsum('...ij,...j', grad_forces[-1, i], forces[-1, i])\
    - np.sqrt(a)*b*horizontal_dt/(2*m)*np.einsum('...ij,...j', grad_forces[-1, i], vel[-1, i])))

    dS_v_N.append(beta*m/((1-a)*fict_beta) * (-np.sqrt(a) * ((pos[-1, i]-pos[-2, i]) / (b*horizontal_dt) + b*horizontal_dt/2 * forces[-1, i]/m) + vel[-1, i]))

    dS_r_array.append(np.concatenate(([dS_r_0[i]], dS_n[i][0], [dS_r_N[i]]), axis=0))
    dS_v_array.append(np.concatenate(([dS_v_0[i]], dS_n[i][1], [dS_v_N[i]]), axis=0))

    result.append(np.array([dS_r_array[i], dS_v_array[i]]))

  stacked_array = np.stack(result, axis=2)
  #print(stacked_array.shape)
  #print(stacked_array[1])

  return stacked_array


#@jit(nopython=True)
def get_dS_helper_manually2(X):
  '''
  Same as get_dS_helper() but not vectorized for checking (it yields same results)
  '''

  # grad_forces size (N_horizontal, N_atoms, N_atoms, 3, 3)
  # X size (N_horizontal, N_atoms, 3)

  grad_forces = get_force_jacobian2(X)
  forces = get_force(X)

  R1 = np.zeros((N_horizontal, N_atoms, 3))
  R2 = np.zeros((N_horizontal, N_atoms, 3))
  R3 = np.zeros((N_atoms, 3))
  R4 = np.zeros((N_atoms, 3))

  a = np.exp(-gamma*horizontal_dt)
  b = np.sqrt(2/(gamma*horizontal_dt)*math.tanh(gamma*horizontal_dt/2))

  pos = X[0, ...]
  vel = X[1, ...]

  dS_n, dS_r_0, dS_v_0, dS_r_N, dS_v_N, dS_r_array, dS_v_array, result = [], [], [], [], [], [], [], []

  for i in range(N_atoms):
    for j in range(N_atoms):
      R1[0, i] += np.dot(grad_forces[0, i, j], pos[1, j] - pos[0, j])
      R1[-1, i] += np.dot(grad_forces[-1, i, j], pos[-1, j] - pos[-2, j])
      R2[0, i] += np.dot(grad_forces[0, i, j], forces[0, j])
      R2[-1, i] += np.dot(grad_forces[-1, i, j], forces[-1, j])
      for t in range(1, len(grad_forces) - 1):
        R1[t, i] += np.dot(grad_forces[t, i, j], (1+a)*pos[t, j] - a*pos[t-1, j] - pos[t+1, j])
        R2[t, i] += np.dot(grad_forces[t, i, j], forces[t, j])
      
      #print(f"i = {i} R1 = {R1[:,i]}")
      #print(f"i = {i} R2 = {R2[:,i]}")

      #R1[:, i] += np.einsum('ijk,ik->ij', grad_forces[1:-1, i, j], (1+a)*pos[1:-1, i] - a*pos[:-2, i] - pos[2:, i])
      #R2[:, i] += np.einsum('ijk,ik->ij', grad_forces[1:-1, i, j], (1+a)*pos[1:-1, i] - a*pos[:-2, i] - pos[2:, i])

      #R2[1:-1, i] += np.dot(grad_forces[1:-1, i, j], forces[1:-1, i])

      R3[i] += np.dot(grad_forces[0, i, j], vel[0, j])
      R4[i] += np.dot(grad_forces[-1, i, j], vel[-1, j])
  

  for i in range(N_atoms):
    #print(R2[1:-1,i])
    dS_n.append([beta*m/((1-a)*fict_beta) * (\
    (1+a)/(b*horizontal_dt)**2 * (2*pos[1:-1, i] - pos[2:, i] - pos[:-2, i])\
    + (1/(2*m))*((1+a)*forces[1:-1, i]-forces[:-2, i]-a*forces[2:, i])\
    + (1/(2*m)) * R1[1:-1, i] \
    + ((b*horizontal_dt)/(2*m))**2 * (1+a) * R2[1:-1, i]\
    + np.sqrt(a)/(b*horizontal_dt)*(vel[2:, i]-vel[:-2, i])),\
    beta*m/((1-a)*fict_beta) * ((1+a) * vel[1:-1, i] + np.sqrt(a)/(b*horizontal_dt) * (pos[:-2, i]-pos[2:, i]))])


    dS_r_0.append(-beta * forces[0, i]/fict_beta + (beta*m/((1-a)*fict_beta)) * (\
    -(1+a)/(b*horizontal_dt)**2 *(pos[1, i]-pos[0, i])\
    - (1/(2*m))*(a*forces[1, i]-forces[0, i]) \
    + np.sqrt(a)/(b*horizontal_dt) * (vel[1, i] + vel[0, i]) \
    - 1/(2*m) * R1[0, i]\
    + np.sqrt(a)*b*horizontal_dt/(2*m) * R3[i]\
    + (b*horizontal_dt)**2/(2*m)**2 * R2[0, i]))

    dS_v_0.append(beta*m*vel[0, i]/fict_beta + beta*m/((1-a)*fict_beta) * np.sqrt(a) * (-(pos[1, i]-pos[0, i]) / (b*horizontal_dt) + b*horizontal_dt/2 * forces[0, i]/m + np.sqrt(a)*vel[0, i]))


    dS_r_N.append(beta*m/((1-a)*fict_beta) * (\
    (1+a)/(b*horizontal_dt)**2 * (pos[-1, i]-pos[-2, i])\
    + (1/(2*m))*(a*forces[-1, i]-forces[-2, i]) \
    - np.sqrt(a)/(b*horizontal_dt) * (vel[-1, i] + vel[-2, i]) \
    + (a/(2*m)) * R1[-1, i] \
    + (b*horizontal_dt)**2/(2*m)**2 * a * R2[-1, i]\
    - np.sqrt(a)*b*horizontal_dt/(2*m)* R4[i]))

    dS_v_N.append(beta*m/((1-a)*fict_beta) * (-np.sqrt(a) * ((pos[-1, i]-pos[-2, i]) / (b*horizontal_dt) + b*horizontal_dt/2 * forces[-1, i]/m) + vel[-1, i]))

    dS_r_array.append(np.concatenate(([dS_r_0[i]], dS_n[i][0], [dS_r_N[i]]), axis=0))
    dS_v_array.append(np.concatenate(([dS_v_0[i]], dS_n[i][1], [dS_v_N[i]]), axis=0))

    result.append(np.array([dS_r_array[i], dS_v_array[i]]))

  stacked_array = np.stack(result, axis=2)

  #print(stacked_array.shape)
  #print(stacked_array[1])

  return stacked_array



def get_S_helper(X, pot_0):

    a = np.exp(-gamma*horizontal_dt)
    b = np.sqrt(2/(gamma*horizontal_dt)*math.tanh(gamma*horizontal_dt/2))

    forces = get_force(X)

    # X of size (2, N_horizontal, N_atoms, 3)

    pos = X[0, ...]
    # pos of size (N_horizontal, N_atoms, 3)

    vel = X[1, ...]
    # vel of size (N_horizontal, N_atoms, 3)

    first_part = beta*(0.5*m*np.sum(vel[0]**2) + pot_0) + N_horizontal*np.log(2*np.pi*(1-a)*b*horizontal_dt/(m*beta))

    sum_part = np.sum(beta*m/(2*(1-a)) * \
    ((pos[1:]-pos[:-1])/(b*horizontal_dt) - b*horizontal_dt/2*forces[:-1]/m - np.sqrt(a)*vel[:-1])**2\
    + (np.sqrt(a)* ( (pos[1:]-pos[:-1])/(b*horizontal_dt) + b*horizontal_dt/2*forces[1:]/m) - vel[1:])**2)

    return (first_part + sum_part)/fict_beta


@jit(nopython=True)
def spring_compute_force_helper(positions, box_length):
      force = np.zeros_like(positions) # (N_atoms, 3)
      box_lengths = np.array([box_length, box_length, box_length])

      potential_energy = 0
      for i in range(N_atoms):
        for j in range(i+1, N_atoms):

            r_ij = positions[j] - positions[i]
            
            # Apply the minimum image convention
            r_ij -= box_lengths * np.round(r_ij / box_lengths)

            r_ij_norm = np.sqrt(np.sum(r_ij**2))

            if r_ij_norm == 0:
                raise ValueError("r_ij_norm = 0")

            # Compute the force magnitude according to the spring potential
            force_mag = - spring_k*(r_ij_norm - spring_l0) 

            V_ij = 0.5*spring_k*(r_ij_norm - spring_l0)**2

            potential_energy += V_ij

            # Add this force to the total force on both particles i and j
            # Note: the force on particle j is the negative of the force on particle i
            force_ij = force_mag * r_ij

            force[i, :] -= force_ij
            force[j, :] += force_ij

      return force, potential_energy



@jit(nopython=True)
def LJ_compute_force_helper(positions, box_length):
      force = np.zeros_like(positions) # (N_atoms, 3)
      box_lengths = np.array([box_length, box_length, box_length])
      potential_energy = 0
      for i in range(N_atoms):
        for j in range(i+1, N_atoms):

            r_ij = positions[j] - positions[i]
            
            # Apply the minimum image convention
            r_ij -= box_lengths * np.round(r_ij / box_lengths)

            r_ij_norm = np.sqrt(np.sum(r_ij**2))

            if r_ij_norm == 0:
                raise ValueError("r_ij_norm = 0")
            if r_ij_norm > lj_cutoff:
                continue

            r_ij_inv = 1. / r_ij_norm
            lj_cutoff_inv = 1. / lj_cutoff

            # Compute the force magnitude according to the Lennard-Jones potential
            force_mag = 24 * lj_epsilon * (2*lj_sigma**12*r_ij_inv**13 - lj_sigma**6*r_ij_inv**7)
            force_mag -= 24 * lj_epsilon * (2*lj_sigma**12*lj_cutoff_inv**13 - lj_sigma**6*lj_cutoff_inv**7)

            V_ij = 4 * lj_epsilon * ((lj_sigma*r_ij_inv)**12 - (lj_sigma*r_ij_inv)**6) - 4 * lj_epsilon * ((lj_sigma*lj_cutoff_inv)**12 + (lj_sigma*lj_cutoff_inv)**6)

            potential_energy += V_ij

            # Add this force to the total force on both particles i and j
            # Note: the force on particle j is the negative of the force on particle i
            force_ij = force_mag * r_ij * r_ij_inv

            force[i, :] -= force_ij
            force[j, :] += force_ij

      return force, potential_energy


@jit(nopython=True)
def compute_potential_energy_helper(X, box_length):

    potential_energy = np.zeros(N_horizontal)

    if type_of_potential == 'HO':
      for t in range(N_horizontal):
        for i in range(N_atoms):
          r = X[0, t, i, :] 
          V_ij = 0.5*m*omega**2*np.sum(r**2)
          potential_energy[t] += V_ij
     
      
    if lennard_jones_on:

      box_lengths = np.array([box_length, box_length, box_length])
      for t in range(N_horizontal):
        for i in range(N_atoms):
          for j in range(i+1, N_atoms):

            r_ij = X[0, t, j, :] - X[0, t, i, :]
            
            # Apply the minimum image convention
            r_ij -= box_lengths * np.round(r_ij / box_lengths)

            r_ij_norm = np.sqrt(np.sum(r_ij**2))

            if r_ij_norm == 0:
                raise ValueError("r_ij_norm = 0")
            if r_ij_norm > lj_cutoff:
                continue

            lj_cutoff_inv = 1. / lj_cutoff

            if r_ij_norm < lj_cap:
              lj_cap_inv = 1. / lj_cap
              A = 24*lj_epsilon*(lj_sigma**6 * lj_cap_inv**7 - 2*lj_sigma**12 * lj_cap_inv**13)
              V_ij = A*(r_ij_norm - lj_cap)
              V_ij += 4 * lj_epsilon * ((lj_sigma*lj_cap_inv)**12 - (lj_sigma*lj_cap_inv)**6) 
              V_ij -= 4 * lj_epsilon * ((lj_sigma*lj_cutoff_inv)**12 + (lj_sigma*lj_cutoff_inv)**6)
            else:
              r_ij_inv = 1. / r_ij_norm
              V_ij = 4 * lj_epsilon * ((lj_sigma*r_ij_inv)**12 - (lj_sigma*r_ij_inv)**6) 
              V_ij -= 4 * lj_epsilon * ((lj_sigma*lj_cutoff_inv)**12 + (lj_sigma*lj_cutoff_inv)**6)

            potential_energy[t] += V_ij

    elif spring_on:
      box_lengths = np.array([box_length, box_length, box_length])
      for t in range(N_horizontal):
        for i in range(N_atoms):
          for j in range(i+1, N_atoms):

            r_ij = X[0, t, j, :] - X[0, t, i, :]
            
            # Apply the minimum image convention
            r_ij -= box_lengths * np.round(r_ij / box_lengths)

            r_ij_norm = np.sqrt(np.sum(r_ij**2))

            if r_ij_norm == 0:
                raise ValueError("r_ij_norm = 0")

            V_ij = 0.5*spring_k*(r_ij_norm - spring_l0)**2
            potential_energy[t] += V_ij

    return potential_energy




def initialize_fcc_lattice2(N_cells, density):

    a=(4/density)**(1./3.) # FCC has 4 atoms per lattice cell
    L = a * N_cells 
    natom = 4 * N_cells**3  # total number of atoms in the box

    j  = 0
    xi = 0.
    yi = 0.
    zi = 0.
    delta=0.0
    rrx = np.random.normal(0., delta, natom)
    rry = np.random.normal(0., delta, natom)
    rrz = np.random.normal(0., delta, natom)

    rx = np.zeros(natom)
    ry = np.zeros(natom)
    rz = np.zeros(natom)

    for nx in range(N_cells):
      for ny in range(N_cells):
        for nz in range(N_cells):
          rx[j] = xi + a*nx + rrx[j]
          ry[j] = yi + a*ny + rry[j]
          rz[j] = zi + a*nz + rrz[j]


          rx[j]/= L
          rx[j]-= np.rint(rx[j])
          ry[j]/= L
          ry[j]-= np.rint(ry[j])
          rz[j]/= L
          rz[j]-= np.rint(rz[j])
          j +=1

          rx[j] = xi + a*nx + rrx[j] + 0.5*a
          ry[j] = yi + a*ny + rry[j] + 0.5*a
          rz[j] = zi + a*nz + rrz[j]

          rx[j]/= L
          rx[j]-= np.rint(rx[j])
          ry[j]/= L
          ry[j]-= np.rint(ry[j])
          rz[j]/= L
          rz[j]-= np.rint(rz[j])
          j +=1

          rx[j] = xi + a*nx + rrx[j] + 0.5*a
          ry[j] = yi + a*ny + rry[j]
          rz[j] = zi + a*nz + rrz[j] + 0.5*a

          rx[j]/= L
          rx[j]-= np.rint(rx[j])
          ry[j]/= L
          ry[j]-= np.rint(ry[j])
          rz[j]/= L
          rz[j]-= np.rint(rz[j])
          j +=1

          rx[j] = xi + a*nx + rrx[j]
          ry[j] = yi + a*ny + rry[j] + 0.5*a
          rz[j] = zi + a*nz + rrz[j] + 0.5*a
          rx[j]/= L
          rx[j]-= np.rint(rx[j])
          ry[j]/= L
          ry[j]-= np.rint(ry[j])
          rz[j]/= L
          rz[j]-= np.rint(rz[j])
          j +=1

    all_points = np.array(np.transpose([rx*L, ry*L, rz*L]))
    # The number of atoms is already correct so no need to shuffle and slice
    return all_points, a, natom, L

def initialize_fcc_lattice(N_cells, density):

    lattice_spacing=(4/density)**(1./3.) # FCC has 4 atoms per lattice cell
    L = lattice_spacing * N_cells 
    num_atoms = 4 * N_cells**3  # total number of atoms in the box

    x = np.linspace(0, L - lattice_spacing, N_cells)  # array of x values
    y = np.linspace(0, L - lattice_spacing, N_cells)  # array of y values
    z = np.linspace(0, L - lattice_spacing, N_cells)  # array of z values

    # Using meshgrid to create 3D grid of lattice points
    xv, yv, zv = np.meshgrid(x, y, z)

    # Stacking together all x, y, and z coordinates for all atoms
    pos = np.stack((xv, yv, zv), axis=-1).reshape(-1, 3)

    # Creating extra points for the fcc lattice (face centered points)
    center_points = lattice_spacing/2 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 0]])
    all_points = np.concatenate([pos + shift for shift in center_points])

    # The number of atoms is already correct so no need to shuffle and slice
    return all_points, lattice_spacing, num_atoms, L

def read_and_check_input_file(name = "input.inp"):

  global density, L, N_cells, horizontal_dt, k_T, gamma, type_of_potential, omega, U_0, a, b, N_horizontal, N_atoms, vertical_dt, M, m, thermostat, fict_gamma, save_config_freq, save_Temp_freq, save_pos_freq, save_vel_freq, save_Ham_freq, save_Obs_freq, beta, fict_beta, fict_k_T, x0_HO, init_path_from_points_from_file, init_points_file, generate_new_traj, writer_traj, writer_H, writer_T, writer_obs, lennard_jones_on, lj_sigma, lj_epsilon, lj_cutoff, freq_output_horizontal, freq_output_vertical, N_vertical, restart_horizontal_from_file, restart_vertical_from_file, vertical, horizontal, lj_cap, natoms, spring_k, spring_l0, spring_on, time_check, init_FCC
 
  input_file = open(path + name, 'r')

  ### Reads parameters from input_file ###
  for line in input_file:
      line = line.strip().split('=')
      if line[0].strip() == 'horizontal':
          horizontal = int(line[1].strip())
      elif line[0].strip() == 'vertical':
          vertical = int(line[1].strip()[0])
      elif line[0].strip() == 'restart_horizontal_from_file':
          restart_horizontal_from_file = int(line[1].strip()[0])
      elif line[0].strip() == 'restart_vertical_from_file':
          restart_vertical_from_file = int(line[1].strip()[0])
      elif line[0].strip() == 'freq_output_horizontal':
          freq_output_horizontal = int(line[1].split()[0])

      ### Initialization of the positions
      elif line[0].strip() == 'L':
          L = int(line[1].split()[0])
      elif line[0].strip() == 'N_cells':
          N_cells = int(line[1].split()[0])
      elif line[0].strip() == 'density':
          density = float(line[1].split()[0])
      elif line[0].strip() == 'natoms':
          natoms = int(line[1].split()[0])

      ### OVRVO parameters ###
      elif line[0].strip() == 'horizontal_dt':
          horizontal_dt = float(line[1].split()[0])
      elif line[0].strip() == 'k_T':
          k_T = float(line[1].split()[0])
      elif line[0].strip() == 'N_horizontal':
          N_horizontal = int(line[1].split()[0])
      elif line[0].strip() == 'gamma':
          gamma = float(line[1].split()[0])
      elif line[0].strip() == 'm':
          m = float(line[1].split()[0])
      elif line[0].strip() == 'N_atoms':
          N_atoms = int(line[1].split()[0])


      elif line[0].strip() == 'init_FCC':
          init_FCC = int(line[1].split()[0])

      ### Parameters for potential ###
      elif line[0].strip() == 'spring_on':
          spring_on = int(line[1].split()[0])
      elif line[0].strip() == 'time_check':
          time_check = int(line[1].split()[0])
      elif line[0].strip() == 'lennard_jones_on':
          lennard_jones_on = int(line[1].split()[0])
      elif line[0].strip() == 'lj_epsilon' and lennard_jones_on == 1:
          lj_epsilon = float(line[1].split()[0])
      elif line[0].strip() == 'lj_sigma' and lennard_jones_on:
          lj_sigma = float(line[1].split()[0])
      elif line[0].strip() == 'lj_cutoff' and lennard_jones_on:
          lj_cutoff = float(line[1].split()[0])
      elif line[0].strip() == 'type_of_potential':
          type_of_potential = line[1].split()[0]
      elif line[0].strip() == 'omega' and type_of_potential == 'HO':
          omega = float(line[1].split()[0])
      elif line[0].strip() == 'x0_HO' and type_of_potential == 'HO':
          x0_HO = float(line[1].split()[0])
      # Capping only affecs vertical dynamics
      elif line[0].strip() == 'lj_cap':
          lj_cap = float(line[1].split()[0])
      elif line[0].strip() == 'spring_k':
          spring_k = float(line[1].split()[0])
      elif line[0].strip() == 'spring_l0':
          spring_l0 = float(line[1].split()[0])

      # BAOAB parameters
      elif line[0].strip() == 'N_vertical':
          N_vertical = int(line[1].split()[0])
      elif line[0].strip() == 'vertical_dt':
          vertical_dt = float(line[1].split()[0])
      elif line[0].strip() == 'M':
          M = float(line[1].split()[0])
      elif line[0].strip() == 'freq_output_vertical':
          freq_output_vertical = int(line[1].split()[0])
         
      ### Thermostatting ###
      elif line[0].strip() == 'fict_gamma':
          fict_gamma = float(line[1].split()[0])
      elif line[0].strip() == 'fict_k_T':
          fict_k_T = float(line[1].split()[0])
      ### Constraining ###
      elif line[0].strip() == 'constraints':
          use_constraints = line[1].split()[0]
      elif line[0].strip() == 'shake' and use_constraints == 'Y':
          use_shake = line[1].split()[0]
          if(use_constraints == 'Y' and use_shake != 'Y'):
              print("WARNING:: Trying to use constraints without shake. The simulation proceeds with unconstrained dynamics.")
      elif line[0].strip() == 'rattle' and use_constraints == 'Y':
          use_rattle = line[1].split()[0]
      elif line[0].strip() == 'constraint_tolerance' and use_constraints == 'Y':
          constraint_tolerance = float(line[1].split()[0])
  
      ### Saving parameters ###
      elif line[0].strip() == 'save_Temp':
          save_Temp_freq = int(line[1].split()[0])
      elif line[0].strip() == 'save_config':
          save_config_freq = int(line[1].split()[0])
      elif line[0].strip() == 'save_Ham':
          save_Ham_freq = int(line[1].split()[0])
      elif line[0].strip() == 'save_Obs':
          save_Obs_freq = int(line[1].split()[0])
      elif line[0].strip() == 'save_pos':
          save_pos_freq = int(line[1].split()[0])
      elif line[0].strip() == 'save_vel':
          save_vel_freq = int(line[1].split()[0])

  if vertical:
    if fict_gamma == 0:
      print('You are running path dynamics with fict_gamma = 0')

  if horizontal and vertical:
    raise ValueError("Please run horizontal and vertical dynamics separately")

  if not(horizontal) and not(vertical):
    raise ValueError("Choose either horizontal or vertical dynamics")

  if vertical and restart_horizontal_from_file:
    raise ValueError("vertical and restart_horizontal_from_file cannot both be 1")

  if horizontal and restart_vertical_from_file:
    raise ValueError("horizontal and restart_vertical_from_file cannot both be 1")


def setup_csv_writer(filename, header):
    outfile = open(filename, 'w')
    writer = csv.writer(outfile)
    writer.writerow(header)
    return writer, outfile


if __name__ == "__main__":
    
    start = time.time()
    read_and_check_input_file("input.inp")

    beta = k_T **-1
    fict_beta = fict_k_T **-1

    if horizontal:
      paths = Paths()
      paths.initialize_horizontal_dynamic()

      paths.OVRVO()
      paths.write_horizontal()
      print("Horizontal dynamic completed")
      quit()

    elif vertical:
      paths = Paths()
      paths.initialize_vertical_dynamic()
      if paths.restart_N_vertical == 0: paths.write_vertical() 
      paths.vertical_iter += 1
      while (paths.vertical_iter < N_vertical + paths.restart_N_vertical):
        #print(f"{paths.vertical_iter} {N_vertical + paths.restart_N_vertical}")
        start_baoab_time = time.time()
        paths.BAOAB()
        end_baoab_time = time.time()
        if time_check: print(f"BAOAB: {end_baoab_time - start_baoab_time}")
      
        if paths.vertical_iter % freq_output_vertical == 0: paths.write_vertical()
        paths.vertical_iter += 1

      print("Vertical dynamic completed")

      end = time.time()
      print(f"total time: {end-start}")

      quit()

