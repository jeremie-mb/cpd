import numpy as np
import math
import os
import sys
import csv
import pandas as pd
import time
from numba import jit

from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

path = os.path.realpath(__file__)
path = path.strip(os.path.basename(sys.argv[0]))

#global dt, k_T, N_horizontal, gamma, type_of_potential, omega, U_0, a, b, do_path_dynamics, N_vertical, N_atoms, vertical_dt, M, thermostat, fict_gamma, save_temp_freq, save_pos_freq, save_vel_freq, save_Ham_freq, beta, fict_beta, fict_k_T, path_point_frequency, use_constraints, use_shake, use_rattle, constraint_tolerance, lennard_jones_on, lj_sigma, lj_epsilon, lj_cutoff

class Paths:
    def __init__(self):
        # Generalized coordinates
        self.X = None
        self.PI = None

        # Functions of self.X and self.PI
        self.dS = None
        self.force = None
        self.force_jacobian = None
        self.force_divergence = None

        # Iterators
        self.horizontal_iter = None
        self.vertical_iter = None

        # Parameters
        self.box_length = None

        # Observables 
        self.instantaneous_temperature = None
        self.mean_temperature = None
        self.potential_energy = None
        self.kinentic_energy = None
        self.total_energy = None

    def initialize_horizontal_dynamic(self):
 
      global N_atoms, box_length
      # For now only external potentials, no pair interaction
      # Random initialization
      if type_of_potential == 'HO':
        r = np.random.normal(loc = 0.0, scale = np.sqrt((1./(m*omega**2))*k_T), size = 3*N_atoms).reshape(N_atoms,3)
        v = np.random.normal(loc = 0.0, scale = np.sqrt(k_T/m), size = 3*N_atoms).reshape(N_atoms, 3)
      elif type_of_potential == 'no_potential' and lennard_jones_on:
        print(f"init_FCC = {init_FCC}")
        if init_FCC == '0':
          '''
          Initialize with the number of atoms and lattice_constant 
          In this case, make sure that lattice_constant ~= LJ equilibrium distance
          '''
          print(f"Initializing positions from the number of atoms")
          lattice_constant = (2**(1. /6 )) * lj_sigma 
          r, box_length = initialize_fcc_lattice(N_atoms, lattice_constant)
        elif init_FCC == '1':
          ''' 
          Initialize with the number of cells and the size of the box
          In this case, you can choose the density but make sure the LJ eq. distance
          is the lattice spacing aka L / N_cells = r_LJ
          '''
          print(f"Initializing positions from the density\n")
          r, lattice_spacing, N_atoms, box_length = initialize_fcc_lattice2(N_cells, density)
          r_eq = (2**(1. /6 )) * lj_sigma 
          print(f"You asked for N_cells = {N_cells} and density = {density}. \n Initializing {N_atoms} in {N_cells} FCC cells in a box of dimension {box_length}.")
          print(f"LJ equilibrium distance is {r_eq} and nearest FCC distance is {lattice_spacing / np.sqrt(2)}")

          ''' Velocities drawn from Boltzmann dist '''
          v = np.random.normal(loc = 0.0, scale = np.sqrt(k_T/m), size = 3*N_atoms).reshape(N_atoms, 3)
      else: 
        raise ValueError("No potential and no Lennard Jones")

      self.horizontal_iter = 0
      self.vertical_iter = 0
      self.X = np.zeros((2, N_horizontal, N_atoms, 3))
      self.kinetic_energy = np.zeros(N_horizontal)
      self.potential_energy = np.zeros(N_horizontal)
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
      c1 = math.exp(-gamma*dt) # c1 = a in the Crooks article
      if gamma == 0:
        c2 = 1 #c2 = b (the time rescaling fator) in the Crooks article
      else:
        c2 = np.sqrt(2/(gamma*dt)*math.tanh(gamma*dt/2))

      r = self.X[0, 0, ...]
      v = self.X[1, 0, ...]

      if self.horizontal_iter != 0:
        raise ValueError("Running OVRVO from initial positions associated with horizontal_iter != 0")

      force_value = self.get_force_OVRVO() # (N_horizontal, N_atoms, 3)

      while self.horizontal_iter < N_horizontal - 1:
        if (self.horizontal_iter % 100 == 0): print(self.horizontal_iter)

        self.horizontal_iter += 1

        #print(f"v before force {v}")

        # O-block
        v = np.sqrt(c1)*v + np.sqrt((1-c1)*k_T/m) * np.random.normal(0,1,3*N_atoms).reshape((N_atoms, 3))
  
        # V-block
        v = v + 0.5*c2*dt*force_value/m
        #print(f"v after force {v}")

        # R-block
        r = r + c2*dt*v

        self.X[0, self.horizontal_iter, ...] = r
        self.X[1, self.horizontal_iter, ...] = v
 
        force_value = self.get_force_OVRVO()
 
        # V-block
        v = v + 0.5*c2*dt*force_value/m

        # O-block
        v = np.sqrt(c1)*v + np.sqrt((1-c1)*k_T/m) * np.random.normal(0,1,3*N_atoms).reshape((N_atoms, 3))

        #r_array.append(r)
        #v_array.append(v)
      
        self.X[0, self.horizontal_iter, :, :] = r
        self.X[1, self.horizontal_iter, :, :] = v

      return 0


    def write(self):
    
      write_mode = 'a' if self.vertical_iter > 1 else 'w'
      first_line = not(os.path.exists("trajectories.csv") and os.path.getsize("trajectories.csv") > 0)

      with open("trajectories.csv", write_mode) as outfile_traj:
        if first_line:
          outfile_traj.write(",".join(["vertical_iter", "atom_index", "horizontal_iter", "x", "y", "z", "vx", "vy", "vz"]) + "\n")
        rows = zip(str(self.vertical_iter)*N_horizontal*N_atoms,
                   [str(i) for i in range(1, N_atoms+1) for alpha in range(N_horizontal) if alpha % freq_output == 0],
                   [f'{alpha}' for i in range(1, N_atoms+1) for alpha in range(int(N_horizontal)) if alpha % freq_output == 0],
                   [f'{self.X[0, alpha, atom_idx -1, 0]:.5f}, {self.X[0, alpha, atom_idx -1, 1]:.5f}, {self.X[0, alpha, atom_idx -1, 2]:.5f}'\
                    for atom_idx in range(1, N_atoms+1) for alpha in range(int(N_horizontal)) if alpha % freq_output == 0],
                   [f'{self.X[1, alpha, atom_idx - 1, 0]:.5f}, {self.X[1, alpha, atom_idx - 1, 1]:.5f}, {self.X[1, alpha, atom_idx -1, 2]:.5f}'\
                    for atom_idx in range(1, N_atoms+1) for alpha in range(int(N_horizontal)) if alpha % freq_output == 0])
        for row in rows:
          outfile_traj.write(",".join(row) + "\n")

      '''
      self.get_kinetic_energy_vector()
      self.get_total_energy_vector()
      temperatures = self.get_instantaneous_real_temperature_vector()

      first_line = not(os.path.exists("horizontal_observables.csv") and os.path.getsize("horizontal_observables.csv") > 0)

      with open("horizontal_observables.csv", write_mode) as outfile_traj:
        if first_line:
            outfile_traj.write(",".join(["vertical_iter", "horizontal_iter", "temperature", "kinetic_energy", "potential_energy", "total_energy"]) + "\n")
        rows = zip([str(self.vertical_iter)]*N_horizontal,
               [f'{int(alpha)}' for alpha in range(N_horizontal) if alpha % freq_output == 0],
               [f'{temperatures[int(alpha)]}' for alpha in range(N_horizontal) if alpha % freq_output == 0],
               [f'{self.kinetic_energy[alpha]}' for alpha in range(N_horizontal) if alpha % freq_output == 0],
               [f'{self.potential_energy[alpha]}' for alpha in range(N_horizontal) if alpha % freq_output == 0],
               [f'{self.total_energy[alpha]}' for alpha in range(N_horizontal) if alpha % freq_output == 0])
        for row in rows:
          outfile_traj.write(",".join(row) + "\n")
      '''
      return 0


    '''
    def write_horizontal(self):

      with open("horizontal_trajectories.csv", 'w') as outfile_traj:
        outfile_traj.write(",".join(["fictitious_time", "atom_index", "horizontal_iter", "x", "y", "z", "vx", "vy", "vz"]) + "\n")
        rows = zip([str(0)]*N_horizontal*N_atoms,
                   [str(i) for i in range(1, N_atoms+1) for alpha in range(N_horizontal) if alpha % freq_output == 0],
                   [f'{alpha}' for i in range(1, N_atoms+1) for alpha in range(int(N_horizontal)) if alpha % freq_output == 0],
                   [f'{self.X[0, alpha, atom_idx -1, 0]:.5f}, {self.X[0, alpha, atom_idx -1, 1]:.5f}, {self.X[0, alpha, atom_idx -1, 2]:.5f}'\
                    for atom_idx in range(1, N_atoms+1) for alpha in range(int(N_horizontal)) if alpha % freq_output == 0],
                   [f'{self.X[1, alpha, atom_idx - 1, 0]:.5f}, {self.X[1, alpha, atom_idx - 1, 1]:.5f}, {self.X[1, alpha, atom_idx -1, 2]:.5f}'\
                    for atom_idx in range(1, N_atoms+1) for alpha in range(int(N_horizontal)) if alpha % freq_output == 0])
        for row in rows:
            outfile_traj.write(",".join(row) + "\n")

      temperatures = self.get_instantaneous_real_temperature_vector()
      self.get_kinetic_energy_vector()
      self.get_total_energy_vector()

      with open("horizontal_observables.csv", 'w') as outfile_traj:
        outfile_traj.write(",".join(["fictitious_time", "horizontal_iter", "real_time", "temperature", "kinetic_energy", "potential_energy", "total_energy"]) + "\n")
        rows = zip([str(0)]*N_horizontal,
               [f'{alpha*dt:.5f}' for alpha in range(N_horizontal) if alpha % freq_output == 0],
               [f'{int(alpha)}' for alpha in range(N_horizontal) if alpha % freq_output == 0],
               [f'{self.get_instantaneous_real_temperature_vector()[int(alpha)]}' for alpha in range(N_horizontal) if alpha % freq_output == 0],

               [f'{self.kinetic_energy[alpha]}' for alpha in range(N_horizontal) if alpha % freq_output == 0],
               [f'{self.potential_energy[alpha]}' for alpha in range(N_horizontal) if alpha % freq_output == 0],
               [f'{self.total_energy[alpha]}' for alpha in range(N_horizontal) if alpha % freq_output == 0])
        for row in rows:
            outfile_traj.write(",".join(row) + "\n")

      return 0
    '''

    def get_force_OVRVO(self):
        if type_of_potential == 'HO':
          external_force = -m*omega**2*self.X[0, self.horizontal_iter, ...]
        elif type_of_potential == 'no_potential':
          external_force = np.zeros_like(self.X[0, self.horizontal_iter, ...])
        else:
          raise ValueError("Unknown potential type")

        if lennard_jones_on:
          force_start_time = time.time()
          lj_force = self.LJ_force_OVRVO_image()
          force_end_time = time.time()
          #print(f"Execution time for LJ_compute_force_helper: {force_end_time - force_start_time} seconds")
        else:
          lj_force = np.zeros_like(self.X[0, self.horizontal_iter, ...])

        self.force = external_force + lj_force
        return self.force

    def LJ_force_OVRVO(self): 
      force = np.zeros_like(self.X[0, self.horizontal_iter, ...]) # (N_atoms, 3)
      for i in range(N_atoms):
        for j in range(i+1, N_atoms):
          r_ij = np.linalg.norm(self.X[0, self.horizontal_iter, i] - self.X[0, self.horizontal_iter, j])
          if r_ij == 0:
            raise ValueError("r_ij = 0")
          if r_ij > lj_cutoff:
            continue

          dir_vec = (self.X[0, self.horizontal_iter, i] - self.X[0, self.horizontal_iter, j]) / r_ij  # Direction vector from particle j to i
          
          # Compute the force magnitude according to the Lennard-Jones potential
          force_mag = 24 * lj_epsilon * ((2*(lj_sigma/r_ij)**13) - (lj_sigma/r_ij)**7)

          # Add this force to the total force on both particles i and j
          # Note: the force on particle j is the negative of the force on particle i
          force[i, :] += force_mag * dir_vec 
          force[j, :] -= force_mag * dir_vec

      return force

    def LJ_force_OVRVO_image(self):
      force, self.potential_energy[self.horizontal_iter] = LJ_compute_force_helper(self.X[0, self.horizontal_iter, ...], box_length)
      return force
      '''
      force = np.zeros_like(self.X[0, self.horizontal_iter, ...]) # (N_atoms, 3)
      box_lengths = np.array([self.box_length, self.box_length, self.box_length])
      self.potential_energy[self.horizontal_iter] = 0
      for i in range(N_atoms):
        for j in range(i+1, N_atoms):
            dist_vec = self.X[0, self.horizontal_iter, i] - self.X[0, self.horizontal_iter, j]

            # Apply the minimum image convention
            #print(f"atom {i} {self.X[0, self.horizontal_iter, i]} and atom {j} {self.X[0, self.horizontal_iter, j]}")
            #print(f"dist_vec is {dist_vec}")
            dist_vec -= box_lengths * np.round(dist_vec / box_lengths)

            r_ij = np.linalg.norm(dist_vec)
            if r_ij == 0:
                #print(f"dist_vec is {dist_vec}")
                #print(f"box_length is size {box_length}")
                raise ValueError("r_ij = 0")
            if r_ij > lj_cutoff:
                continue

            dir_vec = dist_vec / r_ij  # Direction vector from particle j to i

            # Compute the force magnitude according to the Lennard-Jones potential
            force_mag = 24 * lj_epsilon * ((2*(lj_sigma/r_ij)**13) - (lj_sigma/r_ij)**7)

            V_ij = 4 * lj_epsilon * ((lj_sigma / r_ij)**12 - (lj_sigma / r_ij)**6)
            self.potential_energy[self.horizontal_iter] += V_ij

            # Add this force to the total force on both particles i and j
            # Note: the force on particle j is the negative of the force on particle i
            force[i, :] += force_mag * dir_vec
            force[j, :] -= force_mag * dir_vec

      return force
      '''

    def LJ_energy_OVRVO(self): 
      # LIGHTHOUSE calculates the LJ energy of the system at self.horizontal_iter (could be done with LJ_force_OVRVO OPTI)
      potential_energy = 0.
      for i in range(N_atoms):
        for j in range(i+1, N_atoms):
          r_ij = np.linalg.norm(self.X[0, self.horizontal_iter, i] - self.X[0, self.horizontal_iter, j])
          if r_ij == 0:
            raise ValueError("r_ij = 0")
          if r_ij > lj_cutoff:
            continue

          # Compute the force magnitude according to the Lennard-Jones potential
          u = 4 * lj_epsilon * (((lj_sigma/r_ij)**12) - (lj_sigma/r_ij)**6)

          potential_energy += u

      return potential_energy

    def get_instantaneous_real_temperature_vector(self): 
      '''
      X.shape = (2, N_horizontal, N_atoms, 3). Sum over number of atoms and number of components. No constraints  
      Returns a 1-dimensional array of size N_horizontal
      '''
      self.instantaneous_temperature = (1./3)*np.sum(m*self.X[1]**2, axis = (1, 2))/self.X.shape[2] # size (2, N_horizontal, N_atoms, 3)
      return self.instantaneous_temperature 
  
    def get_real_temperature(self): 
      '''
      X.shape = (2, N_horizontal, N_atoms, 3). Sum over path (real time, number of atoms and number of components). No constraints  
      Returns a float
      '''
      self.real_temperature = (1./3)*np.sum(m*self.X[1]**2, axis = (0, 1, 2))/(self.X.shape[1]*self.X.shape[2]) 
      return self.real_temperature

    def get_fictitious_temperature(self): 
      ''' 
      PI.shape = (2, N_vertical, 2*N_atoms, 3)
      Sum over real time, number of fictious atoms, and components
      '''
      return (1./3)*np.sum(self.PI[1]**2/M)/(PI.shape[1]*PI.shape[2]) # PI.shape = (2, N_vertical, 2*N_atoms, 3)

    def get_vertical_hamiltonian(self):

      H = np.sum(m*self.X[1]**2/2, axis = (2,1))

      if type_of_potential == "no_potential" and lennard_jones_on:
        potential_energy = LJ_energy_OVRVO()
        H += potential_energy
      else:
        raise ValueError("Potential energy can only be calculated with no potential and lennard jones")
      return H
     
    def get_kinetic_energy_vector(self):
      self.kinetic_energy = 0.5*m*np.sum(self.X[1, ...]**2, axis = (1,2))
      return self.kinetic_energy 
       
    def get_total_energy_vector(self):
      self.total_energy = self.kinetic_energy + self.potential_energy
      return self.total_energy
 
    def get_force(self):
        if type_of_potential == 'HO':
          external_force = -m*omega**2*self.X[0]
        elif type_of_potential == 'no_potential':
          external_force = np.zeros_like(self.X[0])
        else:
          raise ValueError("Unknown potential type")

        if lennard_jones_on:
          lj_force = self.LJ_force()
        else:
          lj_force = np.zeros_like(self.X[0])

        self.force = external_force + lj_force
        return self.force

    def get_force_jacobian(self): # of size (N_horizontal, N_atoms, 3 , 3)
      # Take the Jacobian of the potential
      if type_of_potential == 'HO':
        external_force_jacobian = np.tile(-m*omega**2 * np.eye(3), (N_horizontal, N_atoms, 1, 1))
      elif type_of_potential == 'no_potential':
        external_force_jacobian = np.zeros(N_horizontal, N_atoms, 3, 3)

      # Add the Jacobian of the pair interaction if on
      if lennard_jones_on == 1:
        lj_force_jacobian = lennard_jones_force_jacobian(self.X[0])
      else:
        lj_force_jacobian = np.zeros((number_of_timestep, N_atoms, 3, 3)) # No Lennard-Jones force

      total_force_jacobian = external_force_jacobian + lj_force_jacobian
      self.force_jacobian = total_force_jacobian
      
      return total_force_jacobian

    def get_force_divergence(self):    
      if type_of_potential == 'HO': 
        external_force_divergence = -3*m*omega**2
      elif type_of_potential == 'no_potential':
        external_force_divergence = 0

      if lennard_jones_on == 1:
        lj_force_divergence = LJ_force_divergence(self)
      
      return external_force_divergence + lj_force_divergence
        

    def LJ_force(self): # self.X[0] is of size (N_horizontal, N_atoms, 3)
      force = np.zeros_like(self.X[0])
      for t in range(N_horizontal):
        for i in range(N_atoms):
          for j in range(i+1, N_atoms):
            r_ij = np.linalg.norm(self.X[0, t, i] - self.X[0, t, j])
            if r_ij == 0:
              raise ValueError("r_ij = 0")
            if r_ij > lj_cutoff:
              continue

            dir_vec = (self.X[0, t, i] - self.X[0, t, j]) / r_ij  # Direction vector from particle j to i
            
            # Compute the force magnitude according to the Lennard-Jones potential
            force_mag = 24 * lj_epsilon * ((2*(lj_sigma/r_ij)**12) - (lj_sigma/r_ij)**6)
            
            # Add this force to the total force on both particles i and j
            # Note: the force on particle j is the negative of the force on particle i
            force[t, i] += force_mag * dir_vec 
            force[t, j] -= force_mag * dir_vec

      return force

    def LJ_force_jacobian(self): 

      jacobians_over_time = []
      for t in range(N_horizontal):

        jacobians = [] # will be of size (N_atoms, 3, 3)
        for i in range(N_atoms):
          jacobian = np.zeros((3, 3))
          for j in range(N_atoms):
            if i != j:
              r_ji = self.X[0, t, i] - self.X[0, t, j]
              r_ji_norm = np.linalg.norm(r_ji)
              u = np.identity(3)  # Unit vectors along x, y, and z

              a = 2 * sigma / r_ji_norm
              b = sigma / r_ji_norm

              for alpha in range(3):  # x=0, y=1, z=2
                term1 = - 24 * (a**12 - b**6) / r_ji_norm * u[alpha]
                term2 = 24 * (self.X[0, t, j, alpha] - self.X[0, t, i, alpha]) * (13 * a**12 - 2 * b**6) / r_ji_norm**2 * r_ji / r_ji_norm

                jacobian[:, alpha] += term1 + term2

          jacobians.append(jacobian)
        # at this stage we have the list jacobians at a given timestep
        jacobians_over_time.append(jacobians)
      return np.array(jacobians_over_time).reshape((N_horizontal, N_atoms, 3, 3))

    def LJ_force_divergence(self):

      traces_over_time = []
      for jacobian_at_t in self.jacobians:
        traces = [np.trace(jacobian) for jacobian in jacobian_at_t]
        traces_over_time.append(traces)
      return np.array(traces_over_time).reshape(N_horizontal, N_atoms)

    def get_dS(self): # X is of size (2, N, N_atoms, 3) 

      force = self.get_force()
      force_jacobian = self.get_force_jacobian()

      a = np.exp(-gamma*dt)
      b = np.sqrt(2/(gamma*dt)*math.tanh(gamma*dt/2))

      pos = self.X[0, ...]
      vel = self.X[1, ...]

      # (N, 3) => (N, N_atoms, 3) vectorization
 
      dS_r_0 = -beta * forces[0,:,:]/fict_beta + (beta*m/((1-a)*fict_beta)) * (
        -(1+a)/(b*dt)**2 *(pos[1,:,:]-pos[0,:,:])
        - (1/(2*m))*(a*forces[1,:,:]-forces[0,:,:]) 
        + np.sqrt(a)/(b*dt) * (vel[1,:,:] + vel[0,:,:]) 
        - 1/(2*m) * np.einsum('...ij,...j', grad_forces[0,:,:,:], pos[1,:,:] - pos[0,:,:])
        + np.sqrt(a)*b*dt/(2*m) * np.einsum('...ij,...j', grad_forces[0,:,:,:], vel[0,:,:])
        + (b*dt)**2/(2*m)**2 * np.einsum('...ij,...j', grad_forces[0,:,:,:], forces[0,:,:]))

      dS_v_0 = beta*m*vel[0,:,:]/fict_beta + beta*m/((1-a)*fict_beta) * np.sqrt(a) * (-(pos[1,:,:]-pos[0,:,:]) / (b*dt) + b*dt/2 * forces[0,:,:]/m + np.sqrt(a)*vel[0,:,:])

      dS_r_N = beta*m/((1-a)*fict_beta) * (
        (1+a)/(b*dt)**2 * (pos[-1,:,:]-pos[-2,:,:])
        + (1/(2*m))*(a*forces[-1,:,:]-forces[-2,:,:]) 
        - np.sqrt(a)/(b*dt) * (vel[-1,:,:] + vel[-2,:,:]) 
        + (a/(2*m)) * np.einsum('...ij,...j', grad_forces[-1,:,:,:], pos[-1,:,:] - pos[-2,:,:])
        + (b*dt)**2/(2*m)**2 * a * np.einsum('...ij,...j', grad_forces[-1,:,:,:], forces[-1,:,:])
        - np.sqrt(a)*b*dt/(2*m)* np.einsum('...ij,...j', grad_forces[-1,:,:,:], vel[-1,:,:]))

      dS_v_N = beta*m/((1-a)*fict_beta) * (-np.sqrt(a) * ((pos[-1,:,:]-pos[-2,:,:]) / (b*dt) + b*dt/2 * forces[-1,:,:]/m) + vel[-1,:,:])

      dS_r_array = np.concatenate(([dS_r_0], dS_n[0], [dS_r_N]), axis=0)
      dS_v_array = np.concatenate(([dS_v_0], dS_n[1], [dS_v_N]), axis=0)

      self.dS_array = np.array([dS_r_array, dS_v_array])

      return self.dS_array


    def initialize_vertical_dynamic(self):
      

      self.PI = np.random.normal(size = 2*N_atoms*N_horizontal*3, scale = np.sqrt(M*fict_k_T)).reshape((2, N_atoms, N_horizontal, 3))
      # Calculate dS
      self.get_dS()

      return 0

    def BAOAB(self):

      # Generalized coordinates self.X
      # Generalized momenta self.PI
      # Path action self.dS

      # dS should be there from the previous timestep

      # B block
      self.PI -= self.dS*(vertical_dt/2)

      # A block
      self.X += self.PI*(vertical_dt/(2*M))
      
      # O block
      random_vector = np.random.multivariate_normal(mean = (0.0, 0.0, 0.0), cov = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], size = (2, N_horizontal, N_atoms))

      self.PI *= math.exp(-fict_gamma*vertical_dt) 
      self.PI += math.sqrt(fict_k_T * M * (1-math.exp(-2*fict_gamma*vertical_dt)))*random_vector

      # A block
      self.X += self.PI*(vertical_dt/(2*M))
      self.get_dS()

      ''' LIGHTHOUSE add shake '''

      # B block
      self.PI -= self.dS*(vertical_dt/2)

      ''' LIGHTHOUSE add rattle '''

      self.iter_vertical += 1

      return new_paths


def initialize_fcc_lattice2(N_cells, density):

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


def initialize_fcc_lattice(N_atoms, nearest_neighbour_distance):
    atoms_per_side = int(np.ceil((N_atoms / 4) ** (1/3)))  # number of atoms per side, rounded up
    lattice_constant = nearest_neighbour_distance * np.sqrt(2)  # calculate lattice constant from nearest neighbour distance
    box_length = lattice_constant * atoms_per_side  # total length of the box is the number of atoms times the lattice spacing

    x = np.linspace(0, box_length - lattice_constant, atoms_per_side)  # array of x values
    y = np.linspace(0, box_length - lattice_constant, atoms_per_side)  # array of y values
    z = np.linspace(0, box_length - lattice_constant, atoms_per_side)  # array of z values

    # Using meshgrid to create 3D grid of lattice points
    xv, yv, zv = np.meshgrid(x, y, z)

    # Stacking together all x, y, and z coordinates for all atoms
    pos = np.stack((xv, yv, zv), axis=-1).reshape(-1, 3)

    # Creating extra points for the fcc lattice (face centered points)
    center_points = lattice_constant/2 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 0]])
    all_points = np.concatenate([pos + shift for shift in center_points])

    # Making sure the number of atoms is correct
    np.random.shuffle(all_points)
    return all_points[:N_atoms], box_length

def read_input_file(name = "input.inp"):

  global density, init_FCC, L, N_cells, dt, k_T, gamma, type_of_potential, omega, U_0, a, b, do_path_dynamics, N_horizontal, N_atoms, vertical_dt, M, m, thermostat, fict_gamma, save_config_freq, save_Temp_freq, save_pos_freq, save_vel_freq, save_Ham_freq, save_Obs_freq, beta, fict_beta, fict_k_T, x0_HO, init_path_from_points_from_file, init_points_file, generate_new_traj, writer_traj, writer_H, writer_T, writer_obs, lennard_jones_on, lj_sigma, lj_epsilon, lj_cutoff, freq_output, N_vertical
 
  input_file = open(path + name, 'r')

  ### Reads parameters from input_file ###
  for line in input_file:
      line = line.strip().split('=')
      if line[0].strip() == 'generate_new_path':
          generate_new_traj = line[1].strip()
      elif line[0].strip() == 'work_name':
          work_name = line[1].split()[0]
      elif line[0].strip() == 'input_work_name':
          input_work_name = line[1].strip().split()[0]
      elif line[0].strip() == 'do_path_dynamics':
          do_path_dynamics = line[1].split()[0]
      elif line[0].strip() == 'freq_output':
          freq_output = int(line[1].split()[0])
          print(freq_output)
  input_file.close()


  input_file = open(path + name, 'r')
  for line in input_file:
      line = line.strip().split('=')

      ### Initialization of the positions
      if line[0].strip() == 'init_FCC':
          init_FCC = line[1].strip().split()[0]
      elif line[0].strip() == 'L':
          L = int(line[1].split()[0])
      elif line[0].strip() == 'N_cells':
          N_cells = int(line[1].split()[0])
      elif line[0].strip() == 'density':
          density = float(line[1].split()[0])

      ### OVRVO parameters ###
      elif line[0].strip() == 'dt':
          dt = float(line[1].split()[0])
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

      ### Parameters for potential ###
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

      if do_path_dynamics == 'Y':
          if line[0].strip() == 'N_vertical':
              N_vertical = int(line[1].split()[0])
          elif line[0].strip() == 'vertical_dt':
              vertical_dt = float(line[1].split()[0])
          elif line[0].strip() == 'M':
              M = float(line[1].split()[0])
         
          ### Thermostatting ###
          elif line[0].strip() == 'thermostat':
              thermostat = line[1].split()[0]
          elif line[0].strip() == 'fict_gamma' and thermostat == 'Y':
              fict_gamma = float(line[1].split()[0])
          elif line[0].strip() == 'fict_gamma' and thermostat == 'N':
              fict_gamma = 0.0
          elif line[0].strip() == 'fict_k_T' and thermostat == 'Y':
              fict_k_T = float(line[1].split()[0])
          elif line[0].strip() == 'fict_k_T' and thermostat == 'N':
              fict_k_T = 1.0
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
      
          #read init points form file
          elif line[0].strip() == 'initialize_path_from_points_from_file':
              init_path_from_points_from_file = line[1].split()[0]
          elif line[0].strip() == 'init_points_file' and init_path_from_points_from_file == 'Y':  
              init_points_file = line[1].split()[0]   
      
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

  if do_path_dynamics == 'Y':
    if gamma == 0:
      print('You are running path dynamics with gamma = 0, which is not possible. Stopping the program')
      quit()


def setup_csv_writer(filename, header):
    outfile = open(filename, 'w')
    writer = csv.writer(outfile)
    writer.writerow(header)
    return writer, outfile

@jit(nopython=True)
def LJ_compute_force_helper(positions, box_length):
      force = np.zeros_like(positions) # (N_atoms, 3)
      box_lengths = np.array([box_length, box_length, box_length])

      potential_energy = 0
      for i in range(N_atoms):
        for j in range(i+1, N_atoms):
            dist_vec = positions[i] - positions[j]

            # Apply the minimum image convention
            dist_vec -= box_lengths * np.round(dist_vec / box_lengths)

            r_ij = np.sqrt(np.sum(dist_vec**2))
            if r_ij == 0:
                raise ValueError("r_ij = 0")
            if r_ij > lj_cutoff:
                continue

            r_ij_inv = 1. / r_ij

            # Compute the force magnitude according to the Lennard-Jones potential
            force_mag = 24 * lj_epsilon * ((2*(lj_sigma * r_ij_inv)**13) - (lj_sigma * r_ij_inv)**7)

            V_ij = 4 * lj_epsilon * ((lj_sigma / r_ij)**12 - (lj_sigma / r_ij)**6)
            potential_energy += V_ij

            # Add this force to the total force on both particles i and j
            # Note: the force on particle j is the negative of the force on particle i
            force_ij = force_mag * dist_vec * r_ij_inv

            force[i, :] += force_ij
            force[j, :] -= force_ij

      return force, potential_energy


def prompt_continue_vertical_dynamics():
    response = input("Do you want to continue with the vertical dynamics? (y/n): ")

    while response not in ['y', 'n']:
        print("Invalid response. Please answer with 'y' or 'n'.")
        response = input("Do you want to continue with the vertical dynamics? (y/n): ")

    if response == 'n':
        print("Not continuing with vertical dynamics. Simulation stopped.")
        exit()
    else:
        print("Continuing with vertical dynamics...")
        return

 

if __name__ == "__main__":
    
    start = time.time()
    read_input_file("input.inp")

    # Define variables
    #beta = k_T **-1
    #fict_beta = fict_k_T **-1
    
    # Create ouput files
    #writer_traj, outfile_traj = setup_csv_writer("trajectories.csv", ["path_number", "path_id", "horizontal_iter", "x", "y", "z", "vx", "vy", "vz"])
    #writer_H, outfile_H = setup_csv_writer("Hamiltonian_for_paths.csv", ["path_number", "H"])
    #writer_T, outfile_T = setup_csv_writer("temperature_for_paths.csv", ["path_number", "T_real", "T_fictitious", "T_real_config"])
    #writer_obs, outfile_obs = setup_csv_writer("observable.csv", ["path_number", "path_id", "lambda_A", "lambda_B", "lambda_Bprime", "observable_x", "observable_y", "observable_z", "run_avg_obs_x", "run_avg_obs_y", "run_avg_obs_z"])
    #writer_multipliers, outfile_multipliers = setup_csv_writer("multipliers.csv", ["path_number", "path_id", "lambda_x0", "lambda_xN", "mu_x0", "mu_xN"])

    # Run horizontal
    paths = Paths()
    paths.initialize_horizontal_dynamic()

    paths.OVRVO()
    paths.write()

    prompt_continue_vertical_dynamics()
    # Vertical
    paths.initialize_vertical_dynamic()
    

    while (self.iter_vertical < N_vertical):
      # Update the class
      paths.BAOAB()
      
      if freq_output_vertical: 
        paths.write()

      self.paths.iter_vertical += 1

    '''
    particle_array = initialize_path_dynamics()
    #particle_array = initialize_path_dynamics(init_traj_name)
    grad_constraints_matrix, lagrange_multipliers, rattle_multipliers = [], [], []
    if(use_constraints == 'Y'): grad_constraints_matrix, lagrange_multipliers, rattle_multipliers = initialize_constraints(particle_array)
    path_sampling(particle_array, lagrange_multipliers, rattle_multipliers, grad_constraints_matrix, outfile_traj, outfile_H, outfile_T)
    print('Done with path dynamics')

    if rank == 0:
        outfile_final_traj = open("final_trajectory.csv", 'w')
        writer_final_traj = csv.writer(outfile_final_traj)
        header_final_traj = ["path_number", "path_id", "horizontal_iter", "x", "y", "z", "vx", "vy", "vz", "Pi_x", "Pi_y", "Pi_z", "Pi_vx", "Pi_vy", "Pi_vz"]
        writer_final_traj.writerow(header_final_traj)
        for part in particle_array:
            writer_final_traj.writerows(np.c_[[N_vertical-1]*len(part.pos_array), [part.idp]*len(part.pos_array), [l*dt for l in range(len(part.pos_array))], part.pos_array, part.vel_array, part.Pi_array[0], part.Pi_array[1]])
        outfile_final_traj.close()
        
        outfile_traj.close()
        outfile_H.close()
        outfile_T.close()
    elif do_path_dynamics == 'N':
      print('Not doing path dynamics as requested in the input file')
    '''
    end = time.time()
    print(f"total time: {end - start}")
