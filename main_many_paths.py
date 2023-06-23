import numpy as np
import math
import os
import sys
import csv
import pandas as pd
import time

from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

path = os.path.realpath(__file__)
path = path.strip(os.path.basename(sys.argv[0]))

global dt, k_T, N, theta, gamma, type_of_potential, omega, U_0, a, b, do_path_dynamics, number_of_paths, delta_t, M, thermostat, fict_gamma, save_temp_freq, save_pos_freq, save_vel_freq, save_Ham_freq, beta, fict_beta, fict_k_T, path_point_frequency, use_constraints, use_shake, use_rattle, constraint_tolerance

class Path_of_particle:
	def __init__(self, idp, pos_array, vel_array, Pi_array, dS_array, lambda_A, lambda_B, lambda_Bprime, obs_running_avg):
		self.idp = idp
		self.pos_array = pos_array
		self.vel_array = vel_array
		self.Pi_array = Pi_array
		self.dS_array = dS_array
		self.lambda_A = lambda_A
		self.lambda_B = lambda_B
		self.lambda_Bprime = lambda_Bprime
		self.obs_running_avg = obs_running_avg
		
'''class Constrains:
	def __init__(self, constraint_vector, grad_constraint_vector, lagrange_multipliers):
		self.constraint_vector = constraint_vector
		self.grad_constraint_vector = grad_constraint_vector
		self.lagrange_multipliers = lagrange_multipliers'''

def initialize_path_dynamics():
    print('Creating new initial trajectories')
    if gamma != 0:
        print('Thermostat on for real dynamics')
    elif gamma == 0:
        print('Thermostat off on for real dynamics')
    
    if rank == 0:
        lambda_As = np.array([0.0])
        idsA = np.array([i for i in range(len(lambda_As))])
        lambda_Bs = np.array([0.0]*len(lambda_As))
        
        lambda_As = np.array_split(lambda_As, size)
        idsA = np.array_split(idsA, size)
        lambda_Bs = np.array_split(lambda_Bs, size)
    else:
        lambda_As = None
        idsA = None
        lambda_Bs = None
    
    lambda_As = comm.scatter(lambda_As, root=0)
    lambda_Bs = comm.scatter(lambda_Bs, root=0)
    idsA = comm.scatter(idsA, root=0)
    
    particle_array = []
    for i in range(len(lambda_As)):    
        lambda_Bprimes = np.array([0.0])
        
        for j in range(len(lambda_Bprimes)):
            pos_array, vel_array = OVRVO(lambda_As[i])
    
            #writer_init_traj.writerows(np.c_[[i for l in range(len(pos_array))], [l*dt for l in range(len(pos_array))], pos_array, vel_array])
      
            Pi_x = np.random.normal(size=N, scale=np.sqrt(M*fict_k_T))
            Pi_vx = np.random.normal(size=N, scale=np.sqrt(M*fict_k_T))
    
            Pi_y = np.random.normal(size=N, scale=np.sqrt(M*fict_k_T))
            Pi_vy = np.random.normal(size=N, scale=np.sqrt(M*fict_k_T))
 
            Pi_z = np.random.normal(size=N, scale=np.sqrt(M*fict_k_T))
            Pi_vz = np.random.normal(size=N, scale=np.sqrt(M*fict_k_T))
 
            #Pi_x -= np.sum(Pi_x)/len(Pi_x)
            #Pi_vx -= np.sum(Pi_vx)/len(Pi_vx)
    
            Pi_pos = np.transpose(np.array([Pi_x, Pi_y, Pi_z]))
            Pi_vel = np.transpose(np.array([Pi_vx, Pi_vx, Pi_vz]))
    
            #pos_array = np.transpose(np.array([data["x"], data["y"], data["z"]]))
            #vel_array = np.transpose(np.array([data["vx"], data["vy"], data["vz"]]))
            Pi_array = np.array([Pi_pos, Pi_vel])
    
            dS_array = get_dS_vector(pos_array, vel_array)
    
            p0 = Path_of_particle(idp=idsA[i]*len(lambda_Bprimes) + j, pos_array=pos_array,  vel_array=vel_array, Pi_array=Pi_array, dS_array=dS_array, lambda_A=lambda_As[i], lambda_B=lambda_Bs[i], lambda_Bprime=lambda_Bprimes[j], obs_running_avg=-dS_array[0,-1])
        
            particle_array.append(p0)
    
    particle_array_init_output = comm.gather(particle_array, root=0)
    #args = (([i for i in range(len(init_x0s))], init_x0s["x0"]))
    print(f"before::gamma: {gamma}")

    ''''with MPIPoolExecutor() as executor:
        particle_list_2 = list(executor.map(run_each_initial_path, [i for i in range(len(init_x0s))], init_x0s["x0"]))
        #particle_list_2 = list(executor.map(run_each_initial_path, init_x0s["x0"]))'''
    
    if rank == 0:
        particle_array_init_output = np.concatenate(particle_array_init_output)
        init_traj_name = "init_trajectory.csv"
        outfile_init_traj = open(init_traj_name, 'w')
        writer_init_traj = csv.writer(outfile_init_traj)
        header_init_traj = ["path_number", "path_id", "real_time", "x", "y", "z", "vx", "vy", "vz", "Pi_x", "Pi_y", "Pi_z", "Pi_vx", "Pi_vy", "Pi_vz"]
        writer_init_traj.writerow(header_init_traj)
        for part in particle_array_init_output:
            #print(part.idp)
            #print(len(part.pos_array))
            #print(len(part.Pi_array[0]))
            writer_init_traj.writerows(np.c_[[0.0]*len(part.pos_array), [part.idp]*len(part.pos_array), [l*dt for l in range(len(part.pos_array))], part.pos_array, part.vel_array, part.Pi_array[0], part.Pi_array[1]])
    
        outfile_init_traj.close()
    print('Done with creating initial path')
    
    #particle_array = []
    
    return particle_array
	
def initialize_constraints(particle_array):#, grad_constraints_matrix, lagrange_multipliers):
	lagrange_multipliers = np.array([[[0.0, 0.0], [0.0, 0.0]]]*len(particle_array))
	rattle_multipliers = np.array([[[0.0, 0.0], [0.0, 0.0]]]*len(particle_array))
	grad_constraints_matrix = []
	'''for part in particle_array: 
		grad_constraints_matrix_part = grad_constraint_vectors(part)
		grad_constraints_matrix.append(grad_constraints_matrix_part)'''
	#grad_constraints_matrix = grad_constraint_vectors(particle_array[0])
	#print(f"initialize_constraints:: grad_constraints_matrix: {grad_constraints_matrix}")
	print(f"len(grad_constraints_matrix): {len(grad_constraints_matrix)}, len(rattle_multipliers): {len(rattle_multipliers)}")
	#print(f"len(grad_constraints_matrix[0]): {len(grad_constraints_matrix[0])}, len(rattle_multipliers): {len(rattle_multipliers)}")
	return grad_constraints_matrix, lagrange_multipliers, rattle_multipliers
	#return lagrange_multipliers, rattle_multipliers
	

def OVRVO(r0):
    r_array = []
    v_array = []
    c1 = math.exp(-gamma*dt) # c1 = a in the Crooks article
    if gamma == 0:
        c2 = 1 #c2 = b (the time rescaling fator) in the Crooks article
    else:
        c2 = np.sqrt(2/(gamma*dt)*math.tanh(gamma*dt/2))

    if type_of_potential == 'HO':
        r = np.random.normal(loc = 0.0, scale = np.sqrt((1./(m*omega**2))*k_T), size = 3)
        v = np.random.normal(loc = 0.0, scale = np.sqrt(k_T/m), size = 3)
    elif type_of_potential == 'DW':
        r = np.array(r0)
        v = np.random.normal(loc = 0.0, scale = np.sqrt(k_T/m), size = 3)

    t = 0.0

    r_array.append(r)
    v_array.append(v)

    for i in range(1,N):
        # O-block
        v = np.sqrt(c1)*v + np.sqrt((1-c1)*k_T/m) * np.random.normal(0,1,3)

        # V-block
        v = v + 0.5*c2*dt*force(r)/m

        # R-block
        r = r + c2*dt*v

        # V-block
        v = v + 0.5*c2*dt*force(r)/m

        # O-block
        v = np.sqrt(c1)*v + np.sqrt((1-c1)*k_T/m) * np.random.normal(0,1,3)

        ### Saving the position and velocity in a .txt file ###
        if i % theta == 0:
            r_array.append(r)
            v_array.append(v)

    return np.array(r_array), np.array(v_array)
'''
def OVRVO(x0_DW):

    r_array = []
    v_array = []
    
    c1 = math.exp(-gamma*dt) # c1 = a in the Crooks article
    if gamma == 0:
        c2 = 1
    else:
        c2 = np.sqrt(2/(gamma*dt)*math.tanh(gamma*dt/2)) #c2 = b (the time rescaling fator) in the Crooks article
    if type_of_potential =='HO':
        x = np.random.normal(loc = 0.0, scale = np.sqrt((1./(m*omega**2))*k_T))
    elif type_of_potential == 'DW':
        x = x0_DW
    v = np.random.normal(loc = 0.0, scale = np.sqrt(k_T/m))
    t = 0.0

    ### Solves the equations for the O-block ###
    def O_block(v):
        v_t_dt = np.sqrt(c1)*v + np.sqrt((1-c1)*k_T/m) * np.random.normal(0,1)

        return v_t_dt

    ### Solves the equations for the V-block ###
    def V_block(x,v,F):
        v_t_dt = v + 0.5*c2*dt*F/m #F is the force

        return v_t_dt

    ### Solves the equations for the R-block ###
    def R_block(x,v):
        x_t_dt = x + c2*dt*v
        
        return x_t_dt

    ### Calculates the force on the particle when in position x ###
    def Force(x):
        if type_of_potential == 'HO':
            F = -m*omega**2*x #Harmonic oscillator
        elif type_of_potential == 'DW':
            F = 4.0*U_0*x*(1.0 - x**2)
            #F = -2*(U_0/(a**2*b**2))*((x-a)*(x-b)**2 + (x-a)**2*(x-b)) #Double well

        return F

    force = Force(x)
    r_array.append(np.array([x, 0.0, 0.0]))
    v_array.append(np.array([v, 0.0, 0.0]))
    for i in range(1,N):
        ### OVRVO ###
        v = O_block(v)
        v = V_block(x,v,force)
        x = R_block(x,v)
        force = Force(x) #Only one force evaluation per timestep
        v = V_block(x,v,force)
        v = O_block(v)

        ### Saving the position and velocity in a .txt file ###
        if i%theta == 0:
            r_array.append(np.array([x, 0.0, 0.0]))
            v_array.append(np.array([v, 0.0, 0.0]))
            #output_traj.write(str(x) + " " + str(v) + " \n")
            
    return np.array(r_array), np.array(v_array)
'''          

def force(r):
  if type_of_potential == 'HO':
    return -m*omega**2*r
  elif type_of_potential == 'DW':
    return 4.0*U_0*r*(1.0 - r**2)

def force_prime(r_n):
  if type_of_potential == 'HO':
    return np.tile(-m*omega**2 * np.eye(3), (len(r_n), 1, 1))
  elif type_of_potential == 'DW':
    new_force_derivatives = np.zeros((len(r), 3, 3))
    new_force_derivatives[:, 0, 0] = 4.0 * U_0 * (1.0 - 3.0 * r[:, 0] ** 2)
    return new_force_derivatives

def force_div(r_n):    
  if type_of_potential == 'HO': 
    return -3*m*omega**2
  elif type_of_potential == 'DW':
    return 4.0*U_0*(1.0 - 3.0*r[:,0]**2) 

def get_dS_vector(pos,vel):
    forces = force(pos)
    grad_forces = force_prime(pos)
    
    a = np.exp(-gamma*dt)
    b = np.sqrt(2/(gamma*dt)*math.tanh(gamma*dt/2))
    dS_n = [beta*m/((1-a)*fict_beta) * (\
        (1+a)/(b*dt)**2 * (2*pos[1:-1] - pos[2:] - pos[:-2])\
        + (1/(2*m))*((1+a)*forces[1:-1]-forces[:-2]-a*forces[2:])\
        + np.einsum('...ij,...j', grad_forces[1:-1], (1/(2*m)) * ((1+a)*pos[1:-1] - a*pos[:-2]-pos[2:]))\
        + np.einsum('...ij,...j', grad_forces[1:-1], ((b*dt)/(2*m))**2 * (1+a) * forces[1:-1])\
        + np.sqrt(a)/(b*dt)*(vel[2:]-vel[:-2])),\
        beta*m/((1-a)*fict_beta) * ((1+a) * vel[1:-1] + np.sqrt(a)/(b*dt) * (pos[:-2]-pos[2:]))]
        
    dS_r_0 = -beta * forces[0]/fict_beta + (beta*m/((1-a)*fict_beta)) * (\
        -(1+a)/(b*dt)**2 *(pos[1]-pos[0])\
        - (1/(2*m))*(a*forces[1]-forces[0]) \
        + np.sqrt(a)/(b*dt) * (vel[1] + vel[0]) \
        - 1/(2*m) * np.dot(grad_forces[0], pos[1] - pos[0])\
        + np.sqrt(a)*b*dt/(2*m) * np.dot(grad_forces[0], vel[0])\
        + (b*dt)**2/(2*m)**2 * np.dot(grad_forces[0], forces[0]))
    
    dS_v_0 = beta*m*vel[0]/fict_beta + beta*m/((1-a)*fict_beta) * np.sqrt(a) * (-(pos[1]-pos[0]) / (b*dt) + b*dt/2 * forces[0]/m + np.sqrt(a)*vel[0])

    dS_r_N = beta*m/((1-a)*fict_beta) * (\
        (1+a)/(b*dt)**2 * (pos[-1]-pos[-2])\
        + (1/(2*m))*(a*forces[-1]-forces[-2]) \
        - np.sqrt(a)/(b*dt) * (vel[-1] + vel[-2]) \
        + (a/(2*m)) * np.dot(grad_forces[-1], pos[-1] - pos[-2])\
        + (b*dt)**2/(2*m)**2 * a * np.dot(grad_forces[-1], forces[-1])\
        - np.sqrt(a)*b*dt/(2*m)* np.dot(grad_forces[-1], vel[-1]))
        
    
    dS_v_N = beta*m/((1-a)*fict_beta) * (-np.sqrt(a) * ((pos[-1]-pos[-2]) / (b*dt) + b*dt/2 * forces[-1]/m) + vel[-1])
    
    dS_r_array = np.concatenate(([dS_r_0], dS_n[0], [dS_r_N]))
    dS_v_array = np.concatenate(([dS_v_0], dS_n[1], [dS_v_N]))
    
    return np.array([dS_r_array, dS_v_array])

def get_S(pos,vel):
    a = np.exp(-gamma*dt)
    b = np.sqrt(2/(gamma*dt)*math.tanh(gamma*dt/2))
    #pot_0 = U_0*(1- pos[0][0]**2)**2
    pot_0 = 0.5*m*omega**2*pos[0,:]**2

    first_part = beta*(0.5*m*np.sum(vel[0]**2) + pot_0) + N*np.log(2*np.pi*(1-a)*b*dt/(m*beta))
    sum_part = np.sum(beta*m/(2*(1-a)) * \
    ((pos[1:]-pos[:-1])/(b*dt) - b*dt/2*force(pos[:-1])/m - np.sqrt(a)*vel[:-1])**2\
    + (np.sqrt(a)* ( (pos[1:]-pos[:-1])/(b*dt) + b*dt/2*force(pos[1:])/m) - vel[1:])**2)
    return (first_part + sum_part)/fict_beta


def write_to_file(name_of_variable, variable, path_number, writer, lagrange_multipliers=[], rattle_multipliers=[]):
    variable_to_output = comm.gather(variable, root=0)
    if name_of_variable == 'config':
        lagrange_multipliers_to_output = comm.gather(lagrange_multipliers, root=0)
        rattle_multipliers_to_output = comm.gather(rattle_multipliers, root=0)
    if(rank == 0):
        if name_of_variable == 'config':
            variable_to_output = np.concatenate(variable_to_output)
            lagrange_multipliers_to_output = np.concatenate(lagrange_multipliers_to_output)
            rattle_multipliers_to_output = np.concatenate(rattle_multipliers_to_output)
            index_pt = 0
            for part in variable_to_output:
                if(path_number % 10000 == 0): writer_traj.writerows(np.c_[[path_number]*len(part.pos_array), [part.idp]*len(part.pos_array), [l*dt for l in range(len(part.pos_array))], part.pos_array, part.vel_array])
                observable = -part.dS_array[0,-1]
                obs_running_avg = part.obs_running_avg
                writer_obs.writerow([path_number, part.idp, part.lambda_A, part.lambda_B, part.lambda_Bprime, observable[0], observable[1], observable[2], obs_running_avg[0], obs_running_avg[1], obs_running_avg[2]])
                
                #print(f"len(lagrange_multipliers): {len(lagrange_multipliers)")
                #index_pt = np.where(variable_to_output==part)
                #writer_multipliers.writerow([path_number, part.idp, lagrange_multipliers_to_output[index_pt][0,0], lagrange_multipliers_to_output[index_pt][0,1], rattle_multipliers_to_output[index_pt][0,0], rattle_multipliers_to_output[index_pt][0,1]])
                index_pt += 1
        elif name_of_variable == 'Temp':
            variable_to_output = np.sum(variable_to_output, axis=0) 
            writer_T.writerow([path_number, variable_to_output[0], variable_to_output[1], variable_to_output[2]])
        elif name_of_variable == 'Ham':
            writer_H.writerow([path_number, np.sum(variable_to_output)])    

def BAOAB(particle_array, lagrange_multipliers, rattle_multipliers, grad_constraints_matrix, path_number):
    for part in particle_array:
        ### B-block ###
        #pos, vel = pos, vel
        index_pt = particle_array.index(part)
        part.Pi_array = part.Pi_array - part.dS_array*(delta_t/2)

        #grad_constraint_matrix_previous_step = get_grad_constraint_matrix_previous_step(part)
        
        ### A-block ###
        part.pos_array = part.pos_array + part.Pi_array[0]*(delta_t/(2*M))
        part.vel_array = part.vel_array + part.Pi_array[1]*(delta_t/(2*M))

        ### O-block ###
        #pos, vel = pos, vel
        random_vector = np.random.multivariate_normal(mean = (0.0, 0.0, 0.0), cov = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], size=(len(part.Pi_array), len(part.Pi_array[0])))
        part.Pi_array = part.Pi_array*math.exp(-fict_gamma*delta_t)+math.sqrt(fict_k_T * M * (1-math.exp(-2*fict_gamma*delta_t)))*random_vector  #*np.random.normal(0,1)
        
        
        ### A-block ###
        part.pos_array = part.pos_array + part.Pi_array[0]*(delta_t/(2*M))
        part.vel_array = part.vel_array + part.Pi_array[1]*(delta_t/(2*M))
        
        
        if(use_constraints=='Y' and use_shake=='Y'): shake(part, lagrange_multipliers[index_pt], grad_constraints_matrix)#[index_pt])

        part.dS_array = get_dS_vector(part.pos_array, part.vel_array) #Need to calculate the new force, so this one is needed. However this will be equal to the first dS in the next BAOAB step
        part.obs_running_avg = part.obs_running_avg + (-part.dS_array[0,-1] - part.obs_running_avg)/(path_number+2)
        ### B-block ###
        #pos, vel = pos, vel
        part.Pi_array = part.Pi_array - part.dS_array*(delta_t/2)
        
        if(use_constraints=='Y' and use_shake=='Y' and use_rattle=='Y'): rattle(part, rattle_multipliers[index_pt], grad_constraints_matrix)#[index_pt])
        
        #constraint
        '''part.pos_array[0] = np.array([-1.0, 0.0, 0.0])
        part.pos_array[len(part.pos_array)-1] = np.array([1.0, 0.0, 0.0])
        part.vel_array[0] = np.array([0.0, 0.0, 0.0])
        part.vel_array[len(part.pos_array)-1] = np.array([0.0, 0.0, 0.0])
        
        part.Pi_array[0,0] = np.array([0.0, 0.0, 0.0])
        part.Pi_array[1,0] = np.array([0.0, 0.0, 0.0])
        
        part.Pi_array[0, len(part.pos_array)-1] = np.array([0.0, 0.0, 0.0])
        part.Pi_array[1, len(part.pos_array)-1] = np.array([0.0, 0.0, 0.0])'''

    #return pos, vel, Pi, dS

def constraint_vector(particle): #must be changed with every chaange of constraints
	sigma_vector_R = np.array([particle.pos_array[0,0] - particle.lambda_A, particle.pos_array[-1,0] - particle.lambda_Bprime])
	sigma_vector_V = np.array([0.0, 0.0])#np.array([particle.vel_array[0,0] - 0.0, particle.vel_array[-1,0] - 0.0])
	sigma_vector = np.array([sigma_vector_R, sigma_vector_V])
	#np.array([[particle_array[0].pos_array[0] - np.array([-1.0, 0.0, 0.0]), particle_array[0].vel_array[0] - np.array([0.0, 0.0, 0.0])], [particle_array[0].pos_array[-1] - np.array([1.0, 0.0, 0.0]), particle_array[0].vel_array[-1] - np.array([0.0, 0.0, 0.0])]])
	return sigma_vector

def grad_constraint_vectors(particle):
	matrix_R = np.array([[[0.0, 0.0, 0.0]]*len(particle.pos_array), [[0.0, 0.0, 0.0]]*len(particle.pos_array)])
	matrix_V = np.array([[[0.0, 0.0, 0.0]]*len(particle.vel_array), [[0.0, 0.0, 0.0]]*len(particle.vel_array)])
	
	matrix_R[0,0] = np.array([1.0, 0.0, 0.0])
	matrix_R[-1,-1] = np.array([1.0, 0.0, 0.0])
	
	#matrix_V[0,0] = np.array([1.0, 0.0, 0.0])
	#matrix_V[-1,-1] = np.array([1.0, 0.0, 0.0])
	
	return np.array([matrix_R, matrix_V])

	
def shake(particle, multipliers, grad_constraints_matrix):
	#print(f"SHAKING::constraint_tolerance: {constraint_tolerance}")
	vector_of_constraints = constraint_vector(particle)
	multipliers[0] = -vector_of_constraints[0]
	particle.pos_array[0,0] += multipliers[0,0] 
	particle.pos_array[-1,0] += multipliers[0,1]
	particle.Pi_array[0][0,0] += multipliers[0,0]/delta_t
	particle.Pi_array[0][-1,0] += multipliers[0,1]/delta_t
	
	'''grad_constraint_matrix_0 = np.copy(grad_constraints_matrix) #grad_constraint_matrix_previous_step
	#print(f"grad_constraint_matrix_0: {grad_constraint_matrix_0}")
	grad_constraint_matrix_R0 = grad_constraint_matrix_0[0].reshape(len(multipliers[0]), len(particle.pos_array)*3)
	grad_constraint_matrix_R0_transposed = np.transpose(grad_constraint_matrix_R0)
	
	#grad_constraint_matrix_V0 = grad_constraint_matrix_0[1].reshape(len(multipliers[1]), len(particle.vel_array)*3)
	#grad_constraint_matrix_V0_transposed = np.transpose(grad_constraint_matrix_V0)
	
	stop_iteration =  False
	multipliers_increment = np.copy(multipliers)
	while(stop_iteration == False):
		
		particle.pos_array += np.dot(grad_constraint_matrix_R0_transposed, multipliers_increment[0]).reshape(len(particle.pos_array), len(particle.pos_array[0]))
		#particle.vel_array += np.dot(grad_constraint_matrix_V0_transposed, multipliers_increment[1]).reshape(len(particle.vel_array), len(particle.vel_array[0]))
		
		
		grad_constraint_matrix_current = grad_constraint_vectors(particle)
		grad_constraint_matrix_R_current = grad_constraint_matrix_current[0].reshape(len(multipliers[0]), len(particle.pos_array)*3)
		#grad_constraint_matrix_V_current = grad_constraint_matrix_current[1].reshape(len(multipliers[1]), len(particle.vel_array)*3)
		vector_of_constraints = constraint_vector(particle)
		
		
		A_matrix_X = np.dot(grad_constraint_matrix_R_current, grad_constraint_matrix_R0_transposed)
		#A_matrix_V = np.dot(grad_constraint_matrix_V_current, grad_constraint_matrix_V0_transposed)
		
		
		multipliers_increment[0] = -np.dot(np.linalg.inv(A_matrix_X), vector_of_constraints[0])
		
		#multipliers_increment[1] = -np.dot(np.linalg.inv(A_matrix_V), vector_of_constraints[1])
		
		if(np.max(np.absolute(multipliers_increment)) < constraint_tolerance): 
			stop_iteration = True
			grad_constraints_matrix = grad_constraint_matrix_current
		
		multipliers += multipliers_increment
		
	particle.Pi_array[0] += np.dot(grad_constraint_matrix_R0_transposed, multipliers[0]).reshape(len(particle.pos_array), len(particle.pos_array[0]))/delta_t'''


def rattle(particle, multipliers, grad_constraints_matrix):
	#print(f"RATTLING::constraint_tolerance: {constraint_tolerance}")
	multipliers[0,0] = -particle.Pi_array[0][0,0]
	multipliers[0,1] = -particle.Pi_array[0][-1,0]
	particle.Pi_array[0][0,0] += multipliers[0,0]
	particle.Pi_array[0][-1,0] += multipliers[0,1]
	
	'''grad_constraint_matrix_0 = np.copy(grad_constraints_matrix) #grad_constraint_matrix_previous_step
	#print(f"grad_constraint_matrix_0: {grad_constraint_matrix_0}")
	grad_constraint_matrix_R0 = grad_constraint_matrix_0[0].reshape(len(multipliers[0]), len(particle.pos_array)*3)
	grad_constraint_matrix_R0_transposed = np.transpose(grad_constraint_matrix_R0)
	
	#grad_constraint_matrix_V0 = grad_constraint_matrix_0[1].reshape(len(multipliers[1]), len(particle.vel_array)*3)
	#grad_constraint_matrix_V0_transposed = np.transpose(grad_constraint_matrix_V0)
	
	A_matrix_X = np.dot(grad_constraint_matrix_R0, grad_constraint_matrix_R0_transposed)
	#A_matrix_V = np.dot(grad_constraint_matrix_V0, grad_constraint_matrix_V0_transposed)
	
	inv_A_dot_grad_sigma_R0 = np.dot(np.linalg.inv(A_matrix_X), grad_constraint_matrix_R0)
	#inv_A_dot_grad_sigma_V0 = np.dot(np.linalg.inv(A_matrix_V), grad_constraint_matrix_V0)
	
	multipliers[0] = -np.dot(inv_A_dot_grad_sigma_R0, particle.Pi_array[0].reshape(len(particle.pos_array)*3, 1)).reshape(1, len(multipliers[0]))
	#multipliers[1] = -np.dot(inv_A_dot_grad_sigma_V0, particle.Pi_array[1].reshape(len(particle.vel_array)*3, 1)).reshape(1, len(multipliers[1]))
	
	
	particle.Pi_array[0] += np.dot(grad_constraint_matrix_R0_transposed, multipliers[0]).reshape(len(particle.pos_array), len(particle.pos_array[0]))
	#particle.Pi_array[1] += np.dot(grad_constraint_matrix_R0_transposed, multipliers[1]).reshape(len(particle.vel_array), len(particle.vel_array[0]))'''
	
	'''stop_iteration =  False
	multipliers_increment = np.copy(multipliers)
	while(stop_iteration == False):
		
		particle.Pi_array[0] += np.dot(grad_constraint_matrix_R0_transposed, multipliers_increment[0]).reshape(len(particle.pos_array), len(particle.pos_array[0]))
		particle.Pi_array[1] += np.dot(grad_constraint_matrix_V0_transposed, multipliers_increment[1]).reshape(len(particle.vel_array), len(particle.vel_array[0]))
		
		
		grad_constraint_matrix_current = grad_constraint_vectors(particle)
		grad_constraint_matrix_R_current = grad_constraint_matrix_current[0].reshape(len(multipliers[0]), len(particle.pos_array)*3)
		grad_constraint_matrix_V_current = grad_constraint_matrix_current[1].reshape(len(multipliers[1]), len(particle.vel_array)*3)
		vector_of_constraints = constraint_vector(particle)
		
		
		A_matrix_X = np.dot(grad_constraint_matrix_R_current, grad_constraint_matrix_R0_transposed)
		A_matrix_V = np.dot(grad_constraint_matrix_V_current, grad_constraint_matrix_V0_transposed)
		
		
		multipliers_increment[0] = -np.dot(np.linalg.inv(A_matrix_X), vector_of_constraints[0])
		
		multipliers_increment[1] = -np.dot(np.linalg.inv(A_matrix_V), vector_of_constraints[1])
		
		if(np.max(np.absolute(multipliers_increment)) < tolerance): 
			stop_iteration = True
			grad_constraints_matrix = grad_constraint_matrix_current
		
		multipliers += multipliers_increment'''
		
	#particle.Pi_array[0] += np.dot(grad_constraint_matrix_R0_transposed, multipliers[0]).reshape(len(particle.pos_array), len(particle.pos_array[0]))/delta_t
	#particle.Pi_array[1] += np.dot(grad_constraint_matrix_V0_transposed, multipliers[1]).reshape(len(particle.vel_array), len(particle.vel_array[0]))/delta_t	

def path_sampling(particle_array, lagrange_multiliers, rattle_multipliers, grad_constraints_matrix, writer_traj, writer_H, writer_T):
    # beta = get_Temp(vel)**-1
    #dS = get_dS_vector(pos,vel)
    #Temp_real, Temp_fictitious, T_real_config, T_real_config_check = get_Temp(particle_array)
    Temp_real, Temp_fictitious, T_real_config = get_Temp(particle_array)
    Ham = get_Ham(particle_array)
    #observable_array = get_observable(particle_array)

    for k in range(number_of_paths):
        if save_config_freq != 0:
            if k %save_config_freq == 0:
                write_to_file('config', particle_array, k, writer_traj, lagrange_multiliers, rattle_multipliers)
                print(f"saving config for path # {k}")
        if save_Temp_freq != 0:
            if k %save_Temp_freq == 0:
                #write_to_file('Temp', [Temp_real, Temp_fictitious, T_real_config, T_real_config_check], k, writer_T)
                write_to_file('Temp', [Temp_real, Temp_fictitious, T_real_config], k, writer_T)
        if save_Ham_freq != 0:
            if k %save_Ham_freq == 0:
                write_to_file('Ham', Ham, k, writer_H)
        '''if save_Obs_freq != 0:
            if k %save_Obs_freq == 0:
                write_to_file('Obs', observable_array, k, writer_obs)'''
                #print(f"saving config for path # {k}")
        #if((k+1)% 10 == 0 and tolerance > 10**-4): tolerance *= 0.5 
        
        BAOAB(particle_array, lagrange_multiliers, rattle_multipliers, grad_constraints_matrix, k)
        #print(f"lagrange_multiliers: {lagrange_multiliers}")
        #Temp_real, Temp_fictitious, T_real_config, T_real_config_check = get_Temp(particle_array)
        Temp_real, Temp_fictitious, T_real_config = get_Temp(particle_array)
        Ham = get_Ham(particle_array)

        if np.any(Ham == np.inf) or np.any(Temp_real == np.inf) or np.any(Temp_fictitious == np.inf):
          print('The path exploded')
          break

        #if Ham == np.inf or Temp_real == np.inf or Temp_fictitious == np.inf:
        #    print('The path exploded')
        #    break

def get_Temp(particle_array): #change for 3 d
    T_real = 0.0
    T_fict = 0.0
    T_real_config = 0.0
    #T_real_config_check = 0.0
    for part in particle_array:
    	T_real += (1./3.)*np.sum(m*part.vel_array**2)/N
    	T_fict += (1./3.)*np.sum(part.Pi_array**2/M)/(2*N)
    	T_real_config -= np.sum(force(part.pos_array[1:-1])**2)/np.sum(force_div(part.pos_array[1:-1]))
    	#T_real_config_check += np.sum(m * omega**2 * part.pos_array**2)/N
    	
    	#print(T_real, T_fict, len(part.vel_array), len(part.Pi_array))
    
    return T_real/len(particle_array), T_fict/len(particle_array), T_real_config/len(particle_array) #, T_real_config_check/len(particle_array)

def get_Ham(particle_array):
    H = 0.0
    for part in particle_array:
        H += np.sum(part.Pi_array**2)/(2*M) + get_S(part.pos_array, part.vel_array)
    return H
    
def get_observable(particle_array):
    observable_array = []
    for part in particle_array:
        observable_array.append(-part.dS_array[0,-1])
    return observable_array

if __name__ == "__main__":
    global dt, k_T, N, theta, gamma, type_of_potential, omega, U_0, a, b, do_path_dynamics, number_of_paths, delta_t, M, m, thermostat, fict_gamma, save_config_freq, save_Temp_freq, save_pos_freq, save_vel_freq, save_Ham_freq, save_Obs_freq, beta, fict_beta, fict_k_T, x0_HO, init_path_from_points_from_file, init_points_file, generate_new_traj, writer_traj, writer_H, writer_T, writer_obs
    
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    #start = time.time()
    
    input_file = open(path + 'input.inp','r')
    print("reading the input file")
    ### Reads parameters from input_file ###
    for line in input_file:
        line = line.strip().split('=')
        if line[0].strip() == 'generate_new_path':
            generate_new_traj = line[1].strip()
        elif line[0].strip() == 'work_name':
            work_name = line[1].split()[0]
        elif line[0].strip() == 'input_work_name':
            input_work_name = line[1].strip().split()[0]
            input_file.close()
            break

    if generate_new_traj == 'Y':
        #constant_file = open(path + 'input_test.inp','r')
        constant_file = open(path + 'input.inp','r')
    
    elif generate_new_traj == 'N':
        traj_file = input_work_name + '_initial_traj.txt'
        constant_file = open(path + work_name + '_constants.txt','r')
    
    for line in constant_file:
        ### OVRVO parameters ###
        line = line.strip().split('=')
        if line[0].strip() == 'dt':
            dt = float(line[1].split()[0])
        elif line[0].strip() == 'k_T':
            k_T = float(line[1].split()[0])
        elif line[0].strip() == 'N':
            N = int(line[1].split()[0])
        elif line[0].strip() == 'theta':
            theta = int(line[1].split()[0])
        elif line[0].strip() == 'gamma':
            gamma = float(line[1].split()[0])
        elif line[0].strip() == 'm':
            m = float(line[1].split()[0])

        ### Parameters for potential ###
        elif line[0].strip() == 'type_of_potential':
            type_of_potential = line[1].split()[0]
        elif line[0].strip() == 'omega' and type_of_potential == 'HO':
            omega = float(line[1].split()[0])
        elif line[0].strip() == 'x0_HO' and type_of_potential == 'HO':
            x0_HO = float(line[1].split()[0])
        elif line[0].strip() == 'U_0' and type_of_potential == 'DW':
            U_0 = float(line[1].split()[0])
        elif line[0].strip() == 'i' and type_of_potential == 'DW':
            i = float(line[1].split()[0])
        elif line[0].strip() == 'j' and type_of_potential == 'DW':
            j = float(line[1].split()[0])
    constant_file.close()
    
    print(f"read_input_file:: dt: {dt}")
    print(f"read_input_file:: rank: {rank}, k_T: {k_T}")

    #input_file = open(path + 'input_test.inp','r')
    input_file = open(path + 'input.inp','r')
    for line in input_file:
        line = line.strip().split('=')
        if line[0].strip() == 'do_path_dynamics':
            do_path_dynamics = line[1].split()[0]
            break
    ### Parameters for path dynamics
    if do_path_dynamics == 'Y':
        print()
        #input_file = open(path + 'input_test.inp','r')
        input_file = open(path + 'input.inp','r')
        for line in input_file:
            line = line.strip().split('=')
            if line[0].strip() == 'number_of_paths':
                number_of_paths = int(line[1].split()[0])
            elif line[0].strip() == 'delta_t':
                delta_t = float(line[1].split()[0])
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
            '''elif line[0].strip() == 'save_pos':
                save_pos_freq = int(line[1].split()[0])
            elif line[0].strip() == 'save_vel':
                save_vel_freq = int(line[1].split()[0])'''

    if generate_new_traj == 'Y':
        constants = open(path + work_name + '_constants.txt','w')
        constants.write("dt = " + str(dt) + '\n')
        constants.write('k_T = ' + str(k_T)+ '\n')
        constants.write('N = ' + str(N)+ '\n')
        constants.write('theta = ' + str(theta)+ '\n')
        constants.write('gamma = ' + str(gamma)+ '\n')
        constants.write('m = ' + str(m)+ '\n')
        if type_of_potential == 'HO':
            constants.write('type_of_potential = HO'+ '\n')
            constants.write('omega = ' + str(omega)+ '\n')
        if type_of_potential == 'DW':
            constants.write('type_of_potential = DW'+ '\n')
            constants.write('U_0 = ' + str(U_0)+ '\n')
            #constants.write('i = ' + str(i)+ '\n')
            #constants.write('j = ' + str(j)+ '\n')


#def main():        
    #comm = MPI.COMM_WORLD
    #size = comm.Get_size()
    #rank = comm.Get_rank()
	
    beta = k_T **-1
    fict_beta = fict_k_T **-1
    #if generate_new_traj == 'Y':
    
    if(rank == 0):
        outfile_traj = open("trajectories.csv", 'w')
        writer_traj = csv.writer(outfile_traj)
        header_traj = ["path_number", "path_id", "real_time", "x", "y", "z", "vx", "vy", "vz"]
        writer_traj.writerow(header_traj)
        #np.savetxt(outfile_traj, ["path_number", "particle_number", "real_time", "x", "v"], delimiter=", ", fmt="%s")

        outfile_H = open("Hamiltonian_for_paths.csv", 'w')
        writer_H = csv.writer(outfile_H)
        header_H = ["path_number", "H"]
        writer_H.writerow(header_H)

        outfile_T = open("temperature_for_paths.csv", 'w')
        writer_T = csv.writer(outfile_T)
        header_T = ["path_number", "T_real", "T_fictitious", "T_real_config"]
        writer_T.writerow(header_T)
        
        outfile_obs = open("observable.csv", 'w')
        writer_obs = csv.writer(outfile_obs)
        header_obs = ["path_number", "path_id", "lambda_A", "lambda_B", "lambda_Bprime", "observable_x", "observable_y", "observable_z", "run_avg_obs_x", "run_avg_obs_y", "run_avg_obs_z"]
        writer_obs.writerow(header_obs)
        
        outfile_multipliers = open("multipliers.csv", 'w')
        writer_multipliers = csv.writer(outfile_multipliers)
        header_multipliers = ["path_number", "path_id", "lambda_x0", "lambda_xN", "mu_x0", "mu_xN"]
        writer_multipliers.writerow(header_multipliers)
    else:
        outfile_traj = None
        outfile_T = None
        outfile_H = None
        outfile_obs = None
        outfile_multipliers = None
        
        writer_traj = None
        writer_T = None
        writer_H = None
        writer_obs = None
        writer_multipliers = None

    #init_traj_name = "init_trajectory.csv"

    if generate_new_traj == 'Y':
        #output_traj = open(path + work_name + '_initial_traj.txt','w')
        #traj_file = work_name + '_initial_traj.txt'
        '''outfile_init_traj = open(init_traj_name, 'w')
        writer_init_traj = csv.writer(outfile_init_traj)
        header_init_traj = ["real_time", "x", "y", "z", "vx", "vy", "vz"]
        writer_init_traj.writerow(header_init_traj)
        print('Creating new initial trajectory')
        if gamma != 0:
            print('Thermostat on for real dynamics')
        elif gamma == 0:
            print('Thermostat off on for real dynamics')
        OVRVO(writer_init_traj)
        print('Done with creating initial path')
        outfile_init_traj.close()'''
        #output_traj.close()
    elif generate_new_traj == 'N':
        if do_path_dynamics == 'N':
            print('You are creating initial path nor doing path dynamics (you are doing nothing). Go check input file')
            quit()
        print('Loading existing trajectory')
        if gamma != 0:
            print('Thermostat on for real dynamics')
        elif gamma == 0:
            print('Thermostat off on for real dynamics')

    if do_path_dynamics == 'Y':
        if gamma == 0:
            print('You are running path dynamics with gamma = 0, which is not possible. Stopping the program')
            quit()
        print('Starting with path dynamcis')
        if thermostat == 'Y':
            print('Thermostat on for path dynamics')
            print(f"fict_gamma = {fict_gamma}")
            print(f"fict_k_T = {fict_k_T}")
        elif thermostat == 'N':
            print('Thermostat off on for path dynamics')
        dt = dt*theta
        #r = np.loadtxt(path + traj_file, usecols = 0)
        #v = np.loadtxt(path + traj_file, usecols = 1)
        #Pi = np.random.normal(loc = 0.0, scale = np.sqrt(M), size=(2,len(r)))
        particle_array = initialize_path_dynamics()
        #particle_array = initialize_path_dynamics(init_traj_name)
        grad_constraints_matrix, lagrange_multipliers, rattle_multipliers = [], [], []
        if(use_constraints == 'Y'): grad_constraints_matrix, lagrange_multipliers, rattle_multipliers = initialize_constraints(particle_array)
        path_sampling(particle_array, lagrange_multipliers, rattle_multipliers, grad_constraints_matrix, outfile_traj, outfile_H, outfile_T)
        print('Done with path dynamics')
    
        if rank == 0:
            outfile_final_traj = open("final_trajectory.csv", 'w')
            writer_final_traj = csv.writer(outfile_final_traj)
            header_final_traj = ["path_number", "path_id", "real_time", "x", "y", "z", "vx", "vy", "vz", "Pi_x", "Pi_y", "Pi_z", "Pi_vx", "Pi_vy", "Pi_vz"]
            writer_final_traj.writerow(header_final_traj)
            for part in particle_array:
                writer_final_traj.writerows(np.c_[[number_of_paths-1]*len(part.pos_array), [part.idp]*len(part.pos_array), [l*dt for l in range(len(part.pos_array))], part.pos_array, part.vel_array, part.Pi_array[0], part.Pi_array[1]])
            outfile_final_traj.close()
            
            outfile_traj.close()
            outfile_H.close()
            outfile_T.close()
    elif do_path_dynamics == 'N':
        print('Not doing path dynamics as requested in the input file')
    #end = time.time()
    #print(f"total time: {end - start}")
