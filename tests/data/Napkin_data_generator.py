seed = 1
num_samples = 1000
import numpy as np
import pandas as pd
#Generate observational data for napkin
def generate_obs_data_for_napkin(seed = seed, num_samples = num_samples):
  np.random.seed(seed)
  #U1 is the latent variable that is a common cause of W and X
  U1 = np.random.normal(loc = 3, scale = 1, size = num_samples)
  #U2 is the latent variable that is a common cause of W and Y
  U2 = np.random.normal(loc = 5, scale = 1, size = num_samples)
  W = np.random.gamma(shape = 1/(1*(U1 * 0.3 + 0.5 * U2)**2), scale = 5*(U1 * 0.3 + 0.5 * U2), size = num_samples)
  R = np.random.normal(loc = W * 0.7, scale = 6, size = num_samples)
  X = np.random.binomial(n = 1, p = 1/(1+np.exp(-2 - 0.23 * U1 - 0.1 * R)), size = num_samples)
  Y = np.random.normal(loc = U2 * 0.5 + X * 3, scale = 6)
  obs_data = pd.DataFrame({'W': W, 'R': R, 'X': X, 'Y':Y})
  return obs_data

#Generate interventional data for napkin
def generate_intv_data_for_napkin(seed = seed, num_samples = num_samples, treatment_assignment = 1):
  np.random.seed(seed)
  #U1 is the latent variable that is a common cause of W and X
  U1 = np.random.normal(loc = 3, scale = 1, size = num_samples)
  #U2 is the latent variable that is a common cause of W and Y
  U2 = np.random.normal(loc = 5, scale = 1, size = num_samples)
  W = np.random.gamma(shape = 1/(1*(U1 * 0.3 + 0.5 * U2)**2), scale = 5*(U1 * 0.3 + 0.5 * U2), size = num_samples)
  R = np.random.normal(loc = W * 0.7, scale = 6, size = num_samples)
  X_intv = np.full(num_samples, treatment_assignment)
  Y_intv = np.random.normal(loc = U2 * 0.5 + X_intv * 3, scale = 6)
  intv_data = pd.DataFrame({'W': W, 'R': R, 'X': X_intv, 'Y':Y_intv})
  return intv_data

#Compute the real ACE value
intv_data_X1 = generate_intv_data_for_napkin(seed = 1, num_samples = 1000, treatment_assignment = 1)
intv_data_X0 = generate_intv_data_for_napkin(seed = 1, num_samples = 1000, treatment_assignment = 0)
real_ace = np.mean(intv_data_X1['Y']) - np.mean(intv_data_X0['Y'])

#Compute the ACE estimated with ananke
ace_anipw = ace_obj_2.compute_effect(obs_data, "anipw")