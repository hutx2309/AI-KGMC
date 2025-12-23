# This script is used to perform Morris sensitivity analysis on the simulated results
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import ast
import random
import pickle
import configparser
from SALib.analyze.morris import analyze

## ========== Read variables from configurations.config and declare other variables =============
config = configparser.ConfigParser()
config.read('config.fig')
GSA_parameters = config['Calibrated_parameters'].get('maiz311')
GSA_result_dir =  config['general'].get('GSA_result_dir')
extracted_output_data_name = config['output'].get('extracted_output_data_name1')
output_file = os.path.join(GSA_result_dir, extracted_output_data_name)
sampled_paras_file_name = config['general'].get('sampled_paras1_file_name')
param_values_file = os.path.join(GSA_result_dir, sampled_paras_file_name) # this file is generated from the step1
samplings = np.load(param_values_file)
sample_size = samplings.shape[0]

params = ast.literal_eval(GSA_parameters)
print(type(params))
# number and Names of parameters that are allowed to vary
vars_si = list(params.keys())
print(vars_si)
params_bound = list(params.values())
n_params = len(vars_si)  # 11 parameters to test

problem = {
    'num_vars': n_params,
    'names': vars_si,
    'bounds': params_bound
}

# load the simulated resulst 
with open(output_file, 'rb') as f:
    sim_results = pickle.load(f)
    
for sample_idx in range(0,6):
    output1 = sim_results[sample_idx]['ECO_GPP']
    #output1 = sim_results[sample_idx]['ECO_RA']+sim_results[sample_idx]['ECO_RH']-sim_results[sample_idx]['ECO_GPP']
    output2 = np.transpose (output1)
    print(output2.shape)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(output2, label=f'Simulation gpp', linestyle='--')
    plt.savefig('GSA_corn_gpp'+str(sample_idx)+'.png', dpi=300)


def morris_analysis():
    outputs = []
    for sample_idx in range(sample_size):
     #   output = np.nansum(sim_results[sample_idx]['ECO_RH']+sim_results[sample_idx]['ECO_RA']-sim_results[sample_idx]['ECO_GPP'])
  #      if sim_results[sample_idx] is None:
  #          print(f"Warning: sim_results[{sample_idx}] is None")
   #     else:
        output = np.nanmean(sim_results[sample_idx]['ECO_GPP'])
        outputs.append(output)
    outputs = np.array(outputs)
    print('outputs',outputs)
    results = analyze(problem, samplings, outputs, num_levels=6, print_to_console=True)
    
    mu = results['mu']  # Mean elementary effect for each parameter
    mu_star = results['mu_star']  # Absolute mean elementary effect for each parameter
    sigma = results['sigma']  # Standard deviation of elementary effects
    names = problem['names']
    mu_star_conf = results['mu_star_conf']  # Confidence intervals for the absolute mean
    # Print results

    # Result Plot
    bar_width = 0.35
    index = np.arange(n_params)
    fig, ax = plt.subplots(figsize=(12, 6))
    # Plotting 'mu_start'
    bars1 = ax.bar(index, mu_star, bar_width, label=r'$\mu^*$ (Mean Absolute Elementary Effect)')
    ax.errorbar(index, mu_star, yerr=mu_star_conf, fmt='none', ecolor='black', capsize=5)
    # Plotting sigma
    bars2 = ax.bar(index + bar_width, sigma, bar_width, label=r'$\sigma$ (Standard Deviation of Elementary Effects)')

    # Add labels and title
    ax.set_xlabel('Parameters')
    ax.set_ylabel(r'Morris Method Sensitivity Indices: $\mu^*$ and $\sigma$')
    ax.set_title('Sensitivity of Maize GPP to 35 Parameters')
    ax.set_xticks(index)
    ax.legend(fontsize=12)
    ax.set_xticklabels(vars_si, rotation=45)
    ax.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()
    plt.savefig("GSA_corn_GPP.svg", dpi=300)

morris_analysis()
