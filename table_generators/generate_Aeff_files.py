import sys,os
sys.path.append(os.path.abspath("./"))
from sn1987a import IBD
from sn1987a import vflux
from sn1987a import detectors
from sn1987a import expdata 
import scipy.integrate as spi
import numpy as np
import os


E_thr_values  = {
    "k2" : np.arange(2.5,9,0.5),
    "bk" : np.arange(8.0,11.0,1),
    "im" : np.arange(14.5,16.0,0.5),
    "ls" : np.arange(5.0, 6.5, 0.5)
}


def run(folder_path = "./tables/Aeff_tables/",experiment_list=("k2","im","bk", "ls"),test_mode=False):
    
    # Check if the directory exists
    if not os.path.exists(folder_path):
        # If it doesn't exist, create it
        os.makedirs(folder_path)

    
    Ev_values   = np.arange(IBD.IBD_threshold+0.004,120,0.01)

    # f is evaluated for each experiment
    for experiment in experiment_list:
        for E_thr in E_thr_values[experiment]: 
            # Open a (new) file associated to event
            filename = f"Aeff_{experiment}_thr{E_thr:02.1f}.tsv"
            f_function_file = open(folder_path + filename, "w")
            f_function_file.write("Ev[MeV]\tAeff[MeV-2]\n")
            
            for Ev in Ev_values:

                # evaluate Aeff
                Aeff_value = detectors.Aeff(experiment,Ev,E_thr)

                # write into file
                string_to_file =  f"{Ev:.3f}\t{Aeff_value:e}\n"
                f_function_file.write(string_to_file)

            print(f"File per esperimento {experiment} con Ethr = {E_thr} finito!")
            f_function_file.close()
            
            if test_mode: 
                # Test mode: if true, the script only evaluates a single f function. Used to play around with settings
                return

if __name__ == "__main__":

    run()

    
    