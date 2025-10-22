import sys,os
sys.path.append(os.path.abspath("./"))

from sn1987a import IBD 
from sn1987a import vflux 
from sn1987a import detectors
from sn1987a import expdata
from sn1987a import reader_cal_S
import scipy.integrate as spi

import numpy as np
import os

s_reader = reader_cal_S.manager()

def g_cal_integranda(y, x, experiment, T0):
    Tc = T0*pow(vflux.f_cal(y,x,1,2),1/4)
    S_cal = s_reader.eval(Tc, experiment,'c')
    return S_cal

def g_cal_integrata(x, experiment, T0):
    integral_value, err = spi.quad(g_cal_integranda, 0, 30, args=(x, experiment, T0))
    return integral_value, err


def run(folder_path = "./tables/cal_g_tables/", experiment_list=("k2","im","bk"),verbose=0):
    
    # Check if the directory exists
    if not os.path.exists(folder_path):
        # If it doesn't exist, create it
        os.makedirs(folder_path)

    experiment_list=("k2","im","bk")
    value_x=np.arange(0, 2, 0.01)
    value_T0=np.arange(0, 10, 0.1)
    j_max = len(value_x)
    #print("="*20)
    #print("Points to evaluate:",len(value_x)*len(value_T0))
    #print("="*20)
    for experiment in experiment_list:
        
        string_to_file = ""
        filename = f"cal_g_c_{experiment}.tsv"
        print(f"[Start] Experiment {experiment}")

        for i,T0 in enumerate(value_T0):
            for j,x in enumerate(value_x):

                
                #if ((i+1)*j_max + j+1)%500==0:
                    #print((i+1)*j_max + j+1,"punti calcolati.")
                g_cal_value, err_g_cal_value = g_cal_integrata(x, experiment, T0)
                if verbose > 0 :
                    print(f"{T0:e}","\t",f"{x:e}","\t",g_cal_value)
                string_to_file +=  f"{T0:e}\t{x:e}\t{g_cal_value:.10e}\t{err_g_cal_value:.10e}\n"
    
        with open(folder_path+filename, "w") as cal_g_file:
            if verbose > 0 :
                print("T0\tx\tG_c(T0, x)")
            cal_g_file.write(f"T0\tt0/tau\tcal_g_function_a\terr\n")
            cal_g_file.write(string_to_file)

        print(f"[Finish] Experiment {experiment}")


    
if __name__ == "__main__":
    run(verbose=0)        