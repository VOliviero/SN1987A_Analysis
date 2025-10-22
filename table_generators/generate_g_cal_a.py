import sys,os
sys.path.append(os.path.abspath("./"))
from sn1987a import vflux
import scipy.integrate as spi
import numpy as np
import os


def g_cal_integranda(y, x, alpha, n):

    g_cal = vflux.f_cal(y, x, alpha, n)
    return g_cal

def g_cal_integrata(x,alpha, n):
    tmin=0
    tmax=300
   
    g_cal, err_g_cal = spi.quad(g_cal_integranda, tmin, tmax, args=(x, alpha , n), epsabs=1e-60, epsrel=1e-10)
    return g_cal, err_g_cal



def run(folder_path = "./tables/cal_g_tables/"):
    
    # Check if the directory exists
    if not os.path.exists(folder_path):
        # If it doesn't exist, create it
        os.makedirs(folder_path)




    values_step=0.01

    value_x=np.arange(0,20+values_step,values_step)

    filename = f"cal_g_a.tsv"
    f_function_file = open(folder_path+filename, "w")
    f_function_file.write(f"t0/tau\tcal_g_function_a\terr\n")

    print("x\tG_a(x)")
    for x in value_x:

                    
        g_cal_value, err_g_cal_value=g_cal_integrata(x, 2, 2)
        print(f"{x:.3f}","\t",g_cal_value)

        string_to_file =  f"{x:.3f}\t{g_cal_value:.10e}\t{err_g_cal_value:.10e}\n"
        f_function_file.write(string_to_file)

    
if __name__ == "__main__":
    run()