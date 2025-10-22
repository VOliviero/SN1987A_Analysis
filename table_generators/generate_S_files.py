import sys,os
sys.path.append(os.path.abspath("./"))
from sn1987a import IBD as IBD
from sn1987a import vflux as vflux
from sn1987a import detectors as detectors
from sn1987a import expdata
from sn1987a import reader_Aeff
import scipy.integrate as spi
import numpy as np
import os


geo_const = vflux.geometric_const

def cal_S_integrand(Ev,phase,Aeff_function,T):
    """
    cal S (caligraphic S) is a function appearing in the expected signal rate as a function of time.
    cal S is a function of Temperature. For cooling, the temperature is a function of time too.
    """
    Aeff        = Aeff_function(Ev)
    g           = vflux.g[phase](Ev,T)

    return Aeff*g

def integrate_cal_S(phase,Aeff_function,T, E_lower, E_upper):
    """
    Evaluates the integral of the integrand function cal_S
    """
    # Integration limits
    Ev_min = E_lower  # Mev
    Ev_max = E_upper#70    # Mev
    integral_value,error_value = spi.quad(cal_S_integrand,
                                          Ev_min,
                                          Ev_max,
                                          args=(phase,Aeff_function,T),
                                          epsabs=1e-100,
                                          epsrel=1e-6,
                                          limit=1000)
    
    return integral_value,error_value
print( IBD.IBD_threshold+0.004)

def run(folder_path = "./tables/cal_s_tables/",experiment_list=("k2","im","bk", "ls"),test_mode=False):
    
    # Check if the directory exists
    if not os.path.exists(folder_path):
        # If it doesn't exist, create it
        os.makedirs(folder_path)

    
    phases          = ("a","c")

    # Threshold Values that will select the appropriate effective area (Aeff) function
    E_thr_values    = { 
            # Taken from Vissani 2015 J. Phys. G: Nucl. Part. Phys. 42 013001
            "k2" : 4.5, # MeV
            "bk" : 10., # MeV
            "im" : 15,  # MeV
            "ls" : 5.0  # MeV
    }
    E_upper_values =  { 
         # To bedecided
            "k2" : 72, # MeV
            "bk" : 72, # MeV
            "im" : 70 , #MeV
            "ls" : 70   # MeV
    }

    E_lower_values =  { 
         # To bedecided
            "k2" : 3.92+0.004, # MeV
            "bk" : 5+0.004, # MeV
            "im" : 16.3+0.004 , #MeV
            "ls" : IBD.IBD_threshold+0.004  # MeV
    }



    # Define temperature range and sampling. Change this if you wish to have a more fine-tuned integral
    function_temperature_step = 0.01
    temperatures = {
                "a" :   np.arange(  0.6,  6.0+function_temperature_step,    function_temperature_step),     # Accretion
                "c" :   np.arange(  0.1,  10+function_temperature_step,    function_temperature_step)      # Cooling
                }

    # calS
    for experiment in experiment_list:

        # Open Aeff file reader corresponding to the experiment
        # We open the file just once!
        Aeff_reader = reader_Aeff.Aeff_reader(experiment,E_thr=E_thr_values[experiment])
        
        for phase in phases: 
            
            # Select the appropriate threshold
            E_thr = E_thr_values[experiment]
            E_up = E_upper_values[experiment]
            E_low = E_lower_values[experiment]
            # Open a (new) file associated to event
            filename = f"cal_S_{experiment}_{phase}.tsv"
            f_function_file = open(folder_path + filename, "w")

            f_function_file.write(f"cal_S_function with E_thr = {E_thr:02.1f} MeV\n")
            f_function_file.write("T[MeV]\tcal_S\tcal_S_error\n")
            f_function_file.write(f"{0.0}\t{0.0}\t{0.0}\n")
           
        
            
            for T in temperatures[phase]:

                # evaluate cal_S
                cal_S_value,integ_error = integrate_cal_S(phase,Aeff_reader.eval,T,E_low,E_up)

                # write into file
                string_to_file =  f"{T:.3f}\t{cal_S_value:.10e}\t{integ_error:.10e}\n"
                f_function_file.write(string_to_file)

            print(f"File per esperimento {experiment} con phase {phase} finito!")
            f_function_file.close()
            
            if test_mode: 
                # Test mode: if true, the script only evaluates a single f function. Used to play around with settings
                return

if __name__ == "__main__":

    run()

    