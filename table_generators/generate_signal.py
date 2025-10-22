import sys,os
sys.path.append(os.path.abspath("./"))
from sn1987a import IBD as IBD
from sn1987a import vflux as vflux
from sn1987a import detectors as detectors
from sn1987a import expdata 
from sn1987a import constants as c
import scipy.integrate as spi
import numpy as np
import os


def integrand_signal_numerator(Ee,phase,experiment,E_i,c_i,T):
    """
    Integrand function in the definition of the correction "Signal" functio
    Liquid scintillator (Baksan) efficiency and response correction added in the definition of eta and res_kernel
    Angular Bias for IMB inside the definition of eta
    """
    Ev          = IBD.Ev(Ee,c_i)                                    # Energy of neutrino as function of energy of positron and angle
    g           = vflux.g[phase](Ev,T)                              # Thermal flux (a,c) as a function of neutrino energy and temperature (a,c)
    j           = IBD.jacobian2(Ev,Ee)                               # jacobian for the change of variable of cross section
    xsec        = IBD.dsigma_bzz(Ev,Ee)                                 # cross section of interaction in the detector for IMB process
    eta         = detectors.eta[experiment](Ee,c_i)                 # efficiency of the detector
    res_kernel  = detectors.res_kernel[experiment](Ee, E_i)         # resolution kernel of the detector
    

    return eta*res_kernel*g*xsec*j*detectors.Np[experiment]*vflux.k[phase]*c.MeV2_to_cm2

def integrate_signal_numerator(phase,experiment,E_i,c_i,T):
    """
    Evaluates the integral of the integrand function at the numerator of the signal_function definitions.
    The integrand function is defined in integrand_signal_numerator
    """
    Ee_min = 2.5    # Mev
    Ee_max = 70    # Mev
    integral_value,error_value = spi.quad(integrand_signal_numerator, Ee_min, Ee_max, args=(phase, experiment, E_i, c_i, T), epsabs=1e-100,epsrel=1e-10) # Rimosso limit = 100. Perché c'era?
    
    return integral_value,error_value





def calculate_signal_denominator(phase,experiment,E_i,c_i,T):
    """
    Evaluates the denominator of the signal_function.
    """
    Ev          = IBD.Ev(E_i,c_i)                                   # Energy of neutrino as function of energy of positron and angle
    g           = vflux.g[phase](Ev,T)                              # Thermal flux (a,c) as a function of neutrino energy and temperature (a,c)
    j           = IBD.jacobian2(Ev,E_i)                              # jacobian for the change of variable of cross section
    xsec        = IBD.dsigma_bzz(Ev,E_i)                                # cross section of interaction in the detector for IMB process
    eta         = detectors.eta[experiment](E_i,c_i)                # efficiency of the detector
    
    return eta*g*xsec*j

def run(folder_path = "./tables/Signal_tables/",experiment_list=("k2","im","bk"),test_mode=False):
    
    # Check if the directory exists
    if not os.path.exists(folder_path):
        # If it doesn't exist, create it
        os.makedirs(folder_path)

    # Define data location and phases (accretion, cooling)
    data = expdata.data
    phases = ("a","c")

    # Define temperature range and sampling. Change this if you wish to have a more fine-tuned integral
    function_temperature_step = 0.1
    temperatures = {
                "a" :   np.arange(  0.1,  6.0+function_temperature_step,    function_temperature_step),     # Accretion
                "c" :   np.arange(  0.1,  10.0+function_temperature_step,    function_temperature_step)      # Cooling
                }


    # signal_function is evaluated for each experiment, each phase (accretion and cooling) and each event
    for experiment in experiment_list:
        for id,n_event,time,E_i,theta_i,bkg in data[experiment]:
            for phase in phases:
                
                # Open a (new) file associated to event
                filename = f"signal_func_{experiment}_{phase}_ev{n_event:02d}.tsv"
                signal_function_file = open(folder_path + filename, "w")
                signal_function_file.write("T[MeV]\tsignal_value\tsignal_error\n")
                signal_function_file.write("0\t0\t0\n")
                
                for T in temperatures[phase]:

                    # evaluate signal_function
                    cos                     = lambda theta : np.cos(theta*np.pi/180)
                    signal_numerator,integ_error = integrate_signal_numerator(phase,experiment,E_i,cos(theta_i),T)
                    #signal_numerator = integrate_signal_numerator(phase,experiment,E_i,cos(theta_i),T)

                    signal_value_at_T            = signal_numerator
                    signal_integ_error           = integ_error

                    # Warning if relative error is more than 1 over 1 milion
                    if signal_integ_error/signal_value_at_T > 1e-6:
                        print(f"[Warning] In file {filename} found error of {signal_integ_error:.1e} for T = {T:.3f}")

                    # write into file
                    string_to_file =  f"{T:.3f}\t{signal_value_at_T:.9e}\t{signal_integ_error:.2e}\n"
                    signal_function_file.write(string_to_file)

                print(f"File ({phase}) per evento numero {n_event} di esperimento {experiment} finito!")
                signal_function_file.close()
                
                if test_mode: 
                    # Test mode: if true, the script only evaluates a single signal_function. Used to play around with settings
                    return

if __name__ == "__main__":

    run()
