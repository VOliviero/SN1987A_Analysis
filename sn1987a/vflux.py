# Definizione funzioni richiamate nella parametrizzazione del flusso,
# sia per la fase di accrescimento che per la fase di cooling.
import os, sys; 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.constants as constants
import math
from sn1987a.constants import mp,mn,me,MeV2_to_barn,MeV2_to_cm2,barn_to_cm2,delta,hc,h,c,Ms

np.seterr(over="ignore")
print("Attenzione, gli overflow error di numpy sono ignorati!")

#Nuova parametrizzazione f_cal, x=t0/tau y=t/tau
#Fuction f_Cal
def f_cal(y,x,alpha,n):
    exp_argument = n*(y**alpha-x**alpha)
    exponential = np.exp(exp_argument)
    constant = alpha*x**(alpha+n) * (1/((y+1e-20)**n))
    """
    if exponential / constant > 1e50:
        return ((1+alpha*x**alpha)/(exponential))**(1/n)
    else:"""
    #return np.heaviside(y,0)*((1+alpha*x**alpha)/(exponential+constant))**(1/n)
    
    return ((1+alpha*x**alpha)/(np.exp((n*(y**alpha-x**alpha)))+alpha*x**(alpha+n) * (1/(y**n))))**(1/n)

def f_cal_a(t, t0, tau_a):
    #return  pow(((1+2*pow((t0/tau_a),2))/(np.exp(2*(pow((t/tau_a),2)-pow((t0/tau_a),2)))+2*pow((t0/tau_a),2)*pow((t0/t),2))),0.5)
    return f_cal(t/tau_a,t0/tau_a,alpha=2,n=2)
    

def f_cal_c(t, t0, tau_c):
    #return  pow(((1+(t0/tau_c))/(np.exp(2*((t/tau_c)-(t0/tau_c)))+(t0/tau_c)*(pow((t0/t),2)))),0.5)
    return f_cal(t/tau_c,t0/tau_c,alpha=1,n=2)
    



# Conversion factor from pc to m
parsec_to_metre = 96939420213600000/np.pi

# Distance of sn1987a in meters
sn_distance = 1.5428e23*51.4/50  # cm

# Fattore geometrico nella definizione dei flussi N di antineutrini
geometric_const =  (1/(4*np.pi*pow(sn_distance,2)))

physics_const = (c/pow(h*c,3))
#######################################################################
#Fase di Accrescimento
#######################################################################

#Sezione d'urto calcolata per le interazioni dei positroni nella supernova
def sigmaeplusn(Ev) :
   return (4.8e-44*pow(Ev,2))/(1 + (Ev/260))


#csi_n(t)
"""def csi_n(t):
    return csi_0*f_cal_a(t) """



# Flusso termale dei positroni fase di accrescimento
def g_A(Ev, Ta):
    return pow(((Ev - delta)/(1 - Ev/mn)),2)/(1 + np.exp(((Ev - delta)/(1 - Ev/mn))/Ta))*sigmaeplusn(Ev) 
   
#######################################################################
#Fase di Cooling
#######################################################################



#Tc(t)
"""def T_c(t):
    return T0*pow(f_cal_c(t),1/4)"""

# Flusso termale dei positroni fase di cooling
def g_C(Ev, Tc):  
    
    g_C=pow(Ev,2)/(1 + np.exp(Ev/Tc))
    if np.isnan(g_C):
        g_C=0

    #try:
       # espressione
    #except ValueError:
        # gc=0
    return g_C



g = {
    "a" : g_A,
    "c" : g_C
}

#######################################################################
#flusso totale
#######################################################################

'''#flusso di accrescimento
def flux_a(t, Ev):
    Ta = 0.6*T0
    Nn = Ms/mn*csi_n(t)
    gA = 8*np.pi*g_A(Ev,Ta)
    
    return Nn*gA

#flusso di cooling
def flux_c(t, Ev):
    
    gC = 4*np.pi*g_C(Ev,T_c(t))
    R_ns = np.pi*pow(Rns, 2)
    return gC*R_ns'''

k_a = 8*np.pi*geometric_const *physics_const * Ms/mn

k_c = 4*(np.pi**2)*geometric_const *physics_const

k = {
    "a" : k_a,
    "c" : k_c
}

def flux_a(Ev, Ta, csin): 
    k_a = 8*np.pi*geometric_const *physics_const * Ms/mn
    Nn = csin
    gA =g_A(Ev, Ta)#include già sigmaeplusn
    return Nn*gA*k_a 

def flux_a_pagliaroli(Ev, Ta, csin): 
    k_a = 8*np.pi*geometric_const *physics_const 
    Nn = csin
    gA =g_A(Ev, Ta)#include già sigmaeplusn
    return Nn*gA*k_a 



#flusso di cooling
def flux_c(Ev, Tc, Rns):    #(t, Ev, tau_c=tau_c, T0=T0,  Rns=Rns):
    k_c = 4*(np.pi**2)*geometric_const *physics_const
    gC=g_C(Ev, Tc)
    R_ns = pow(Rns, 2)
    return gC*R_ns*k_c


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    tau_a_list   = np.array([0.1,0.2,0.3])
    tau_c_list   = np.array([7,8,9])
    t_max_list   = np.array([0.1,0.2,0.3])
    time         = np.linspace(-1,10,1000)
    
    fig,axs =plt.subplots(2,1)
    for t_max in t_max_list:
        for tau_a in tau_a_list:
            axs[0].plot(time,f_cal_a(time,t_max,tau_a))
        for tau_c in tau_c_list:
            axs[1].plot(time,f_cal_c(time,t_max,tau_c))
    
    plt.show()

    