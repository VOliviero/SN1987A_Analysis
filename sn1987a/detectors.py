"""
This module contains information and functions related to the three experiments which
detected antineutrinos from the SN1987a event.

# Naming Convention
_k2 : Kamiokande
_bk : Baksan
_im : IMB

"""
import os, sys; 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numpy import heaviside
import numpy as np
import sn1987a.IBD as ibd
from sn1987a.constants import mn,me,mp
from scipy.special import erf
import scipy.integrate as sci

# Target Properties
amu=1.66053906660*10**-24 #grammi

mc=12.011*amu
mo=15.999*amu
mh=1.00748*amu 
delta= (mn**2 - mp**2 - me**2)/(2*mp)

ton=1000*1000; #grammi

#Np_bk=200*ton/(mc*9 + mh*20 )*5/32
#Np_k2=2140*ton/(mo+ mh*2 )*1/9
Np_bk =(200*ton/21.297e-23)*20*(1-0.000145) 
print(Np_bk)
#Np_k2=(2140*ton/2.9915e-23)*2*(1-0.000145)
Np_k2=1.4305039689681664e32
#Np_im=(6800*ton/2.9915e-23)*2*(1-0.000145)
Np_im=4.5455e32
Np_ls=0.0838e32

Np = {
    "k2" : Np_k2,
    "bk" : Np_bk,
    "im" : Np_im,
    "ls" : Np_ls
}

names = {
    "k2" : "Kamiokande-II",
    "bk" : "Baksan",
    "im" : "IMB",
    "ls" : "LSD"
}





# Heaviside theta definition
Heaviside = lambda x : heaviside(x,0.5)

#######################################
# Dead Time
#######################################

imb_dead_time_s = 0.035

#######################################
# Efficiency
#######################################

'''def eff_k2(Ee,c=0):
    """Returns the efficiency η of the Kamiokande detector as a function of the positron energy Ee [MeV]"""
    return 0.93 * ( 1 - 0.2/Ee - (2.5/Ee)**2 ) * Heaviside(Ee-2.6)'''

def eff_k2(Ee,c=0):
    ## new efficiency ##
    """Returns the efficiency η of the Kamiokande detector as a function of the positron energy Ee [MeV]"""
    return 0.93 * ( 1 - 0.29/Ee - (2.37/Ee)**2 ) * Heaviside(Ee-2.6)

def eff_im(Ee,c,e_im=15.5):
    """Returns the efficiency η of the IMB detector as a function of the positron energy Ee [MeV] and a threshold value e_im.
    Corrected for the angular bias of IMB (PRD 37, 3361)"""
    scaled_Ee = Ee/e_im - 1
    efficiency = ( 0.379 * scaled_Ee - 6*10**-3 * scaled_Ee**4 + 10**-3 * scaled_Ee**5 ) * Heaviside(Ee-e_im)
    
    #c = (((Ee+delta)*mp)/(Ev*(Ee**2-me**2)**0.5))+(Ee/(Ee**2-me**2)**0.5)-mp/(Ee**2-me**2)**0.5
    angular_bias_correction = (1 + 0.1 * c)
    return efficiency * angular_bias_correction

def eff_bk(Ee,c):
    """Returns the efficiency η of the Baksan detector as a function of the positron energy Ee [MeV]"""
    Ee = Ee + me # Correction for liquid scintillator
    return 1.0

def eff_ls(Ee,c):
    """Returns the efficiency η of the LSD detector as a function of the positron energy Ee [MeV]"""
    
    return 0

η_k2 = eff_k2
η_im = eff_im
η_bk = eff_bk
η_ls = eff_ls

eta = {
    "k2":eff_k2,
    "im":eff_im,
    "bk":eff_bk,
    "ls":eff_ls
}


#######################################
# Resolution and Resolution Kernel
#######################################

def resol_function(Ee,statistical,systematic):
    return statistical * np.sqrt(Ee/10.0) + systematic * (Ee/10.0)

#def resolution_k2(Ee):
    #return resol_function(Ee,1.27,1.0)
## new sigma ##    
def resolution_k2(Ee):
    return resol_function(Ee,1.27,1.0)

def resolution_im(Ee):
    return resol_function(Ee,3.3,0.2)

def resolution_bk(Ee):
    return resol_function(Ee,0.0,2.0)

def resolution_ls(Ee):
    return resol_function(Ee,0.0,1.0)

res = {
    "k2" : resolution_k2,
    "im" : resolution_im,
    "bk" : resolution_bk,
    "ls" : resolution_ls
}

norm_gauss = lambda mu, x, sigma : 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-(x-mu)**2 / ( 2 * sigma**2))

# Response kernel for the measured energy in K2
def res_kernel_k2(Ee, Ei):
    return norm_gauss(Ee,Ei,resolution_k2(Ee))

# Response kernel for the measured energy in IMB
def res_kernel_im(Ee, Ei):
    return norm_gauss(Ee,Ei,resolution_im(Ee))

# Response kernel for the measured energy in BKS
def res_kernel_bk(Ee, Ei):
    Ee = Ee + me # Correction for liquid scintillator
    return norm_gauss(Ee,Ei,resolution_bk(Ee))

res_kernel = {
    "k2" : res_kernel_k2,
    "im" : res_kernel_im,
    "bk" : res_kernel_bk
}

#######################################
# Total Efficiency
#######################################
# Total Efficiency is the integral of eta(Ee)*res_kernel(Ee,Ei,sigma(Ee)) in Ei
# It appears in the calculation of the effective area for each detector
# The integral gives rise to an error function since res_kernel is gaussian in Ei
# REF page 6 in Vissani 2015 J. Phys. G: Nucl. Part. Phys. 42 013001
# REF page 19 in Vissani Symmetry 2021, 13, 1851
#
# ! note: cosine is set to 0 

def tot_eff_k2(Ee,E_thr,c=0):
    return eff_k2(Ee,c) * 0.5 * ( 1 + erf((Ee-E_thr)/(2**0.5 * resolution_k2(Ee))))

def tot_eff_im(Ee,E_thr,Ev):
    c = (((Ee+delta)*mp)/(Ev*(Ee**2-me**2)**0.5))+(Ee/(Ee**2-me**2)**0.5)-mp/(Ee**2-me**2)**0.5
    return eff_im(Ee,c) * 0.5 * ( 1 + erf((Ee-E_thr)/(2**0.5 * resolution_im(Ee))))

def tot_eff_bk(Ee,E_thr,c=0):
    return eff_bk(Ee,c) * 0.5 * ( 1 + erf((Ee-E_thr+me)/(2**0.5 * resolution_bk(Ee+me))))

def tot_eff_ls(Ee, E_thr, c=0):
    csi_ls_up = 0.5*(1+erf((Ee-(E_thr+1.5))/(2**0.5*resolution_ls(Ee))))
    csi_ls_low = 0.5*(1+erf((Ee-E_thr)/(2**0.5*resolution_ls(Ee))))
    return (56/72)*csi_ls_up+(16/72)*csi_ls_low


tot_eff = {
    "k2" : tot_eff_k2,
    "bk" : tot_eff_bk,
    "im" : tot_eff_im,
    "ls" : tot_eff_ls
}
#######################################
# Effective Area
#######################################
# Effective area is the "global cross section" of a successfully measured IBD event in a detector
# It is given by Nprotons * total_efficiency(Ee,E_thr) * dSigmaIBD/dEe(Ev,Ee) integrated in Ee from Ee_min to Ee_max (kinematically)
def Aeff(experiment,Ev,E_thr):
    
    def integrand(Ee,E_thr, Ev):
        return tot_eff[experiment](Ee,E_thr, Ev)*ibd.dsigma_bzz(Ev,Ee)*ibd.MeV2_to_cm2

    Ee_min = ibd.Ee_min(Ev)
    Ee_max = ibd.Ee_max(Ev)
    integral,_ = sci.quad(integrand,Ee_min,Ee_max,args=(E_thr, Ev),epsabs=1e-100,epsrel=1e-10)
    
    return Np[experiment] * integral
'''
def areaefficace_k2(Ee,Ev):
    Delta = mn - mp
    Ev=(Ee+Delta)/(1-Ee/mp)
    J=(1+Ev/mp)**2/(1+Delta/mp) #as defined from Comparative Analisys ...
    return Np_k2*J*eff_k2(Ee)*ibd.dsigma(Ev,Ee)

def areaefficace_bk(Np,Ee):
    Delta = mn - mp
    Ev=(Ee+Delta)/(1-Ee/mp)
    J=(1+Ev/mp)**2/(1+Delta/mp)
    return Np*J*eff_bk(Ee)*ibd.dsigma(Ev,Ee)

def areaefficace_im(Np,Ee):
    Delta = mn - mp
    Ev = (Ee+Delta)/(1-Ee/mp)
    J = (1+Ev/mp)**2/(1+Delta/mp) 
    return Np*J*eff_im(Ee)*ibd.dsigma(Ev,Ee)
    '''

