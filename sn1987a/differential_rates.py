import os, sys; 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sn1987a.IBD as IBD
import sn1987a.vflux as vflux
import sn1987a.detectors as detectors
from sn1987a import constants as c

def dN_dtdEedc_a(experiment, Ei, cosine, Ta, csi_n):
    Ev      = IBD.Ev(Ei,cosine)
    flux    = vflux.flux_a(Ev, Ta, csi_n)
    Np      = detectors.Np[experiment]
    dsigma  = IBD.dsigma(Ev, Ei) * c.MeV2_to_cm2
    J       = IBD.jacobian2(Ev, Ei)
    return flux*Np*dsigma*J

def dN_dtdEedc_c(experiment, Ei, cosine, Tc,  Rns):
    Ev      = IBD.Ev(Ei,cosine)
    flux    = vflux.flux_c(Ev, Tc, Rns)
    Np      = detectors.Np[experiment]
    dsigma  = IBD.dsigma(Ev, Ei) * c.MeV2_to_cm2
    J       =  IBD.jacobian2(Ev, Ei)
    return flux*Np*dsigma*J

efficiency  =   detectors.eta['k2'](20,0.95106) *detectors.res_kernel_k2()

def dN_tot(experiment, Ei, cosine, Ta, csi_n, Tc, Rns):
    return dN_dtdEedc_a(experiment, Ei, cosine, Ta, csi_n) + dN_dtdEedc_c(experiment, Ei, cosine, Tc,  Rns)
T0 = 4.6
csi_0 = 0.018
tau_a = 0.52
tau_c = 5.6
t_start = 0.035
time_absolute = t_start
t_max = 0.1
Rns = 17.0e5
Ta = 0.6 * T0
Tc          =   T0    * vflux.f_cal_c(t=time_absolute, t0=t_max, tau_c=tau_c)**(1/4)
csi_n       =   csi_0 * vflux.f_cal_a(t=time_absolute, t0=t_max, tau_a=tau_a)
print(dN_tot('k2', 20, 0.95106, Ta, csi_n, Tc, Rns))