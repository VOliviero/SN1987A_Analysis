import os, sys; 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math import log as Log # Log in Mathematica is logarithm in base e
from math import sqrt
import numpy as np
from sn1987a.constants import mp,mn,me,MeV2_to_barn,MeV2_to_cm2,barn_to_cm2,delta,hc,h,c


# Kinematic Variables
IBD_threshold   = ((mn+me)**2-mp**2)/(2*mp) # ~1.806

s               = lambda Ev : 2 * mp * Ev + mp**2
Ev_CM           = lambda Ev : (s(Ev) - mp**2)/(2 * sqrt(s(Ev)))
Ee_CM           = lambda Ev : (s(Ev) - mn**2 + me**2)/(2 * sqrt(s(Ev)))
pe_CM           = lambda Ev : sqrt(s(Ev) - (mn - me)**2) * sqrt(s(Ev) - (mn + me)**2) / (2 * sqrt(s(Ev)))                 
Ee_min          = lambda Ev : Ev - delta - Ev_CM(Ev)/mp * (Ee_CM(Ev) + pe_CM(Ev))
Ee_max          = lambda Ev : Ev - delta - Ev_CM(Ev)/mp * (Ee_CM(Ev) - pe_CM(Ev))


# Jacobian for the change of variables Ev->Ee
def jacobian(c, Ee):
    Ev = (Ee+delta)/(1-((Ee-np.sqrt((Ee**2 - me**2))*c)/mp))
    J_ve=(Ev**2/mp)*np.sqrt((Ee**2 - me**2))/(Ee + delta)
    
    return J_ve

#Ev = (Ee+delta)/(1-((Ee-np.sqrt((Ee**2 - me**2))*c)/mp))
def jacobian2(Ev, Ee):
    J_ve=(Ev**2/mp)*np.sqrt((Ee**2 - me**2))/(Ee + delta)
    return J_ve    


delto=1.29333236
def jacobian3(Ev, Ee):
    J_ve=(Ev**2/mp)*np.sqrt((Ee**2 - me**2))/(Ee + delto)
    return J_ve    


def dsigma_gm(Ev,Ee):
    """
    Cross section dσ/dEe (Ev, Ee)
    It's already differential in Ee (and not in t) (checked by comparing article cross section with Shota's Code)
    Returns dσ/dEe in MeV^-3 , with inputs Ev in MeV, Ee in Mev.
    Obtained with FortranForm[dXsec[Ev, Ee] //. {1. y_ -> y, -1. k_ -> -k, E^x_ -> exp[x]}] in Shota's Mathematica code.
    """
    return (-7.764519626241806e-34*((1.1507389656833256e49 + 8.866755713088948e48*Ee - 1.9758466521107417e48*(1.294223737146638 + Ee - Ev)**2 + 1.3302817613847957e47*(1.294223737146638 + Ee - Ev)**3 - 2.3335136163208056e45*(1.294223737146638 + Ee - Ev)**4 - 7.786839845256933e42*(1.294223737146638 + Ee - Ev)**5 + 5.13672147477018e40*(1.294223737146638 + Ee - Ev)**6 + 8.322646399254691e37*(1.294223737146638 + Ee - Ev)**7 - 6.031059118677985e35*(1.294223737146638 + Ee - Ev)**8 + 7.975159091865054e32*(1.294223737146638 + Ee - Ev)**9 - 1.992870080710164e29*(1.294223737146638 + Ee - Ev)**10 - 1.4290831766074329e26*(1.294223737146638 + Ee - Ev)**11 - 8.866755713088948e48*Ev)/((17059.49198320237 - 1876.54417632*Ee + 1876.54417632*Ev)**2*(1.1211713319832024e6 - 1876.54417632*Ee + 1876.54417632*Ev)**4) + Ee**2*(-1.2444840327321736e13 + 8.381290880839006e8*(1.294223737146638 + Ee - Ev)**2 + 3.272689101919159e6*(1.294223737146638 + Ee - Ev)**3 + 2.1091450036113533e10*(-Ee + Ev) - 2.5986966653012757e24/(597.4659942095673 - Ee + Ev)**4) + Ev**2*(-1.2444840327321736e13 + 8.381290880839006e8*(1.294223737146638 + Ee - Ev)**2 + 3.272689101919159e6*(1.294223737146638 + Ee - Ev)**3 + 2.1091450036113533e10*(-Ee + Ev) - 2.5986966653012757e24/(597.4659942095673 - Ee + Ev)**4) + Ev*(3.505521534064767e9 + 6.355014937216012e6*Ee - 231832.94335803072*(1.294223737146638 + Ee - Ev)**2 - 910.7852080935395*(1.294223737146638 + Ee - Ev)**3 - 6.355014937216012e6*Ev + (-7.51067519263486e21 + 7.232139715594883e20*Ee - 7.232139715594883e20*Ev)/((-9.090908809115762 + Ee - Ev)*(597.4659942095673 - Ee + Ev)**4) - (6.080084011174354e17*(1.294223737146638 + Ee - Ev)*(177.16673692369145 + Ee - Ev))/(597.4659942095673 - Ee + Ev)**2) + Ee*(3.505521534064767e9 + 6.355014937216012e6*Ee - 231832.94335803072*(1.294223737146638 + Ee - Ev)**2 - 910.7852080935395*(1.294223737146638 + Ee - Ev)**3 - 2.488968700965841e13*Ev + 1.6762581761678011e9*(1.294223737146638 + Ee - Ev)**2*Ev + 6.545378203838318e6*(1.294223737146638 + Ee - Ev)**3*Ev + 4.2182900072227066e10*Ev*(-Ee + Ev) + (-7.51067519263486e21 + 7.232139715594883e20*Ee - 7.232139715594883e20*Ev)/((-9.090908809115762 + Ee - Ev)*(597.4659942095673 - Ee + Ev)**4) - (5.197393330602551e24*Ev)/(597.4659942095673 - Ee + Ev)**4 - (6.080084011174354e17*(1.294223737146638 + Ee - Ev)*(177.16673692369145 + Ee - Ev))/(597.4659942095673 - Ee + Ev)**2))*(1 + 0.0023228194642461804*(6 + 0.43834015581233*(1/Ee)**1.5 + 1.5*Log(469.13604408/Ee))))/Ev**2

def dsigma(Ev,Ee):
    return dsigma_bzz(Ev,Ee)

def dσ(Ev,Ee):
    """
    Returns dσ/dEe in MeV^-3 , with inputs Ev in MeV, Ee in Mev
    """
    return dsigma(Ev,Ee)

def dσ_cm2overMeV(Ev,Ee):
    """
    Returns dσ/dEe in cm^2/MeV with inputs Ev in MeV, Ee in Mev
    """
    return dsigma(Ev,Ee)*MeV2_to_cm2


def Ev(Ee,c):
    """ 
    Energy of the neutrino in IMB reaction given Ee (positron energy) and c (cosine of the scattering angle).
    Returns Ev in MeV
    Ee in MeV
    c in range [-1,1]
    """
    Pe      = (Ee**2 - me**2)**0.5
    return (Ee + delta)/(1- (Ee-Pe*c)/mp)

#parametri per la sezione d'urto
# NOTA la differenza tra gm e bzz è il valore di Vud e gA che nel caso GM erano "nuovi conservativi" nel caso bzz erano "nuovi"
GF=1.16637*10**(-11)
soglia=((mn+me)**2-mp**2)/(2*mp)
Vud=0.97425
alpha=(137.035999174)**(-1)
Delta=mn-mp
M=938.921

# cross section by shota
def Cpert(Ee) : 
     return 1 + alpha/np.pi *(6 + 1.5 *np.log(mp/(2 *Ee)) + 1.2 *(me/Ee)**(1.5))

def tg2AmpSq1st(Ee, Ev) :
    t= mn**2 - mp**2 - 2*mp*(Ev - Ee)
    MA = 1007.86
    gA = -1.27597
    f1= 1 + (2.4*t)/10**6
    f2=3.706*(1 + (3.2*t)/10**6)
    Xi= 2*mp*(Ev + Ee) - me**2
    f3=0
    g3=0
    g1= gA/(1 - t/MA**2)**2
    g2=2*g1*(M**2/(139.6**2 - t))
    
    return -(1/M**2)*2*(4*f1**2* M**2 *(4 *M**2* (Delta**2 + me**2 - t) + 
     me**4 - Delta**2*me**2 - Xi**2 - 
     t**2 + Delta**2* t) + 
  4 *f1* M* (2* f2* M* (me**4 + me**2 *(t - Delta**2) + 
        2*t*(Delta**2 - 
           t)) + Delta* f2 *me**2* Xi + 
     2 *f3* me**2 *(4*Delta*M**2 + 
        2* M* Xi + Delta*(me**2 - t)) + 
     2 *(2* g1* M + Delta*g3) *(2 *Delta* M* me**2 + Xi* t)) + 4* f2**2* M**2* me**4 - 4* f2**2 *M**2* t**2 + 
  4 *Delta**2 *f2**2* M**2* t + 
  4 *Delta* f2**2* M* me**2 *Xi + 
  f2**2* me**2 *t**2 - Delta**2 *f2**2 *me**2* t - 
  f2**2* t**3 + Delta**2 *f2**2* t**2 + f2**2*Xi**2 *t + 
  8 *Delta* f2* f3* M* me**4 + 4 *f2* f3* me**2 *Xi* t + 
  4*g1* M* (4* f2* M* (2*Delta* M *me**2 + Xi* t) + 
     2 *g2* me**2 *(Delta* Xi + 
        2* M* (Delta**2 + me**2 - t)) + 
     g3* (-4* Delta* M**2* (me**2 - 2 *t) + 
        2 *M* me**2 *Xi + Delta* (me**4 + me**2 *t - 2* t**2))) + 
  16 *Delta**2 *f2* g3* M**2 *me**2 + 
  8* Delta* f2* g3* M *Xi* t - 16 *f3**2* M**2* me**4 + 
  16* f3**2* M**2* me**2* t + 4* f3**2* me**4* t - 4*f3**2* me**2* t**2 - 
  4* g1**2 *M**2* (4* M**2 *(-Delta**2 + me**2 - t) - 
     me**4 - Delta**2 *me**2 + Xi**2 + 
     t**2 + Delta**2 *t) - 4* Delta**2 *g2**2* me**4 + 
  4 *g2**2* me**4* t - 4* g2**2* me**2* t**2 + 4* Delta**2* g2**2* me**2* t + 
  8 *Delta*g2* g3* M* me**4 + 4* g2* g3* me**2*Xi* t - 
  4* g3**2 *M**2 *me**2* t + 4* g3**2* M**2 *t**2 + 4*Delta**2 *g3**2* M**2* t 
  +4 *Delta* g3**2 *M *me**2* Xi + Delta**2 *g3**2* me**4 + 
  g3**2* me**2* t**2 - g3**2 *t**3 - Delta**2* g3**2* t**2 + 
  g3**2* Xi**2 *t)


def dsigma_bzz(Ev, Ee):
    return (GF**2)*(Vud**2)/(128*np.pi*mp*Ev**2)*Cpert(Ee)*tg2AmpSq1st(Ee,Ev)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    errors = []
    print(f"Ee\tEv\tSigma GM\tSigma BZZ\tdiff")
    for Ee in np.arange(3,80):
        Ev_val = Ev(Ee,0)
        dsigma_with_bzz = dsigma_bzz(Ee,Ev_val)
        dsigma_with_gm = dsigma_gm(Ee,Ev_val)
        errors.append(2*(dsigma_with_bzz-dsigma_with_gm)/(dsigma_with_bzz+dsigma_with_gm))
        print(f"{Ee:.1f}\t{Ev_val:.3f}\t{dsigma_with_bzz:+5e}\t{dsigma_with_gm:+5e}\t{(dsigma_with_bzz-dsigma_with_gm):+.2e}")
   
    errors = np.array(errors)
    print(f"\nMax Error: {np.max(np.abs(errors))} %")
    print(f"Average Error: {np.average(np.abs(errors))} %\n")
