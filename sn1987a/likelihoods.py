import os, sys; 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sn1987a.expdata as expdata
import sn1987a.vflux as vflux
import sn1987a.detectors as detectors
import sn1987a.reader_f_func as reader_f_func
import sn1987a.reader_cal_G as reader_cal_G
import sn1987a.reader_cal_S as reader_cal_S
import sn1987a.differential_rates as differential_rates
import numpy as np
import sn1987a.reader_Signal as Signal_reader

f_manager = reader_f_func.manager()
G_manager = reader_cal_G.manager()
S_manager = reader_cal_S.manager()
Signal_manager = Signal_reader.manager()

def logL_generic(experiment,        # bk, im or k2
                 tau_a,             # decay time accretion
                 tau_c,             # decay time colling
                 T0,                # initial temperature
                 Rns,               # raggio stella di neutroni (cm)
                 csi_0,             # frazione inziale di neutroni che partecipano a IBD
                 t_max,             # maximum time
                 t_start,           # start time
                 skip_event_id      = None,
                 ):       
    
    #####################################################
    # LIKELIHOOD PART 0 : Preparation and constants evaluation
    #####################################################

    logLikelihood = 0

    # Dead time correction on Ntot
    # This value is updated for each event
    N_tot_deadtime_correction = 0

    # Total livetime accounting for muon background in IMB
    livetime_factor = 0.9055 if experiment == "im" else 1

    # Evaluating Ta and calS_a, which depend on Ta only
    Ta              =   T0    * 0.6
    calS_a     = S_manager.eval(T=Ta,
                                experiment=experiment,
                                phase="a",)
    
    # Evaluating geometric factors and unit conversion
    k_a         = 8 * np.pi     * vflux.geometric_const * vflux.physics_const * vflux.Ms/vflux.mn
    k_c         = 4 * np.pi**2  * vflux.geometric_const * vflux.physics_const
    
    #####################################################
    # LIKELIHOOD PART 1 : Event specific terms
    #####################################################
   
    # Retrieve the data as pandas dataframe
    dataframe = expdata.get_df(experiment)

    for _,event in dataframe.iterrows(): 
        """
        Note on iterrow:
        the first return is an integer which identifies the event, i discard it in _
        event is a structure which contains the following variables
         exp , id, exp_id , E_MeV , angle_deg , cos_angle , time_ms , time_s
        """

        
        if event.id == skip_event_id:
            continue

        # Evaluate absolute time of the event
        time_absolute = event.time_s + t_start #serve per Tc

        Tc          =   T0    * vflux.f_cal_c(t=time_absolute, t0=t_max, tau_c=tau_c)**(1/4)
        csi_n       =   csi_0 * vflux.f_cal_a(t=time_absolute, t0=t_max, tau_a=tau_a)



        efficiency  =   detectors.eta[experiment](event.E_MeV,c=event.cos_angle)

        f_func_a    =   f_manager.eval(Ta, experiment, "a", event.exp_id)
        f_func_c    =   f_manager.eval(Tc, experiment, "c", event.exp_id)

        signal_a    =   Signal_manager.eval(Ta, experiment, "a", event.exp_id)
        signal_c    =   Signal_manager.eval(Tc, experiment, "c", event.exp_id)
        
        signal_f =   sum((
                                f_func_a * differential_rates.dN_dtdEedc_a(experiment, event.E_MeV, event.cos_angle, Ta, csi_n),
                                f_func_c * differential_rates.dN_dtdEedc_c(experiment, event.E_MeV, event.cos_angle, Tc, Rns),
                                ))

        signal   =   signal_a * csi_n + signal_c * (Rns**2)      #for new_tab
        #print(f_func_c*differential_rates.dN_dtdEedc_c(experiment, event.E_MeV, event.cos_angle, Tc, Rns), signal_c)
        background      = event.background/2 #Background. Da confermare se ci va il /2 oppure no.
        Ni_f              = signal_f * efficiency + background
        Ni              = signal + background      #for new_tab
        print(signal)
        #print(f"Event {event.id} | Ni = {Ni}| Ni_f = {Ni_f} | Tc = {Tc} | csi_n = {csi_n} | signal_a = {signal_a} | signal_c = {signal_c} | background = {background}")
        
        logLikelihood  += -2 * np.log(Ni)

        # Evaluate contribution to dead-time from this event if IMB
        if experiment=="im":
            dead_time_corrected_time = time_absolute + detectors.imb_dead_time_s / 2
            Tc_dead_time =  T0    * vflux.f_cal_c(t=dead_time_corrected_time, t0=t_max, tau_c=tau_c)**(1/4)
            calS_c_dead_time         = S_manager.eval(T = Tc_dead_time,
                                        experiment=experiment,
                                        phase="c",)
            N_tot_deadtime_correction +=  detectors.imb_dead_time_s * sum((
                                            k_a * calS_a * csi_0 * vflux.f_cal_a(dead_time_corrected_time, t0=t_max, tau_a=tau_a),
                                            k_c * (Rns**2) * calS_c_dead_time
                                            ))
    

    #####################################################
    # LIKELIHOOD PART 2 : Total number of events
    #####################################################

    if np.isnan(T0):
        raise Exception("T0 is not a number")
    
    # Accretion
    calG_a     = G_manager.eval(t0_over_tau = t_max/tau_a,
                                experiment=experiment,
                                phase="a")
    
    # Cooling
    calG_c     = G_manager.eval(T0=T0,
                                t0_over_tau = t_max/tau_c,
                                experiment=experiment,
                                phase="c")

    N_a         = k_a * csi_0  * tau_a * (calS_a  * calG_a)
    N_c         = k_c * Rns**2 * tau_c * (          calG_c) 
    N_tot       = (N_a + N_c - N_tot_deadtime_correction) * livetime_factor
    
    logLikelihood   +=  2 * N_tot
    return logLikelihood


def logL_k2(tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_k2, **kwargs):
    return logL_generic("k2",tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_k2, skip_event_id=kwargs.get("skip_event_id"))

def logL_im(tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_im, **kwargs):
    return logL_generic("im",tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_im, skip_event_id=kwargs.get("skip_event_id"))

def logL_bk(tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_bk, **kwargs):
    return logL_generic("bk",tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_bk, skip_event_id=kwargs.get("skip_event_id"))

def logL_ls(tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_ls=0, **kwargs):
    return logL_generic("ls",tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_ls, skip_event_id=kwargs.get("skip_event_id"))

def logL_combined(tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_k2, t_start_im, t_start_bk, skip_event_id=None, *args,**kwargs):
    
    return sum((
                logL_k2(tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_k2, skip_event_id=skip_event_id),
                logL_bk(tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_bk, skip_event_id=skip_event_id),
                logL_im(tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_im, skip_event_id=skip_event_id),
    ))

'''def logL_combined_timediff_k(tau_a, tau_c, T0, Rns, csi_0, t_max, sum_k, diff_k, t_start_bk, skip_event_id=None, *args, **kwargs):
    
    t_start_k2 = np.abs( 0.5 * (sum_k + diff_k))
    t_start_im = np.abs(0.5 * (sum_k - diff_k))
    
    return sum((
        logL_k2(tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_k2, skip_event_id=skip_event_id),
        logL_bk(tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_bk, skip_event_id=skip_event_id),
        logL_im(tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_im, skip_event_id=skip_event_id),
    ))'''




def logL_combined_timediff_k(tau_a, tau_c, T0, Rns, csi_0, t_max, diff_k, t_start_im, t_start_bk, skip_event_id=None, *args, **kwargs):
    
    t_start_k2 = diff_k + t_start_im
    
  
    return   sum((
                        logL_k2(tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_k2, skip_event_id=skip_event_id),
                        logL_bk(tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_bk, skip_event_id=skip_event_id),
                        logL_im(tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_im, skip_event_id=skip_event_id),
                     ))


'''def logL_combined_timediff_k(tau_a, tau_c, T0, Rns, csi_0, t_max, diff_k, t_start_im, t_start_bk, skip_event_id=None, *args, **kwargs):
    
    t_start_k2 = diff_k + t_start_im

    # Calcolo della likelihood senza vincoli
    logL_k2_val = logL_k2(tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_k2, skip_event_id=skip_event_id)
    logL_bk_val = logL_bk(tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_bk, skip_event_id=skip_event_id)
    logL_im_val = logL_im(tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_im, skip_event_id=skip_event_id)
    
    logL_combined = logL_k2_val + logL_bk_val + logL_im_val
    
    # Controllo del vincolo
    if diff_k >-t_start_im:
        # Se il vincolo è rispettato, restituiamo il valore calcolato
        return logL_combined
    else:
        # Penalizzazione in caso di violazione del vincolo
        penalized_value = 263.7 * np.exp( 2*(diff_k - t_start_im)**2)
        
        return penalized_value '''


                   

def logL_combined_timediff_b(tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_k2, t_start_im, diff_b, skip_event_id=None, *args, **kwargs):
    
    t_start_bk = diff_b + t_start_im
     
    
    return sum((
        logL_k2(tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_k2, skip_event_id=skip_event_id),
        logL_bk(tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_bk, skip_event_id=skip_event_id),
        logL_im(tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_im, skip_event_id=skip_event_id),
    ))



def logL_combined_skip_one_event(skip_event_id):
    def logL_combined_with_missing_event(tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_k2, t_start_im, t_start_bk):
        return logL_combined(tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_k2, t_start_im, t_start_bk,skip_event_id=skip_event_id)
    return logL_combined_with_missing_event


def logL_combined_LSDts(tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_k2, t_start_im, t_start_bk,*args,**kwargs):
    t_start_ls=0
    return sum((
                logL_k2(tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_k2),
                logL_bk(tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_bk),
                logL_im(tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_im),
                logL_ls(tau_a, tau_c, T0, Rns, csi_0, t_max, t_start_ls)
                ))

def LogL_combined_tmax_fixed(tau_a, tau_c, T0, Rns, csi_0, t_start_k2, t_start_im, t_start_bk,*args,**kwargs):
 
    return sum((
                logL_k2(tau_a, tau_c, T0, Rns, csi_0, 0.1, t_start_k2),
                logL_bk(tau_a, tau_c, T0, Rns, csi_0, 0.1, t_start_bk),
                logL_im(tau_a, tau_c, T0, Rns, csi_0, 0.1, t_start_im),
                ))
                             

def logL_lmfit(params,*args,**kwargs):
    pvals = params.valuesdict()
    return logL_combined_LSDts(pvals["tau_a"], pvals["tau_c"], pvals["T0"], pvals["Rns"], pvals["csi_0"], pvals["t_max"], pvals["t_start_k2"], pvals["t_start_im"], pvals["t_start_bk"])


def logL_generic_cooling(experiment,        # bk, im or k2
                 tau_c,             # decay time colling
                 T0,                # initial temperature
                 Rns,               # raggio stella di neutroni (cm)
                
                 t_max,             # maximum time
                 t_start,           # start time
                 skip_event_id      = None,
                 ):       
    
    #####################################################
    # LIKELIHOOD PART 0 : Preparation and constants evaluation
    #####################################################

    logLikelihood = 0

    # Dead time correction on Ntot
    # This value is updated for each event
    N_tot_deadtime_correction = 0

    # Total livetime accounting for muon background in IMB
    livetime_factor = 0.9055 if experiment == "im" else 1

    # Evaluating Ta and calS_a, which depend on Ta only
    Ta              =   T0    * 0.6
    calS_a     = S_manager.eval(T=Ta,
                                experiment=experiment,
                                phase="a",)
    
    # Evaluating geometric factors and unit conversion
    k_c = 4 * np.pi**2  * vflux.geometric_const * vflux.physics_const
    
    #####################################################
    # LIKELIHOOD PART 1 : Event specific terms
    #####################################################
   
    # Retrieve the data as pandas dataframe
    dataframe = expdata.get_df(experiment)

    for _,event in dataframe.iterrows(): 
        """
        Note on iterrow:
        the first return is an integer which identifies the event, i discard it in _
        event is a structure which contains the following variables
         exp , id, exp_id , E_MeV , angle_deg , cos_angle , time_ms , time_s
        """

        
        if event.id == skip_event_id:
            continue

        # Evaluate absolute time of the event
        time_absolute = event.time_s + t_start #serve per Tc

        Tc          =   T0    * vflux.f_cal_c(t=time_absolute, t0=t_max, tau_c=tau_c)**(1/4)



        efficiency  =   detectors.eta[experiment](event.E_MeV,c=event.cos_angle)

        f_func_c    =   f_manager.eval(Tc, experiment, "c", event.exp_id)

        signal_c    =   Signal_manager.eval(Tc, experiment, "c", event.exp_id)
        
        

        signal   =    signal_c * (Rns**2)      #for new_tab
        #print(f_func_c*differential_rates.dN_dtdEedc_c(experiment, event.E_MeV, event.cos_angle, Tc, Rns), signal_c)
        background = event.background/2 #Background. Da confermare se ci va il /2 oppure no.
        #Ni_f              = signal_f * efficiency + background
        Ni              = signal + background      #for new_tab
        #print(f"Event {event.id} | Ni = {Ni}| Ni_f = {Ni_f} | Tc = {Tc} | csi_n = {csi_n} | signal_a = {signal_a} | signal_c = {signal_c} | background = {background}")
        
        logLikelihood  += -2 * np.log(Ni)

        # Evaluate contribution to dead-time from this event if IMB
        if experiment=="im":
            dead_time_corrected_time = time_absolute + detectors.imb_dead_time_s / 2
            Tc_dead_time =  T0    * vflux.f_cal_c(t=dead_time_corrected_time, t0=t_max, tau_c=tau_c)**(1/4)
            calS_c_dead_time         = S_manager.eval(T = Tc_dead_time,
                                        experiment=experiment,
                                        phase="c",)
            N_tot_deadtime_correction +=  detectors.imb_dead_time_s * k_c * (Rns**2) * calS_c_dead_time
                                            
    

    #####################################################
    # LIKELIHOOD PART 2 : Total number of events
    #####################################################

    if np.isnan(T0):
        raise Exception("T0 is not a number")
    
   
    
    # Cooling
    calG_c     = G_manager.eval(T0=T0,
                                t0_over_tau = t_max/tau_c,
                                experiment=experiment,
                                phase="c")

    
    N_c         = k_c * Rns**2 * tau_c * (          calG_c) 
    N_tot       = ( N_c - N_tot_deadtime_correction) * livetime_factor
    
    logLikelihood   +=  2 * N_tot
    return logLikelihood


def logL_k2_c(tau_c, T0, Rns, t_max, t_start_k2, **kwargs):
    return logL_generic_cooling("k2", tau_c, T0, Rns,  t_max, t_start_k2, skip_event_id=kwargs.get("skip_event_id"))

def logL_im_c( tau_c, T0, Rns, t_max, t_start_im, **kwargs):
    return logL_generic_cooling("im",tau_c, T0, Rns, t_max, t_start_im, skip_event_id=kwargs.get("skip_event_id"))

def logL_bk_c(tau_c, T0, Rns, t_max, t_start_bk, **kwargs):
    return logL_generic_cooling("bk", tau_c, T0, Rns, t_max, t_start_bk, skip_event_id=kwargs.get("skip_event_id"))

def logL_ls_c(tau_c, T0, Rns, t_max, t_start_ls=0, **kwargs):
    return logL_generic_cooling("ls",tau_c, T0, Rns, t_max, t_start_ls, skip_event_id=kwargs.get("skip_event_id"))

def logL_combined_c(tau_c, T0, Rns, t_max, t_start_k2, t_start_im, t_start_bk, skip_event_id=None, *args,**kwargs):
    
    return sum((
                logL_k2_c( tau_c, T0, Rns,  t_max, t_start_k2, skip_event_id=skip_event_id),
                logL_bk_c( tau_c, T0, Rns,  t_max, t_start_bk, skip_event_id=skip_event_id),
                logL_im_c( tau_c, T0, Rns,  t_max, t_start_im, skip_event_id=skip_event_id),
    ))






benchMarkValues_veronica = {
                 "tau_a"     : 0.53,   # decay time accretion
                 "tau_c"     : 7.74,   # decay time colling
                 "T0"        : 3.54,   # initial temperature
                 "Rns"       : 2e6,   # raggio stella di neutroni (cm)
                 "csi_0"     : 0.09,   # frazione inziale di neutroni che partecipano a IBD
                 "t_max"     : 0.1,   # maximum time
                 "t_start_k2": 0.05,
                 "t_start_im": 0.05,
                 "t_start_bk": 0.05,
                 }

def benchmarkLikelihood(benchmark = benchMarkValues_veronica):
    print("Benchmarks values: ")
    for key in benchmark:
        print(key+f"\t{benchmark[key]}")
    print(" ")
    
    print(f"Valore calcolato  K2: {logL_k2(**benchmark):.03f}", "\t(BZZ : 123.038)")
    print(f"Valore calcolato IMB: {logL_im(**benchmark):.03f}")
    print(f"Valore calcolato BAK: {logL_bk(**benchmark):.03f}")
    print(f"Valore calcolato LSD: {logL_ls(**benchmark):.03f}")
    print(f"Valore calcolato LL combined: {logL_combined(**benchmark):.03f}")

if __name__ == "__main__":

    benchmarkLikelihood()