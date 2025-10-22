import numpy as np

# Masses and constants
mp      = 938.27208816  # MeV
mn      = 939.56542052  # Mev
me      = 0.51099895000 # MeV

delta   = (mn**2 - mp**2 - me**2)/(2*mp) # ~1.2940 MeV # 1.29333236

# Conversion Factors 
MeV2_to_barn   = 389.379372 # 1 MeV^-2 ~ 400 barn
barn_to_cm2    = 1e-24 # 1 cm^2 = 10^-24 MeV
MeV2_to_cm2    = MeV2_to_barn * barn_to_cm2

# Physical Quantities
hc = 2*np.pi*197.3e-13 #MeV*cm
h = (6.62607015/(1.602e-19))*1e-40
c = 29979245800 

# Solar Mass
Ms = 2.195e60*me # MeV

benchmark_values = {
"csi_0"   : 0.02,
"Rns"     : 1.2e6,#cm
"tau_a"   : 0.3,  #secondi
"tau_c"   : 5.5,   #secondi
"na"      : 2,
"nc"      : 2,
"alpha_a" : 2,
"alpha_c" : 1,
"T0"      : 5, #MeV
}