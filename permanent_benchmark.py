import numpy as np
from thewalrus import perm
from piquassoboost.sampling.Boson_Sampling_Utilities import ChinHuhPermanentCalculator, GlynnPermanent, GlynnPermanentDouble, GlynnRepeatedPermanentCalculator, GlynnPermanentSingleDFE, GlynnPermanentDualDFE, GlynnPermanentInf, GlynnRepeatedPermanentCalculatorDouble, BBFGRepeatedPermanentCalculatorDouble, BBFGRepeatedPermanentCalculatorLongDouble, BBFGPermanentLongDouble, BBFGPermanentDouble, GlynnRepeatedSingleDFEPermanentCalculator
import piquasso as pq
import random
from scipy.stats import unitary_group

import time


DEPTH = 40

def pairwise(t):
    return zip(t[::2], t[1::2])

def generate_random_unitary( dim ):

    with pq.Program() as program:
        for column in range(DEPTH):
            for modes in pairwise(range(column % 2, dim)):
                theta = random.uniform(0, 2 * np.pi)
                phi = random.uniform(0, 2 * np.pi)

                pq.Q(*modes) | pq.Beamsplitter(theta=theta, phi=phi)

        pq.Q() | pq.Sampling()

    state = pq.SamplingState(1, 1, 1, 1, 0, 0)
    state.apply(program, shots=1)


    return state.interferometer







# generate the random matrix
dim = 30
A = unitary_group.rvs(dim)#generate_random_unitary(dim)
Arep = A

iter_loops = 10

#np.save("mtx", A )
#A = np.load("mtx.npy")
#Arep = A

#Arep = np.zeros((dim,dim), dtype=np.complex128)
#for idx in range(dim):
#    Arep[:,idx] = A[:,0]


# multiplicities of input/output states
input_state = np.ones(dim, np.int64)
output_state = np.ones(dim, np.int64)
output_state[0] = 1
output_state[1] = 2
input_state[2] = 2

# ChinHuh permanent calculator
permanent_ChinHuh_calculator = ChinHuhPermanentCalculator( A, input_state, output_state )
time_Cpp = 1000000
for idx in range(iter_loops):
    start = time.time()   

    permanent_ChinHuh_Cpp = 0#permanent_ChinHuh_calculator.calculate()

    time_loc = time.time() - start
    start = time.time()   
       
    if time_Cpp > time_loc:
        time_Cpp = time_loc




# Glynn repeated permanent calculator
permanent_BBFG_calculator_repeated = BBFGRepeatedPermanentCalculatorDouble( Arep, input_state=input_state, output_state=output_state )
time_BBFG_repeated_double = 1000000000
for idx in range(iter_loops):
    start = time.time()   

    permanent_BBFG_repeated_double = permanent_BBFG_calculator_repeated.calculate()

    time_loc = time.time() - start
    start = time.time()   
       
    if time_BBFG_repeated_double > time_loc:
        time_BBFG_repeated_double = time_loc


# Glynn repeated permanent calculator
permanent_BBFG_calculator_repeated = BBFGRepeatedPermanentCalculatorLongDouble( Arep, input_state=input_state, output_state=output_state )
time_BBFG_repeated_long_double = 1000000000
for idx in range(iter_loops):
    start = time.time()   

    permanent_BBFG_repeated_long_double = permanent_BBFG_calculator_repeated.calculate()

    time_loc = time.time() - start
    start = time.time()   
       
    if time_BBFG_repeated_long_double > time_loc:
        time_BBFG_repeated_long_double = time_loc



# Glynn repeated permanent calculator
permanent_Glynn_calculator_repeated = GlynnRepeatedPermanentCalculator( Arep, input_state=input_state, output_state=output_state )
time_Glynn_repeated = 1000000000
for idx in range(iter_loops):
    start = time.time()   

    permanent_Glynn_repeated = permanent_Glynn_calculator_repeated.calculate()

    time_loc = time.time() - start
    start = time.time()   
       
    if time_Glynn_repeated > time_loc:
        time_Glynn_repeated = time_loc



# Glynn repeated permanent calculator
permanent_Glynn_calculator_repeated = GlynnRepeatedPermanentCalculatorDouble( Arep, input_state=input_state, output_state=output_state )
time_Glynn_repeated_double = 1000000000
for idx in range(iter_loops):
    start = time.time()   

    permanent_Glynn_repeated_double = permanent_Glynn_calculator_repeated.calculate()

    time_loc = time.time() - start
    start = time.time()   
       
    if time_Glynn_repeated_double > time_loc:
        time_Glynn_repeated_double = time_loc
        
        
# GlynnRepeatedSingleDFEPermanentCalculator
permanent_Glynn_DFE_calculator_repeated = GlynnRepeatedSingleDFEPermanentCalculator( Arep, input_state=input_state, output_state=output_state )
time_Glynn_repeated_singleDFE = 1000000000
for idx in range(iter_loops):
    start = time.time()   

    permanent_Glynn_repeated_singleDFE = permanent_Glynn_DFE_calculator_repeated.calculate()

    time_loc = time.time() - start
    start = time.time()   
       
    if time_Glynn_repeated_singleDFE > time_loc:
        time_Glynn_repeated_singleDFE = time_loc


        
# calculate the permanent using walrus library
time_walrus = 1000000000        
for idx in range(iter_loops):
    start = time.time()   
    permanent_walrus_quad_Ryser = perm(Arep, quad=True, method="ryser")
    time_loc = time.time() - start
    start = time.time()   
       
    if time_walrus > time_loc:
        time_walrus = time_loc

time_walrus_BBFG = 1000000        
for idx in range(iter_loops):
    start = time.time()   
    permanent_walrus_quad_BBFG = 0#perm(Arep, quad=True, method="bbfg")
    time_loc = time.time() - start
    start = time.time()   
       
    if time_walrus_BBFG > time_loc:
        time_walrus_BBFG = time_loc


# calculate the permanent using walrus library
time_walrus_double = 1000000000        
for idx in range(iter_loops):
    start = time.time()   
    permanent_walrus_Ryser = perm(Arep, quad=False, method="ryser")
    time_loc = time.time() - start
    start = time.time()   
       
    if time_walrus_double > time_loc:
        time_walrus_double = time_loc







# recursive Glynn permanent calculator
permanent_Glynn_calculator = GlynnPermanent( Arep )
time_Glynn_longdouble = 1000000000
for idx in range(iter_loops):
    start = time.time()   

    permanent_Glynn_longdouble = permanent_Glynn_calculator.calculate()

    time_loc = time.time() - start
    start = time.time()   
       
    if time_Glynn_longdouble > time_loc:
        time_Glynn_longdouble = time_loc



# recursive Glynn permanent calculator double
permanent_Glynn_calculator = GlynnPermanentDouble( Arep )
time_Glynn_double = 1000000000
for idx in range(iter_loops):
    start = time.time()   

    permanent_Glynn_double = permanent_Glynn_calculator.calculate()

    time_loc = time.time() - start
    start = time.time()   
       
    if time_Glynn_double > time_loc:
        time_Glynn_double = time_loc




# BBFG permanent calculator
permanent_BBFG_calculator = BBFGPermanentDouble( Arep )
time_BBFG_double = 1000000000
for idx in  range(iter_loops):
    start = time.time()   

    permanent_BBFG_double = permanent_BBFG_calculator.calculate()

    time_loc = time.time() - start
    start = time.time()   
       
    if time_BBFG_double > time_loc:
        time_BBFG_double = time_loc



# BBFG permanent calculator double
permanent_BBFG_calculator = BBFGPermanentLongDouble( Arep )
time_BBFG_longdouble = 1000000000
for idx in range(iter_loops):
    start = time.time()   

    permanent_BBFG_longdouble = permanent_BBFG_calculator.calculate()

    time_loc = time.time() - start
    start = time.time()   
       
    if time_BBFG_longdouble > time_loc:
        time_BBFG_longdouble = time_loc


print(' ')
print( permanent_BBFG_repeated_double )
print( permanent_BBFG_repeated_long_double )
print( permanent_Glynn_repeated )
print( permanent_Glynn_repeated_double )
print( permanent_ChinHuh_Cpp )
print( permanent_Glynn_repeated_singleDFE )

print( permanent_walrus_quad_Ryser )
print( permanent_walrus_Ryser )
print( permanent_walrus_quad_BBFG )
print( permanent_Glynn_longdouble )
print( permanent_Glynn_double )
print( permanent_BBFG_double )
print( permanent_BBFG_longdouble )


# Glynn DFE permanent calculator
permanent_Glynn_calculator = GlynnPermanentSingleDFE( Arep )
time_Glynn_DFE = 1000000000
for idx in range(iter_loops):
    start = time.time()   

    permanent_Glynn_DFE = permanent_Glynn_calculator.calculate()
    time_loc = time.time() - start
    start = time.time()   
       
    if time_Glynn_DFE > time_loc:
        time_Glynn_DFE = time_loc


# Glynn dual DFE permanent calculator
permanent_Glynn_calculator = GlynnPermanentDualDFE( Arep )
time_Glynn_dual_DFE = 1000000000
for idx in range(iter_loops):
    start = time.time()   

    permanent_Glynn_dual_DFE = permanent_Glynn_calculator.calculate()
    time_loc = time.time() - start
    start = time.time()   
       
    if time_Glynn_dual_DFE > time_loc:
        time_Glynn_dual_DFE = time_loc


        

print( permanent_Glynn_DFE )
print( permanent_Glynn_dual_DFE )

# Glynn Inf permanent calculator
if (dim<=20):
    permanent_Glynn_calculator = GlynnPermanentInf( Arep )
    time_Glynn_InfinitePrecision = 1000000000
    for idx in range(iter_loops):
        start = time.time()   

        permanent_Glynn_InfinitePrecision = permanent_Glynn_calculator.calculate()
        time_loc = time.time() - start
        start = time.time()   
       
        if time_Glynn_InfinitePrecision > time_loc:
            time_Glynn_InfinitePrecision = time_loc


        

    print( permanent_Glynn_InfinitePrecision )



print(' ')
print('*******************************************')
print('Time elapsed with BBFG repeated double : ' + str(time_BBFG_repeated_double))
print('Time elapsed with BBFG repeated long double : ' + str(time_BBFG_repeated_long_double))
print('Time elapsed with piquasso Glynn repeated double: ' + str(time_Glynn_repeated_double))
print('Time elapsed with piquasso Glynn repeated long double: ' + str(time_Glynn_repeated))
print('Time elapsed with piquasso Chin-Huh: ' + str(time_Cpp))
print('Time elapsed with DFE repeated Glynn: ' + str(time_Glynn_repeated_singleDFE))

print('Time elapsed with walrus: ' + str(time_walrus))
print('Time elapsed with walrus double: ' + str(time_walrus_double))
print('Time elapsed with walrus BBFG : ' + str(time_walrus_BBFG))
print('Time elapsed with piquasso Glynn_longdouble: ' + str(time_Glynn_longdouble))
print('Time elapsed with piquasso Glynn_double: ' + str(time_Glynn_double))
print('Time elapsed with piquasso BBFG_longduble: ' + str(time_BBFG_longdouble))
print('Time elapsed with piquasso BBFG_double: ' + str(time_BBFG_double))
print('Time elapsed with DFE Glynn: ' + str(time_Glynn_DFE))
print('Time elapsed with dual DFE Glynn: ' + str(time_Glynn_dual_DFE))
if (dim<=20):
    print('Time elapsed with infinite precision Glynn: ' + str(time_Glynn_InfinitePrecision))
#print( "speedup: " + str(time_walrus/time_Cpp) )
print( "speedup Glynn: " + str(time_walrus/time_Glynn_longdouble) )
print(' ')
print(' ')


#print( 'Relative difference between quad walrus and piquasso result: ' + str(abs(permanent_walrus_quad_Ryser-permanent_ChinHuh_Cpp)/abs(permanent_ChinHuh_Cpp)*100) + '%')
print( 'Relative difference between quad walrus and piquasso Glynn result: ' + str(abs(permanent_walrus_quad_Ryser-permanent_Glynn_longdouble)/abs(permanent_Glynn_longdouble)*100) + '%')


