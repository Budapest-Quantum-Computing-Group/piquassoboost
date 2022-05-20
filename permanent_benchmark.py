import numpy as np
from thewalrus import perm
from piquassoboost.sampling.Boson_Sampling_Utilities import ChinHuhPermanentCalculator, GlynnPermanent, GlynnPermanentDoubleCPU, GlynnRepeatedPermanentCalculator, GlynnPermanentSingleDFE, GlynnPermanentDualDFE, GlynnPermanentInf, GlynnRepeatedPermanentCalculatorDouble, BBFGRepeatedPermanentCalculatorDouble
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
dim = 4
A = unitary_group.rvs(dim)#generate_random_unitary(dim)
Arep = A

iter_loops = 1

#np.save("mtx", A )
#A = np.load("mtx.npy")
#Arep = A

#Arep = np.zeros((dim,dim), dtype=np.complex128)
#for idx in range(dim):
#    Arep[:,idx] = A[:,0]


# multiplicities of input/output states
input_state = np.ones(dim, np.int64)
output_state = np.ones(dim, np.int64)


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


# multiplicities of input/output states
input_state = np.ones(dim, np.int64)
output_state = np.ones(dim, np.int64)
output_state[0] = 2
output_state[2] = 3

# Glynn repeated permanent calculator
permanent_BBFG_calculator_repeated = BBFGRepeatedPermanentCalculatorDouble( Arep, input_state=input_state, output_state=output_state )
time_BBFG_Cpp_repeated_double = 1000000000
for idx in range(iter_loops):
    start = time.time()   

    permanent_BBFG_Cpp_repeated_double = permanent_BBFG_calculator_repeated.calculate()

    time_loc = time.time() - start
    start = time.time()   
       
    if time_BBFG_Cpp_repeated_double > time_loc:
        time_BBFG_Cpp_repeated_double = time_loc
        
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




# multiplicities of input/output states
input_state = np.ones(dim, np.int64)
output_state = np.ones(dim, np.int64)


# Glynn repeated permanent calculator
permanent_Glynn_calculator_repeated = GlynnRepeatedPermanentCalculator( Arep, input_state=input_state, output_state=output_state )
time_Glynn_Cpp_repeated = 1000000000
for idx in range(iter_loops):
    start = time.time()   

    permanent_Glynn_Cpp_repeated = permanent_Glynn_calculator_repeated.calculate()

    time_loc = time.time() - start
    start = time.time()   
       
    if time_Glynn_Cpp_repeated > time_loc:
        time_Glynn_Cpp_repeated = time_loc



# Glynn repeated permanent calculator
permanent_Glynn_calculator_repeated = GlynnRepeatedPermanentCalculatorDouble( Arep, input_state=input_state, output_state=output_state )
time_Glynn_Cpp_repeated_double = 1000000000
for idx in range(iter_loops):
    start = time.time()   

    permanent_Glynn_Cpp_repeated_double = permanent_Glynn_calculator_repeated.calculate()

    time_loc = time.time() - start
    start = time.time()   
       
    if time_Glynn_Cpp_repeated_double > time_loc:
        time_Glynn_Cpp_repeated_double = time_loc


# Glynn permanent calculator
permanent_Glynn_calculator = GlynnPermanent( Arep )
time_Glynn_Cpp = 1000000000
for idx in range(iter_loops):
    start = time.time()   

    permanent_Glynn_Cpp = permanent_Glynn_calculator.calculate()

    time_loc = time.time() - start
    start = time.time()   
       
    if time_Glynn_Cpp > time_loc:
        time_Glynn_Cpp = time_loc



# Glynn permanent calculator double
permanent_Glynn_calculator = GlynnPermanentDoubleCPU( Arep )
time_Glynn_Cpp_double = 1000000000
for idx in range(iter_loops):
    start = time.time()   

    permanent_Glynn_Cpp_double = permanent_Glynn_calculator.calculate()

    time_loc = time.time() - start
    start = time.time()   
       
    if time_Glynn_Cpp_double > time_loc:
        time_Glynn_Cpp_double = time_loc


print(' ')
print( permanent_BBFG_Cpp_repeated_double )
print( permanent_walrus_quad_Ryser )
print( permanent_walrus_Ryser )
print( permanent_walrus_quad_BBFG )
print( permanent_ChinHuh_Cpp )
print( permanent_Glynn_Cpp_repeated )
print( permanent_Glynn_Cpp_repeated_double )
print( permanent_Glynn_Cpp )
print( permanent_Glynn_Cpp_double )

'''
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
'''
# Glynn Inf permanent calculator
if (dim<=1):
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
print('Time elapsed with walrus: ' + str(time_walrus))
print('Time elapsed with walrus double: ' + str(time_walrus_double))
print('Time elapsed with walrus BBFG : ' + str(time_walrus_BBFG))
print('Time elapsed with BBFG double : ' + str(time_BBFG_Cpp_repeated_double))
print('Time elapsed with piquasso Chin-Huh: ' + str(time_Cpp))
print('Time elapsed with piquasso Glynn: ' + str(time_Glynn_Cpp))
print('Time elapsed with piquasso Glynn_double: ' + str(time_Glynn_Cpp_double))
print('Time elapsed with piquasso Glynn repeated: ' + str(time_Glynn_Cpp_repeated))
print('Time elapsed with piquasso Glynn repeated double: ' + str(time_Glynn_Cpp_repeated_double))
#print('Time elapsed with DFE Glynn: ' + str(time_Glynn_DFE))
#print('Time elapsed with dual DFE Glynn: ' + str(time_Glynn_dual_DFE))
if (dim<=1):
    print('Time elapsed with infinite precision Glynn: ' + str(time_Glynn_InfinitePrecision))
#print( "speedup: " + str(time_walrus/time_Cpp) )
print( "speedup Glynn: " + str(time_walrus/time_Glynn_Cpp) )
print(' ')
print(' ')


#print( 'Relative difference between quad walrus and piquasso result: ' + str(abs(permanent_walrus_quad_Ryser-permanent_ChinHuh_Cpp)/abs(permanent_ChinHuh_Cpp)*100) + '%')
print( 'Relative difference between quad walrus and piquasso Glynn result: ' + str(abs(permanent_walrus_quad_Ryser-permanent_Glynn_Cpp)/abs(permanent_Glynn_Cpp)*100) + '%')


