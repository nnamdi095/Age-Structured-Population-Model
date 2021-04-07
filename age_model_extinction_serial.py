import numpy as np
import time

start = time.time()

"""
from mpi4py import MPI  
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size() """

# CREATE THE PROJECTION MATRIX
L = np.array([[0, 0.133, 1.082, 1.194, 1.590, 1.590, 1.590, 1.590],
              [0.380, 0, 0, 0, 0, 0, 0, 0], 
              [0, 0.653, 0, 0, 0, 0, 0, 0],  
              [0, 0, 0.850, 0, 0, 0, 0, 0],
              [0, 0, 0, 0.400, 0, 0, 0, 0],
              [0, 0, 0, 0, 0.589, 0, 0, 0],
              [0, 0, 0, 0, 0, 0.589, 0, 0],
              [0, 0, 0, 0, 0, 0, 0.589, 0]])

#AGE DISTRIBUTION AT, t = 0
x_0 = np.array([[20], [10], [9], [9], [3], [3], [2], [2]])

#FINDING THE STABLE POPULATION DISTRIBUTION
kmax = 1000
k = 1

while k < kmax:
    Lk = np.linalg.matrix_power(L, k)
    x_t = np.matmul(Lk, x_0)
    #computing total population
    total_pop = np.sum(x_t)
    #initializing convergence condition
    if total_pop <= 1:
        break
    #reassingment
    k += 1




#FINDING PROJECTED POPULATION GROWTH RATE
eigen_values, eigen_vectors = np.linalg.eig(L)
lambdah = np.linalg.norm(eigen_values, np.inf)
yearly_decline = (lambdah - 1) * 100


survival = [0.38, 0.653, 0.85, 0.4, 0.589, 0.589, 0.589]

params = [1.05, 1.1, 1.15, 1.2, 0.95, 0.9, 0.85, 0.8]


change_list = []
for x in survival:
    for y in params:
        p_new = x*y
        change_list.append(p_new)

change_arr = np.vstack(change_list)

change_arr

P1_new = change_arr[0: 8]
P2_new = change_arr[8: 16]
P3_new = change_arr[16: 24]
P4_new = change_arr[24: 32]
P5_new = change_arr[32: 40]
P6_new = change_arr[40: 48]
P7_new = change_arr[48: 56]




rate_1 = []
L1 = np.copy(L)
for entry in P1_new:
    L1[1,0] = entry
    eigen_value1, eigen_vector1 = np.linalg.eig(L1)
    lambdah1 = np.linalg.norm(eigen_value1, np.inf)
    rate_1.append(lambdah1)

rate_2 = []
L2 = np.copy(L)
for entry in P2_new:
    L2[2,1] = entry
    eigen_value2, eigen_vector2 = np.linalg.eig(L2)
    lambdah2 = np.linalg.norm(eigen_value2, np.inf)
    rate_2.append(lambdah2)

rate_3 = []
L3 = np.copy(L)
for entry in P3_new:
    L3[3,2] = entry
    eigen_value3, eigen_vector3 = np.linalg.eig(L3)
    lambdah3 = np.linalg.norm(eigen_value3, np.inf)
    rate_3.append(lambdah3)

rate_4 = []
L4 = np.copy(L)
for entry in P4_new:
    L4[4,3] = entry
    eigen_value4, eigen_vector4 = np.linalg.eig(L4)
    lambdah4 = np.linalg.norm(eigen_value4, np.inf)
    rate_4.append(lambdah4)

rate_5 = []
L5 = np.copy(L)
for entry in P5_new:
    L5[5,4] = entry
    eigen_value5, eigen_vector5 = np.linalg.eig(L5)
    lambdah5 = np.linalg.norm(eigen_value5, np.inf)
    rate_5.append(lambdah5)
    
rate_6 = []
L6 = np.copy(L)
for entry in P6_new:
    L6[6,5] = entry
    eigen_value6, eigen_vector6 = np.linalg.eig(L6)
    lambdah6 = np.linalg.norm(eigen_value6, np.inf)
    rate_6.append(lambdah6)

rate_7 = []
L7 = np.copy(L)
for entry in P7_new:
    L7[7,6] = entry
    eigen_value7, eigen_vector7 = np.linalg.eig(L7)
    lambdah7 = np.linalg.norm(eigen_value7, np.inf)
    rate_7.append(lambdah7)



# change in lambdah
lam1 = np.vstack(rate_1).reshape(-1,1)
d_lam1 = lam1 - lambdah
#change in survivability
dP1 = P1_new -  survival[0]
#sensitivity
sense1 = d_lam1/dP1



lam2 = np.vstack(rate_2).reshape(-1, 1)
d_lam2 = lam2 - lambdah
#change in survivability
dP2 = P2_new -  survival[1]
#sensitivity
sense2 = d_lam2/dP2

lam3 = np.asarray(rate_3).reshape(-1, 1)
d_lam3 = lam3 - lambdah
#change in survivability
dP3 = P3_new -  survival[2]
#sensitivity
sense3 = d_lam3/dP3


lam4 = np.asarray(rate_4).reshape(-1, 1)
d_lam4 = lam4 - lambdah
#change in survivability
dP4 = P4_new -  survival[3]
#sensitivity
sense4 = d_lam4/dP4


lam5 = np.asarray(rate_5).reshape(-1, 1)
d_lam5 = lam5 - lambdah
#change in survivability
dP5 = P5_new -  survival[4]
#sensitivity
sense5 = d_lam5/dP5


lam6 = np.asarray(rate_6).reshape(-1, 1)
d_lam6 = lam6 - lambdah
#change in survivability
dP6 = P6_new -  survival[5]
#sensitivity
sense6 = d_lam6/dP6

lam7 = np.asarray(rate_7).reshape(-1, 1)
d_lam7 = lam7 - lambdah
#change in survivability
dP7 = P7_new -  survival[6]
#sensitivity
sense7 = d_lam7/dP7




end = time.time()

print("Decline rate: ", lambdah)
print("\nAnnual decline: ", yearly_decline)
print("\nYears of extinction: ", k)

print("\nSensitivity 1: ", sense1.round(4))
print("\nSensitivity 2: ", sense2.round(4))
print("\nSensitivity 3: ", sense3.round(4))
print("\nSensitivity 4: ", sense4.round(4))
print("\nSensitivity 5: ", sense5.round(4))
print("\nSensitivity 6: ", sense6.round(4))
print("\nSensitivity 7: ", sense7.round(4))

print ("\nExecution time: ", str(end-start))

