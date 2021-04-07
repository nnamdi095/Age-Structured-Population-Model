import numpy as np
import time

from mpi4py import MPI  
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
dest = 0

start = time.time()


"""
#sample 1
L = np.array([[0, 5, 4], [0.15, 0, 0], [0, 0.5, 0]])
x_0 = np.array([3000, 440, 350])

"""

#sample 2
L = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 322.80],
              [0.966, 0.0, 0.0, 0.0, 0.0, 0.0], 
              [0.13, 0.01, 0.125, 0, 0, 3.448],  
              [0.007, 0.0, 0.125, 0.238, 0.0, 30.170],
              [0.008, 0.0, 0.0, 0.245, 0.167, 0.862],
              [0.0, 0.0, 0.0, 0.023, 0.75, 0.0]])


#AGE DISTRIBUTION AT, t = 0

x_0 = np.array([[1000], [1500], [200], [300], [600], [25]])


x_1 = np.copy(x_0)

kmax = 1000
k = 1

x_percent_old = np.zeros_like(x_0)



while k < kmax:
    Lk = np.linalg.matrix_power(L, k)
    x_t = np.matmul(Lk, x_0)
    x_percent = x_t/x_1
    
    #computing matrix v
    total_pop = np.sum(x_t)
    v = x_t/total_pop
    
    
    #initializing convergence condition
    percent_change = np.abs(np.subtract(x_percent_old, x_percent))
    if round(np.linalg.norm(percent_change, np.inf),5) == 10**(-5):
        break
     
    #reassingment
    x_percent_old = x_percent
    x_1 = x_t
    k += 1


end = time.time() 

print(k)
print("Execution time: ", str(end - start))

eigen_values, eigen_vectors = np.linalg.eig(L)

lambdah = np.linalg.norm(eigen_values, np.inf)
yearly_decline = (lambdah - 1) * 100

lambdah
yearly_decline