import numpy as np

M = 5
N = 2
K = 3
L = 4

aShape = [K,L,M]
bShape = [L,K,N]

a = np.arange(np.prod(aShape)).reshape(aShape)
b = np.arange(np.prod(bShape)).reshape(bShape)
c = np.tensordot(a,b, axes=([0,1],[1,0]))

print(c.shape)
print(c)
