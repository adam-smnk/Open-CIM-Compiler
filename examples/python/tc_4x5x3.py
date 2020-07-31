import numpy as np

M = 3;
N = 2;
K = 4;
L = 5;
P = 3;
Q = 4;

aShape = [P,K,L,M]
bShape = [K,Q,L,N,P]

a = np.arange(np.prod(aShape)).reshape(aShape)
b = np.arange(np.prod(bShape)).reshape(bShape)
c = np.tensordot(a,b, axes=([0,1,2],[4,0,2]))

print(c.shape)
print(c)

