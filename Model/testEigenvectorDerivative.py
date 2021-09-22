import numpy
import scipy.sparse
import matplotlib.pyplot as plt
import bionetwork
import torch
from scipy.linalg import eig
from scipy.sparse.linalg import eigs
import plotting

networkSize = 100
deltaSize = numpy.double(1e-10)
initialEig = 0.95
density = 0.1

#Generate random test data
network = scipy.sparse.random(networkSize, networkSize, density)
scipy.sparse.lil_matrix.setdiag(network, 0)
networkList = scipy.sparse.find(network)
networkList = numpy.array((networkList[1], networkList[0])) #we flip the network for rowise ordering

weights = numpy.random.normal(0, 0.5, len(networkList[0]))
A = scipy.sparse.csr_matrix((weights, networkList), shape=(networkSize, networkSize), dtype='float64')
e = eig(A.todense(), left = False, right=False)
e = max(abs(e))
A.data = initialEig * A.data/e


def fullProblemParams():
    e, w, v = eig(A.todense(), left = True, right=True)
    selected = numpy.argmax(numpy.abs(e))
    selected = (abs(e) > (abs(e[selected]) - 1e-10))
    eValue = e[selected]
    w = w[:,selected]
    v = v[:,selected]

    # if numpy.sum(selected) == 1:
    #     eValue = e[selected]
    #     w = w[:,selected]
    #     v = v[:,selected]
    # else:
    #     eValue = numpy.sum(e[selected])
    #     w = numpy.sum(w[:,selected], axis=1, keepdims=True)
    #     v = numpy.sum(v[:,selected], axis=1, keepdims=True)
    #     w = w/numpy.linalg.norm(w, ord=2)
    #     v = v/numpy.linalg.norm(v, ord=2)
    return eValue, w, v

def eigsProblemParams(A):
    e, v = eigs(A, k=1, which='LM')
    eValue = e[0]
    e2, w = eigs(A.T, k=1, sigma=eValue, OPpart='r')
    print(e, e2)
    return eValue, w, v


def calculateDelta(e, w, v, networkList):
    divisor = w.T.dot(v).flatten()
    delta = numpy.multiply(w[networkList[0]], v[networkList[1]])/divisor
    direction = e/abs(e)
    delta = (delta/direction).real
    return delta

#%%
weights = torch.from_numpy(A.data)
originalWeights = weights.detach().clone()
storeEig = numpy.zeros(len(weights), dtype='float64')

weights = torch.autograd.Variable(weights, requires_grad=True)
refEig = bionetwork.spectralRadius.apply(weights, A, networkList)
eigReg = torch.square(refEig) * 0.5/refEig.item()
eigReg.backward()
calculatedDelta = weights.grad.data

# A.data = originalWeights.detach().numpy()
# eValue, w, v = eigsProblemParams(A)
# calculatedDelta = calculateDelta(eValue, w, v, networkList)

e = eig(A.todense(), left = False, right=False)
eValue = e[numpy.argmax(numpy.abs(e))]

#print('Verify that these numbers are the same:', eValue, refEig.item())
for i in range(len(storeEig)):
    if (i % 100) == 0:
        print(i)
    A.data = originalWeights.clone().numpy()
    A.data[i] = A.data[i] + deltaSize
    e = eig(A.todense(), left=False, right=False)
    storeEig[i] = numpy.max(numpy.abs(e))

deltaY = storeEig - numpy.abs(eValue)
deltaX = (deltaSize - 0.0)
empiricalDerivative = deltaY/deltaX

#%%
plt.rcParams["figure.figsize"] = (6,6)
A.data = originalWeights.detach().numpy()
plt.scatter(e.real, e.imag)
plt.scatter(eValue.real, eValue.imag)
tmp = plt.Circle((0, 0), numpy.abs(eValue), color='tab:orange', fill=False)
plt.gca().add_patch(tmp)
tmp = plt.Circle((0, 0), 1, color='k', fill=False)
plt.gca().add_patch(tmp)
plt.xlabel('Real')
plt.ylabel('Imag')
#%%
plt.figure()
plt.scatter(calculatedDelta, empiricalDerivative)
plotting.lineOfIdentity()
plt.xlabel('Calculated from eigen vector')
plt.ylabel('Numerical')


