import numpy
import scipy.sparse
from scipy.sparse.linalg import eigs
from scipy.linalg import eig
import matplotlib.pyplot as plt
import time
import bionetwork
import torch
import pandas


steps = 500

networkSize = 100
density = 0.2
errorSize = 1e-3
noise = 1e-10
initialEig = 1.2
targetEig = 0.95
expFactor = 20

#Generate random test data

network = scipy.sparse.random(networkSize, networkSize, density)
scipy.sparse.lil_matrix.setdiag(network, 0)
networkList = scipy.sparse.find(network)
networkList = numpy.array((networkList[1], networkList[0])) #we flip the network for rowise ordering

weights = numpy.random.normal(0, 0.5, len(networkList[0]))
A = scipy.sparse.csr_matrix((weights, networkList), shape=(networkSize, networkSize), dtype='float32')
e1, v = eigs(A, k=1)
A.data = initialEig * A.data/abs(e1)

storeEig = numpy.zeros(steps)

aBefore = A.copy()

#%%
A = aBefore.copy()



start = time.time()
weights = torch.autograd.Variable(torch.from_numpy(A.data), requires_grad=True)
scaleFactor = 1/numpy.exp(expFactor * targetEig)
optimizer = torch.optim.Adam([weights], lr=0.001, weight_decay=0)

for i in range(steps):
    optimizer.zero_grad()
    A.data = weights.detach().numpy()
    spectralRadius = bionetwork.spectralRadius.apply(weights, A, networkList)
    storeEig[i] = abs(spectralRadius.item())

    if spectralRadius.item()>targetEig:
        loss = errorSize * scaleFactor * (torch.exp(expFactor*spectralRadius)-1)
        #loss = errorSize * torch.abs(spectralRadius)
        #loss = errorSize * torch.square(spectralRadius)
    else:
        storeEig[i:] =storeEig[i]
        break
    loss.backward()
    optimizer.step()
    weights.data = weights.data + torch.randn(weights.shape) * noise

    if i % 5 == 0:
        print('i={:.0f}, e={:.4f}, l={:.5f}'.format(i, storeEig[i], loss.item()))

e = bionetwork.spectralRadius.apply(weights, A, networkList)

print('Time:',  time.time()-start)

aAfter = A.copy()
aAfter.data = weights.detach().numpy()

#%%
folder = 'figures/Figure 3/'

plt.rcParams["figure.figsize"] = (3,3)
plt.plot(range(len(storeEig)), storeEig, 'o')
plt.plot([0, steps], [targetEig, targetEig])
plt.ylabel('Spectral radius')
plt.xlabel('Steps')


# plt.figure()
# plt.plot(aBefore.data, aBefore.data-aAfter.data, 'o', color='black')
# plt.xlabel('weights before')
# plt.ylabel('delta')
# plt.title('Weights')
# plt.plot([-1, 1], [-0, 0])
# plt.grid(True)

plt.tight_layout()

plt.rcParams["figure.figsize"] = (6,6)
plt.figure()
eBefore = eig(aBefore.todense(), left = False, right=False)
eAfter = eig(aAfter.todense(), left = False, right=False)

dfBefore = pandas.DataFrame((eBefore.real, eBefore.imag), index=['Real', 'Imaginary']).T
dfAfter = pandas.DataFrame((eAfter.real, eAfter.imag), index=['Real', 'Imaginary']).T


axisLim = 1.2
for i in [1, 2]:
    plt.subplot(1,2,i)
    if i == 1:
        plt.scatter(eBefore.real, eBefore.imag)
        plt.title('Before')
    else:
        plt.scatter(eAfter.real, eAfter.imag)
        plt.title('After')


    tmp = plt.Circle((0, 0), 1, color='k', fill=False)
    plt.gca().add_patch(tmp)
    plt.gca().set_aspect('equal', 'box')
    plt.xlim([-axisLim, axisLim])
    plt.ylim([-axisLim, axisLim])
plt.tight_layout()

plt.savefig(folder + 'B.svg')
dfBefore.to_csv(folder + 'B_before.tsv', sep='\t')
dfAfter.to_csv(folder + 'B_after.tsv', sep='\t')

plt.rcParams["figure.figsize"] = (3,3)
plt.figure()
df = pandas.DataFrame((aAfter.data, aBefore.data), index=['Before', 'After']).T
plt.plot(aAfter.data, aBefore.data, 'o', alpha=0.2)
plt.xlabel('Weights after')
plt.ylabel('Weights before')
distance = numpy.sum(numpy.square(aAfter.data - aBefore.data))
#plt.text(-1, 1, 'SS = {:.3f}'.format(distance))
#plt.plot([-1, 1], [-1, 1], transform=ax2.transAxes)
plt.plot([-1, 1], [-1, 1], 'k-')
#plt.grid(True)
plt.gca().set_aspect('equal', 'box')
plt.savefig(folder + 'C.svg')
df.to_csv(folder + 'C.tsv', sep='\t')

#%%
plt.rcParams["figure.figsize"] = (8,4)
plt.figure()
numberOfSteps = 100
amplitude = 0.001
j = 0

resultsBefore = numpy.zeros((numberOfSteps+1, networkSize))
resultsBefore[0,:] = amplitude * numpy.random.randn(networkSize)
resultsBefore[0,0] = amplitude



for i in range(numberOfSteps):
    resultsBefore[i+1,:] = aBefore.dot(resultsBefore[i,:])

plt.subplot(1,2,1)

for i in range(numberOfSteps-1):
    plt.plot(resultsBefore[i:(i+2), j], resultsBefore[(i+1):(i+3), j], 'k-', alpha=0.2)

plt.plot(resultsBefore[0, j], resultsBefore[1, j], 'o', color='b')
plt.text(resultsBefore[0, j], resultsBefore[1, j], 'Start')
plt.plot(resultsBefore[-2, j], resultsBefore[-1, j], 'o', color='r')
plt.text(resultsBefore[-2, j], resultsBefore[-1, j], 'End')
plt.gca().set_aspect('equal', 'box')
plt.axis('equal')
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.xlabel('Deviation_n')
plt.ylabel('Deviation_n+1')


resultsAfter = numpy.zeros((numberOfSteps+1, networkSize))
resultsAfter[0,:] = resultsBefore[0,:].copy()

for i in range(numberOfSteps):
    resultsAfter[i+1,:] = aAfter.dot(resultsAfter[i,:])

plt.subplot(1,2,2)
colorProperty = ['k', 'b']
for i in range(numberOfSteps-1):
    plt.plot(resultsAfter[i:(i+2), j], resultsAfter[(i+1):(i+3), j], 'k-', alpha=0.2)

plt.plot(resultsAfter[0, j], resultsAfter[1, j], 'o', color='b')
plt.text(resultsAfter[0, j], resultsAfter[1, j], 'Start')
plt.plot(resultsAfter[-2, j], resultsAfter[-1, j], 'o', color='r')
plt.text(resultsAfter[-2, j], resultsAfter[-1, j], 'End')
plt.gca().set_aspect('equal', 'box')
plt.axis('equal')
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.xlabel('Deviation_n')
plt.ylabel('Deviation_n+1')
#plt.plot([0, 0.3], [0, 0.3], 'k-')
plt.tight_layout()
plt.savefig(folder + 'D.svg')
df = pandas.DataFrame((resultsBefore[:,j], resultsAfter[:,j]), index=['Before', 'After']).T
df.to_csv(folder + 'D_before.tsv', sep='\t')


