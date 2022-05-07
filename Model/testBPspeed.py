import torch
import bionetwork
import numpy
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas

def makeNetwork(curSize):
    parameters = bionetwork.trainingParameters(iterations=100, clipping=1)
    interactionsPerNode = 10
    sparsity = interactionsPerNode/curSize
    networkList, nodeNames = bionetwork.getRandomNet(curSize, sparsity)
    MOA = numpy.full(networkList.shape, False, dtype=bool)
    net = bionetwork.bionet(networkList, len(nodeNames), MOA, parameters, 'MML', torch.double)
    return net


folder = 'figures/SI Figure 10/'

networkSize = numpy.array([1000, 3000, 5000, 7000, 9000, 11000, 13000, 15000, 17000, 19000])
batchsize = 3
repeats = 10


criterion = torch.nn.MSELoss()


resultsF = numpy.zeros((len(networkSize), repeats))
resultsB = numpy.zeros((len(networkSize), repeats))

numberOfWeights = numpy.zeros((len(networkSize), repeats))

for i in range(len(networkSize)):
    for j in range(repeats):
        net = makeNetwork(networkSize[i])
        net.preScaleWeights(0.9)

        #start = time.perf_counter()
        input1 = torch.randn(batchsize, net.A.shape[0], dtype=torch.double, requires_grad=True)

        start = time.perf_counter()
        prediction1 = net(input1)
        resultsF[i, j] = time.perf_counter() - start

        predictionForLoss = torch.randn(input1.shape).double()
        predictionForLoss.requires_grad = False

        start = time.perf_counter()
        loss1 = criterion(prediction1, predictionForLoss)
        a = loss1.backward()
        resultsB[i, j] = time.perf_counter() - start

        numberOfWeights[i, j] = net.A.data.shape[0]
    print(networkSize[i], resultsF[i, :], resultsB[i, :])


#%%
plt.rcParams["figure.figsize"] = (3,3)
plt.figure()
meanTimeF = numpy.mean(resultsF, axis=1)
stdTimeF = numpy.std(resultsF, axis=1)

meanTimeB = numpy.mean(resultsB, axis=1)
stdTimeB = numpy.std(resultsB, axis=1)

X = networkSize.reshape(-1,1).repeat(repeats, axis=1).flatten().reshape(-1, 1)
Y = 0.5*(resultsF+resultsB).flatten().reshape(-1, 1)
reg = LinearRegression(fit_intercept=False).fit(X, Y)
print(reg.score(X, Y))

plt.errorbar(networkSize, meanTimeF, yerr=stdTimeF)
plt.errorbar(networkSize, meanTimeB, yerr=stdTimeB)
plt.legend({'Forward', 'Backward'}, frameon=False)
plt.xlabel('Number of nodes')
plt.ylabel('Time [s]')
plt.xlim([0, max(networkSize)+1000])
X = numpy.array([0, max(networkSize)])
Y = reg.coef_.flatten() * X
plt.plot(X, Y, 'k-')
plt.ylim(bottom=0)

plt.savefig(folder + 'A.svg')


replicates = ['Rep_{:d}'.format(x+1) for x in range(repeats)]
df = pandas.DataFrame(resultsF, columns=replicates, index=networkSize).T
df.to_csv(folder + 'A_Forward.tsv', sep='\t')

df = pandas.DataFrame(resultsB, columns=replicates, index=networkSize).T
df.to_csv(folder + 'A_Backward.tsv', sep='\t')


# plt.figure()
# plt.scatter(numberOfWeights.flatten(), resultsF.flatten())
# plt.scatter(numberOfWeights.flatten(), resultsB.flatten())
# plt.legend({'Forward', 'Backward'})
# plt.xlabel('Number of interactions')
# plt.ylabel('Time [s]')