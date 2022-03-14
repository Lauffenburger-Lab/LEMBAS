import torch
import numpy
import matplotlib.pyplot as plt
import bionetwork
import plotting
import saveSimulations
import pandas
from scipy.stats import pearsonr

#import pandas as pd
batchSize = 1

#Load network
networkList, nodeNames, modeOfAction = bionetwork.loadNetwork("data/toyNFKB-Model.txt")
annotation = pandas.read_csv('data/toyNFKB-Annotation.txt', sep='\t', low_memory=False)
bionetParams = bionetwork.trainingParameters(iterations = 10, clipping=1, leak=0.01)

inputAmplitude = 1
projectionAmplitude = 1.3


inName = annotation.loc[annotation.ligand, 'name'].values
outName = annotation.loc[annotation.TF, 'name'].values

model = bionetwork.model(networkList, nodeNames, modeOfAction, inputAmplitude, projectionAmplitude, inName, outName, bionetParams)
model.inputLayer.weights.requires_grad = False
model.projectionLayer.weights.requires_grad = False
model.network.bias.requires_grad = False
model.network.bias[:] = 0

parameterizedModel = bionetwork.model(networkList, nodeNames, modeOfAction, inputAmplitude, projectionAmplitude, inName, outName, bionetParams)
parameterizedModel = bionetwork.loadParam('data/toyNFKB-Parameters.txt', parameterizedModel, nodeNames)
allWeights = parameterizedModel.network.weights.data





#%%
#simulate dynamical process
plt.rcParams["figure.figsize"] = (5,4)
decayRate = numpy.array([0.2, 0.11])
productionRate = numpy.array([0.05, 0.2])

delay = 10
T = list(numpy.arange(-10, 120))
samplePoints = numpy.array([-1, 10, 30, 90])
#samplePoints = numpy.array([-1, 10, 40])

sampleValues = numpy.zeros((0, 3))

state = numpy.zeros((len(T), 3))
state[numpy.isin(T, numpy.arange(70)), 0] = 1 #TNF

for i in range(1, len(T)):     
    curX = torch.tensor([state[i-1, 0], state[i-1, 2]]).reshape(1,-1).clone()
    curY, _ = parameterizedModel(curX.double())
    
    production = numpy.zeros(2)
    production[0] = curY.item()
    if i > delay:
        production[1] = 4 * state[i-delay, 1]
    production = production * productionRate

    degradation = decayRate * state[i-1,1:3]
    state[i, 1:3] = state[i-1, 1:3] + production - degradation

    if numpy.any(samplePoints== T[i]):
        sampleValues = numpy.append(sampleValues, state[i, :].reshape(1,-1), axis=0)


normState = state/numpy.max(state, axis=0)
normSample = sampleValues/numpy.max(state, axis=0)

normalizedTime = T/max(T)
normalizedSamplePoints = samplePoints/max(T)

plt.plot(normalizedTime, normState[:,0], '--')
plt.plot(normalizedTime, normState[:,1])
plt.plot(normalizedTime, normState[:,2])
plt.scatter(normalizedSamplePoints, normSample[:,1], color=[0,0,0])

plt.xlabel('time')
plt.ylabel('amplitude')
plt.legend(['TNF', 'NFKBn',  'p100', 'sample'], frameon=False)
plt.savefig('figures/NFKB/trajectory.svg')

conditionLabels = ['ctrl early', 'TNF early', 'TNF late', 'ctrl late']

normSample2 = sampleValues/numpy.max(sampleValues, axis=0)

X = torch.tensor(normSample2[:,[0, 2]])
Y =  torch.tensor(normSample2[:,1]).reshape(-1,1)
N = X.shape[0]


#%%
#Generate data
X_ctr_early = [0, 0]
X_TNF_early = [1, 0]
X_TNF_late = [1, 1]
X_ctrl_late= [0,  1]
Xtest = torch.tensor([X_ctr_early, X_TNF_early, X_TNF_late, X_ctrl_late]).double()
Ytest, YFull = parameterizedModel(Xtest)


#Setup optimizer
batchSize = 3
maxIter = 1000
MoAFactor = 1
L2beta = 1e-5
criterion = torch.nn.MSELoss(reduction='mean')

optimizer = torch.optim.Adam(model.parameters(), lr=1) #, eps=1e-10 , amsgrad=True
resetState = optimizer.state.copy()

stats = plotting.initProgressObject(maxIter)

mLoss =  criterion(torch.mean(Y, dim=0)*torch.ones(Y.shape), Y)

curState = torch.rand((X.shape[0], model.network.bias.shape[0]), dtype=torch.double, requires_grad=False)

e = 0
for e in range(e, maxIter):
    curLr = bionetwork.oneCycle(e, maxIter, maxHeight = 2e-3, minHeight = 1e-5, peak = 100)
    optimizer.param_groups[0]['lr'] = curLr

    curLoss = []
    #curEig = []
    trainloader = bionetwork.getSamples(N, batchSize)
    for dataIndex in trainloader:
        model.train()
        optimizer.zero_grad()   
        dataIn = X[dataIndex, :].view(len(dataIndex), X.shape[1])
        dataOut = Y[dataIndex, :].view(len(dataIndex), Y.shape[1])

        Yhat, YhatFull = model(dataIn)
        fitLoss = criterion(dataOut, Yhat)
        curState[dataIndex,:] = YhatFull.detach()

        #signConstraint = MoAFactor * torch.sum(torch.abs(model.network.weights[model.network.getViolations(model.network.weights)]))

        stateLoss = 1e-5 * bionetwork.uniformLoss(curState, dataIndex, YhatFull, maxConstraintFactor = 1)

        weightLoss = L2beta * torch.sum(torch.square(model.network.weights))

        loss = fitLoss + weightLoss + stateLoss  #signConstraint

        loss.backward()

        optimizer.step()

        curLoss.append(fitLoss.item())


    stats = plotting.storeProgress(stats, e, loss=curLoss, lr=curLr, violations=torch.sum(model.network.getViolations(model.network.weights)).item())

    model.eval()
    stats['violations'][e] = torch.sum(model.network.getViolations()).item()
    Yhat, YhatFull = model(Xtest)
    fitLoss = criterion(Ytest, Yhat)
    stats['test'][e] = fitLoss.item()


    if e % 50 == 0:
        plotting.printStats(e, stats)

    if numpy.logical_and(e % 100 == 0, e>0):
        optimizer.state = resetState.copy()

plotting.finishProgress(stats)


#%%
model.eval()

plt.rcParams["figure.figsize"] = (3,3)
plt.figure()

T = numpy.array(range(stats['loss'].shape[0]))
plt.semilogy(T, plotting.movingaverage(stats['loss'], 10))
#plt.semilogy(T, plotting.movingaverage(stats['test'], 10))
plt.plot([0, len(T)], numpy.array([1, 1])*mLoss.item(), 'black', linestyle='--')
plt.xlim([0, len(T)])
plt.ylim([1e-6, 1])
#plt.text(T[-1], 1e-6, 'train {:.5f}\ntest {:.5f}'.format(stats['loss'][-1], stats['test'][-1]), ha='right', va='bottom')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('figures/NFKB/training.svg')

#plt.legend(numpy.array(['Train', 'Test', 'Mean']), frameon=False)

# plt.figure()
# plt.plot([0, maxIter], [0, 0], 'black')
# plt.plot(T, stats['rate'])
# #plt.plot(T, stats['rate']*plotting.movingaverage(stats['loss'], 50)/numpy.nanmean(stats['loss']))
# plt.ylabel('Learning rate')
# plt.xlabel('Epoch')


plt.figure()
plt.scatter(model.network.weights.detach().numpy(), parameterizedModel.network.weights.detach().detach().numpy())
plotting.lineOfIdentity()
plt.xlabel('Fit weights')
plt.ylabel('Reference weights')
#plt.gca().axis('equal')
plt.ylim([-1, 1.5])
plt.xlim([-1, 1.5])
plt.gca().set_xticks([-1,0,1.5])
plt.gca().set_yticks([-1,0,1.5])
r, p = pearsonr(model.network.weights.detach().numpy().flatten(), parameterizedModel.network.weights.detach().numpy().flatten())
plt.text(-0.9, 1, 'r {:.2f}\np {:.2e}'.format(r, p))
plt.savefig('figures/NFKB/parameters.svg')
#saveSimulations.save('simulations', 'toynetNFKB', {'X':X, 'Y':Y, 'Model':model})