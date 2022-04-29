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
networkList, nodeNames, modeOfAction = bionetwork.loadNetwork("data/ToyNetRecurrent-Model.txt")
annotation = pandas.read_csv('data/ToyNetRecurrent-Annotation.txt', sep='\t', low_memory=False)
bionetParams = bionetwork.trainingParameters(iterations = 150, clipping=1, leak=0.01)

inputAmplitude = 1
projectionAmplitude = 1.3


inName = annotation.loc[annotation.ligand, 'name'].values
outName = annotation.loc[annotation.TF, 'name'].values

model = bionetwork.model(networkList, nodeNames, modeOfAction, inputAmplitude, projectionAmplitude, inName, outName, bionetParams)
model.inputLayer.weights.requires_grad = False
model.projectionLayer.weights.requires_grad = False


parameterizedModel = bionetwork.model(networkList, nodeNames, modeOfAction, inputAmplitude, projectionAmplitude, inName, outName, bionetParams)
parameterizedModel = bionetwork.loadParam('data/ToyNetRecurrent-Parameters.txt', parameterizedModel, nodeNames)

allWeights = parameterizedModel.network.weights.data

#Randomly initalize model
#model.network.weights.data = 0.1 * (torch.rand(4)-0.5)
#model.network.bias.data = torch.zeros((5,1))

#Generate data
inputPoints = numpy.linspace(0, 1, num=10)
X = numpy.array(numpy.meshgrid(inputPoints, inputPoints)).T.reshape(-1, 2)
X = torch.tensor(X).double()
Y, YFull = parameterizedModel(X)
Y = Y.detach()

referenceSpectralRadius = bionetwork.getAllSpectralRadius(parameterizedModel, YFull)
print(max(referenceSpectralRadius))


testSelection = 'Random'
printSelection = True
if testSelection=='Random':
    nrOfTest = 20
    testSamples = numpy.random.permutation(len(Y))[0:nrOfTest]
elif testSelection=='VerticalStripe':
    condition = numpy.logical_or((X[:,0] == X[40,0]), (X[:,0] == X[50,0]))
    testSamples = numpy.argwhere(condition).flatten()
elif testSelection=='HorizontalStripe':
    condition = numpy.logical_or((X[:,1] == X[4,1]), (X[:,1] == X[5,1]))
    testSamples = numpy.argwhere(condition).flatten()
elif testSelection=='Boundary':
     testSamples = numpy.array([0, 1, 10, 11, 12, 22, 23, 33, 34, 44, 45, 54, 55, 64, 65, 74, 75, 84, 85, 94, 95])
elif testSelection=='SWQ':
     testSamples = numpy.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31, 32, 33, 34])

trainSamples = numpy.array(range(X.shape[0]))
trainSamples= trainSamples[numpy.isin(trainSamples, testSamples)==False]


if printSelection:
    YTestPattern = torch.zeros(Y.shape)
    YTestPattern[testSamples] = 1
    plotting.contourPlot(X[:, 0], X[:, 1], YTestPattern)
    plt.xticks([0,5,10], labels=['0', '0.5', '1'])
    plt.yticks([0,5,10], labels=['0', '0.5', '1'])
    plt.gca().set_xticks(list(range(0,11)), minor=True)
    plt.gca().set_yticks(list(range(0,11)), minor=True)
    plt.xlabel(inName[0])
    plt.ylabel(inName[1])
    plt.gcf().axes[1].set_label(outName[0])
    plt.gca().xaxis.grid(True, 'both', linewidth=1, color=[0,0,0])
    plt.gca().yaxis.grid(True, 'both', linewidth=1, color=[0,0,0])



Xtest = X[testSamples,:]
Ytest = Y[testSamples]
X = X[trainSamples,:]
Y = Y[trainSamples]
N = X.shape[0]


trainloader = torch.utils.data.DataLoader(range(N), batch_size=batchSize, shuffle=True)

#%%
#Setup optimizer
maxIter = 1000
MoAFactor = 1
L2beta = 1e-8
spectralFactor = 0
criterion = torch.nn.MSELoss(reduction='mean')


optimizer = torch.optim.Adam(model.parameters(), lr=1) #, eps=1e-10 , amsgrad=True
resetState = optimizer.state.copy()

storeWeights = torch.zeros(maxIter, model.network.weights.shape[0])
storeBias = torch.zeros(maxIter, model.network.bias.shape[0])
stats = plotting.initProgressObject(maxIter)

mLoss =  criterion(torch.mean(Y, dim=0)*torch.ones(Y.shape), Y)

curState = torch.rand((X.shape[0], model.network.bias.shape[0]), dtype=torch.double, requires_grad=False)

e = 0
for e in range(e, maxIter):
    curLr = bionetwork.oneCycle(e, maxIter, maxHeight = 2e-3, minHeight = 1e-8, peak = 200)
    optimizer.param_groups[0]['lr'] = curLr

    storeWeights[e,:] = model.network.weights.detach().flatten()
    storeBias[e,:] = model.network.bias.detach().flatten()

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




        signConstraint = MoAFactor * torch.sum(torch.abs(model.network.weights[model.network.getViolations(model.network.weights)]))
        ligandConstraint = 1e-1 * torch.sum(torch.square(model.network.bias[model.inputLayer.nodeOrder,0]))

        biasLoss = L2beta * torch.sum(torch.square(model.network.bias))
        weightLoss = L2beta * torch.sum(torch.square(model.network.weights))

        #spectralRadiusLoss, spectralRadius = bionetwork.spectralLoss(model, YhatFull, model.network.weights, expFactor = 21)

        loss = fitLoss + signConstraint + ligandConstraint + biasLoss + weightLoss #+ spectralFactor * spectralRadiusLoss

        loss.backward()

        optimizer.step()

        #curEig.append(spectralRadius.item())
        curLoss.append(fitLoss.item())


    model.eval()
    stats['violations'][e] = torch.sum(model.network.getViolations()).item()
    Yhat, YhatFull = model(Xtest)
    fitLoss = criterion(Ytest, Yhat)
    stats['test'][e] = fitLoss.item()

    stats = plotting.storeProgress(stats, e, loss=curLoss, lr=curLr, violations=torch.sum(model.network.getViolations(model.network.weights)).item())

    if e % 50 == 0:
        plotting.printStats(e, stats)

    if numpy.logical_and(e % 100 == 0, e>0):
        optimizer.state = resetState.copy()


plotting.finishProgress(stats)


#%%
folder = 'figures/Figure 2/'
plt.figure()
#plt.scatter(YFull[:,numpy.argwhere(numpy.isin(nodeNames, 'L1'))].detach().numpy(), YFull[:,numpy.argwhere(numpy.isin(nodeNames, 'S1'))].detach().numpy())
#plt.figure()

#Plot stats
# plt.rcParams["figure.figsize"] = (6,3)
# plt.subplot(1, 2, 1)
#Data
plt.rcParams["figure.figsize"] = (3,3)
df = plotting.contourPlot(torch.cat((X[:, 0], Xtest[:, 0])), torch.cat((X[:, 1], Xtest[:, 1])), torch.cat((Y, Ytest)))
plt.xticks([0,5,10], labels=['0', '0.5', '1'])
plt.yticks([0,5,10], labels=['0', '0.5', '1'])
plt.gca().set_xticks(list(range(0,11)), minor=True)
plt.gca().set_yticks(list(range(0,11)), minor=True)
plt.xlabel(inName[0])
plt.ylabel(inName[1])
plt.gcf().axes[1].set_label(outName[0])
plt.gca().xaxis.grid(True, 'both', linewidth=1, color=[0,0,0])
plt.gca().yaxis.grid(True, 'both', linewidth=1, color=[0,0,0])

plt.savefig(folder + 'D.svg')
df.to_csv(folder + 'D.tsv', sep='\t')


#Fit
plt.figure()
# plt.subplot(1, 2, 2)
# plt.rcParams["figure.figsize"] = (6,3)
Yhat, YhatFull = model(torch.cat((X, Xtest)))
plotting.contourPlot(torch.cat((X[:, 0], Xtest[:, 0])), torch.cat((X[:, 1], Xtest[:, 1])), Yhat.detach())
plt.xticks([0,5,10], labels=['0', '0.5', '1'])
plt.yticks([0,5,10], labels=['0', '0.5', '1'])
plt.gca().set_xticks(list(range(0,11)), minor=True)
plt.gca().set_yticks(list(range(0,11)), minor=True)
plt.xlabel(inName[0])
plt.ylabel(inName[1])
plt.gcf().axes[1].set_label(outName[0])
plt.gca().xaxis.grid(True, 'both', linewidth=1, color=[0,0,0])
plt.gca().yaxis.grid(True, 'both', linewidth=1, color=[0,0,0])



plt.rcParams["figure.figsize"] = (3,3)
plt.figure()

T = numpy.array(range(stats['loss'].shape[0]))
df = pandas.DataFrame((plotting.movingaverage(stats['loss'], 10), plotting.movingaverage(stats['test'], 10)), columns=T, index=['loss', 'test']).T
plt.semilogy(T, df['loss'])
plt.semilogy(T, df['test'])
plt.plot([0, len(T)], numpy.array([1, 1])*mLoss.item(), 'black', linestyle='--')
plt.xlim([0, len(T)])
plt.ylim([1e-6, 1])
plt.text(T[-1], 1e-6, 'train {:.5f}\ntest {:.5f}'.format(stats['loss'][-1], stats['test'][-1]), ha='right', va='bottom')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(numpy.array(['Train', 'Test', 'Mean']), frameon=False)
plt.savefig(folder + 'E.svg')
df.to_csv(folder + 'E.tsv', sep='\t')

plt.figure()
plt.plot([0, maxIter], [0, 0], 'black')
plt.plot(T, stats['rate'])
#plt.plot(T, stats['rate']*plotting.movingaverage(stats['loss'], 50)/numpy.nanmean(stats['loss']))
plt.ylabel('Learning rate')
plt.xlabel('Epoch')


# plt.figure()
# plt.plot([0, maxIter], [1, 1], 'black')
# plt.plot([0, len(T)], numpy.mean(referenceSpectralRadius) * numpy.array([1, 1]), 'black', linestyle='--')
# plt.plot([0, len(T)], numpy.max(referenceSpectralRadius) * numpy.array([1, 1]), 'black', linestyle='--')
# plotting.shadePlot(T, plotting.movingaverage(stats['eig'], 5), plotting.movingaverage(stats['eigSTD'], 5))
# plt.ylabel('Spectral radius')
# plt.xlabel('Epoch')


plt.figure()
colorOrder = plt.rcParams['axes.prop_cycle'].by_key()['color']
referenceWeights = parameterizedModel.network.weights.data.numpy()

for i in range(len(referenceWeights)):
    plt.plot(T, storeWeights[:, i].numpy(), color=colorOrder[i])
    plt.plot(T[-1], referenceWeights[i], 'o', color=colorOrder[i], alpha=0.5)
plt.plot([0, maxIter], [0, 0], 'black')
plt.xlabel('Epoch')
plt.ylabel('Weight')


plt.figure()
referenceBias = parameterizedModel.network.bias.data.numpy()
for i in range(len(referenceBias)):
    plt.plot(T, storeBias[:, i].numpy(), color=colorOrder[i])
    plt.plot(T[-1], referenceBias[i], 'o', color=colorOrder[i], alpha=0.5)
plt.plot([0, maxIter], [0, 0], 'black')
plt.xlabel('Epoch')
plt.ylabel('Bias')



plt.figure()
Yhat, YhatFull = model(X)
YhatTest = model(Xtest)[0].detach()
Yhat = Yhat.detach()
dfTrain = pandas.DataFrame((Yhat.detach().numpy().flatten(), Y.detach().numpy().flatten()), index=['fit', 'reference']).T
dfTest = pandas.DataFrame((YhatTest.detach().numpy().flatten(), Ytest.detach().numpy().flatten()), index=['fit', 'reference']).T
plt.scatter(dfTrain['fit'], dfTrain['reference'])
plt.scatter(dfTest['fit'], dfTest['reference'])
plt.legend(numpy.array(['Train', 'Test']), frameon=False)
plotting.lineOfIdentity()
plt.xlabel('Fit')
plt.ylabel('Reference data')
plt.gca().axis('equal')
plt.gca().set_xticks([0,0.5,1])
plt.gca().set_yticks([0,0.5,1])
r, p = pearsonr(YhatTest.detach().numpy().flatten(), Ytest.detach().numpy().flatten())
plt.text(0.7, 0.1, 'r {:.2f}\np {:.2e}'.format(r, p))
plt.savefig(folder + 'F.svg')
dfTrain.to_csv(folder + 'F_train.tsv', sep='\t')
dfTest.to_csv(folder + 'F_test.tsv', sep='\t')

plt.figure()
dfWeight = pandas.DataFrame((storeWeights[-1, :].numpy(), referenceWeights), index=['fit', 'reference']).T
dfBias = pandas.DataFrame((storeBias[-1, :].numpy(), referenceBias.flatten()), index=['fit', 'reference']).T
plt.scatter(dfWeight['fit'], dfWeight['reference'])
plt.scatter(dfBias['fit'], dfBias['reference'])
plt.legend(numpy.array(['Weights', 'Bias']), frameon=False)
plotting.lineOfIdentity()
plt.xlabel('Fitted parameters')
plt.ylabel('Reference parameters')
plt.gca().axis('equal')

plt.savefig(folder + 'G.svg')
dfWeight.to_csv(folder + 'G_weight.tsv', sep='\t')
dfBias.to_csv(folder + 'G_bias.tsv', sep='\t')



saveSimulations.save('simulations', 'toynetRecurrent', {'X':X, 'Y':Y, 'Model':model})