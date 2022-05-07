import torch
import numpy
import matplotlib.pyplot as plt
import bionetwork
import plotting
import pandas as pd
import seaborn as sns
import time
import copy
from scipy.stats import pearsonr
import pandas

def getSample(filePath, sampleName):
    file = open(filePath + 'name.txt', mode='r')
    curSample = file.read()
    file.close()
    select = sampleName == curSample
    return (select, curSample)


networkList, nodeNames, modeOfAction = bionetwork.loadNetwork('data/ToyNetRecurrent-Model.txt')
bionetParams = bionetwork.trainingParameters(iterations = 150, clipping=1, spectralLimit = 0.995, leak=0.01)



inputAmplitude = 1
projectionAmplitude = 1.3
simultaniousInput = 2


folder = 'figures/SI Figure 8/'

annotation = pd.read_csv('data/ToyNetRecurrent-Annotation.txt', sep='\t')
uniprot2gene = dict(zip(annotation['code'], annotation['name']))

inName = annotation.loc[annotation['ligand'],'code'].values
outName = annotation.loc[annotation['TF'],'code'].values
inName = numpy.intersect1d(nodeNames, inName)
outName = numpy.intersect1d(nodeNames, outName)
dictionary = dict(zip(nodeNames, list(range(len(nodeNames)))))
inNameGenes = [uniprot2gene[x] for x in inName]
outNameGenes = [uniprot2gene[x] for x in outName]

def uniformLoss(data):
    #uniform distribution range [0 1] u=0.5 var=1/12
    targetMean = 0.5
    targetVar= 1/12
    targetMin = 0.01
    targetMax = 0.99

    meanFactor = 0.5
    varFactor = 1
    rangeFactor = 1

    nodeMean = torch.mean(data, dim=0)
    nodeVar = torch.mean(torch.square(data-nodeMean), dim=0)
    minVal, _ = torch.min(data, dim=0)
    maxVal, _ = torch.max(data, dim=0)

    meanLoss = torch.sum(torch.square(nodeMean - targetMean))
    varLoss =  torch.sum(torch.square(nodeVar - targetVar))
    maxLoss = torch.sum(torch.square(maxVal - targetMax))
    minloss = torch.sum(torch.square(minVal- targetMin))

    loss = meanFactor * meanLoss + varFactor * varLoss + rangeFactor * minloss + rangeFactor * maxLoss
    return loss


def getCorrCoef(Yhat):
    #fact = 1.0 / (Yhat.size(1) - 1)
    epsilon = 1e-6
    fact = 1.0 / Yhat.shape[1]
    m = Yhat - torch.mean(Yhat, dim=1, keepdim=True)
    c = fact * m.matmul(m.t())
    d = torch.diagonal(c)
    #Sometimes corrCoff suffered from numeric instability during first iterations due to d sqrt(x)/dx = inf at x = 0
    stddev = torch.sqrt(d + epsilon)
    c = c/stddev.reshape(-1, 1)
    c = c/stddev.reshape(1, -1)
    c = c.fill_diagonal_(1)
    return c

#%%

model = bionetwork.model(networkList, nodeNames, modeOfAction, inputAmplitude, projectionAmplitude, inName, outName, bionetParams)
model.inputLayer.weights.requires_grad = False
model.projectionLayer.weights.requires_grad = False


parameterizedModel = bionetwork.model(networkList, nodeNames, modeOfAction, inputAmplitude, projectionAmplitude, inName, outName, bionetParams)
parameterizedModel = bionetwork.loadParam('data/ToyNetRecurrent-Parameters.txt', parameterizedModel, nodeNames)

#Add additional randomization compared with default to prevent leakage
#model.network.weights.data = 0.1 + 0.1 * torch.rand(model.network.weights.shape).double()
#model.network.weights.data[modeOfAction[1]] = -model.network.weights.data[modeOfAction[1]]
#model.network.bias.data = torch.zeros(model.network.bias.shape).double()


priorWeights = model.network.weights.clone().detach()
priorBias = model.network.bias.clone().detach()

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay=0)
resetState = optimizer.state.copy()

inputPoints = numpy.linspace(0, 1, num=10)
X = numpy.array(numpy.meshgrid(inputPoints, inputPoints)).T.reshape(-1, 2)
X = numpy.array(numpy.meshgrid(inputPoints, inputPoints)).T.reshape(-1, 2)
X = torch.tensor(X).double()
Y, YFull = model(X)

L2 = 1e-3
MoAFactor = 0.1
spectralFactor = 1e-3
spectralTarget = numpy.exp(numpy.log(10**-6)/bionetParams['iterations'])
lbRegularization = 0.001

maxIter = 5000

stats = plotting.initProgressObject(maxIter)

start = time.time()
e = 0
for e in range(e, maxIter):
    optimizer.zero_grad()

    Yhat, YhatFull = model(X)

    spectralRadiusLoss, spectralRadius = bionetwork.spectralLoss(model.network, YhatFull.detach(), model.network.weights)
    spectralRadiusLoss = spectralFactor * spectralRadiusLoss

    absFilter = torch.abs(model.network.weights.detach())>lbRegularization
    weightLoss = L2 * torch.sum(torch.square(model.network.weights[absFilter]))

    #conditionCorrelation = torch.mean(torch.square(getCorrCoef(YhatFull)+1e-6)) #Reduce correlation between conditions
    #TFcorrelation = torch.mean(torch.square(getCorrCoef(YhatFull.T)+1e-6)) #Reduce correlation between TFs

    uniformYfull = uniformLoss(YhatFull)
    uniformTF = uniformLoss(0.9 * Yhat) #0.9 makes sure that end results includes 1

    signConstraint = torch.sum(torch.abs(model.network.weights[model.network.getViolations()]))

    loss = uniformTF + 0.01 * uniformYfull + MoAFactor * signConstraint + weightLoss + spectralRadiusLoss # + 0.2 * TFcorrelation

    loss.backward()

    optimizer.step()

    stats['loss'][e] = uniformTF.item()
    stats['rate'][e] = optimizer.param_groups[0]['lr']
    stats['violations'][e] = torch.sum(model.network.getViolations()).item()
    stats['eig'][e] = spectralRadius.item()

    if e % 100 == 0:
        plotting.printStats(e, stats)

    if numpy.logical_and(e % 100 == 0, e>0):
        optimizer.state = resetState.copy()

model.network.weights.data[modeOfAction[0,:]] = torch.abs(model.network.weights.data[modeOfAction[0,:]])
model.network.weights.data[modeOfAction[1,:]] = -torch.abs(model.network.weights.data[modeOfAction[1,:]])


finalWeights = model.network.weights.detach()
finalBias = model.network.bias.detach()

print('Time:', time.time()-start)

referenceState = copy.deepcopy(model.network.state_dict())
#%%

#plotting.plotHeatmap(bionetwork.activation(X.numpy(), 0), inName)
conditions = bionetwork.generateConditionNames(X, inNameGenes)

plt.rcParams["figure.figsize"] = (3,3)
plt.figure()
T = numpy.array(range(stats['loss'].shape[0]))
#plt.plot(T, plotting.movingaverage(stats['loss'], 10))
plt.plot(T, plotting.movingaverage(stats['loss'], 10), plotting.movingaverage(stats['lossSTD'], 10))

plt.xlim([0, len(T)])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim(bottom=0)
#plt.savefig('synthNetModel/loss.svg')

plt.figure()
plt.plot([0, maxIter], [0, 0], 'black')
plt.plot(T, stats['rate'])
plt.ylabel('Learning rate')
plt.xlabel('Epoch')
#plt.savefig('synthNetModel/lr.svg')

plt.figure()
plt.plot([0, len(T)], [spectralTarget, spectralTarget], 'black', linestyle='--')
#plt.plot(T, stats['eig'])
#plt.plot(T, stats['eig'], stats['eigSTD'])
plt.plot(T, plotting.movingaverage(stats['eig'], 20))
plt.ylabel('Spectral radius')
plt.xlabel('Epoch')
plt.ylim([0, 1])
#plt.savefig('synthNetModel/sr.svg')


inputPoints = numpy.linspace(0, 1, num=10)
X = numpy.array(numpy.meshgrid(inputPoints, inputPoints)).T.reshape(-1, 2)
X = numpy.array(numpy.meshgrid(inputPoints, inputPoints)).T.reshape(-1, 2)
X = torch.tensor(X).double()
Y, YFull = model(X)


plt.figure()
h = plotting.contourPlot(X[:, 0].detach(), X[:, 1].detach(), Y.detach())
plt.xticks([0,5,10], labels=['0', '0.5', '1'])
plt.yticks([0,5,10], labels=['0', '0.5', '1'])
plt.gca().set_xticks(list(range(0,11)), minor=True)
plt.gca().set_yticks(list(range(0,11)), minor=True)
plt.xlabel(inName[0])
plt.ylabel(inName[1])
plt.gcf().axes[1].set_label(outName[0])
plt.gca().xaxis.grid(True, 'both', linewidth=1, color=[0,0,0])
plt.gca().yaxis.grid(True, 'both', linewidth=1, color=[0,0,0])
plt.savefig(folder + 'B.svg')
h.to_csv(folder + 'B.tsv', sep='\t')

plt.figure()
plt.hist(Y.detach().numpy().flatten())


plt.rcParams["figure.figsize"] = (3,3)
plt.figure()
referenceWeights = parameterizedModel.network.weights.detach().numpy().flatten()
fittedWeights = model.network.weights.detach().numpy().flatten()
referenceBias = parameterizedModel.network.bias.detach().numpy().flatten()
fittedBias = model.network.bias.detach().numpy().flatten()

dfWeight = pandas.DataFrame((fittedWeights, referenceWeights), index=['fit', 'reference']).T
dfBias = pandas.DataFrame((fittedBias, referenceBias), index=['fit', 'reference']).T
plt.scatter(dfWeight['fit'], dfWeight['reference'], alpha=0.8)
plt.scatter(dfBias['fit'], dfBias['reference'], alpha=0.8)
plt.legend(numpy.array(['Weights', 'Bias']), frameon=False)
plotting.lineOfIdentity()
plt.xlabel('Automatic')
plt.ylabel('Manual')
plt.gca().axis('equal')

r, p = pearsonr(referenceWeights, fittedWeights)
plt.text(1, -1, 'r {:.2f}\np {:.2e}'.format(r, p))

plt.savefig(folder + 'D.svg')
dfWeight.to_csv(folder + 'D_weight.tsv', sep='\t')
dfBias.to_csv(folder + 'D_bias.tsv', sep='\t')



plt.figure()
referenceY, YFull = parameterizedModel(X)
plt.scatter(Y.detach().numpy().flatten(), referenceY.detach().numpy().flatten(), color=[0.5,0.5,0.5], alpha=0.8)
plotting.lineOfIdentity()
r, p = pearsonr(Y.detach().flatten(), referenceY.detach().flatten())
plt.text(0, 0.9, 'r {:.2f}\np {:.2e}'.format(r, p))
plt.xlabel('Automatic')
plt.ylabel('Manual')
plt.gca().axis('equal')
plt.savefig(folder + 'C.svg')
df = pandas.DataFrame((Y.detach().numpy().flatten(), referenceY.detach().numpy().flatten()), index=['Automatic', 'Manual']).T
df.to_csv(folder + 'C.tsv', sep='\t')

plt.figure()
#Test alternative parametrization
modifiedState = copy.deepcopy(referenceState)
mu = numpy.mean(modifiedState['weights'].detach().numpy())
sig = numpy.std(modifiedState['weights'].detach().numpy())
modifiedState['weights'] = torch.tensor(numpy.random.normal(mu, sig, modifiedState['weights'].shape[0]))
#modifiedState['weights'] = modifiedState['weights'][numpy.random.permutation(modifiedState['weights'].shape[0])]
#reversedSign = model.network.getViolations(modifiedState['weights'])
#modifiedState['weights'][reversedSign] = -modifiedState['weights'][reversedSign]
#modifiedState['bias'] = modifiedState['bias'][numpy.random.permutation(modifiedState['bias'].shape[0]),:]
perturbedModel = copy.deepcopy(model)
perturbedModel.network.load_state_dict(modifiedState)
Y, YFull = perturbedModel(X)

h = plotting.contourPlot(X[:, 0].detach(), X[:, 1].detach(), Y.detach())
plt.xticks([0,5,10], labels=['0', '0.5', '1'])
plt.yticks([0,5,10], labels=['0', '0.5', '1'])
plt.gca().set_xticks(list(range(0,11)), minor=True)
plt.gca().set_yticks(list(range(0,11)), minor=True)
plt.xlabel(inName[0])
plt.ylabel(inName[1])
plt.gcf().axes[1].set_label(outName[0])
plt.gca().xaxis.grid(True, 'both', linewidth=1, color=[0,0,0])
plt.gca().yaxis.grid(True, 'both', linewidth=1, color=[0,0,0])
plt.savefig(folder + 'A.svg')
h.to_csv(folder + 'A.tsv', sep='\t')
#bionetwork.saveParam(model, nodeNames, 'synthNetModel/equationParams.txt')
