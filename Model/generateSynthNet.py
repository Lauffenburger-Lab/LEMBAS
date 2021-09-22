import torch
import numpy
import matplotlib.pyplot as plt
import bionetwork
import plotting
import pandas as pd
import seaborn as sns
import time

def getSample(filePath, sampleName):
    file = open(filePath + 'name.txt', mode='r')
    curSample = file.read()
    file.close()
    select = sampleName == curSample
    return (select, curSample)


networkList, nodeNames, modeOfAction = bionetwork.loadNetwork('data/KEGGnet-Model.tsv')
bionetParams = bionetwork.trainingParameters(iterations = 150, clipping=1, spectralLimit = 0.995, leak=0.01)




N = 200
simultaniousInput = 5
inputAmplitude = 3
projectionAmplitude = 1.2


annotation = pd.read_csv('data/KEGGnet-Annotation.tsv', sep='\t')
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

model = bionetwork.model(networkList, nodeNames, modeOfAction, inputAmplitude, projectionAmplitude, inName, outName, bionetParams, torch.double)
model.inputLayer.weights.requires_grad = False
model.projectionLayer.weights.requires_grad = False

#Add additional randomization compared with default to prevent leakage
model.network.weights.data = 3 * torch.rand(model.network.weights.shape).double()
model.network.weights.data[modeOfAction[1]] = -model.network.weights.data[modeOfAction[1]]
model.network.bias.data = 0.01 * torch.rand(model.network.bias.shape).double()
model.network.preScaleWeights()


priorWeights = model.network.weights.clone().detach()
priorBias = model.network.bias.clone().detach()

optimizer = torch.optim.Adam(model.parameters(), lr = 2e-3, weight_decay=0)
resetState = optimizer.state.copy()

MoAFactor = 2
spectralFactor = 1e-2
spectralTarget = numpy.exp(numpy.log(10**-4)/bionetParams['iterations'])

maxIter = 5000

stats = plotting.initProgressObject(maxIter)

start = time.time()
e = 0
for e in range(e, maxIter):
    #optimizer.param_groups[0]['lr'] = oneCycle(e, maxIter)
    optimizer.zero_grad()
    model.network.weights.data = model.network.weights.data + torch.randn(model.network.weights.shape) * 1e-8 #breaks symetries

    #Test sampling new random input for each iter
    X = bionetwork.generateRandomInput(model, N, simultaniousInput)
    Yhat, YhatFull = model(X)

    spectralRadiusLoss, spectralRadius = bionetwork.spectralLoss(model, YhatFull, model.network.weights, expFactor = 21)
    spectralRadiusLoss = spectralRadiusLoss * spectralFactor

    biasLoss = 1e-6 * torch.sum(torch.square(model.network.bias))
    weightLoss = 1e-6 * torch.sum(torch.square(model.network.weights))

    conditionCorrelation = torch.mean(torch.square(getCorrCoef(Yhat)+1e-6)) #Reduce correlation between conditions
    TFcorrelation = torch.mean(torch.square(getCorrCoef(Yhat.T)+1e-6)) #Reduce correlation between TFs
    uniformYfull = uniformLoss(YhatFull)
    uniformTF = uniformLoss(Yhat)
    uniformCondition = uniformLoss(Yhat.T)

    signConstraint = torch.sum(torch.abs(model.network.weights[model.network.getViolations()]))
    ligandConstraint = torch.sum(torch.square(model.network.bias[model.inputLayer.nodeOrder]))

    loss = uniformTF + 0.1 * uniformCondition + conditionCorrelation + TFcorrelation + spectralRadiusLoss + 1e-2 * ligandConstraint + MoAFactor * signConstraint + biasLoss + weightLoss

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.network.bias.grad, 20, 2)
    torch.nn.utils.clip_grad_norm_(model.network.weights.grad, 50, 2)


    optimizer.step()

    stats['loss'][e] = conditionCorrelation.item()
    stats['rate'][e] = optimizer.param_groups[0]['lr']
    stats['violations'][e] = torch.sum(model.network.getViolations()).item()
    stats['eig'][e] = spectralRadius.item()

    if e % 100 == 0:
        plotting.printStats(e, stats)

    if numpy.logical_and(e % 200 == 0, e>0):
        optimizer.state = resetState.copy()

    if numpy.logical_and(e % 5000 == 0, e>0):
        model = bionetwork.reduceSpectralRadius(model, spectralTarget, X)

model.network.weights.data[modeOfAction[0,:]] = torch.abs(model.network.weights.data[modeOfAction[0,:]])
model.network.weights.data[modeOfAction[1,:]] = -torch.abs(model.network.weights.data[modeOfAction[1,:]])


afterWeights = model.network.weights.detach().clone()
afterBias = model.network.bias.detach().clone()



#Enforce spectral radius < target
X = bionetwork.generateRandomInput(model, 200, simultaniousInput)

localY, _ = model(X)
model = bionetwork.reduceSpectralRadius(model, spectralTarget, X)
plt.hist(numpy.std(localY.detach().numpy(), axis=0))

finalWeights = model.network.weights.detach()
finalBias = model.network.bias.detach()

print('Time:', time.time()-start)


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
plt.plot([0, maxIter], [1, 1], 'black')
plt.plot([0, len(T)], [spectralTarget, spectralTarget], 'black', linestyle='--')
#plt.plot(T, stats['eig'])
#plt.plot(T, stats['eig'], stats['eigSTD'])
plt.plot(T, plotting.movingaverage(stats['eig'], 20))
plt.ylabel('Spectral radius')
plt.xlabel('Epoch')
#plt.savefig('synthNetModel/sr.svg')

Yhat, YhatFull = model(X)
df = pd.DataFrame(Yhat.T.detach().numpy(), index=outNameGenes, columns=conditions)
sns.clustermap(df, cmap='gray', vmin=0, vmax=1, yticklabels=True)

plt.figure()
plt.hist(model.network.bias.data.numpy().flatten(), 100)
plt.ylabel('Bias')
#plt.savefig('synthNetModel/biasHist.svg')

plt.figure()
plt.hist(model.network.weights.data.numpy().flatten(), 100)
plt.ylabel('Weights')
#plt.savefig('synthNetModel/weightsHist.svg')

plt.figure()
sns.clustermap(pd.DataFrame(getCorrCoef(Yhat).T.detach().numpy()), cmap='RdBu_r', vmin=-1, vmax=1)


plt.figure()
sr = bionetwork.getAllSpectralRadius(model, YhatFull)
plt.hist(sr,20)

plt.figure()
plt.scatter(Yhat.detach().numpy().flatten(), localY.detach().numpy().flatten(), alpha=0.1)
plt.xlabel('Before')
plt.ylabel('After')
plt.title('SR reduction')

plt.rcParams["figure.figsize"] = (4,2)
plt.figure()
plt.subplot(1,2,1)
plt.scatter(numpy.abs(finalWeights), numpy.abs(priorWeights), alpha=0.5)
plt.subplot(1,2,2)
plt.scatter(numpy.abs(finalBias), numpy.abs(priorBias), alpha=0.5)

#bionetwork.saveParam(model, nodeNames, 'synthNetModel/equationParams.txt')
plt.figure()
plt.hist(model.network.bias[model.inputLayer.nodeOrder].detach().numpy().flatten())
#%% Test KO
# plt.figure()
# sortedState = torch.zeros(YhatFull.shape)
# for i in range(YhatFull.shape[1]):
#     sortedState[:,i] = torch.sort(YhatFull[:,i])[0]

# sns.heatmap(pd.DataFrame(sortedState.detach().T.numpy()), cmap='gray', vmin=0, vmax=1)