import torch
import numpy
import bionetwork
import plotting
import pandas
import saveSimulations
import matplotlib.pyplot as plt
import copy
import seaborn as sns

#Load network
networkList, nodeNames, modeOfAction = bionetwork.loadNetwork('data/KEGGnet-Model.tsv')
annotation = pandas.read_csv('data/KEGGnet-Annotation.tsv', sep='\t')
uniprot2gene = dict(zip(annotation['code'], annotation['name']))
bionetParams = bionetwork.trainingParameters(iterations = 60, clipping=5, leak=0.01, spectralTarget=0.9)
spectralCapacity = numpy.exp(numpy.log(1e-2)/bionetParams['iterations'])
inputAmplitude = 3
projectionAmplitude = 1.2


inName = annotation.loc[annotation['ligand'],'code'].values
outName = annotation.loc[annotation['TF'],'code'].values
inName = numpy.intersect1d(nodeNames, inName)
outName = numpy.intersect1d(nodeNames, outName)
outNameGene = [uniprot2gene[x] for x in outName]
nodeNameGene = [uniprot2gene[x] for x in nodeNames]


parameterizedModel = bionetwork.model(networkList, nodeNames, modeOfAction, inputAmplitude, projectionAmplitude, inName, outName, bionetParams)
parameterizedModel = bionetwork.loadParam('synthNetScreen/equationParams.txt', parameterizedModel, nodeNames)


#Generate data
def generateData(parameterizedModel, N = 500, simultaniousInput = 5):
    numberOfInputs = parameterizedModel.inputLayer.weights.shape[0]
    X = torch.zeros(N, len(inName), dtype=torch.double)
    for i in range(1, N): #skip 0 to include a ctrl sample i.e. zero input
        X[i, (i-1) % numberOfInputs] = torch.rand(1, dtype=torch.double) #stimulate each receptor at least once
        X[i, numpy.random.randint(0, numberOfInputs, simultaniousInput-1)] = torch.rand(simultaniousInput-1, dtype=torch.double)    
    Y, YfullRef = parameterizedModel(X)
    Y = Y.detach()
    X = X.detach()
    YfullRef = YfullRef.detach()
    return X, Y, YfullRef


criterion1 = torch.nn.MSELoss(reduction='mean')

# X, Y, YfullRef = generateData(parameterizedModel, 500)
# importantWeights = numpy.zeros(parameterizedModel.network.weights.shape[0])
# referenceWeights = parameterizedModel.network.weights.detach().clone()
# for i in range(len(importantWeights)):
#     #zero out weight
#     parameterizedModel.network.weights.data[i] = 0
#     Yhat, Yfull = parameterizedModel(X)
#     importantWeights[i] = criterion1(Yhat, Y).item()    
#     #reset weight
#     parameterizedModel.network.weights.data[i] = referenceWeights[i]
#     print(i, importantWeights[i])
    
#importantWeights = importantWeights[0:10]
#selectedWeights = numpy.flip(numpy.argsort(importantWeights))
#selectedWeights = selectedWeights[0:10]
selectedWeights = numpy.array([163, 607,  29,  80, 375, 793, 760, 370, 274, 276], dtype=int)

groundTruth = networkList[:, selectedWeights[0]]
print(numpy.array(nodeNameGene)[groundTruth])
#Identify consistently important interactions
# def getImportantNodes(parameterizedModel, networkList):
#     totalSamples = 100
#     totalN = 500
#     countWeights = numpy.zeros(parameterizedModel.network.weights.shape[0])
#     topWeights = 10
    
#     for i in range(totalSamples):
#         X, Y, YfullRef = generateData(parameterizedModel, totalN)
#         variabilityOfProteins = torch.std(YfullRef, axis=0)
#         jointVariability = variabilityOfProteins[networkList[0]] * variabilityOfProteins[networkList[1]]
#         variabilityAndWeight = jointVariability * model.network.weights
#         selectedWeights = numpy.flip(numpy.argsort(numpy.abs(variabilityAndWeight.detach().numpy())))
#         selectedWeights = selectedWeights[0:topWeights]
#         countWeights[selectedWeights] += 1
#     return countWeights
    
# weightImportance = getImportantNodes(parameterizedModel, networkList)  
#print(weightImportance)

N = 400

X, Y, YfullRef = generateData(parameterizedModel, N)
Xtest, Ytest, YfullRef = generateData(parameterizedModel, N)

folder = 'figures/SI Figure 12/'

#%%

def executeErrorModel(model2, errorModel, dataIn, stateIn):
    dataError = errorModel(stateIn)
    #dataError = errorModel(dataIn)
    Yin = model2.inputLayer(dataIn)
    Yin = Yin + dataError
    YhatFull = model2.network(Yin)
    Yhat = model2.projectionLayer(YhatFull)
    return Yhat, YhatFull, dataError 

model2 = copy.deepcopy(parameterizedModel)
model2.network.weights.data[selectedWeights[0]] = 0 

latentSize = 200
batchSize = 50
maxIter = 2000
L1 = 1e-4

Yhat, referenceState = model2(Xtest)
Yhat, referenceStateTest = model2(Xtest)
referenceState = referenceState.detach()
referenceStateTest = referenceStateTest.detach()
baseLine = criterion1(Yhat, Ytest)


#Define network:s
# errorModel = torch.nn.Sequential(*[torch.nn.Linear(len(inName), latentSize, bias=True),
#                                 torch.nn.LeakyReLU(),
#                                 torch.nn.Linear(latentSize, latentSize, bias=True),
#                                 torch.nn.LeakyReLU(),
#                                 torch.nn.Linear(latentSize, latentSize, bias=True),
#                                 torch.nn.LeakyReLU(),        
#                                 torch.nn.Linear(latentSize, latentSize, bias=True),
#                                 torch.nn.LeakyReLU(),                                    
#                                 torch.nn.Linear(latentSize, len(nodeNames), bias=True)])
# errorModel = errorModel.double()

errorModel = torch.nn.Sequential(*[torch.nn.Linear(referenceState.shape[1], latentSize, bias=True),
                                torch.nn.LeakyReLU(),
                                torch.nn.Linear(latentSize, latentSize, bias=True),
                                torch.nn.LeakyReLU(),
                                torch.nn.Linear(latentSize, latentSize, bias=True),
                                torch.nn.LeakyReLU(),        
                                torch.nn.Linear(latentSize, latentSize, bias=True),
                                torch.nn.LeakyReLU(),                                    
                                torch.nn.Linear(latentSize, len(nodeNames), bias=True)])
errorModel = errorModel.double()

#Setup optimizer
optimizer = torch.optim.Adam(errorModel.parameters(), lr=1, weight_decay=1e-3)
resetState = optimizer.state.copy()




#Evaluate network
stats = plotting.initProgressObject(maxIter)
e=0
for e in range(e, maxIter):
    curLr = bionetwork.oneCycle(e, maxIter, maxHeight = 1e-4, minHeight = 1e-6, peak = 500)
    optimizer.param_groups[0]['lr'] = curLr

    curLoss = []
    trainloader = bionetwork.getSamples(N, batchSize)
    for dataIndex in trainloader:
        optimizer.zero_grad()
        dataIn = X[dataIndex, :].view(len(dataIndex), X.shape[1])
        dataOut = Y[dataIndex, :].view(len(dataIndex), Y.shape[1])
        stateIn = referenceState[dataIndex, :].view(len(dataIndex), referenceState.shape[1])
        
        Yhat, YhatFull, dataError  = executeErrorModel(model2, errorModel, dataIn, stateIn)
        errorSparsity = L1 * torch.sum(torch.abs(dataError))
        fitLoss = criterion1(Yhat, dataOut)
        loss = fitLoss + errorSparsity
        loss.backward()
        optimizer.step()
        curLoss.append(fitLoss.item())

    stats = plotting.storeProgress(stats, e, loss=curLoss, lr=curLr)

    if (e % 200 == 0 and e > 0 and e < maxIter*0.5):
        optimizer.state = resetState.copy()

    if e % 50 == 0:
        Yhat, YhatFull, dataError = executeErrorModel(model2, errorModel, Xtest, referenceStateTest)  
        fitLoss = criterion1(Ytest, Yhat)
        stats['test'][e] = fitLoss.item()
        plotting.printStats(e, stats)

stats = plotting.finishProgress(stats)

#%%
plt.rcParams["figure.figsize"] = (4,4)
plt.figure()
T = numpy.array(range(stats['loss'].shape[0]))
plotting.shadePlot(T, stats['loss'], stats['lossSTD'])
nanFilter = numpy.isnan(stats['test'])==False
plt.plot(T[nanFilter], plotting.movingaverage(stats['test'][nanFilter], 5))
#plt.plot([0, len(T)], numpy.array([1, 1])*mLoss.item(), 'black', linestyle='--')
plt.plot([0, len(T)], numpy.array([1, 1])*baseLine.item(), 'red', linestyle='--')

plt.xlim([0, len(T)])
plt.ylim(bottom=1e-6)
plt.yscale('log')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(numpy.array(['Train', 'Test', 'Mean']), frameon=False)


plt.figure()
Yhat, YhatFull, dataError  = executeErrorModel(model2, errorModel, X, referenceState)
YtestHat, YtestHatFull, dataError = executeErrorModel(model2, errorModel, Xtest, referenceStateTest)
plotting.plotComparison(Yhat, Y, YtestHat, Ytest)

plt.figure()
nodeNameGeneArray = numpy.array(nodeNameGene, dtype=object)
dataErrorValues = dataError.detach().numpy()
modelValues = YhatFull.detach().numpy()
# jointValues = numpy.concatenate((modelValues, dataErrorValues), axis=1)
# dataNames = 'Error_' + nodeNameGeneArray
# modelNames = 'Model_' + nodeNameGeneArray
# jointNames = numpy.concatenate((modelNames, dataNames), axis=0)

topN = 5
meanAbsActivity = numpy.mean(numpy.abs(dataErrorValues), axis=0)
candidatesTarget = numpy.flip(numpy.argsort(meanAbsActivity))[0:topN]
df = pandas.DataFrame(meanAbsActivity, index=nodeNameGeneArray, columns=['Mean activity'])

df = df.iloc[candidatesTarget,:]
sns.barplot(data=df.T, color='#1f77b4')
plt.ylabel('Mean abs activity')
plt.savefig(folder + "A.svg")   
df.to_csv(folder + 'A.tsv', sep='\t')


#%%
def sensitivityAnalysis(model, errorModel, dataIn, referenceState, selectedNode):
    upValue = 0.1
    dataError = errorModel(referenceState)
    
    Yin = model.inputLayer(dataIn)
    Yin = Yin + dataError
    YinUp = Yin.clone()
    YinUp[:,selectedNode] = YinUp[:,selectedNode] + upValue

    ctrl = model.network(Yin)
    up = model.network(YinUp)
    sensitivity = (up-ctrl)/(1 + upValue)
    return sensitivity

correlatedNodes = numpy.corrcoef(dataErrorValues[:,candidatesTarget[0]], y=modelValues.T)
correlatedNodes = correlatedNodes[0,1:]
correlatedNodes[candidatesTarget[0]] = 0

candidatesSources = numpy.flip(numpy.argsort(numpy.abs(correlatedNodes)))[0:20]
# plt.figure()
# df = pandas.DataFrame(numpy.abs(correlatedNodes[candidatesSources]), index=nodeNameGeneArray[candidatesSources], columns=['Correlation'])
# sns.barplot(data=df.T, orient='h')
# plt.xlabel('abs correlation')


plt.figure()
#sensitivityAnalysis
sensitivity = sensitivityAnalysis(model2, errorModel, X, referenceState, candidatesTarget[0])
meanAbsSensitivity = numpy.mean(numpy.abs(sensitivity.detach().numpy()), axis=0)
sensitiveNodes = numpy.argwhere(meanAbsSensitivity>1e-2).flatten()


insensitiveCandididates = candidatesSources[numpy.isin(candidatesSources, sensitiveNodes)==False]
plt.figure()
df = pandas.DataFrame(numpy.abs(correlatedNodes), index=nodeNameGeneArray, columns=['Correlation'])
df = df.iloc[insensitiveCandididates,:]
sns.barplot(data=df.T, orient='h', color='#1f77b4')
plt.xlabel('abs correlation')
#plt.savefig("figures/missingInteractions/posthocTarget.svg")   
plt.savefig(folder + "B.svg")   
df.to_csv(folder + 'B.tsv', sep='\t')



# correlationStructure[numpy.isnan(correlationStructure)] = 0
# numpy.fill_diagonal(correlationStructure, 0)
# #correlationStructure[0:len(dataNames), 0:len(dataNames)] = 0 #ignore model-model correlations
# df = pandas.DataFrame(correlationStructure[0:len(dataNames), :], index=dataNames, columns=jointNames)
# valueRange= numpy.max(numpy.abs(df.values))
# #tresh = 
# tresh = 1e-4 * valueRange
# df = df.loc[numpy.mean(numpy.abs(df))>tresh,:]
# df = df.loc[:,numpy.mean(numpy.abs(df), axis=0)>tresh]
# sns.clustermap(df, cmap='RdBu_r', vmin=-valueRange, vmax=valueRange)

# correlationStructure = numpy.cov(jointValues.T)
# correlationStructure[numpy.isnan(correlationStructure)] = 0
# numpy.fill_diagonal(correlationStructure, 0)
# #correlationStructure[0:len(dataNames), 0:len(dataNames)] = 0 #ignore model-model correlations
# df = pandas.DataFrame(correlationStructure[0:len(dataNames), :], index=dataNames, columns=jointNames)
# valueRange= numpy.max(numpy.abs(df.values))
# #tresh = 
# tresh = 1e-4 * valueRange
# df = df.loc[numpy.mean(numpy.abs(df))>tresh,:]
# df = df.loc[:,numpy.mean(numpy.abs(df), axis=0)>tresh]
# sns.clustermap(df, cmap='RdBu_r', vmin=-valueRange, vmax=valueRange)

# groundTruth = networkList[:, selectedWeights[0]]
# print(nodeNameGeneArray[groundTruth])

#df = pandas.DataFrame(dataErrorValues, columns=nodeNameGene)
#sns.clustermap(df, cmap='RdBu_r', vmin=0, vmax=1)

# correlationStructure = numpy.cov(dataErrorValues.T)
# correlationStructure[numpy.isnan(correlationStructure)] = 0
# numpy.fill_diagonal(correlationStructure, 0)
# df = pandas.DataFrame(correlationStructure, columns=nodeNameGene, index=nodeNameGene)
# valueRange= numpy.max(numpy.abs(correlationStructure))
# tresh = valueRange
# df = df.loc[numpy.sum(numpy.abs(df))>tresh,:]
# df = df.loc[:,numpy.sum(numpy.abs(df), axis=0)>tresh]
# sns.clustermap(df, cmap='RdBu_r', vmin=-valueRange, vmax=valueRange)




#%%
def executeErrorModel(model2, errorModel, dataIn, noiseLevel):
    dataError = errorModel(dataIn)
    Yin = model2.inputLayer(dataIn)
    Yin = Yin + dataError
    Yin = Yin + noiseLevel * torch.randn(Yin.shape)
    YhatFull = model2.network(Yin)
    Yhat = model2.projectionLayer(YhatFull)
    return Yhat, YhatFull, dataError 

#Setup optimizer
batchSize = 50
MoAFactor = 0.1
maxIter = 3000
L1 = 1e-4 #for the sparsity of signal inputation from the fully conected layer
L2 =1e-8 #for model
spectralFactor = 1e-3
latentSize = 200
noiseLevel = 1e-3

model = bionetwork.model(networkList, nodeNames, modeOfAction, inputAmplitude, projectionAmplitude, inName, outName, bionetParams, 'MML', torch.double)
model.inputLayer.weights.requires_grad = False
model.projectionLayer.weights.requires_grad = False
#model.network.balanceWeights()
model.network.preScaleWeights(0.7)


errorModel = torch.nn.Sequential(*[torch.nn.Linear(X.shape[1], latentSize, bias=True),
                                torch.nn.LeakyReLU(),
                                torch.nn.Linear(latentSize, latentSize, bias=True),
                                torch.nn.LeakyReLU(),
                                torch.nn.Linear(latentSize, latentSize, bias=True),
                                torch.nn.LeakyReLU(),        
                                torch.nn.Linear(latentSize, latentSize, bias=True),
                                torch.nn.LeakyReLU(),                                    
                                torch.nn.Linear(latentSize, len(nodeNames), bias=True)])
errorModel = errorModel.double()

nodeDegreOut = numpy.sum(numpy.array(model.network.A.todense() != 0), axis=0)
nodeDegreOut = torch.tensor(nodeDegreOut) + 1

criterion1 = torch.nn.MSELoss(reduction='mean')

optimizer1 = torch.optim.Adam(model.parameters(), lr=1, weight_decay=0)
optimizer2 = torch.optim.Adam(errorModel.parameters(), lr=1e-4, weight_decay=1e-3)
resetState1 = optimizer1.state.copy()
resetState2 = optimizer2.state.copy()

mLoss = criterion1(torch.mean(Y, dim=0)*torch.ones(Y.shape), Y)
print(mLoss)

stats = plotting.initProgressObject(maxIter)

curState = torch.rand((N, model.network.bias.shape[0]), dtype=torch.double, requires_grad=False)

e = 0
for e in range(e, maxIter):
    curLr = bionetwork.oneCycle(e, maxIter, maxHeight = 1e-3, startHeight=1e-4, endHeight=1e-6, peak = 1000)
    optimizer1.param_groups[0]['lr'] = curLr

    curLoss = []
    curEig = []
    trainloader = bionetwork.getSamples(N, batchSize)  #max(10, round(N * e/maxIter)
    for dataIndex in trainloader:
        model.network.weights.data[selectedWeights[0]] = 0 #simulate missing interaction
        
        dataIn = X[dataIndex, :].view(len(dataIndex), X.shape[1])
        dataOut = Y[dataIndex, :].view(len(dataIndex), Y.shape[1])
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        

        Yhat, YhatFull, dataError  = executeErrorModel(model, errorModel, dataIn, noiseLevel)

        curState[dataIndex, :] = YhatFull.detach()

        fitLoss = criterion1(dataOut, Yhat)

        signConstraint = MoAFactor * torch.sum(torch.abs(model.network.weights[model.network.getViolations(model.network.weights)]))

        errorSparsity = L1 * torch.sum(nodeDegreOut * torch.sum(torch.abs(dataError), axis=0))
        #stateLoss = 1e-5 * bionetwork.uniformLoss(curState, dataIndex, YhatFull, maxConstraintFactor = 1, targetMax = 1/projectionAmplitude)
        stateLoss = 1e-5 * bionetwork.uniformLossBatch(YhatFull, maxConstraintFactor = 1, targetMax = 1/projectionAmplitude)
        

        biasLoss = L2 * torch.sum(torch.square(model.network.bias))
        #absFilter = torch.abs(model.network.weights.detach())>0.001
        #weightLoss = L2 * torch.sum(torch.square(model.network.weights[absFilter]))
        #weightLoss = L2 * (torch.sum(torch.square(model.network.weights)) + torch.sum(1/(torch.square(model.network.weights) + 0.5)))
        weightLoss = L2 * torch.sum(torch.square(model.network.weights))

        spectralRadiusLoss, spectralRadius = bionetwork.spectralLoss(model.network, YhatFull.detach(), model.network.weights, expFactor = 10)
        spectralRadiusLoss = spectralFactor * spectralRadiusLoss


        ligandConstraint = 1e-4 * torch.sum(torch.square(model.network.bias[model.inputLayer.nodeOrder,:]))
        loss = fitLoss + signConstraint + biasLoss + weightLoss + stateLoss + spectralRadiusLoss + ligandConstraint + errorSparsity 

        loss.backward()
        optimizer1.step()
        optimizer2.step()
        model.network.weights.data[selectedWeights[0]] = 0 #simulate missing interaction
        
        curEig.append(spectralRadius.item())
        curLoss.append(fitLoss.item())

        stats = plotting.storeProgress(stats, e, curLoss, curEig, curLr, violations=torch.sum(model.network.getViolations(model.network.weights)).item())

    if e % 50 == 0:
        model.eval()
        Yhat, YhatFull, dataError  = executeErrorModel(model, errorModel, Xtest, 0)
        Yhat, YhatFull = model(Xtest)
        fitLoss = criterion1(Ytest, Yhat)
        stats['test'][e] = fitLoss.item()
        plotting.printStats(e, stats)
        
    if e % 200 == 0 and e > 0:
        optimizer1.state = resetState1.copy()
        optimizer2.state = resetState2.copy()

stats = plotting.finishProgress(stats)

#%%
plt.rcParams["figure.figsize"] = (5,5)
plt.figure()

T = numpy.array(range(stats['loss'].shape[0]))
plotting.shadePlot(T, stats['loss'], stats['lossSTD'])
nanFilter = numpy.isnan(stats['test'])==False
plt.plot(T[nanFilter], stats['test'][nanFilter])
plt.plot([0, len(T)], numpy.array([1, 1])*mLoss.item(), 'black', linestyle='--')
plt.xlim([0, len(T)])
plt.ylim(bottom=1e-6)
plt.yscale('log')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(numpy.array(['Train', 'Test', 'Mean']), frameon=False)

Yhat, _ = model(X)
modelPerformance = criterion1(Yhat, Y)

deltaWeight = model.network.weights - parameterizedModel.network.weights
deltaWeight[selectedWeights[0]] = 0  #Ignore the weight we changed
deltaBias= model.network.bias - parameterizedModel.network.bias

treshWeight = 1e-1
treshBias = 4e-2
differentialWeights = numpy.abs(deltaWeight.detach().numpy())>treshWeight
differentialBias = numpy.abs(deltaBias.detach().numpy().flatten())>treshBias

print(numpy.array(nodeNameGene)[networkList[:, differentialWeights]])
print(numpy.array(nodeNameGene)[differentialBias])


plt.rcParams["figure.figsize"] = (6,6)
plt.figure()
nodeNameGeneArray = numpy.array(nodeNameGene, dtype=object)
dataErrorValues = dataError.detach().numpy()
modelValues = YhatFull.detach().numpy()
# jointValues = numpy.concatenate((modelValues, dataErrorValues), axis=1)
# dataNames = 'Error_' + nodeNameGeneArray
# modelNames = 'Model_' + nodeNameGeneArray
# jointNames = numpy.concatenate((modelNames, dataNames), axis=0)

meanAbsActivity = numpy.mean(numpy.abs(dataErrorValues), axis=0)
candidatesTarget = numpy.flip(numpy.argsort(meanAbsActivity))[0:10]
df = pandas.DataFrame(meanAbsActivity[candidatesTarget], index=nodeNameGeneArray[candidatesTarget], columns=['Mean activity'])
sns.barplot(data=df.T, orient='h', color='#1f77b4')
plt.xlabel('Mean abs(activity)')
plt.savefig(folder + "C.svg")   
df.to_csv(folder + 'C.tsv', sep='\t')

