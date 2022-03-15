import torch
import numpy
import matplotlib.pyplot as plt
import bionetwork
import plotting
import pandas
import time
import copy

class drugLayer(torch.nn.Module):
    def __init__(self, nodeList, drugInName, drugNetwork):
        super(drugLayer, self).__init__()
        targetInName = numpy.unique(drugNetwork['target'])
        drugMatrix = -torch.ones((len(targetInName), len(drugInName)), requires_grad=True, dtype=torch.double)
        drugMask = torch.zeros((len(targetInName), len(drugInName)), dtype=torch.double)
        for i in range(drugNetwork.shape[0]):
            row = numpy.argwhere(drugNetwork.iloc[i,:]['target']==targetInName)
            col = numpy.argwhere(drugNetwork.iloc[i,:]['source']==drugInName)
            drugMask[row, col] = 1    
        drugMatrix.data = drugMatrix.data * drugMask.data

        #df = pandas.DataFrame(drugMask.numpy(), columns=drugInName, index=targetInName)
        #sns.heatmap(df)         
      
        dictionary = dict(zip(nodeList, list(range(len(nodeList)))))
        self.nodeOrder = numpy.array([dictionary[x] for x in targetInName])        
      
        self.drugMatrix = torch.nn.Parameter(drugMatrix)
        self.drugMask = drugMask
        self.drugInName = drugInName
        self.targetInName = targetInName
        self.dtype = torch.double
        self.size_out = len(nodeList) 

    def forward(self, X):
        yIn = torch.zeros((X.shape[0],  self.size_out), dtype=self.dtype)
        yIn[:, self.nodeOrder] = torch.matmul(self.drugMatrix, X.T).T
        return yIn
    
    def signRegularization(self, MoAFactor):
        drugWeightFilter = self.drugMatrix.detach()>0
        return MoAFactor * torch.sum(torch.abs(self.drugMatrix[drugWeightFilter]))
    
    def L2Regularization(self, L2):
        return L2 * torch.sum(torch.square(self.drugMatrix))

class cellLayer(torch.nn.Module):
    def __init__(self, numberOfGenes):
        super(cellLayer, self).__init__()

        weights = torch.ones(numberOfGenes, requires_grad=True, dtype=torch.double)
        weights.data = 1e-4 *weights.data 
        bias = torch.zeros(numberOfGenes, requires_grad=True, dtype=torch.double)
        self.weights = torch.nn.Parameter(weights)
        self.bias = torch.nn.Parameter(bias)

    def forward(self, dataCell):
        cellIn = dataCell*self.weights + self.bias
        
        #leaky cutoff, corresponds to leaky relu but in the oposit direction
        cellInFilter = cellIn.detach()>0
        cellIn[cellInFilter] = 0.01 * cellIn[cellInFilter] 
        return cellIn
    
    def signRegularization(self, MoAFactor):
        weightFilter = self.weights.detach()<0
        return MoAFactor * torch.sum(torch.abs(self.weights[weightFilter]))
    
    def L2Regularization(self, L2):
        L2weight = torch.sum(torch.square(self.weights))
        L2bias = torch.sum(torch.square(self.bias))
        return L2 * (L2weight + L2bias)

class mutationLayer(torch.nn.Module):
    def __init__(self, mutations, nodeNamesGene):
        super(mutationLayer, self).__init__()

        weights = 1e-4 * torch.randn(mutations.shape[0], requires_grad=True, dtype=torch.double)
        projection = torch.zeros((len(nodeNamesGene), mutations.shape[0]), dtype=torch.double)

        mutationTarget = mutations.index.values.copy()
        for i in range(len(mutationTarget)):
            mutationTarget[i] = mutationTarget[i].split('-')[1]
            targetIndex = numpy.argwhere(mutationTarget[i] == nodeNamesGene).flatten()
            projection[targetIndex, i] = 1

        self.weights = torch.nn.Parameter(weights)
        self.projection = projection
        self.mutationTarget = mutationTarget

    def forward(self, mutationIn):
        mutationIn = (mutationIn * self.weights)
        mutationIn = torch.matmul(self.projection, mutationIn.T).T
        return mutationIn

    def L2Regularization(self, L2):
        return L2 * torch.sum(torch.square(self.weights))
    
class viabilityLayer(torch.nn.Module):
    def __init__(self, nodeNames, viabilityNetwork, scaleFactor):
        super(viabilityLayer, self).__init__()
        viabilityNodes = numpy.unique(viabilityNetwork['source'].values)
        viabilityNegative = numpy.unique(viabilityNetwork['source'].values[viabilityNetwork['inhibition']==1])

        dictionary = dict(zip(nodeNames, list(range(len(nodeNames)))))
        self.nodeOrder = numpy.array([dictionary[x] for x in viabilityNodes])
        viabilityNodes = numpy.array(nodeNames)[self.nodeOrder] #sort viability nodes in same order as nodeNames

        weights = scaleFactor * torch.ones(len(viabilityNodes), dtype=torch.double)
        bias = torch.tensor(0, requires_grad=True, dtype=torch.double)
        self.bias = torch.nn.Parameter(bias)
        self.weights = torch.nn.Parameter(weights)
        self.viabilityNodes = viabilityNodes
        
        negativeMap = numpy.isin(viabilityNodes, viabilityNegative)
        self.weights.data[negativeMap] = -self.weights.data[negativeMap]
        self.signMap = torch.sign(self.weights).clone()

    def forward(self, YhatFull):
        Yhat = self.weights * YhatFull[:, self.nodeOrder]
        viability = torch.sum(Yhat, axis=1).reshape(-1,1) + self.bias
        return viability
    
    def signRegularization(self, MoAFactor):
        signMissMatch = torch.logical_not(torch.sign(self.weights.detach()) == self.signMap)
        return MoAFactor * torch.sum(torch.abs(self.weights[signMissMatch]))
    
    def L2Regularization(self, L2):
        L2weight = torch.sum(torch.square(self.weights))
        L2bias = torch.sum(torch.square(self.bias))        
        return L2 * (L2weight + L2bias)
    
class fullModel(torch.nn.Module):
    def __init__(self, nodeNames, nodeNamesGene, drugInName, networkList, modeOfAction, viabilityNetwork, drugNetwork, mutations):
        super(fullModel, self).__init__()
        viabilityProjectionFactor = 0.1
        bionetParams = bionetwork.trainingParameters(iterations = 100, clipping=1, targetPrecision=1e-6, leak=0.01)
        
        self.signalingModel = bionetwork.bionet(networkList, len(nodeNames), modeOfAction, bionetParams, 'MML', torch.double)
        self.drugModel = drugLayer(nodeNames, drugInName, drugNetwork)
        self.cellModel = cellLayer(len(nodeNames))
        self.mutationModel = mutationLayer(mutations, nodeNamesGene)
        self.viabilityModel = viabilityLayer(nodeNames, viabilityNetwork, viabilityProjectionFactor)

    def forward(self, dataIn, dataCell, dataMutation, noiseLevel):
        Yin = self.drugModel(dataIn) + self.cellModel(dataCell) + self.mutationModel(dataMutation)
        Yin = Yin + noiseLevel * torch.randn(Yin.shape)
        YhatFull = self.signalingModel(Yin)
        viabilityHat = self.viabilityModel(YhatFull)
        return viabilityHat, YhatFull
    
    def L2Regularization(self, L2):
        lbRegularization = 0.001
        
        drugL2 = self.drugModel.L2Regularization(L2)
        cellL2 = self.cellModel.L2Regularization(L2)
        mutationL2 = self.mutationModel.L2Regularization(L2)
        
        #Signaling model L2
        absFilter = torch.abs(model.signalingModel.weights.detach())>lbRegularization
        weightLoss = L2 * torch.sum(torch.square(model.signalingModel.weights[absFilter.detach()]))
        biasLoss = L2 * torch.sum(torch.square(model.signalingModel.bias))
        
        viabilityL2 = self.viabilityModel.L2Regularization(L2)
        L2Loss = drugL2 + cellL2 + mutationL2 + weightLoss + biasLoss + viabilityL2   
        return L2Loss
    
    def signRegularization(self, MoAFactor):
        drugSign = self.drugModel.signRegularization(MoAFactor)
        cellSign = self.cellModel.signRegularization(MoAFactor) 
        signalingSign = self.signalingModel.signRegularization(MoAFactor)
        viability = self.viabilityModel.signRegularization(MoAFactor) 
        signConstraints = drugSign + cellSign + signalingSign + viability
        return signConstraints


inputAmplitude = 1
projectionAmplitude = 0.1

#Setup optimizer
batchSize = 50
MoAFactor = 0.1
spectralFactor = 1e-3
maxIter = 40
noiseLevel = 1e-3
L2 = 1e-6

#Load network
networkList, nodeNames, modeOfAction = bionetwork.loadNetwork('viabilityModel/ligandScreen-Model.tsv')
annotation = pandas.read_csv('viabilityModel/ligandScreen-Annotation.tsv', sep='\t')
uniprot2gene = dict(zip(annotation['code'], annotation['name']))
allLigands = annotation.loc[annotation['ligand'], 'code'].values

nodeNamesGene = numpy.array([uniprot2gene[x] for x in nodeNames])


drugInput = pandas.read_csv('viabilityModel/drug.tsv', sep='\t', low_memory=False, index_col=0)
viabilityOutput = pandas.read_csv('viabilityModel/viability.tsv', sep='\t', low_memory=False, index_col=0)
drugInName = drugInput.columns.values.copy()



drugNetwork = pandas.read_csv('viabilityModel/drugLayer.tsv', sep='\t', low_memory=False)
viabilityNetwork = pandas.read_csv('viabilityModel/viabilityLayer.tsv', sep='\t', low_memory=False)
nodeFilter = numpy.isin(viabilityNetwork['source'], nodeNames)
viabilityNetwork = viabilityNetwork.loc[nodeFilter,:].copy()


#Load cell line data
cellLineMember = pandas.read_csv('viabilityModel/cellLine.tsv', sep='\t', low_memory=False, index_col=0)
cellLineLevels = pandas.read_csv('viabilityModel/cellLineRKPMZscored_subset.tsv', sep='\t', low_memory=False, index_col=0)
cellLineLevels = cellLineLevels.drop_duplicates()
missingValues = numpy.setdiff1d(nodeNamesGene, cellLineLevels.index.values)
#Zero padding:
df = pandas.DataFrame(numpy.zeros((len(missingValues), cellLineLevels.shape[1])), index=missingValues, columns=cellLineLevels.columns)
cellLineLevels = cellLineLevels.append(df)
cellLineLevels = cellLineLevels.loc[nodeNamesGene,:]

geneData = cellLineLevels.values.dot(cellLineMember.values.T).T

#Load mutation data
mutations = pandas.read_csv('viabilityModel/mutations.tsv', sep='\t', low_memory=False, index_col=0)
mutationData = mutations.values.dot(cellLineMember.values.T).T

#Joint model
model = fullModel(nodeNames, nodeNamesGene, drugInName, networkList, modeOfAction, viabilityNetwork, drugNetwork, mutations)
model.signalingModel.preScaleWeights()
model.signalingModel.bias.data[numpy.isin(nodeNames, allLigands)] = 1  #Begin with signalal from all ligands

criterion = torch.nn.MSELoss()

sampleName = drugInput.index.values
X = torch.tensor(drugInput.values.copy(), dtype=torch.double)
Y = torch.tensor(viabilityOutput.values, dtype=torch.double)
Xcell = torch.tensor(geneData, dtype=torch.double)
Xmutation = torch.tensor(mutationData, dtype=torch.double)


# currentFold = 1
# CVtest = pandas.read_csv('viabilityModel/CVtest.tsv', sep='\t', low_memory=False, index_col=0)
# testMask = numpy.isin(CVtest['testfold'].values, currentFold)
# Xtest = X[testMask,].clone()
# Ytest = Y[testMask,].clone()
# Xcelltest = Xcell[testMask,].clone()
# Xmutationtest = Xmutation[testMask,].clone()

# X = X[testMask==False,].clone()
# Y = Y[testMask==False,].clone()
# Xcell= Xcell[testMask==False,].clone()
# Xmutation = Xmutation[testMask==False,].clone()

#Probabily large enought batches to use batch level regularization
def uniformLoss(data, targetMin = 0, targetMax = 0.99, maxConstraintFactor = 1):
    targetMean = (targetMax-targetMin)/2
    targetVar= (targetMax-targetMin)**2/12

    nodeMean = torch.mean(data, dim=0)
    nodeVar = torch.mean(torch.square(data-nodeMean), dim=0)
    maxVal, _ = torch.max(data, dim=0)
    minVal, _ = torch.min(data, dim=0)

    meanLoss = torch.sum(torch.square(nodeMean - targetMean))
    varLoss =  torch.sum(torch.square(nodeVar - targetVar))
    maxLoss = torch.sum(torch.square(maxVal - targetMax))
    minloss = torch.sum(torch.square(minVal- targetMin))
    maxConstraint = -maxConstraintFactor * torch.sum(maxVal[maxVal.detach()<=0]) #max value should never be negative

    loss = meanLoss + varLoss + minloss + maxLoss + maxConstraint
    return loss


#%%

referenceState = copy.deepcopy(model.state_dict())
optimizer = torch.optim.Adam(model.parameters(), lr=1, weight_decay=0)
resetState = optimizer.state.copy()

mLoss = criterion(torch.mean(Y, dim=0)*torch.ones(Y.shape), Y)
print(mLoss)


stats = plotting.initProgressObject(maxIter)
N = X.shape[0]
#curState = torch.rand((X.shape[0], model.signalingModel.bias.shape[0]), dtype=torch.double, requires_grad=False)

e = 0
for e in range(e, maxIter):
    curLr = bionetwork.oneCycle(e, maxIter, maxHeight = 2e-3, minHeight = 2e-3, peak = 5)
    optimizer.param_groups[0]['lr'] = curLr

    curLoss = []
    curEig = []
    trainloader = bionetwork.getSamples(N, batchSize)
    for dataIndex in trainloader:
        optimizer.zero_grad()
        model.drugModel.drugMatrix.data = model.drugModel.drugMatrix.data * model.drugModel.drugMask.data #Sparsity is here applied using mask, this could be improved
        dataIn = X[dataIndex, :].view(len(dataIndex), X.shape[1])
        dataOut = Y[dataIndex, :].view(len(dataIndex), Y.shape[1])
        dataCell = Xcell[dataIndex, :].view(len(dataIndex), Xcell.shape[1])
        dataMutation = Xmutation[dataIndex, :].view(len(dataIndex), Xmutation.shape[1])

        dataCell = dataCell + noiseLevel * torch.randn(dataCell.shape)

        viabilityHat, YhatFull = model(dataIn, dataCell, dataMutation, noiseLevel)

                
        #curState[dataIndex, :] = YhatFull.detach()

        fitLoss = criterion(dataOut, viabilityHat)

        stateLoss = 1e-5 * uniformLoss(YhatFull)
        L2Loss = model.L2Regularization(L2)
        
        
        signConstraint = model.signRegularization(MoAFactor)                                                                                                                                                                   
                                                                                                                                                                                                                                               
        spectralRadiusLoss, spectralRadius = bionetwork.spectralLoss(model.signalingModel, YhatFull.detach(), model.signalingModel.weights, expFactor = 5)

        loss = fitLoss + signConstraint + stateLoss + L2Loss  + spectralFactor * spectralRadiusLoss 

        loss.backward()

        optimizer.step()

        curEig.append(spectralRadius.item())
        curLoss.append(fitLoss.item())

    stats = plotting.storeProgress(stats, e, curLoss, curEig, curLr, violations=model.signalingModel.getNumberOfViolations())
    
    # if numpy.logical_or(e % 5 == 0, e == (maxIter-1)):
    #     viabilityHat, YhatFull = model(Xtest, Xcelltest, Xmutationtest, 0)
    #     fitLoss = criterion(Ytest, viabilityHat)
    #     stats['test'][e] = fitLoss.item()
    plotting.printStats(e, stats)

    if numpy.logical_and(e % 2 == 0, e>0):
        optimizer.state = resetState.copy()

stats = plotting.finishProgress(stats)
#model.eval()
#Yhat, YhatFull = model(X)

#torch.save(model, 'CVmacrophage/model_reference.pt')

#%%
spectralCapacity = numpy.exp(numpy.log(1e-6)/model.signalingModel.param['iterations'])

plt.rcParams["figure.figsize"] = (3,3)
plt.figure()


T = numpy.array(range(stats['loss'].shape[0]))
#plotting.shadePlot(T, plotting.movingaverage(stats['loss'], 5), plotting.movingaverage(stats['lossSTD'], 10))
plotting.shadePlot(T, stats['loss'], stats['lossSTD'])
nanFilter = numpy.isnan(stats['test'])==False
plt.plot(T[nanFilter], stats['test'][nanFilter])
plt.plot([0, len(T)], numpy.array([1, 1])*mLoss.item(), 'black', linestyle='--')
plt.xlim([0, len(T)])
plt.ylim(bottom=1e-3)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.yscale('log')

plt.figure()
plt.plot(T, stats['rate'])
plt.plot([0, maxIter], [0, 0], 'black')
plt.legend(numpy.array(['lr', 'loss adjusted lr']), frameon=False)
plt.ylabel('Learning rate')
plt.xlabel('Epoch')


plt.figure()
plt.plot([0, maxIter], [1, 1], 'black')
plt.plot([0, len(T)], spectralCapacity * numpy.array([1, 1]), 'black', linestyle='--')
plotting.shadePlot(T, plotting.movingaverage(stats['eig'], 5), plotting.movingaverage(stats['eigSTD'], 5))
plt.ylabel('Spectral radius')
plt.xlabel('Epoch')


plt.figure()
viabilityHat, YhatFull = model(X, Xcell, Xmutation, 0)

plt.figure()
plt.scatter(viabilityHat.detach().numpy(), Y.detach().numpy(), color=[0.5,0.5,0.5], alpha=0.1)
plotting.lineOfIdentity()
plotting.addCorrelation(viabilityHat, Y)
plt.xlabel('Fit')
plt.ylabel('Data')
plt.gca().axis('equal')


fitData = pandas.DataFrame(viabilityHat.detach().numpy(), index=sampleName, columns=['viability'])
fitData.to_csv('viabilityModel/CV/fit_ref.tsv', sep='\t')

#TestData
#viabilityHatTest, YhatFullTest = model(Xtest, Xcelltest, Xmutationtest, 0)
# plt.rcParams["figure.figsize"] = (5,5)
# plotting.plotComparison(viabilityHat.flatten(), Y.flatten(), viabilityHatTest.flatten(), Ytest.flatten())




#plt.figure()
#srModel = bionetwork.getAllSpectralRadius(model, YhatFull)
# plt.hist(srModel)
# plt.ylabel('SR model')

#Independent test set for validaiton
# drugInputValidation = pandas.read_csv('FrohlicModel/drugIndependent.tsv', sep='\t', low_memory=False, index_col=0)
# viabilityValidation = pandas.read_csv('FrohlicModel/viabilityIndependen.tsv', sep='\t', low_memory=False, index_col=0)
# Xvalidation = torch.tensor(drugInputValidation.values.copy(), dtype=torch.double)
# Yvalidation = torch.tensor(viabilityValidation.values, dtype=torch.double)
# XcellValidation = torch.zeros((Yvalidation.shape[0], geneData.shape[1]), dtype=torch.double) #cell data is missing

# viabilityHat, YhatFull = model(X, Xcell, Xmutation, 0)
# YinValidation = mergeDrugAndCell(model, drugMatrix, geneWeights, geneBias, Xvalidation, XcellValidation)
# YhatFullValidation = model.network(YinValidation)
# YhatValidation = calculateViability(model, viabilityBias, YhatFullValidation)
# plotting.plotComparison(Yhat.flatten(), Y.flatten(), YhatValidation.flatten(), Yvalidation.flatten())


#plotting.displayData(Y, sampleName, outNameGene)
#plotting.compareDataAndModel(X.detach(), Y.detach(), Yhat.detach(), sampleName, outNameGene)

torch.save(model, 'viabilityModel/CV/model_ref.pt')

#%%
#Cross validation
CVtest = pandas.read_csv('viabilityModel/CVtest.tsv', sep='\t', low_memory=False, index_col=0)
allFolds = numpy.unique(CVtest['testfold'].values)
#allFolds = [3]
totalTime = numpy.zeros(len(allFolds))

for i in range(len(allFolds)):
    model.load_state_dict(copy.deepcopy(referenceState))
    
    startTime = time.time()
    currentFold = allFolds[i]
    print('Current fold', currentFold)    
    testMask = numpy.isin(CVtest['testfold'].values, currentFold)
    
    X = torch.tensor(drugInput.values.copy(), dtype=torch.double)
    Y = torch.tensor(viabilityOutput.values, dtype=torch.double)
    Xcell = torch.tensor(geneData, dtype=torch.double)
    Xmutation = torch.tensor(mutationData, dtype=torch.double)
    
    Xtest = X[testMask,].clone()
    Ytest = Y[testMask,].clone()
    Xcelltest = Xcell[testMask,].clone()
    Xmutationtest = Xmutation[testMask,].clone()
    
    X = X[testMask==False,].clone()
    Y = Y[testMask==False,].clone()
    Xcell= Xcell[testMask==False,].clone()
    Xmutation = Xmutation[testMask==False,].clone()
    
    N = X.shape[0]
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=0)
    resetState = optimizer.state.copy()
    
    #curState = torch.rand((X.shape[0], model.signalingModel.bias.shape[0]), dtype=torch.double, requires_grad=False)
    
    for e in range(maxIter):
        trainloader = bionetwork.getSamples(N, batchSize)
        for dataIndex in trainloader:
            optimizer.zero_grad()
            model.drugModel.drugMatrix.data = model.drugModel.drugMatrix.data * model.drugModel.drugMask.data #Sparsity is here applied using mask, this could be improved
            dataIn = X[dataIndex, :].view(len(dataIndex), X.shape[1])
            dataOut = Y[dataIndex, :].view(len(dataIndex), Y.shape[1])
            dataCell = Xcell[dataIndex, :].view(len(dataIndex), Xcell.shape[1])
            dataMutation = Xmutation[dataIndex, :].view(len(dataIndex), Xmutation.shape[1])
            dataCell = dataCell + noiseLevel * torch.randn(dataCell.shape)
    
            viabilityHat, YhatFull = model(dataIn, dataCell, dataMutation, noiseLevel)
                    
            #curState[dataIndex, :] = YhatFull.detach()
    
            fitLoss = criterion(dataOut, viabilityHat)
    
            signConstraint = model.signalingModel.signRegularization(MoAFactor)
    
            stateLoss = 1e-5 * uniformLoss(YhatFull)
            #stateLoss = 1e-5 * bionetwork.uniformLoss(curState, dataIndex, YhatFull, maxConstraintFactor = 10)
            L2Loss = model.L2Regularization(L2)
            signConstraint = model.signRegularization(MoAFactor)                                                                                                                                                                                     
            spectralRadiusLoss, spectralRadius = bionetwork.spectralLoss(model.signalingModel, YhatFull.detach(), model.signalingModel.weights, expFactor = 5)
    
            loss = fitLoss + signConstraint + spectralFactor * spectralRadiusLoss + stateLoss + L2Loss
    
            loss.backward()
    
            optimizer.step()
    
        if numpy.logical_and(e % 2 == 0, e>0):
            optimizer.state = resetState.copy()
    
    viabilityHat, YhatFull = model(X, Xcell, Xmutation, 0)

    fitData = pandas.DataFrame(viabilityHat.detach().numpy(), index=sampleName[testMask==False], columns=['viability'])
    fitData.to_csv('viabilityModel/CV/fit_' + str(currentFold)  + '.tsv', sep='\t')
    
    viabilityHatTest, YhatFullTest = model(Xtest, Xcelltest, Xmutationtest, 0)
    
    predictionData = pandas.DataFrame(viabilityHatTest.detach().numpy(), index=sampleName[testMask], columns=['viability'])
    predictionData.to_csv('viabilityModel/CV/prediction_' + str(currentFold)  + '.tsv', sep='\t')
    
    torch.save(model, 'viabilityModel/CV/model_' + str(currentFold)  + '.pt')
    stopTime = time.time()
    totalTime[i] = stopTime - startTime
    print('\t', totalTime[i]/60, 'minutes')

predictionData = pandas.DataFrame(totalTime, index=allFolds, columns=['total time'])
predictionData.to_csv('viabilityModel/CV/time.tsv', sep='\t')