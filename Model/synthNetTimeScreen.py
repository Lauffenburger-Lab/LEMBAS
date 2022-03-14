import torch
import numpy
import bionetwork
import plotting
import pandas
import argparse
from scipy.stats import pearsonr
import time

#Get data number
parser = argparse.ArgumentParser(prog='Macrophage simulation')
parser.add_argument('--selectedCondition', action='store', default=None)
args = parser.parse_args()
curentId = int(args.selectedCondition)


testCondtions = pandas.read_csv('synthNetTime/conditions.tsv', sep='\t', low_memory=False)
simultaniousInput = 5
N = int(testCondtions.loc[curentId == testCondtions['Index'],:]['DataSize'].values)
print(curentId, N)

inputAmplitude = 3
projectionAmplitude = 1.2

#Setup optimizer
noiseLevel = 10
L2 = 1e-6
batchSize = 50
MoAFactor = 0.1
spectralFactor = 1e-3
totalUpdates = 10000
maxIter = int(totalUpdates/(N/batchSize))

#Load network
networkList, nodeNames, modeOfAction = bionetwork.loadNetwork('data/KEGGnet-Model.tsv')
annotation = pandas.read_csv('data/KEGGnet-Annotation.tsv', sep='\t')
bionetParams = bionetwork.trainingParameters(iterations = 150, clipping=1, leak=0.01)


inName = annotation.loc[annotation['ligand'],'code'].values
outName = annotation.loc[annotation['TF'],'code'].values
inName = numpy.intersect1d(nodeNames, inName)
outName = numpy.intersect1d(nodeNames, outName)
model = bionetwork.model(networkList, nodeNames, modeOfAction, inputAmplitude, projectionAmplitude, inName, outName, bionetParams)
model.inputLayer.weights.requires_grad = False
model.projectionLayer.weights.requires_grad = False
model.network.preScaleWeights()


parameterizedModel = bionetwork.model(networkList, nodeNames, modeOfAction, inputAmplitude, projectionAmplitude, inName, outName, bionetParams)
parameterizedModel = bionetwork.loadParam('synthNetScreen/equationParams.txt', parameterizedModel, nodeNames)

#Generate data
X = torch.zeros(N, len(inName), dtype=torch.double)
for i in range(1, N): #skip 0 to include a ctrl sample i.e. zero input
    X[i, (i-1) % len(inName)] = torch.rand(1, dtype=torch.double) #stimulate each receptor at least once
    X[i, numpy.random.randint(0, len(inName), simultaniousInput-1)] = torch.rand(simultaniousInput-1, dtype=torch.double)

Y, YfullRef = parameterizedModel(X)
Y = Y.detach()



#%%
criterion1 = torch.nn.MSELoss(reduction='mean')

optimizer = torch.optim.Adam(model.parameters(), lr=1, weight_decay=0)
resetState = optimizer.state.copy()

mLoss = criterion1(torch.mean(Y, dim=0)*torch.ones(Y.shape), Y)
print(mLoss)

#Evaluate network
generalResults = numpy.zeros((totalUpdates, 4))
targetedEvaluation = pandas.DataFrame(columns=['iteration', 'mse', 'correlation', 'start', 'stop'])

curentUpdate = 0
e = 0
baseTime = time.time()
for e in range(e, maxIter):
    trainloader = bionetwork.getSamples(N, batchSize)  #max(10, round(N * e/maxIter)
    for dataIndex in trainloader:
        generalResults[curentUpdate, 0] = curentUpdate
        generalResults[curentUpdate, 1] = time.time() - baseTime
        curLr = bionetwork.oneCycle(curentUpdate, totalUpdates, maxHeight = 1e-3, startHeight=1e-4, endHeight=1e-6, peak = 500)
        optimizer.param_groups[0]['lr'] = curLr

        dataIn = X[dataIndex, :].view(len(dataIndex), X.shape[1]).clone()
        dataOut = Y[dataIndex, :].view(len(dataIndex), Y.shape[1]).clone()

        model.train()
        optimizer.zero_grad()

        Yin = model.inputLayer(dataIn)
        Yin = Yin + noiseLevel * curLr * torch.randn(Yin.shape)
        YhatFull = model.network(Yin)
        Yhat = model.projectionLayer(YhatFull)

        fitLoss = criterion1(dataOut, Yhat)

        signConstraint = MoAFactor * torch.sum(torch.abs(model.network.weights[model.network.getViolations(model.network.weights)]))
        ligandConstraint = 1e-5 * torch.sum(torch.square(model.network.bias[model.inputLayer.nodeOrder]))

        stateLoss = 1e-5 * bionetwork.uniformLossBatch(YhatFull, targetMax = 1/projectionAmplitude, maxConstraintFactor = 1)
        spectralRadiusLoss, spectralRadius = bionetwork.spectralLoss(model.network, YhatFull.detach(), model.network.weights, expFactor = 10)

        weightLoss = L2 * torch.sum(torch.square(model.network.weights))
        biasLoss = L2 * torch.sum(torch.square(model.network.bias))

        loss = fitLoss + signConstraint + ligandConstraint  + spectralFactor * spectralRadiusLoss + stateLoss + weightLoss + biasLoss

        loss.backward()
        optimizer.step()

        if numpy.logical_and(curentUpdate % 500 == 0, e>0):
            optimizer.state = resetState.copy()     
        
        generalResults[curentUpdate, 2] = time.time() - baseTime
        generalResults[curentUpdate, 3] = fitLoss.item()
        
        if curentUpdate % 50 == 0:
            startTime = time.time() - baseTime
            Yhat, YhatFull = model(X)
            r, p = pearsonr(Yhat.detach().numpy().flatten(), Y.numpy().flatten())
            fitLoss = criterion1(Y, Yhat).item()
            stopTime = time.time() - baseTime
            dfRow = {'iteration': curentUpdate , 'correlation': r, 'mse': fitLoss, 'start': startTime, 'stop':stopTime}
            targetedEvaluation = targetedEvaluation.append(dfRow, ignore_index = True)
            print(curentUpdate, fitLoss)
           
        curentUpdate+=1

optimizer.zero_grad()
model(X)
torch.save(model, 'synthNetTime/model_' + str(curentId) + '.pt')
torch.save(X, 'synthNetTime/X_' + str(curentId) + '.pt')

targetedEvaluation.to_csv('synthNetTime/targetedEvaluation_' + str(curentId) +  '.tsv', sep='\t', index=False)

df = pandas.DataFrame(generalResults, columns=['index', 'start', 'stop', 'mse'])   
df.to_csv('synthNetTime/generalResults_' + str(curentId) +  '.tsv', sep='\t', index=False)