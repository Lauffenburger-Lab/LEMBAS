import torch
import numpy
import bionetwork
import plotting
import pandas
import argparse


#Get data number
parser = argparse.ArgumentParser(prog='Macrophage simulation')
parser.add_argument('--selectedCondition', action='store', default=None)
args = parser.parse_args()
curentId = int(args.selectedCondition)


testCondtions = pandas.read_csv('synthNetScreen/conditions.tsv', sep='\t', low_memory=False)
simultaniousInput = int(testCondtions.loc[curentId == testCondtions['Index'],:]['Ligands'].values)
N = int(testCondtions.loc[curentId == testCondtions['Index'],:]['DataSize'].values)
print(curentId, simultaniousInput, N)

inputAmplitude = 3
projectionAmplitude = 1.2

#Setup optimizer
noiseLevel = 10
batchSize = 5
MoAFactor = 0.1
spectralFactor = 1e-3
maxIter = 10000

#Load network
networkList, nodeNames, modeOfAction = bionetwork.loadNetwork('data/KEGGnet-Model.tsv')
annotation = pandas.read_csv('data/KEGGnet-Annotation.tsv', sep='\t')
bionetParams = bionetwork.trainingParameters(iterations = 150, clipping=1, leak=0.01)



inName = annotation.loc[annotation['ligand'],'code'].values
outName = annotation.loc[annotation['TF'],'code'].values
inName = numpy.intersect1d(nodeNames, inName)
outName = numpy.intersect1d(nodeNames, outName)
model = bionetwork.model(networkList, nodeNames, modeOfAction, inputAmplitude, projectionAmplitude, inName, outName, bionetParams, torch.double)
model.inputLayer.weights.requires_grad = False
model.projectionLayer.weights.requires_grad = False
model.network.preScaleWeights()


parameterizedModel = bionetwork.model(networkList, nodeNames, modeOfAction, inputAmplitude, projectionAmplitude, inName, outName, bionetParams, torch.double)
parameterizedModel = bionetwork.loadParam('synthNetScreen/equationParams.txt', parameterizedModel, nodeNames)


#Generate data
X = torch.zeros(N, len(inName), dtype=torch.double)
for i in range(1, N): #skip 0 to include a ctrl sample i.e. zero input
    X[i, (i-1) % len(inName)] = torch.rand(1, dtype=torch.double) #stimulate each receptor at least once
    X[i, numpy.random.randint(0, len(inName), simultaniousInput-1)] = torch.rand(simultaniousInput-1, dtype=torch.double)

controlIndex = 0
Y, YfullRef = parameterizedModel(X)
Y = Y.detach()


#%%


spectralTarget = numpy.exp(numpy.log(10**-2)/bionetParams['iterations'])
criterion1 = torch.nn.MSELoss(reduction='mean')

optimizer = torch.optim.Adam(model.parameters(), lr=1, weight_decay=0)
resetState = optimizer.state.copy()

mLoss = criterion1(torch.mean(Y, dim=0)*torch.ones(Y.shape), Y)
print(mLoss)

#Evaluate network
stats = plotting.initProgressObject(maxIter)

curState = torch.rand((N, model.network.bias.shape[0]), dtype=torch.double, requires_grad=False)
e = 0
for e in range(e, maxIter):
    curLr = bionetwork.oneCycle(e, maxIter, maxHeight = 2e-3, minHeight = 1e-8, peak = 1000)
    optimizer.param_groups[0]['lr'] = curLr

    curLoss = []
    curEig = []
    trainloader = bionetwork.getSamples(N, batchSize)  #max(10, round(N * e/maxIter)
    for dataIndex in trainloader:
        dataIn = X[dataIndex, :].view(len(dataIndex), X.shape[1])
        dataOut = Y[dataIndex, :].view(len(dataIndex), Y.shape[1])

        model.train()
        optimizer.zero_grad()

        Yin = model.inputLayer(dataIn)
        Yin = Yin + noiseLevel * curLr * torch.randn(Yin.shape)
        YhatFull = model.network(Yin)
        Yhat = model.projectionLayer(YhatFull)

        curState[dataIndex, :] = YhatFull.detach()

        fitLoss = criterion1(dataOut, Yhat)

        signConstraint = MoAFactor * torch.sum(torch.abs(model.network.weights[model.network.getViolations(model.network.weights)]))
        ligandConstraint = 1e-5 * torch.sum(torch.square(model.network.bias[model.inputLayer.nodeOrder]))


        stateLoss = 1e-4 * bionetwork.uniformLoss(curState, dataIndex, YhatFull, maxConstraintFactor = 50)
        biasLoss = 1e-8 * torch.sum(torch.square(model.network.bias))
        weightLoss = 1e-8 * (torch.sum(torch.square(model.network.weights)) + torch.sum(1/(torch.square(model.network.weights) + 0.5)))

        spectralRadiusLoss, spectralRadius = bionetwork.spectralLoss(model, YhatFull, model.network.weights, expFactor = 21)

        loss = fitLoss + signConstraint + ligandConstraint + weightLoss + biasLoss + spectralFactor * spectralRadiusLoss + stateLoss# + rangeAplification + stdAmplification + meanAmplification +

        loss.backward()
        optimizer.step()

        curEig.append(spectralRadius.item())
        curLoss.append(fitLoss.item())



    stats['loss'][e] = numpy.mean(numpy.array(curLoss))
    stats['lossSTD'][e] = numpy.std(numpy.array(curLoss))
    stats['eig'][e] = numpy.mean(numpy.array(curEig))
    stats['eigSTD'][e] = numpy.std(numpy.array(curEig))
    stats['rate'][e] = optimizer.param_groups[0]['lr']
    stats['violations'][e] = torch.sum(model.network.getViolations(model.network.weights)).item()

    if e % 100 == 0:
        plotting.printStats(e, stats)

    if numpy.logical_and(e % 200 == 0, e>0):
        optimizer.state = resetState.copy()

stats = plotting.finishProgress(stats)
model.eval()
model(X)

torch.save(model, 'synthNetScreen/model_' + str(curentId) + '.pt')
torch.save(X, 'synthNetScreen/X_' + str(curentId) + '.pt')