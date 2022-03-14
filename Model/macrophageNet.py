import torch
import numpy
import matplotlib.pyplot as plt
import bionetwork
import plotting
import pandas
import saveSimulations


inputAmplitude = 3
projectionAmplitude = 1.2

#Setup optimizer
batchSize = 5
MoAFactor = 0.1
spectralFactor = 1e-3
maxIter = 5000
noiseLevel = 10
L2 = 1e-6



#Load network
networkList, nodeNames, modeOfAction = bionetwork.loadNetwork('data/macrophage-Model.tsv')
annotation = pandas.read_csv('data/macrophage-Annotation.tsv', sep='\t')
uniprot2gene = dict(zip(annotation['code'], annotation['name']))
bionetParams = bionetwork.trainingParameters(iterations = 100, clipping=1, targetPrecision=1e-6, leak=0.01)
spectralCapacity = numpy.exp(numpy.log(1e-2)/bionetParams['iterations'])


inName = annotation.loc[annotation['ligand'],'code'].values
outName = annotation.loc[annotation['TF'],'code'].values
inName = numpy.intersect1d(nodeNames, inName)
outName = numpy.intersect1d(nodeNames, outName)
outNameGene = [uniprot2gene[x] for x in outName]
nodeNameGene = [uniprot2gene[x] for x in nodeNames]

ligandInput = pandas.read_csv('data/macrophage-Ligands.tsv', sep='\t', low_memory=False, index_col=0)
TFOutput = pandas.read_csv('data/macrophage-TFs.tsv', sep='\t', low_memory=False, index_col=0)

#Subset input and output to intersecting nodes
inName = ligandInput.columns.values
outName = TFOutput.columns.values
inName = numpy.intersect1d(nodeNames, inName)
outName = numpy.intersect1d(nodeNames, outName)
inNameGene = [uniprot2gene[x] for x in inName]
outNameGene = [uniprot2gene[x] for x in outName]
ligandInput = ligandInput.loc[:,inName]
TFOutput = TFOutput.loc[:,outName]

sampleName = ligandInput.index.values
model = bionetwork.model(networkList, nodeNames, modeOfAction, inputAmplitude, projectionAmplitude, inName, outName, bionetParams)
model.inputLayer.weights.requires_grad = False
model.network.preScaleWeights()


X = torch.tensor(ligandInput.values.copy(), dtype=torch.double)
Y = torch.tensor(TFOutput.values, dtype=torch.double)


#%%
criterion = torch.nn.MSELoss(reduction='mean')

optimizer = torch.optim.Adam(model.parameters(), lr=1, weight_decay=0)
resetState = optimizer.state.copy()

mLoss = criterion(torch.mean(Y, dim=0)*torch.ones(Y.shape), Y)
print(mLoss)


stats = plotting.initProgressObject(maxIter)
N = X.shape[0]
curState = torch.rand((X.shape[0], model.network.bias.shape[0]), dtype=torch.double, requires_grad=False)

e = 0
for e in range(e, maxIter):
    curLr = bionetwork.oneCycle(e, maxIter, maxHeight = 2e-3, minHeight = 1e-8, peak = 1000)
    optimizer.param_groups[0]['lr'] = curLr

    curLoss = []
    curEig = []
    trainloader = bionetwork.getSamples(N, batchSize)
    for dataIndex in trainloader:
        model.train()
        model.network.weights.data = model.network.weights.data + 1e-8 * torch.randn(model.network.weights.shape) #breaks potential symmetries
        optimizer.zero_grad()

        dataIn = X[dataIndex, :].view(len(dataIndex), X.shape[1])
        dataOut = Y[dataIndex, :].view(len(dataIndex), Y.shape[1])

        Yin = model.inputLayer(dataIn)
        Yin = Yin + noiseLevel * curLr * torch.randn(Yin.shape)
        YhatFull = model.network(Yin)
        Yhat = model.projectionLayer(YhatFull)

        curState[dataIndex, :] = YhatFull.detach()

        fitLoss = criterion(dataOut, Yhat)

        signConstraint = MoAFactor * torch.sum(torch.abs(model.network.weights[model.network.getViolations(model.network.weights)]))
        ligandConstraint = 1e-3 * torch.sum(torch.square(model.network.bias[model.inputLayer.nodeOrder]))

        stateLoss = 1e-5 * bionetwork.uniformLoss(curState, dataIndex, YhatFull, maxConstraintFactor = 50)
        biasLoss = L2 * torch.sum(torch.square(model.network.bias))
        weightLoss = L2 * (torch.sum(torch.square(model.network.weights)) + torch.sum(1/(torch.square(model.network.weights) + 0.5)))
        projectionLoss = 1e-6 * torch.sum(torch.square(model.projectionLayer.weights - projectionAmplitude))

        spectralRadiusLoss, spectralRadius = bionetwork.spectralLoss(model.network, YhatFull, model.network.weights, expFactor = 21)

        loss = fitLoss + signConstraint + ligandConstraint + weightLoss + biasLoss + spectralFactor * spectralRadiusLoss + stateLoss + projectionLoss

        loss.backward()

        optimizer.step()

        curEig.append(spectralRadius.item())
        curLoss.append(fitLoss.item())

        stats = plotting.storeProgress(stats, e, curLoss, curEig, curLr, violations=torch.sum(model.network.getViolations(model.network.weights)).item())

    if e % 50 == 0:
        plotting.printStats(e, stats)

    if numpy.logical_and(e % 200 == 0, e>0):
        optimizer.state = resetState.copy()



stats = plotting.finishProgress(stats)
model.eval()
Yhat, YhatFull = model(X)

torch.save(model, 'CVmacrophage/model_reference.pt')

#%%
plt.rcParams["figure.figsize"] = (3,3)
plt.figure()


T = numpy.array(range(stats['loss'].shape[0]))
plotting.shadePlot(T, plotting.movingaverage(stats['loss'], 5), plotting.movingaverage(stats['lossSTD'], 10))
#plt.plot(T[nanFilter], plotting.movingaverage(stats['loss'][nanFilter], 5))
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
Yhat, YhatFull = model(X)
plt.scatter(Yhat.detach().numpy(), Y.detach().numpy(), alpha=0.2)
plotting.lineOfIdentity()
plotting.addCorrelation(Yhat, Y)
plt.xlabel('Fit')
plt.ylabel('Data')
plt.gca().axis('equal')
plt.gca().set_xticks([0, 0.5, 1])
plt.gca().set_yticks([0, 0.5, 1])

plt.figure()
Yctrl, _ = model(torch.zeros((1, X.shape[1]), dtype=torch.double))
signalDistance = torch.sum(torch.abs(Yhat-Yctrl),dim=1).detach().numpy().flatten()
fitDistance = torch.sum(torch.square(Yhat-Y),dim=1).detach().numpy().flatten()
plt.scatter(signalDistance, fitDistance)


plt.figure()
srModel = bionetwork.getAllSpectralRadius(model, YhatFull)

plt.hist(srModel)
plt.ylabel('SR model')

plt.rcParams["figure.figsize"] = (10,10)
plt.figure()
rank = plotting.compareAllTFs(Yhat, Y, outNameGene)

plt.figure()
rank = plotting.compareAllTFs(Yhat.T, Y.T, sampleName)

plotting.displayData(Y, sampleName, outNameGene)
#plotting.compareDataAndModel(X.detach(), Y.detach(), Yhat.detach(), sampleName, outNameGene)

saveSimulations.save('simulations', 'macrophageNet', {'X':X, 'Y':Y, 'Model':model})