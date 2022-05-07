import torch
import numpy
import bionetwork
import plotting
import pandas
import saveSimulations
import matplotlib.pyplot as plt
import copy

#Load network
networkList, nodeNames, modeOfAction = bionetwork.loadNetwork('data/KEGGnet-Model.tsv')
annotation = pandas.read_csv('data/KEGGnet-Annotation.tsv', sep='\t')
uniprot2gene = dict(zip(annotation['code'], annotation['name']))
bionetParams = bionetwork.trainingParameters(iterations = 150, clipping=1, leak=0.01)
spectralCapacity = numpy.exp(numpy.log(1e-2)/bionetParams['iterations'])
inputAmplitude = 3
projectionAmplitude = 1.2


inName = annotation.loc[annotation['ligand'],'code'].values
outName = annotation.loc[annotation['TF'],'code'].values
inName = numpy.intersect1d(nodeNames, inName)
outName = numpy.intersect1d(nodeNames, outName)
outNameGene = [uniprot2gene[x] for x in outName]
nodeNameGene = [uniprot2gene[x] for x in nodeNames]

model = bionetwork.model(networkList, nodeNames, modeOfAction, inputAmplitude, projectionAmplitude, inName, outName, bionetParams)
#model.network = bionetwork.orthogonalizeWeights(model.network)
model.inputLayer.weights.requires_grad = False
model.projectionLayer.weights.requires_grad = False
model.network.preScaleWeights()
model.train()

modelReg = copy.deepcopy(model)
modelNoreg = copy.deepcopy(model)


parameterizedModel = bionetwork.model(networkList, nodeNames, modeOfAction, inputAmplitude, projectionAmplitude, inName, outName, bionetParams)
parameterizedModel = bionetwork.loadParam('synthNetScreen/equationParams.txt', parameterizedModel, nodeNames)


#Generate data
N = 10
simultaniousInput = 5
X = torch.zeros(N, len(inName), dtype=torch.double)
for i in range(1, N): #skip 0 to include a ctrl sample i.e. zero input
    X[i, (i-1) % len(inName)] = torch.rand(1, dtype=torch.double) #stimulate each receptor at least once
    X[i, numpy.random.randint(0, len(inName), simultaniousInput-1)] = torch.rand(simultaniousInput-1, dtype=torch.double)

controlIndex = 0
Y, YfullRef = parameterizedModel(X)
Y = Y.detach()

folder = 'figures/SI Figure 11/'

#%%
#Setup optimizer
batchSize = 5
MoAFactor = 0.1
spectralFactor = 1e-3
maxIter = 8000
noiseLevel = 10

spectralTarget = numpy.exp(numpy.log(10**-2)/bionetParams['iterations'])
criterion1 = torch.nn.MSELoss(reduction='mean')

optimizerReg = torch.optim.Adam(modelReg.parameters(), lr=1, weight_decay=0)
resetStateReg = optimizerReg.state.copy()
optimizerNoreg = torch.optim.Adam(modelNoreg.parameters(), lr=1, weight_decay=0)
resetStateNoreg = optimizerNoreg.state.copy()


mLoss = criterion1(torch.mean(Y, dim=0)*torch.ones(Y.shape), Y)
print(mLoss)

statsNoreg = plotting.initProgressObject(maxIter)
statsReg = plotting.initProgressObject(maxIter)

curStateReg = torch.rand((N, model.network.bias.shape[0]), dtype=torch.double, requires_grad=False)
curStateNoreg = torch.rand((N, model.network.bias.shape[0]), dtype=torch.double, requires_grad=False)

e = 0
for e in range(e, maxIter):
    curLr = bionetwork.oneCycle(e, maxIter, maxHeight = 2e-3, minHeight = 1e-8, peak = 1000)
    optimizerNoreg.param_groups[0]['lr'] = curLr
    optimizerReg.param_groups[0]['lr'] = curLr

    trainloader = bionetwork.getSamples(N, batchSize)  #max(10, round(N * e/maxIter)

    for trainCondition in ['Noreg', 'Reg']:
        curLoss = []
        curEig = []

        if trainCondition == 'Noreg':
            optimizer = optimizerNoreg
            model = modelNoreg
            curState = curStateNoreg
        else:
            optimizer = optimizerReg
            model = modelReg
            curState = curStateReg

        for dataIndex in trainloader:
            dataIn = X[dataIndex, :].view(len(dataIndex), X.shape[1])
            dataOut = Y[dataIndex, :].view(len(dataIndex), Y.shape[1])

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

            spectralRadiusLoss, spectralRadius = bionetwork.spectralLoss(model.network, YhatFull, model.network.weights, expFactor = 21)

            if trainCondition == 'Noreg':
                spectralRadiusLoss = torch.tensor(0.0)
            else:
                spectralRadiusLoss = spectralFactor * spectralRadiusLoss

            loss = fitLoss + spectralRadiusLoss +  signConstraint + ligandConstraint + weightLoss + biasLoss + stateLoss

            loss.backward()

            optimizer.step()

            curEig.append(spectralRadius.item())
            curLoss.append(fitLoss.item())

        if trainCondition == 'Noreg':
            statsNoreg = plotting.storeProgress(statsNoreg, e, curLoss, curEig, curLr)
            curStateReg =curStateNoreg
        else:
            statsReg = plotting.storeProgress(statsReg, e, curLoss, curEig, curLr)
            curStateReg = curState

    if e % 50 == 0:
        plotting.printStats(e, statsNoreg)
        plotting.printStats(e, statsReg)

    if numpy.logical_and(e % 100 == 0, e>0):
        optimizerReg.state = resetStateReg.copy()
        optimizerNoreg.state = resetStateNoreg.copy()

statsReg = plotting.finishProgress(statsReg)
statsNoreg = plotting.finishProgress(statsNoreg)

#%%
plt.rcParams["figure.figsize"] = (3,3)
plt.figure()
averaginingWindow = 50

T = numpy.array(range(statsReg['loss'].shape[0]))
plotting.shadePlot(T, plotting.movingaverage(statsReg['loss'], averaginingWindow), plotting.movingaverage(statsReg['lossSTD'], averaginingWindow))
plotting.shadePlot(T, plotting.movingaverage(statsNoreg['loss'], averaginingWindow), plotting.movingaverage(statsNoreg['lossSTD'], averaginingWindow))
plt.plot([0, len(T)], numpy.array([1, 1])*mLoss.item(), 'black', linestyle='--')
plt.xlim([0, len(T)])
plt.ylim(bottom=1e-6)
plt.yscale('log')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(numpy.array(['Regularized', 'Un-Regularized', 'Mean']), frameon=False)
plt.savefig(folder + 'A_loss.svg')


plt.figure()
#plt.plot([0, len(T)], spectralCapacity * numpy.array([1, 1]), 'black', linestyle='--')
#plt.plot([0, len(T)], spectralTarget * numpy.array([1, 1]), [0.850, 0.325, 0.098], linestyle='--')
plotting.shadePlot(T, plotting.movingaverage(statsReg['eig'], averaginingWindow), plotting.movingaverage(statsReg['eigSTD'], averaginingWindow))
plotting.shadePlot(T, plotting.movingaverage(statsNoreg['eig'], averaginingWindow), plotting.movingaverage(statsNoreg['eigSTD'], averaginingWindow))
plt.plot([0, len(T)], model.network.param['spectralTarget'] * numpy.array([1, 1]), 'black', linestyle='--')


plt.legend(numpy.array(['Regularized', 'Un-Regularized', 'Capacity']), frameon=False)
plt.ylabel('Spectral radius')
plt.xlabel('Epoch')
plt.xlim([0, len(T)])
plt.ylim(bottom=0, top=1.1)
plt.savefig(folder + 'A_SR.svg')

df = pandas.DataFrame((statsReg['loss'], statsNoreg['loss'], statsReg['eig'], statsNoreg['eig']), columns=T, index=['Regularized Loss', 'Un-Regularized Loss', 'Regularized SR', 'Un-Regularized SR']).T
df.to_csv(folder + 'A.tsv', sep='\t')


modelNoreg(X) #required to shift from the view input
modelReg(X) #required to shift from the view input
saveSimulations.save('simulations', 'equationNet', {'X':X, 'Y':Y, 'Model-Noreg':modelNoreg,  'Model-Reg':modelReg})



