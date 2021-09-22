import torch
import numpy
import matplotlib.pyplot as plt
import time
import bionetwork
import plotting
import copy
import saveSimulations

batchSize = 10

#Load network
networkList, nodeNames, modeOfAction = bionetwork.loadNetwork("data/pipeNet.txt")
bionetParams = bionetwork.trainingParameters(iterations = 6, clipping=1, leak=0.01)


class net(torch.nn.Module):
    def __init__(self, networkList, nodeNames, inName, outName, bionetParams, valType):
        super(net, self).__init__()
        self.inputLayer = bionetwork.projectInput(nodeNames, inName, valType)
        self.network = bionetwork.bionet(networkList, len(nodeNames), modeOfAction, bionetParams, valType)
        self.projectionLayer = bionetwork.projectOutput(nodeNames, outName, 2, valType)

    def forward(self, X):
        fullX = self.inputLayer(X)
        fullY = self.network(fullX)
        Yhat = self.projectionLayer(fullY)
        return Yhat, fullY


inputAmplitude = 3
projectionAmplitude = 2.5

inName = numpy.array(['A'])
outName = numpy.array(['E'])


model = bionetwork.model(networkList, nodeNames, modeOfAction, inputAmplitude, projectionAmplitude, inName, outName, bionetParams, torch.double)
model.projectionLayer.weights.requires_grad = False

parameterizedModel = copy.deepcopy(model)
parameterizedModel.network.weights.data[modeOfAction[0,:]] = torch.tensor([1, 2, 0.5, 1.5], dtype=torch.double) #torch.ones(4, dtype=torch.double)
parameterizedModel.network.bias.data = torch.tensor([0, -0.5, 0.2, -0.1, 0]).reshape([-1, 1])



#Generate data
X = torch.tensor(numpy.linspace(0, 2, num=100)).reshape([-1, 1])
Y = parameterizedModel(X)[0].detach()


nrOfTest = 20
randomOrder = numpy.random.permutation(len(Y))
trainSamples = randomOrder[nrOfTest:(len(Y)+1)]
testSamples = randomOrder[0:nrOfTest]
Xtest = X[testSamples,:]
Ytest = Y[testSamples]
X = X[trainSamples,:]
Y = Y[trainSamples]
N = X.shape[0]


trainloader = torch.utils.data.DataLoader(range(N), batch_size=batchSize, shuffle=True)


#Setup optimizer
MoAFactor = 10
maxIter = 2000
criterion = torch.nn.MSELoss(reduction='mean')


optimizer = torch.optim.Adam(model.parameters(), lr=1e-8) #, amsgrad=True
resetState = optimizer.state.copy()
#scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, 200, gamma=0.2)

#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.1, epochs=maxIter, pct_start=0.3, final_div_factor=1e6, steps_per_epoch=1, cycle_momentum=False)
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-8, max_lr=0.01, step_size_up=500/100, step_size_down=4500/100, mode='triangular2', cycle_momentum=False)
def oneCycle(e, maxIter):
    maxHeight = 2e-3
    minHeight = 1e-8
    peak = 1000
    phaseLength = 0.95 * maxIter

    if e<=peak:
        effectiveE = e/peak
        lr = (maxHeight-minHeight) * 0.5 * (numpy.cos(numpy.pi*(effectiveE+1))+1) + minHeight
    elif e<=phaseLength:
        effectiveE = (e-peak)/(phaseLength-peak)
        lr = (maxHeight-minHeight) * 0.5 * (numpy.cos(numpy.pi*(effectiveE+2))+1) + minHeight
    else:
        lr = minHeight
    beta = 0.9 # + 0.05 * (1 - ((lr-minHeight)/maxHeight))
    return lr, beta


def uniformLoss(curState, dataIndex, YhatFull):
    data = curState.detach().clone()
    data[dataIndex, :] = YhatFull

    targetMean = 0.5
    targetVar= 1/12
    targetMin = 0
    targetMax = 0.99

    meanFactor = 1
    varFactor = 1
    minFactor = 0.9
    maxFactor = 1

    nodeMean = torch.mean(data, dim=0)
    nodeVar = torch.mean(torch.square(data-nodeMean), dim=0)
    maxVal, _ = torch.max(data, dim=0)
    minVal, _ = torch.min(data, dim=0)
    absMin = torch.abs(torch.min(minVal.detach()))
    #nodeSum = torch.min(torch.sum(data, dim=0), torch.zeros(data.shape[1], dtype=data.dtype))

    meanLoss = torch.sum(torch.square(nodeMean - targetMean))
    varLoss =  torch.sum(torch.square(nodeVar - targetVar))
    maxLoss = torch.sum(torch.square(maxVal - targetMax)) + 0.1 * torch.sum(1/(maxVal + 0.01 + absMin)) #distance between max and 0
    minloss = torch.sum(torch.square(minVal- targetMin)) + 0.1 * torch.sum(1/((maxVal-minVal) + 0.01)) #distance between min and max
    #sumloss = torch.sum(torch.square(nodeSum)) #ensure integral is positive

    loss = meanFactor * meanLoss + varFactor * varLoss + minFactor * minloss + maxFactor * maxLoss #+ sumloss
    return loss

#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-8, T_max=5000/100)

#lambda1 = lambda epoch: numpy.sqrt(1 - epoch/maxIter)
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

#Evaluate network

storeWeights = torch.zeros(maxIter, model.network.weights.shape[0])
storeBias = torch.zeros(maxIter, model.network.bias.shape[0])

stats = plotting.initProgressObject(maxIter)
curLossLevel = 1
mLoss =  criterion(torch.mean(Y, dim=0)*torch.ones(Y.shape), Y)

curState = torch.ones((N, model.network.bias.shape[0]), dtype=torch.double, requires_grad=False)
start = time.time()
for e in range(maxIter):
    curLr, curBeta = oneCycle(e, maxIter)
    optimizer.param_groups[0]['betas'] = (curBeta, 0.999)
    optimizer.param_groups[0]['lr'] = curLr

#    storeWeights[e,:] = torch.mean(model.network.weights, dim=1).detach().flatten()
#    storeBias[e,:] = torch.mean(model.network.bias, dim=1).detach().flatten()
    storeWeights[e,:] = model.network.weights.detach().flatten()
    storeBias[e,:] = model.network.bias.detach().flatten()

    curLoss = []
    for dataIndex in trainloader:
        dataIn = X[dataIndex, :]
        dataOut = Y[dataIndex, :]
        model.train()
        optimizer.zero_grad()
        #dataIn = dataIn+torch.randn(dataIn.shape)*0.1
        #dataOut = dataOut+torch.randn(dataIn.shape)*0.1

        # Yin = model.inputLayer(dataIn)
        # Yin = Yin + 2 * curLr * torch.randn(Yin.shape)
        # YhatFull = model.network(Yin)
        # Yhat = model.projectionLayer(YhatFull)
        Yhat, YhatFull = model(dataIn)

        curState[dataIndex,:] = YhatFull.detach()
        fitLoss = criterion(dataOut, Yhat)
        signConstraint = MoAFactor * torch.sum(torch.abs(model.network.weights[model.network.getViolations(model.network.weights)]))

        #stateLoss = 1e-4 * uniformLoss(curState, dataIndex, YhatFull)

        biasLoss = 1e-8 * torch.sum(torch.square(model.network.bias))
        weightLoss = 1e-8 * torch.sum(torch.square(model.network.weights))

        loss = fitLoss + signConstraint + biasLoss + weightLoss  #+ stateLoss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.network.bias, 0.5, 2)
        torch.nn.utils.clip_grad_norm_(model.network.weights, 0.5, 2)
        #model.network.bias.grad = model.network.bias.grad * 2 * torch.rand(model.network.bias.shape) + 1e-8 * torch.randn(model.network.bias.shape)
        #model.network.weights.grad = model.network.weights.grad * 2 * torch.rand(model.network.weights.shape) + 1e-8 * torch.randn(model.network.weights.shape)

        optimizer.step()
        curLoss.append(fitLoss.item())



    stats['loss'][e] = numpy.mean(numpy.array(curLoss))
    stats['rate'][e] = optimizer.param_groups[0]['lr']


    model.eval()
    stats['violations'][e] = torch.sum(model.network.getViolations(model.network.weights)).item()
    Yhat, YhatFull = model(Xtest)
    fitLoss = criterion(Ytest, Yhat)
    stats['test'][e] = fitLoss.item()
    if e % 100 == 0:
        plotting.printStats(e, stats)
    if numpy.logical_and(e % 100 == 0, e>0):
        optimizer.state = resetState.copy()
        #scheduler.step()

print('Time:', time.time()-start)

#%%
XA = torch.tensor(numpy.linspace(0, 2, num=50)).reshape([-1, 1])
YA = parameterizedModel(XA)[0].detach()
Ytrain = model(X)[0].detach()
Ytest = model(Xtest)[0].detach()

#Plot stats
plt.rcParams["figure.figsize"] = (3,3)
plt.figure()
plt.plot(XA, YA, 'black')
plt.scatter(X, Ytrain)
plt.scatter(Xtest, Ytest)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(['Reference', 'Train', 'Test'], frameon=False)

plt.figure()
T = numpy.array(range(stats['loss'].shape[0]))
plt.semilogy(T, plotting.movingaverage(stats['loss'], 5))
plt.semilogy(T, plotting.movingaverage(stats['test'], 5))
plt.plot([0, len(T)], numpy.array([1, 1])*mLoss.item(), 'black', linestyle='--')
plt.xlim([0, len(T)])
plt.ylim([1e-6, 1])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(numpy.array(['Train', 'Test', 'Mean']), frameon=False)

plt.figure()
plt.plot([0, maxIter], [0, 0], 'black')
plt.plot(T, stats['rate'])
plt.ylabel('Learning rate')
plt.xlabel('Epoch')


plt.figure()
plt.plot(T, storeWeights[:, 0].numpy())
plt.plot(T, storeWeights[:, 1].numpy())
plt.plot(T, storeWeights[:, 2].numpy())
plt.plot(T, storeWeights[:, 3].numpy())
plt.xlabel('Epoch')
plt.ylabel('Weight')
#plt.legend(numpy.array(['A->B', 'B->C', 'C->D']), frameon=False)


plt.figure()
trueWeights = parameterizedModel.network.weights.data
trueBias = parameterizedModel.network.bias.data.flatten()
finalWeights = model.network.weights.data.flatten()
finalBias = model.network.bias.data.flatten()
finalWeightsSTD = 0
finalBiasSTD = 0

plt.axhline(0, color='black', label='_nolegend_')
plt.axvline(0, color='black', label='_nolegend_')
plt.scatter(finalWeights, trueWeights)
plt.scatter(finalBias, trueBias)
plt.errorbar(finalWeights, trueWeights, xerr=finalWeightsSTD, linestyle='None', marker='None', color='black')
plt.errorbar(finalBias, trueBias, xerr=finalBiasSTD, linestyle='None', marker='None', color='black')

plt.plot([-2, 2], [-2, 2], 'black', label='_nolegend_')
plt.xlabel('Fit')
plt.ylabel('Reference')
plt.legend(numpy.array(['Weight', 'Bias']), frameon=False)
plt.gca().axis('equal')


plt.figure()
plt.plot(T, storeBias[:, 0].numpy())
plt.plot(T, storeBias[:, 1].numpy())
plt.plot(T, storeBias[:, 2].numpy())
plt.plot(T, storeBias[:, 3].numpy())
plt.plot(T, storeBias[:, 4].numpy())
#plt.legend(numpy.array(['A', 'B', 'C', 'D']), frameon=False)
plt.xlabel('Epoch')
plt.ylabel('Bias')


plt.figure()
_, stateRef = parameterizedModel(X)
_, state = model(X)
for i in range(state.shape[1]):
    A = state[:,i].detach().numpy()
    B = stateRef[:,i].detach().numpy()
    order = numpy.argsort(A)
    plt.plot(A[order], B[order], 'o-')
plt.xlabel('State Fit')
plt.ylabel('State Reference')
plt.legend(nodeNames, frameon=False)

saveSimulations.save('simulations', 'pipeNet', {'X':X, 'Y':Y, 'Xtest':Xtest, 'Ytest':Ytest, 'Model':model})