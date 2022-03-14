import torch
import torch.nn as nn
import scipy.sparse
import numpy.random
import pandas
import numpy
import activationFunctions
from scipy.sparse.linalg import eigs
#from scipy.linalg import norm
from scipy.linalg import eig


#import matplotlib.pyplot as plt
#import seaborn

def gradCliping(grad, n):
    clipingFilter = grad<-n
    grad[clipingFilter] = numpy.tanh(grad[clipingFilter]+n) - n
    clipingFilter = grad>n
    grad[clipingFilter] = numpy.tanh(grad[clipingFilter]-n) + n
    return grad




##########Spectral radius

class spectralRadius(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, A, networkList):
        ctx.dim = weights.shape
        ctx.tol = 10**-6
        weights = weights.detach().numpy().flatten()

        ctx.networkList = networkList
        ctx.weights = weights
        ctx.A = A
        ctx.A.data = ctx.weights


        try:
            e, v = eigs(ctx.A, k=1, which='LM', ncv=100, tol=ctx.tol)
            v = v[:,0]
            e = e[0]
        except  (KeyboardInterrupt, SystemExit):
            raise
        except:
            print('Forward fail (did not find any eigenvalue with eigs)')
            tmpA = ctx.A.toarray()
            e, v, w = lreig(tmpA) #fall back to solving full eig problem

        spectralRadius = numpy.abs(e)
        ctx.e = e
        ctx.v = v
        ctx.w = numpy.empty(0)

        return torch.from_numpy(numpy.asarray(spectralRadius))

    @staticmethod
    def backward(ctx, grad_output):
        v = ctx.v
        e = ctx.e
        w = ctx.w
        networkList = ctx.networkList
        tmpA = ctx.A
        tmpA.data = ctx.weights
        tmpA = tmpA.T  #tmpA.T.toarray()

        if w.shape[0]==0:
            try:
                eT = e
                if numpy.isreal(eT): #does for some reason not converge if imag = 0
                    eT = eT.real
                e2, w = eigs(tmpA, k=1, sigma=eT, OPpart='r', tol=ctx.tol)
                selected = 0 #numpy.argmin(numpy.abs(e2-eT))
                w = w[:,selected]
                e2 = e2[selected]
                #Check if same eigenvalue
                if abs(e-e2)>(ctx.tol*10):
                    print('Backward fail (eigs left returned different eigenvalue)')
                    w = numpy.empty(0)
                    #e, v, w = lreig(tmpA) #fall back to solving whole eig problem
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                print('Backward fail (did not find any eigenvalue with eigs)')
                #e, v, w = lreig(tmpA) #fall back to solving full eig problem
                delta = numpy.zeros(ctx.weights.shape)


        if w.shape[0] != 0:
            divisor = w.T.dot(v).flatten()
            if abs(divisor) == 0:
                delta = numpy.zeros(ctx.weights.shape)
                print('Empty eig')
            else:
                delta = numpy.multiply(w[networkList[0]], v[networkList[1]])/divisor
                direction = e/numpy.abs(e)
                delta = (delta/direction).real
        else:
            #print('Empty eig')
            delta = numpy.zeros(ctx.weights.shape)

        #deltaFilter = numpy.not_equal(numpy.sign(delta), numpy.sign(ctx.weights))
        #delta[deltaFilter] = 0

        delta = torch.tensor(delta, dtype = grad_output.dtype)

        constrainNorm = True
        if constrainNorm:
            norm = torch.norm(delta, 2)
            if norm>10:
                delta = delta/norm #typical seems to be ~0.36
            #delta = delta * numpy.abs(ctx.weights)
            #delta = delta/norm(delta)


        dW = grad_output * delta

        return dW, None, None, None




class bionetworkFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, bias, A, networkList, parameters, activation, deltaActivation):
        #Load into memory
        ctx.weights = weights.detach().numpy()
        A.data = ctx.weights
        ctx.networkList = networkList
        ctx.A = A
        ctx.deltaActivation = deltaActivation

        bIn = x.transpose(0, 1).detach().numpy() + bias.detach().numpy()

        xhat = numpy.zeros(bIn.shape, dtype = bIn.dtype)
        #xhat = numpy.random.rand(bIn.shape[0], bIn.shape[1]).astype(bIn.dtype)
        xhatBefore = xhat.copy()

        for i in range(parameters['iterations']):
            if i>40: #normally takes around 40 iterations to reach steady state
                if i>41:
                    if numpy.sum(numpy.abs(xhat-xhatBefore))<1e-6:
                        break            
                xhatBefore = xhat.copy()            
            xhat = A.dot(xhat)
            xhat += bIn
            xhat = activation(xhat, parameters['leak'])

        output = torch.from_numpy(xhat)
        output.requires_grad_()
        output = output.transpose(0, 1)

        #Pass to backward
        ctx.xRaw = A.dot(xhat) + bIn     #When converged this is the same as taking inv(activation(xhat))
        ctx.x = xhat
        ctx.parameters = parameters
        return output

    @staticmethod
    def backward(ctx, grad_output):

        AT = ctx.A
        AT.data = ctx.weights.data
        AT = AT.T

        gradIn = grad_output.transpose(0, 1).detach().numpy()
        grad = numpy.zeros(gradIn.shape)
        #grad = numpy.random.randn(gradIn.shape[0], gradIn.shape[1]).astype(bIn.dtype)
        deltaX = ctx.xRaw.copy()
        deltaX = ctx.deltaActivation(deltaX, ctx.parameters['leak'])
        gradBefore = grad.copy() 

        for i in range(ctx.parameters['iterations']):
            if i>20:   #normally takes around 30 iterations to reach steady state
                if i>21:
                    if numpy.sum(numpy.abs(grad-gradBefore))<1e-6:            
                        break
                gradBefore = grad.copy()   
            grad = deltaX * (AT.dot(grad) + gradIn)
            #as a precaution clipping with tanh for |gradients| > clipping val
            grad = gradCliping(grad, ctx.parameters['clipping'])
            #norm = numpy.linalg.norm(grad, ord='fro')
            #normFactor = numpy.where(norm>ctx.parameters['clipping'], ctx.parameters['clipping']/norm, 1)
            #grad = grad * normFactor

        output = torch.from_numpy(grad).transpose(0, 1)

        #Construct gradients
        grad_weight = torch.from_numpy(numpy.sum(numpy.multiply(ctx.x[ctx.networkList[1],:], grad[ctx.networkList[0],:]), axis=1))
        grad_bias = torch.from_numpy(numpy.sum(grad, axis=1)).unsqueeze(1)

        return output, grad_weight, grad_bias, None, None, None, None, None
    
 
class model(torch.nn.Module):
    def __init__(self, networkList, nodeNames, modeOfAction, inputAmplitude, projectionFactor, inName, outName, bionetParams, activationFunction='MML', valType=torch.double):
        super(model, self).__init__()
        self.inputLayer = projectInput(nodeNames, inName, inputAmplitude, valType)
        self.network = bionet(networkList, len(nodeNames), modeOfAction, bionetParams, activationFunction, valType)
        self.projectionLayer = projectOutput(nodeNames, outName, projectionFactor, valType)

    def forward(self, X):
        fullX = self.inputLayer(X)
        fullY = self.network(fullX)
        Yhat = self.projectionLayer(fullY)
        return Yhat, fullY


def spectralLoss(signalingModel, YhatFull, weights, expFactor = 20, lb=0.5):
    bionetParams = signalingModel.param

    randomIndex = numpy.random.randint(YhatFull.shape[0])
    activationFactor = signalingModel.oneStepDeltaActivationFactor(YhatFull[randomIndex,:], bionetParams['leak']).detach()
    weightFactor = activationFactor[signalingModel.networkList[0]]
    multipliedWeightFactor = weights * weightFactor
    spectralRadius = signalingModel.getSpectralRadius(multipliedWeightFactor)
    #spectralClampFactor = 1/torch.max(spectralRadius.detach()/bionetParams['spectralLimit'], torch.tensor(1.0).double()) #Prevents infinte penalty
    #spectralRadiusLoss =  (1/(1 - spectralClampFactor*spectralRadius) - 1)

    # if spectralRadius>bionetParams['spectralTarget']:
    #     spectralRadiusLoss = torch.abs(spectralRadius)
    # else:
    #     spectralRadiusLoss = torch.tensor(0.0)

    scaleFactor = 1/numpy.exp(expFactor *  bionetParams['spectralTarget'])

    if spectralRadius>lb:
        spectralRadiusLoss = scaleFactor * (torch.exp(expFactor*spectralRadius)-1)
    else:
        spectralRadiusLoss = torch.tensor(0.0)

    return spectralRadiusLoss, spectralRadius


def uniformLoss(curState, dataIndex, YhatFull, targetMin = 0, targetMax = 0.99, maxConstraintFactor = 10):
    data = curState.detach().clone()
    data[dataIndex, :] = YhatFull
    loss = uniformLossBatch(data, targetMin = targetMin, targetMax = targetMax, maxConstraintFactor = maxConstraintFactor)
    return loss

def uniformLossBatch(YhatFull, targetMin = 0, targetMax = 0.99, maxConstraintFactor = 10):
    targetMean = (targetMax-targetMin)/2
    targetVar= (targetMax-targetMin)**2/12

    factor = 1
    meanFactor = factor
    varFactor = factor
    minFactor = factor
    maxFactor = factor
    maxConstraintFactor = factor * maxConstraintFactor

    nodeMean = torch.mean(YhatFull, dim=0)
    nodeVar = torch.mean(torch.square(YhatFull-nodeMean), dim=0)
    maxVal, _ = torch.max(YhatFull, dim=0)
    minVal, _ = torch.min(YhatFull, dim=0)

    meanLoss = meanFactor * torch.sum(torch.square(nodeMean - targetMean))
    varLoss =  varFactor * torch.sum(torch.square(nodeVar - targetVar))
    maxLoss = maxFactor * torch.sum(torch.square(maxVal - targetMax))
    minloss = minFactor * torch.sum(torch.square(minVal- targetMin))
    maxConstraint = -maxConstraintFactor * torch.sum(maxVal[maxVal.detach()<=0]) #max value should never be negative

    loss = meanLoss + varLoss + minloss + maxLoss + maxConstraint
    return loss
# class bionetworkFunction2(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, weights, bias, A, networkList, parameters):
#         steps = parameters['iterations']
#         A.data = weights.detach().numpy()

#         bIn = x.transpose(0, 1).detach().numpy() + bias.detach().numpy()
#         xHat = numpy.zeros((steps+1,) + bIn.shape, dtype = bIn.dtype)
#         xRaw = numpy.zeros((steps+1,) + bIn.shape, dtype = bIn.dtype)

#         for i in range(steps):
#             xRaw[i, :, :] = A.dot(xHat[i, :, :])
#             xRaw[i, :, :] += bIn
#             xHat[i+1, :, :] = activation(xRaw[i, :, :], parameters['leak'])

#         output = torch.from_numpy(xHat[steps, :, :])
#         output = output.transpose(0, 1)
#         output.requires_grad_()

#         #Pass to backward
#         ctx.networkList = networkList
#         ctx.A = A
#         ctx.xRaw = xRaw
#         ctx.x = xHat
#         ctx.parameters = parameters
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         steps = ctx.parameters['iterations']
#         AT = ctx.A.T
#         gradIn = grad_output.transpose(0, 1).detach().numpy()
#         grad = gradIn.copy()
#         grad_weight = numpy.zeros(ctx.networkList.shape[1], dtype = gradIn.dtype)
#         grad_bias = numpy.zeros(grad.shape[0], dtype = gradIn.dtype)
#         output = numpy.zeros(gradIn.shape, dtype = gradIn.dtype)
#         deltaX = deltaActivation(ctx.xRaw, ctx.parameters['leak'])
#         #curNorm = norm(grad, 1)


#         for i in range(steps):
#             backIndex = steps-i-1
#             #grad = curNorm  * grad/norm(grad, 1) * numpy.exp((i/steps) * numpy.log(10**-1))
#             grad = deltaX[backIndex,:,:]*grad

#             output += grad
#             gradData = grad[ctx.networkList[0], :]  #this is annoyingly innefficent
#             xData = ctx.x[backIndex, ctx.networkList[1], :] #this is annoyingly innefficent
#             grad_weight += numpy.sum(numpy.multiply(xData, gradData), axis=1)

#             grad_bias += numpy.sum(grad, axis=1)

#             grad = AT.dot(grad)

#             #print(i, norm(grad, 1))
#             #as a precaution clipping with tanh for |gradients| > clipping val
#             #grad = gradCliping(grad, ctx.parameters['clipping'])
#             #The last grad corresponds to X * 0 so it can be skipped


#         output = torch.from_numpy(output).transpose(0, 1)
#         grad_weight = torch.from_numpy(grad_weight)
#         grad_bias = torch.from_numpy(grad_bias).unsqueeze(1)
#         return output, grad_weight, grad_bias, None, None, None


# def initializeWeights(networkList, modeOfAction, size, type):
#     #The idea is to scale input to a source so that well connected nodes have lower weights
#     baseWeight = 0.5
#     weights = baseWeight + 0.005 * torch.randn(networkList.shape[1])
#     weights.type(type)
#     bias = 0.05 + 0.001 * torch.randn((size, 1))
#     bias.type(type)

#     weights[modeOfAction[1] == True] =  -0.2 * weights[modeOfAction[1] == True]

#     for i in range(size):
#         affectedWeights = networkList[0,:] == i
#         n = numpy.sum(affectedWeights)
#         if n != 0:
#             #weights[affectedWeights] = weights[affectedWeights] * numpy.sqrt(2/n)
#             inFromWeights = baseWeight * sum(weights[affectedWeights])
#             factor = (baseWeight - bias[i])/inFromWeights #assums state of each node is ~0.5
#             if factor>0:
#                 weights[affectedWeights] = weights[affectedWeights] * factor;
#             else:
#                 bias[i] = baseWeight - inFromWeights #only inhibition, relies on bias for signal


#     #for this to work as intended network list must be sorted on index 0
#     A = scipy.sparse.csr_matrix((weights.detach().numpy(), networkList), shape=(size, size))

#     weights = weights.numpy()
#     N = 1000
#     stepSize = 0.01
#     tresh = 0.95
#     for i in range(N):
#          dW, e = eigRegularization(A, weights, networkList)
#          A.data = A.data-stepSize*dW.detach().numpy()
#          weights = A.data
#          if e<tresh:
#              break

#     weights = torch.tensor(weights)

#     return weights, bias

def generateRandomInput(model, N, simultaniousInput):
    sizeX = model.inputLayer.weights.shape[0]
    X = torch.zeros((N, sizeX), dtype=torch.double)
    for i in range(1, N): #leave first block blank
        selected = numpy.random.randint(sizeX, size=simultaniousInput)
        X[i, selected] = torch.rand(simultaniousInput, dtype=torch.double)
    return X

def reduceSpectralRadius(model, spectralTarget, localX, maxIter = 100):
    N = localX.shape[0]
    leak = model.network.param['leak']
    networkList = model.network.networkList

    localY, localFull = model(localX)
    localY = localY.clone().detach()
    criterion = torch.nn.MSELoss(reduction='mean')
    maxSr = numpy.inf
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005, weight_decay=0)
    noise = 1e-8
    srFactor =  1e-5

    print('Reducing spectral radius for all input (target ', spectralTarget, '):')
    for k in range(maxIter):
        optimizer.zero_grad()
        Yhat, YhatFull = model(localX)
        fitLoss = criterion(Yhat, localY)
        signConstraint = torch.sum(torch.abs(model.network.weights[model.network.getViolations()]))
        spectralRadius = torch.zeros(N, dtype=torch.double)
        spectralRegulation = torch.zeros(N, dtype=torch.double)
        for i in range(N):
            activationFactor = model.network.oneStepDeltaActivationFactor(localFull[i,:].flatten(), leak).detach()
            weightFactor = activationFactor[networkList[0]]
            multipliedWeightFactor = model.network.weights * weightFactor
            sr = model.network.getSpectralRadius(multipliedWeightFactor)
            spectralRadius[i] = sr.item()
            if spectralRadius[i]>spectralTarget:
                spectralRegulation[i] = srFactor * torch.abs(sr)
                #spectralClampFactor = 1/torch.max(spectralRadius[i]/bionetParams['spectralLimit'], torch.tensor(1.0).double()) #Prevents infinte penalty
                #spectralRegulation[i] = (1/(1 - spectralClampFactor*sr) - 1)
            else:
                spectralRegulation[i] = torch.tensor(0, dtype=torch.double, requires_grad=True)

        loss = 1e-5 * fitLoss + torch.sum(spectralRegulation) + signConstraint
        loss.backward()
        optimizer.step()
        model.network.weights.data = model.network.weights.data + torch.randn(model.network.weights.shape) * noise
        maxSr = numpy.max(spectralRadius.detach().numpy())
        print(k, maxSr)
        if maxSr<spectralTarget:
            break

    return model

def oneCycle(e, maxIter, maxHeight = 1e-3, startHeight=1e-5, endHeight=1e-5, minHeight = 1e-7, peak = 1000):
    phaseLength = 0.95 * maxIter
    if e<=peak:
        effectiveE = e/peak
        lr = (maxHeight-startHeight) * 0.5 * (numpy.cos(numpy.pi*(effectiveE+1))+1) + startHeight
    elif e<=phaseLength:
        effectiveE = (e-peak)/(phaseLength-peak)
        lr = (maxHeight-endHeight) * 0.5 * (numpy.cos(numpy.pi*(effectiveE+2))+1) + endHeight
    else:
        lr = endHeight
    return lr


def getSamples(N, batchSize):
    order = numpy.random.permutation(N)
    outList = []
    while len(order)>0:
        outList.append(order[0:batchSize])
        order = order[batchSize:]
    return outList

def getAllSpectralRadius(model, YhatFull):
    leak = model.network.param['leak']
    sr = numpy.zeros(YhatFull.shape[0])
    activationFactor = model.network.oneStepDeltaActivationFactor(YhatFull, leak)
    for i in range(len(sr)):
        weightFactor = activationFactor[i, model.network.networkList[0]]
        sr[i] = model.network.getSpectralRadius(model.network.weights * weightFactor).item()
    return sr

class bionet(nn.Module):
    def __init__(self, networkList, size, modeOfAction, parameters, activationFunction, dtype):
        super().__init__()
        self.param = parameters

        self.size_in = size
        self.size_out = size
        self.networkList = networkList
        self.modeOfAction = torch.tensor(modeOfAction)
        self.type = dtype

        # initialize weights and biases
        weights, bias = self.initializeWeights()

        #for this to work as intended network list must be sorted on index 0
        self.A = scipy.sparse.csr_matrix((weights.detach().numpy(), networkList), shape=(size, size), dtype='float64')

        self.weights = nn.Parameter(weights)
        self.bias = nn.Parameter(bias)
        
        if activationFunction == 'MML':
            self.activation = activationFunctions.MMLactivation
            self.delta = activationFunctions.MMLDeltaActivation
            self.oneStepDeltaActivationFactor = activationFunctions.MMLoneStepDeltaActivationFactor
        elif activationFunction == 'leakyRelu':
            self.activation = activationFunctions.leakyReLUActivation
            self.delta = activationFunctions.leakyReLUDeltaActivation
            self.oneStepDeltaActivationFactor = activationFunctions.leakyReLUoneStepDeltaActivationFactor     
        elif activationFunction == 'sigmoid':
            self.activation = activationFunctions.sigmoidActivation
            self.delta = activationFunctions.sigmoidDeltaActivation
            self.oneStepDeltaActivationFactor = activationFunctions.sigmoidOneStepDeltaActivationFactor

    def forward(self, x):
        return bionetworkFunction.apply(x, self.weights, self.bias, self.A, self.networkList, self.param, self.activation, self.delta)

    def getWeight(self, nodeNames, source, target):
        self.A.data = self.weights.detach().numpy()
        locationSource = numpy.argwhere(numpy.isin(nodeNames, source))[0]
        locationTarget = numpy.argwhere(numpy.isin(nodeNames, target))[0]
        weight = self.A[locationTarget, locationSource][0]
        return weight

    def getViolations(self, weights = None):
        if weights == None:
            weights = self.weights.detach()
        wrongSignActivation = torch.logical_and(weights<0, self.modeOfAction[0] == True)#.type(torch.int)
        wrongSignInhibition = torch.logical_and(weights>0, self.modeOfAction[1] == True)#.type(torch.int)
        return torch.logical_or(wrongSignActivation, wrongSignInhibition)
    
    def getNumberOfViolations(self):
        return torch.sum(self.getViolations(self.weights))
    
    def signRegularization(self, MoAFactor):
        return MoAFactor * torch.sum(torch.abs(self.weights[self.getViolations(self.weights)]))

    def balanceWeights(self):
        positiveWeights = self.weights.data>0
        negativeWeights = positiveWeights==False
        positiveSum = torch.sum(self.weights.data[positiveWeights])
        negativeSum = -torch.sum(self.weights.data[negativeWeights])
        factor = positiveSum/negativeSum
        self.weights.data[negativeWeights] = factor * self.weights.data[negativeWeights]    

    # def getSpectralLoss(self):
    #     self.A.data = self.weights.detach().numpy()
    #     e, v, w = lreigs(self.A)
    #     spectralRadius = numpy.abs(e)
    #     divisor = w.T.dot(v)

    #     if abs(divisor)>10**-4:
    #         w = w/divisor #Ensures wT*v=1
    #         delta = numpy.multiply(w[self.networkList[0]], v[self.networkList[1]])
    #         delta = numpy.squeeze(delta)
    #         direction = spectralRadius/e
    #         delta = (direction * delta).real
    #         deltaFilter = numpy.not_equal(numpy.sign(delta), numpy.sign(self.weights.detach().numpy()))
    #         delta[deltaFilter] = 0
    #     else:
    #         print('missmatch using L2 instead')
    #         delta = numpy.sign(self.weights.detach().numpy())  #Could occure if degenerate eigenvalues
    #     delta = delta/norm(delta, 1)
    #     delta = torch.tensor(delta).reshape([-1,1])
    #     return delta, spectralRadius
    def getSpectralRadius(self, weights):
        return spectralRadius.apply(weights, self.A, self.networkList)

    def getRevSpectralRadius(self, weights):
        return spectralRadius.apply(weights, self.A.T, self.networkList)

    def initializeWeights(self):
        #The idea is to scale input to a source so that well connected nodes have lower weights
        #weights = 0.5 + 0.1 * (torch.rand(self.networkList.shape[1])-0.5)
        #bias = 0.01 + 0.001 * (torch.rand(self.size_in,1)-0.5)

        weights = 0.1 + 0.1 * torch.rand(self.networkList.shape[1], dtype=self.type)
        weights[self.modeOfAction[1,:]] = -weights[self.modeOfAction[1,:]]
        bias = 1e-3 * torch.ones((self.size_in, 1), dtype=self.type)
        #values, counts = np.unique(self.networkList[0,:], return_counts=True)

        for i in range(self.size_in):
            affectedIn = self.networkList[0,:] == i
            if numpy.any(affectedIn):
                if torch.all(weights[affectedIn]<0):
                    bias.data[i] = 1 #only affected by inhibition, relies on bias for signal

        # for i in range(self.size_in):
        #     affectedIn = self.networkList[0,:] == i
        #     fanIn = max(sum(affectedIn), 1)
        #     affectedOut = self.networkList[0,:] == i
        #     fanOut = max(sum(affectedOut), 1)
        #     weights.data[affectedIn] = weights.data[affectedIn] * numpy.sqrt(2.0/numpy.sqrt(fanIn * fanOut))

        return weights, bias


    def preScaleWeights(self, targetRadius = 0.8):
        spectralRadius = self.getSpectralRadius(self.weights)
        factor = targetRadius/spectralRadius.item()
        self.weights.data = self.weights.data * factor
        # print('Pre-scaling eig')
        # optimizer = torch.optim.Adam([self.weights], lr=0.001)
        # weightFactor = self.weights
        # for i in range(1000):
        #     optimizer.zero_grad()
        #     spectralRadius = self.getSpectralRadius(weightFactor)
        #     if i % 20 == 0:
        #         print('i={:.0f}, e={:.4f}'.format(i, spectralRadius.item()))

        #     if spectralRadius.item()>targetRadius:
        #         spectralRadius.backward()
        #         optimizer.step()
        #     else:
        #         break


    # def getSpectralR(self, Xfull):
    #     rawX =


# def ismember(a, b):
#     bind = {}
#     for i, elt in enumerate(b):
#         if elt not in bind:
#             bind[elt] = i
#     return [bind.get(itm, None) for itm in a]

class projectInput(nn.Module):
    def __init__(self, nodeList, inputNames, amplitude, type):
        super().__init__()

        self.size_in = len(inputNames)
        self.size_out = len(nodeList)
        self.type = type
        dictionary = dict(zip(nodeList, list(range(len(nodeList)))))
        self.nodeOrder = numpy.array([dictionary[x] for x in inputNames])
        weights = amplitude * torch.ones(len(inputNames), dtype=type)
        self.weights = nn.Parameter(weights)

    def forward(self, x):
        curIn = torch.zeros([x.shape[0],  self.size_out], dtype=self.type)
        curIn[:, self.nodeOrder] = self.weights * x
        return curIn

class projectOutput(nn.Module):
    def __init__(self, nodeList, outputNames, scaleFactor, type):
        super().__init__()

        self.size_in = len(nodeList)
        self.size_out = len(outputNames)
        self.type = type

        dictionary = dict(zip(nodeList, list(range(len(nodeList)))))
        self.nodeOrder = numpy.array([dictionary[x] for x in outputNames])

        #bias = torch.zeros(len(outputNames), dtype=type)
        weights = scaleFactor * torch.ones(len(outputNames), dtype=type)
        self.weights = nn.Parameter(weights)

    def forward(self, x):
        curOut = self.weights * x[:, self.nodeOrder]
        return curOut


def getEigenvalue(model):
    try:
        eigenValue = abs(eigs(model.A, k=1)[0])
        eigenvalue = torch.from_numpy(eigenValue)
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        eigenvalue = numpy.nan
    return eigenvalue


# def matrixRegularization(AT, networkList, deltaX, regFactor):
#     if regFactor > 0:
#         localA = AT.T.copy()
#         generalizedDelta = numpy.amax(deltaX, axis=1)
#         #localA = generalizedDelta.dot(AT)
#         weights = localA.data.copy()
#         weightFactor = generalizedDelta[networkList[0]]
#         localA.data = weightFactor * localA.data

#         dW, e = eigRegularization(localA, weights, networkList)
#         dW = regFactor * dW
#     else:
#         dW = torch.zeros(networkList.shape[1])
#     return dW

# def eigRegularization(A, weights, networkList):
#     e, v, w = lreigs(A)
#     spectralRadius = abs(e)

#     divisor = w.T.dot(v)

#     if abs(divisor)>10**-4:
#         w = w/divisor #Ensures wT*v=1
#         delta = numpy.multiply(w[networkList[0]], v[networkList[1]])
#         delta = numpy.squeeze(delta)
#         direction = abs(e)/e
#         delta = (delta * direction).real
#         delta[numpy.sign(delta) != numpy.sign(weights)] = 0
#     else:
#         print('missmatch using L2 instead')
#         delta = 0.1 * numpy.sign(weights)  #Could occure if degenerate eigenvalues

#     delta = numpy.multiply(delta, abs(weights))
#     #delta = delta/norm(delta)
#     delta = (spectralRadius**2) * delta
#     dW = torch.from_numpy(delta)
#     return dW, spectralRadius

# def lreigs(A):
#     try:
#         e, v = eigs(A, k=1, tol=10**-5, ncv = 20)
#         e = e[0]
#         tmpA = A.toarray().T  #faster than sparse in my numerical tests
#         eT = e + 10**-7
#         if numpy.isreal(eT):
#             eT = eT.real
#         e2, w = eigs(tmpA, k=1, sigma=eT, tol=10**-5, ncv=20, OPpart='r')
#         e2 = e2[0]
#         #Check if same eigenvalue
#         if abs(e-e2)>10**-3:
#             print('fail (eigs left returned different eigenvalue)')
#             e, w, v = lreig(tmpA) #fall back to solving whole eig problem
#     except (KeyboardInterrupt, SystemExit):
#         raise
#     except:
#         print('fail (did not find any eigenvalue with eigs)')
#         tmpA = A.toarray()
#         e, v, w = lreig(tmpA) #fall back to solving whole eig problem

#     # #Check if not orthogonal (implies degenerate eigenvalues)
#     # divisor = w.T.dot(v)
#     # if (abs(divisor)<10**-4):
#     #     print('fail (divisor lower than tolerance maybe a degenerate eigenvalue)')
#     #     e, w, v = lreig(tmpA) #fall back to solving whole eig problem
#     return e, v, w

def lreig(A):
    #fall back if eigs fails
    e, w, v = eig(A, left = True)
    selected = numpy.argmax(numpy.abs(e))
    eValue = e[selected]
    # selected = (e == eValue)

    # if numpy.sum(selected) == 1:
    w = w[:,selected]
    v = v[:,selected]
    # else:
    #     w = numpy.sum(w[:,selected], axis=1, keepdims=True)
    #     v = numpy.sum(v[:,selected], axis=1, keepdims=True)
    #     w = w/norm(w)
    #     v = v/norm(v)
    return eValue, v, w

def getRandomNet(networkSize, sparsity):
    network = scipy.sparse.random(networkSize, networkSize, sparsity)
    scipy.sparse.lil_matrix.setdiag(network, 0)
    networkList = scipy.sparse.find(network)
    networkList = numpy.array((networkList[1], networkList[0])) #we flip the network for rowise ordering
    nodeNames = [str(x+1) for x in range(networkSize)]
    #weights = torch.from_numpy(networkList[2])
    return networkList, nodeNames

def loadNetwork(filename, banList = []):
    net = pandas.read_csv(filename, sep='\t', index_col=False)
    net = net[~ net["source"].isin(banList)]
    net = net[~ net["target"].isin(banList)]

    sources = list(net["source"])
    targets = list(net["target"])
    stimulation = numpy.array(net["stimulation"])
    inhibition = numpy.array(net["inhibition"])
    modeOfAction = 0.1 * numpy.ones(len(sources))
    modeOfAction[stimulation==1] = 1
    modeOfAction[inhibition==1] = -1

    networkList, nodeNames, weights = makeNetworkList(sources, targets, modeOfAction)  #0 == Target 1 == Source due to numpy sparse matrix structure
    modeOfAction = numpy.array([[weights==1],[weights==-1]]).squeeze()

    return networkList, nodeNames, modeOfAction

def makeNetworkList(sources, targets, weights):
    nodeNames = list(numpy.unique(sources + targets))
    dictionary = dict(zip(nodeNames, list(range(len(nodeNames)))))
    sourceNr = numpy.array([dictionary[x] for x in sources]) #colums
    targetNr = numpy.array([dictionary[x] for x in targets]) #rows
    size = len(nodeNames)
    A = scipy.sparse.csr_matrix((weights, (sourceNr, targetNr)), shape=(size, size))
    networkList = scipy.sparse.find(A)
    weights = networkList[2]
    networkList = numpy.array((networkList[1], networkList[0]))  #0 == Target 1 == Source due to numpy sparse matrix structure
    return networkList, nodeNames, weights


def trainingParameters(**attributes):
    #set defaults
    params = {'iterations': 150, 'leak': 0.01, 'clipping': 1,  'targetPrecision': 1e-4}

    for curKey in params.keys():
        if curKey in attributes.keys():
            params[curKey] = attributes[curKey]

    if 'spectralTarget' in attributes.keys():
        params['spectralTarget'] = attributes['spectralTarget']
    else:
        params['spectralTarget'] = numpy.exp(numpy.log(params['targetPrecision'])/params['iterations'])

    return params



def saveParam(model, nodeList, fileName):
    nodeList = numpy.array(nodeList).reshape([-1, 1])

    #Weights
    networkList = model.network.networkList
    sources = nodeList[networkList[1]]
    targets = nodeList[networkList[0]]
    paramType = numpy.array(['Weight'] * len(sources)).reshape([-1, 1])
    values = model.network.weights.detach().numpy().reshape([-1, 1])
    data1 = numpy.concatenate((sources, targets, paramType, values), axis=1)

    #Bias
    sources = nodeList
    targets = numpy.array([''] * len(sources)).reshape([-1, 1])
    paramType = numpy.array(['Bias'] * len(sources)).reshape([-1, 1])
    values = model.network.bias.detach().numpy().reshape([-1, 1])
    data2 = numpy.concatenate((sources, targets, paramType, values), axis=1)

    #Projection
    projectionList = model.projectionLayer.nodeOrder
    sources = nodeList[projectionList]
    targets = numpy.array([''] * len(sources)).reshape([-1, 1])
    paramType = numpy.array(['Projection'] * len(sources)).reshape([-1, 1])
    values = model.projectionLayer.weights.detach().numpy().reshape([-1, 1])
    data3 = numpy.concatenate((sources, targets, paramType, values), axis=1)

    #Input projection
    projectionList = model.inputLayer.nodeOrder
    sources = nodeList[projectionList]
    targets = numpy.array([''] * len(sources)).reshape([-1, 1])
    paramType = numpy.array(['Input'] * len(sources)).reshape([-1, 1])
    values = model.inputLayer.weights.detach().numpy().reshape([-1, 1])
    data4 = numpy.concatenate((sources, targets, paramType, values), axis=1)

    data = numpy.concatenate((data1, data2, data3, data4))

    pd = pandas.DataFrame(data)
    pd.columns = ['Source', 'Target', 'Type', 'Value']
    pd.to_csv(fileName, sep = '\t', quoting = None, index = False)

def loadParam(fileName, model, nodeNames):
    dictionary = dict(zip(nodeNames, list(range(len(nodeNames)))))
    data = pandas.read_csv(fileName, delimiter = '\t')

    #Reset model to zero
    model.inputLayer.weights.data = torch.zeros(model.inputLayer.weights.shape)
    model.network.weights.data = torch.zeros(model.network.weights.shape)
    model.network.bias.data = torch.zeros(model.network.bias.shape)
    model.projectionLayer.weights.data = torch.zeros(model.projectionLayer.weights.shape)


    inputLookup = model.inputLayer.nodeOrder
    networkLookup = model.network.networkList #model.network.A.nonzero()
    projectionLookup = model.projectionLayer.nodeOrder

    for i in range(data.shape[0]):
        curRow = data.iloc[i,:]
        source = dictionary[curRow['Source']]
        value = curRow['Value']
        if curRow['Type'] == 'Weight':
            target = dictionary[curRow['Target']]
            weightNr = numpy.argwhere(numpy.logical_and(networkLookup[1,:] == source, networkLookup[0,:] == target))
            model.network.weights.data[weightNr] = value
        elif curRow['Type'] == 'Bias':
            model.network.bias.data[source] = value
        elif curRow['Type'] == 'Projection':
            model.projectionLayer.weights.data[projectionLookup == source] = value
        elif curRow['Type'] == 'Input':
            model.inputLayer.weights.data[inputLookup == source] = value
    return model


def getMeanLoss(criterion, Y):
    averagePerOutput = torch.mean(Y, dim=0)
    medianPerOutput = torch.from_numpy(numpy.median(Y, axis=0))
    errorFromPredictingTheMean = criterion(Y, averagePerOutput)
    errorFromPredictingTheMedian = criterion(Y, medianPerOutput)
    return (errorFromPredictingTheMean.item(), errorFromPredictingTheMedian.item())


def sensitivityAnalysis(model, nodeNames, X, conditionName, fileName):
    downValue = -10
    upValue = 10
    nodeNames = numpy.array(nodeNames, dtype=object)
    n = len(nodeNames)

    Xfull = model.inputLayer(X)
    ctrlMatrix = numpy.zeros((2, n))
    downMatrix = downValue * numpy.identity(n)
    upMatrix = upValue * numpy.identity(n)

    joinedMatrix = numpy.concatenate((ctrlMatrix, downMatrix, upMatrix))
    upNames = nodeNames + '_u'
    downNames = nodeNames + '_d'
    ctrlName = numpy.array(['null', 'ctrl'])
    joinedNames = numpy.concatenate((ctrlName, downNames, upNames))
    joinedMatrix = torch.tensor(joinedMatrix)
    joinedMatrix = joinedMatrix + Xfull
    joinedMatrix[0,:] = 0

    sensitivityFull = model.network(joinedMatrix)
    df = pandas.DataFrame(sensitivityFull.detach().numpy(), index=joinedNames, columns=nodeNames)
    df = df.round(decimals=3)
    df.to_csv(fileName, sep='\t')

def generateConditionNames(X, inName):
    inName = numpy.array(inName)
    X = X.detach().numpy()
    names = numpy.empty(X.shape[0], dtype=object)
    for i in range(len(names)):
        curSelection = X[i,:]>0
        if sum(curSelection) == 0:
            names[i] = '(none)'
        else:
            curLigands = list(inName[curSelection])
            names[i] = '_'.join(curLigands)
    return names


#For reference:
class bionetworkAutoGrad(nn.Module):
    def __init__(self, networkList, size, reps=150):
        super().__init__()
        self.size_in = size
        self.size_out = size

        #requires_grad=False
        bias = torch.Tensor(size, 1)
        self.bias = nn.Parameter(bias, requires_grad = True)
        self.reps = reps
        self.leak = 0.01

        # initialize weights and biases
        weights = torch.Tensor(len(networkList[0]))
        nn.init.uniform_(weights, -0.1, 0.1) # weight init
        nn.init.uniform_(self.bias, -0.1, 0.1)  # bias init

        self.A = torch.sparse.FloatTensor(torch.from_numpy(networkList).long(), weights, torch.Size([size, size])).double()
        self.A = nn.Parameter(self.A, requires_grad = True)
        self.A.values = weights

    def forward(self, x):
        #Load into memory
        bIn = x.transpose(0, 1)
        curB = torch.add(bIn, self.bias)
        xhat = torch.zeros(bIn.shape, dtype=bIn.dtype)

        #print(self.A)
        for i in range(self.reps):
            xhat = torch.sparse.mm(self.A, xhat)
            xhat = torch.add(xhat, curB)
            xhat[xhat<0] = self.leak*xhat[xhat<0]
            xhat[xhat>0.5] = 0.5 * (1 + (1/(0.5/(xhat[xhat>0.5]-0.5) + 1)))
            #xhat[xhat>0] = 1/(0.5/(xhat[xhat>0]) + 1)

        output = xhat.transpose(0, 1)
        return output


