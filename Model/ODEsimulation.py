import torch
import numpy
import pandas
import seaborn as sns
from scipy import interpolate
import matplotlib.pyplot as plt
import time
import plotting
from scipy.stats import pearsonr

def plotACondition(X, Y, Yhat, YtestHat, Ytest):
    plt.figure()
    ax1=plt.subplot(2, 2, 1)
    df = pandas.DataFrame.from_dict(numpy.array([X[:,0].numpy(), X[:,1].numpy(), Y[:,0].numpy()]).T)
    df.columns = ['X_value','Y_value','Z_value']
    pivotted= df.pivot('Y_value','X_value','Z_value')
    pivotted.columns = pivotted.columns.values.round(3)
    pivotted.index = pivotted.index.values.round(3)
    sns.heatmap(pivotted, cmap='gray', vmin=0, vmax=1)
    plt.title('data')

    ax1=plt.subplot(2, 2, 2)
    df = pandas.DataFrame.from_dict(numpy.array([X[:,0].numpy(), X[:,1].numpy(), Yhat[:,0].numpy()]).T)
    df.columns = ['X_value','Y_value','Z_value']
    pivotted = df.pivot('Y_value','X_value','Z_value')
    pivotted.columns = pivotted.columns.values.round(3)
    pivotted.index = pivotted.index.values.round(3)
    sns.heatmap(pivotted, cmap='gray', vmin=0, vmax=1)
    plt.title('model')

    ax1=plt.subplot(2, 2, 3)
    delta = Y.numpy()-Yhat.numpy()
    df = pandas.DataFrame.from_dict(numpy.array([X[:,0].numpy(), X[:,1].numpy(), delta[:,0]]).T)
    df.columns = ['X_value','Y_value','Z_value']
    pivotted = df.pivot('Y_value','X_value','Z_value')
    pivotted.columns = pivotted.columns.values.round(3)
    pivotted.index = pivotted.index.values.round(3)
    sns.heatmap(pivotted, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('data-model')

    ax1=plt.subplot(2, 2, 4)
    #plt.scatter(Yhat.numpy(), Y.numpy(), alpha=0.5)
    #plt.scatter(YtestHat.numpy(), Ytest.numpy(), alpha=0.5)
    #plt.title('data vs model')
    #plotting.lineOfIdentity()
    plotting.plotComparison(Yhat, Y, YtestHat, Ytest)
    plt.tight_layout()


def interpolateData(df, resolution, normalize):
    epsilon=0
    df = df.copy()
    df.columns = df.columns.astype('float')
    df.index = df.index.astype('float')
    X = df.index
    Y = df.columns
    points = (X, Y)
    x, y = numpy.meshgrid(X,Y)
    z = df.values

    outputGrid = numpy.linspace(epsilon, 1-epsilon, resolution)
    xR, yR = numpy.meshgrid(outputGrid, outputGrid, indexing='ij')

    f = interpolate.interpn(points, z, (xR, yR), method='linear')
    if normalize:
        f = f/numpy.max(f)

    torchX = torch.tensor((xR.flatten(), yR.flatten()), dtype=torch.float).T
    torchY = torch.tensor(f.flatten(), dtype=torch.float).reshape(-1,1)
    return torchX, torchY


#For reference:
class linearNetwork(torch.nn.Module):
    def __init__(self, hiddenLayers, width, aFunction):
        super().__init__()
        self.size_in = 2
        self.size_out = 1
        self.leak = 0.01
        self.hiddenLayers = hiddenLayers
        self.aFunction = aFunction
        self.scaleFactor = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.double))

        if hiddenLayers == 0:
            self.layers = torch.nn.ModuleList([torch.nn.Linear(self.size_in, self.size_out, bias=True)])
        else:
            self.layers = torch.nn.ModuleList([torch.nn.Linear(self.size_in, width, bias=True)])
            for i in range(hiddenLayers-1):
                self.layers.append(torch.nn.Linear(width, width, bias=True))
            self.layers.append(torch.nn.Linear(width, self.size_out, bias=True))

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.activationFunction(x)
            x = self.scaleFactor * x
        return x

    def activationFunction(self, x):
        if self.aFunction == 'MML':
            x = self.MMLactivation(x)
        elif self.aFunction == 'relu':
            x = torch.nn.functional.leaky_relu(x)
        elif self.aFunction == 'sigmoid':
            x = torch.sigmoid(x)
        return x

    # def activationFunction(self, x):
    #     x = torch.atan(x)
    #     return x

    # def activationFunction(self, x):
    #     #xFilter = x<0
    #     #x[xFilter] = self.leak*x[xFilter]
    #     #xFilter = x>0
    #     #k = 2.0
    #     #x[xFilter] = (1.5708 + torch.atan(k*(x[xFilter]-1)))/(2*1.5708)
    #     x = (1.5 + torch.atan(x-2))/(2*1.5)
    #     #x[xFilter] = 1/(1 + torch.exp(-x[xFilter]+5))
    #     #xFilter = x>0.5
    #     #x[xFilter] = 0.5 * (1 + (1/(0.5/(x[xFilter]-0.5) + 1)))
    #     # xFilter = x>0
    #     # x[xFilter] = 1/(0.5/(x[xFilter]) + 1)
    #     return x

    def MMLactivation(self, x):
        xFilter = x<=0
        x[xFilter] = self.leak*x[xFilter]
        xFilter = x>0.5
        x[xFilter] = 1 - (0.25/x[xFilter])
        return x

def NNloop(model, maxIter, lr, noiseLevel, trainloader):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(list(model.parameters()), lr=lr, weight_decay=1e-5)
    resetState = optimizer.state.copy()
    storeLoss = torch.ones(maxIter)
    for e in range(maxIter):
        curLoss = 0
        for dataIn, dataOut in trainloader:
            optimizer.zero_grad()
            #dataIn = dataIn + noiseLevel * (torch.randn(dataIn.shape) * dataIn.detach())
            #dataOut = dataOut + noiseLevel * torch.randn(dataOut.shape)
            output = model(dataIn)
            loss = criterion(output, dataOut)
            loss.backward()
            optimizer.step()

            for param in model.parameters():
                param.data = param.data + noiseLevel * torch.randn(param.shape)

            curLoss+= loss.item()
        storeLoss[e] = curLoss/numpy.ceil(len(trainset)/trainloader.batch_size)

        if numpy.logical_and(e % 100 == 0, e>0):
            optimizer.state = resetState.copy()
    return model, storeLoss



normalizeData = True

replicates = 10
maxIter = 5000
noiseLevel = 0
lr = 0.002
hiddenNodes = 5

folder = 'data/evaluatedODE'
ODEfiles = ['independentActivation.tsv',
            'independentDeActivation.tsv',
            'cooperativeActivation.tsv',
            'competitiveInhibition.tsv']
ODENames = ['A', 'I', 'CA', 'CI']

activationFunction = ['sigmoid', 'relu', 'MML']
numberOfLayers = [0, 1]



#%%
results = numpy.zeros((len(ODEfiles), len(activationFunction), len(numberOfLayers), replicates))

start = time.time()
for i in range(len(ODEfiles)):
    print(ODEfiles[i])
    df = pandas.read_csv(folder + '/' + ODEfiles[i], sep='\t', low_memory=False, index_col=0)
    X, Y = interpolateData(df, 7, normalizeData)
    Xtest, Ytest = interpolateData(df, 20, normalizeData)
    trainset = torch.utils.data.TensorDataset(X, Y)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True) #round(len(trainset)/20),
    for j in range(len(activationFunction)):
        for k in range(len(numberOfLayers)):
            for l in range(replicates):
                model = linearNetwork(numberOfLayers[k], hiddenNodes, activationFunction[j])
                model, storeLoss = NNloop(model, maxIter, lr, noiseLevel, trainloader)
                YtestHat = model(Xtest).detach()
                r, p = pearsonr(YtestHat.flatten(), Ytest.flatten())
                results[i, j, k, l] = r

meanResults = numpy.mean(results, axis =3)
stdResults = numpy.std(results, axis =3)
print('Time:', time.time()-start)

resultsLayer0 = meanResults[:,:,0]
resultsLayer1 = meanResults[:,:,1]


df0 = pandas.DataFrame(resultsLayer0, index=ODENames, columns = activationFunction)
df1 = pandas.DataFrame(resultsLayer1, index=ODENames, columns = activationFunction)


plt.rcParams["figure.figsize"] = (3,2)
sns.heatmap(df0, annot=True, fmt="2.3f", cmap='gray', vmin=0.85, vmax=1)
plt.figure()
sns.heatmap(df1, annot=True, fmt="2.3f", cmap='gray', vmin=0.85, vmax=1)

# #%%
# plt.semilogy(range(len(storeLoss)), storeLoss, 'o', color='black')
# plt.title('Loss')


#%%
plt.rcParams["figure.figsize"] = (6,6)

# Yhat = model(X).detach()

# Xtest, Ytest = interpolateData(df, 20, normalizeData)
# YtestHat = model(Xtest).detach()
# #plotACondition(X, Y, Yhat, YtestHat, Ytest)

reference =numpy.zeros((len(ODEfiles), 20*20))
results = numpy.zeros((len(ODEfiles), len(numberOfLayers), 20*20))

start = time.time()
for i in range(len(ODEfiles)):
    print(ODEfiles[i])
    df = pandas.read_csv(folder + '/' + ODEfiles[i], sep='\t', low_memory=False, index_col=0)
    X, Y = interpolateData(df, 7, normalizeData)
    Xtest, Ytest = interpolateData(df, 20, normalizeData)
    trainset = torch.utils.data.TensorDataset(X, Y)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True) #round(len(trainset)/20),
    reference[i,:] = Ytest.detach().numpy().flatten()
    for j in range(len(numberOfLayers)):
        model = linearNetwork(numberOfLayers[j], hiddenNodes, 'MML')
        model, storeLoss = NNloop(model, maxIter, lr, noiseLevel, trainloader)
        YtestHat = model(Xtest).detach()
        results[i,j,:] = YtestHat.detach().numpy().flatten()
print('Time:', time.time()-start)


titles = ['IA', 'II', 'CA', 'CI']
plt.rcParams["figure.figsize"] = (12,3)
plt.figure()
for i in range(len(ODEfiles)):
    plt.subplot(1, 4, 1+i)
    plt.scatter(results[i, 0,:], reference[i,:])
    plt.scatter(results[i, 1,:], reference[i,:])
    r1, p = pearsonr(results[i, 0,:], reference[i,:])
    r2, p = pearsonr(results[i, 1,:], reference[i,:])
    plt.text(0.6, 0.1, '0: r={:.2f}\n1: r={:.2f}'.format(r1, r2))
    print(r1, r2)
    plt.gca().axis('equal')
    plt.xlabel('Fit')
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plotting.lineOfIdentity()
    plt.title(titles[i])

    if i == 0:
        plt.ylabel('Reference data')
        plt.legend(['0 layers', '1 layer'], frameon=False)

#    plt.gca().set_xticks([0,0.5,1])
#    plt.gca().set_yticks([0,0.5,1])
