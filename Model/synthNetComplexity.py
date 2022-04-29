import torch
import numpy
import matplotlib.pyplot as plt
import bionetwork
import pandas
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
#from sklearn.manifold import TSNE
import umap
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import seaborn as sns
import copy
from scipy.sparse.linalg import eigs
from scipy.linalg import eig
import scipy.sparse as sparse
import activationFunctions

#Load network
networkList, nodeNames, modeOfAction = bionetwork.loadNetwork('data/KEGGnet-Model.tsv')
annotation = pandas.read_csv('data/KEGGnet-Annotation.tsv', sep='\t')
uniprot2gene = dict(zip(annotation['code'], annotation['name']))


inputAmplitude = 3
projectionAmplitude = 1.2

inName = annotation.loc[annotation['ligand'],'code'].values
outName = annotation.loc[annotation['TF'],'code'].values
inName = numpy.intersect1d(nodeNames, inName)
outName = numpy.intersect1d(nodeNames, outName)
inNameGenes = [uniprot2gene[x] for x in inName]
outNameGenes = [uniprot2gene[x] for x in outName]
nodeNameGene = [uniprot2gene[x] for x in nodeNames]

bionetParams = bionetwork.trainingParameters(iterations = 150, clipping=1, leak=0.01)
model = bionetwork.model(networkList, nodeNames, modeOfAction, inputAmplitude, projectionAmplitude, inName, outName, bionetParams, 'MML', torch.double)
model = bionetwork.loadParam('synthNetModel/equationParams.txt', model, nodeNames)


folder = 'figures/Figure 4/'

#%%
N=100
simultaniousInput = 5
data = torch.zeros(N, len(inName), dtype=torch.double)

for i in range(N):
    data[i, numpy.random.randint(0, len(inName), simultaniousInput)] = torch.rand(simultaniousInput, dtype=torch.double)

conditions = bionetwork.generateConditionNames(data, inNameGenes)

Yhat, YhatFull = model(data)
df = pandas.DataFrame(Yhat.T.detach().numpy(), index=outNameGenes, columns=conditions)
h = sns.clustermap(df, cmap='RdBu_r', vmin=0, vmax=1) #, yticklabels=True
plt.savefig(folder + 'C.svg')
h.data2d.to_csv(folder + 'C.tsv', sep='\t')

#%%
#Illustrate transients
A = model.network.A
A.data = model.network.weights.detach().numpy()

#Define input
u = model.inputLayer(data)
u = u[50,:].reshape(-1, 1)
u = u.detach().numpy() + model.network.bias.detach().numpy()

maxIter = 50
yFull = numpy.zeros((u.shape[0], maxIter))
for i in range(maxIter-1):
    yFull[:, i+1] = A.dot(yFull[:,i]) + u.flatten()
    yFull[:, i+1] = activationFunctions.MMLactivation(yFull[:, i+1])
y = yFull[model.projectionLayer.nodeOrder, :]
df = pandas.DataFrame(y, index=numpy.array(nodeNameGene)[model.projectionLayer.nodeOrder])
sns.clustermap(df, cmap='RdBu_r', vmin=0, vmax=1, col_cluster=False, yticklabels=True)
plt.ylabel('TF activity')


#%%
#Demonstrate random model
refState = copy.deepcopy(model.state_dict())
curState = model.state_dict()
weights = curState['network.weights'] 
curState['network.weights']  = weights[numpy.random.permutation(len(weights))]
#respect sign:
curState['network.weights'][modeOfAction[0]] = numpy.abs(curState['network.weights'][modeOfAction[0]])
curState['network.weights'][modeOfAction[1]] = -numpy.abs(curState['network.weights'][modeOfAction[1]])

curState['network.weights'] = curState['network.weights']
bias = curState['network.bias'] 
curState['network.bias']  = bias[numpy.random.permutation(bias.shape[0]),:]

model.load_state_dict(curState)

#model.network.balanceWeights()
#model.network.preScaleWeights()

Yhat, YhatFull = model(data)
df = pandas.DataFrame(Yhat.T.detach().numpy(), index=outNameGenes, columns=conditions)
h = sns.clustermap(df, cmap='RdBu_r', vmin=0, vmax=1) #, yticklabels=True
plt.savefig(folder + 'B.svg')
h.data2d.to_csv(folder + 'B.tsv', sep='\t')

plt.savefig('figures/generativeSynthnet/random.svg')

plt.figure()
meanY = torch.mean(YhatFull, axis=0).detach().numpy()

model.load_state_dict(refState)

#%%
#Demonstrate effect of random structure and parameters
def runIterations(A, u, activationFunction='none', iterations=150):
    yFull = numpy.zeros(u.shape)
    for j in range(iterations):
        yFull = A.dot(yFull) + u
        if activationFunction == 'relu':
            yFull = activationFunctions.leakyReLUActivation(yFull)
        elif activationFunction == 'MML':
            yFull = activationFunctions.MMLactivation(yFull)
    return yFull

def scaleSpectralRadius(A, targetSR = 0.95):
    e, v = eig(A.toarray())
    A.data = A.data * targetSR/numpy.max(numpy.abs(e))
    return A

def calulateCorrelation(yFull):
    y = model.projectionLayer(torch.tensor(yFull.T))
    correlation = numpy.mean(numpy.abs(numpy.corrcoef(y.detach().numpy())))
    return correlation
    
def correlationInRandomMatrix(A, u, matrixDensity, activationFunction):
    mask = sparse.random(A.shape[0], A.shape[1], density = matrixDensity).toarray() == 0
    Arandom = numpy.random.normal(0, 1, A.shape)
    Arandom[mask] = 0
    Arandom = sparse.csr_matrix(Arandom)
    
    Arandom = scaleSpectralRadius(Arandom)
    yFull = runIterations(Arandom, u, activationFunction)
    correlation = calulateCorrelation(yFull)
    return correlation

def correlationInBioMatrix(A, u, activationFunction):
    #Randomize parameters and respect sign
    A.data = numpy.random.randn(model.network.weights.shape[0])
    A.data[modeOfAction[0]] = numpy.abs(A.data[modeOfAction[0]])
    A.data[modeOfAction[1]] = -numpy.abs(A.data[modeOfAction[1]])
    
    A = scaleSpectralRadius(A)
    yFull = runIterations(A, u, activationFunction)
    #print(numpy.sum(abs(yFull)<0.05))
    correlation = calulateCorrelation(yFull)
    return correlation


plt.rcParams["figure.figsize"] = (5,3)
plt.figure()
refState = copy.deepcopy(model.state_dict())
model.load_state_dict(refState)
A = model.network.A.copy()
matrixDensity = model.network.weights.shape[0]/numpy.square(A.shape[0])
#matrixDensity = 0.05

#Define input
u = model.inputLayer(data)
referenceY = model.network(u).T
u = u.detach().numpy().T
#

#b = model.network.bias.detach().numpy()
#u = u + b
#testY = runIterations(A, u, 'MML')
#plt.scatter(testY.flatten(), referenceY.detach().numpy().flatten())

plt.figure()
referenceCorr = calulateCorrelation(referenceY.detach().numpy())
# referenceCorr = numpy.zeros(3)
# referenceCorr[0] = calulateCorrelation(runIterations(A, u, 'none'))
# referenceCorr[1] = calulateCorrelation(runIterations(A, u, 'relu'))
# referenceCorr[2] = calulateCorrelation(runIterations(A, u, 'MML'))

nIterations = 20
results = numpy.zeros((nIterations, 6))
for i in range(nIterations):
    results[i, 0] = correlationInRandomMatrix(A, u, matrixDensity, 'none')
    results[i, 1] = correlationInRandomMatrix(A, u, matrixDensity, 'relu')
    results[i, 2] = correlationInRandomMatrix(A, u, matrixDensity, 'MML')

    results[i, 3] = correlationInBioMatrix(A, u, 'none')
    results[i, 4] = correlationInBioMatrix(A, u, 'relu')
    results[i, 5] = correlationInBioMatrix(A, u, 'MML')    

df = pandas.DataFrame(results, columns=['random', 'random\n(relu)', 'random\n(MML)', 'kegg',  'kegg\n(relu)',  'kegg\n(MML)'])
ax = sns.boxplot(data=df)
ax = sns.swarmplot(data=df, color=[0,0,0], size=2)
plt.ylabel('mean abs correlation coefficient')
plt.ylim([0, 1])
#xlocations = [3, 4, 5]
#plt.scatter(xlocations, referenceCorr)
#plt.text(4, referenceCorr[0], 'trained')
plt.plot([-0.5, 5.5], [referenceCorr, referenceCorr], 'k--')
plt.text(2.5, referenceCorr -0.05, 'model with optimized\nparameters')
df.to_csv('figures/generativeSynthnet/correlationRandomParameters.tsv', sep='\t')
plt.savefig('figures/generativeSynthnet/correlationRandomParameters.svg')

# yhatFull = runIterations(A, u, 'MML')
# yhatFull = numpy.mean(yhatFull, axis=1)
# activationFactor = activationFunctions.MMLoneStepDeltaActivationFactor(torch.tensor(yhatFull), 0.01).detach()
# weightFactor = activationFactor[networkList[0]]
# Aadjusted = A.copy()
# Aadjusted.data = Aadjusted.data * weightFactor.numpy()
# e, v = eigs(Aadjusted, 6)

#e, v = eigs(A, 6)
# meanY = torch.mean(y, axis=1)
# plt.scatter(numpy.repeat(meanY.detach().numpy().reshape(-1,1), y.shape[1], axis=1), y.detach().numpy())

# df = pandas.DataFrame(y.T.detach().numpy(), index=outNameGenes, columns=conditions)
# sns.clustermap(df, cmap='RdBu_r')

# # e, v = eigs(model.network.A, 1)
# # plt.scatter(meanY, v.real)


#%%

useUMap = False
correlationBasedDistance = True


N=2000
simultaniousInput = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
totalSamples = N * len(simultaniousInput)
data = torch.zeros(totalSamples, len(inName), dtype=torch.double)
sampleLabel = numpy.zeros(totalSamples)

k = 0
for i in range(len(simultaniousInput)):
    curSamples = simultaniousInput[i]
    for j in range(N):
        data[k, numpy.random.randint(0, len(inName), curSamples)] = torch.rand(curSamples, dtype=torch.double)
        sampleLabel[k] = i
        k+=1

#names = bionetwork.generateConditionNames(data, [uniprot2gene[x] for x in inName])

model.eval()
Ypredict, YpredictFull = model(data)
Ypredict = Ypredict.detach().numpy()

if useUMap:
    projection = PCA(n_components=8)
    principalComponents = projection.fit_transform(Ypredict)
    projection = umap.UMAP(n_neighbors=20)
    principalComponents = projection.fit_transform(principalComponents)
else:
    projection = PCA(n_components=8)
    principalComponents = projection.fit_transform(Ypredict)

print(projection.explained_variance_ratio_)
print(sum(numpy.array(projection.explained_variance_ratio_)))

#%%
def PCAString(i, projection):
    return 'PC {0:d} ({1:0.2f}%)'.format(i, 100*projection.explained_variance_ratio_[i-1])

selectedInputs = [5, 2, 1]
plt.rcParams["figure.figsize"] = (6,6)
minAndMax = [numpy.floor(numpy.min(principalComponents)), numpy.ceil(numpy.max(principalComponents))]

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors.insert(2, '#000000')
plt.figure()

for i in range(4):
    plt.subplot(2, 2, i + 1)
    for j in range(len(selectedInputs)):
        curData = numpy.argwhere(numpy.isin(simultaniousInput, selectedInputs[j]))[0]
        plt.plot(principalComponents[sampleLabel==curData, i], principalComponents[sampleLabel==curData, i+1], 'o', color=colors[j])
        plt.xlim(minAndMax)
        plt.ylim(minAndMax)
        plt.xlabel(PCAString(i+1, projection))
        plt.ylabel(PCAString(i+2, projection))
    if i==0:
        plt.legend(selectedInputs)

plt.tight_layout()


plt.rcParams["figure.figsize"] = (3,3)
plt.figure()
i=0
minAndMax = [numpy.floor(numpy.min(principalComponents[:,0:2])), numpy.ceil(numpy.max(principalComponents[:,0:2]))]

for j in range(len(selectedInputs)):
    curData = numpy.argwhere(numpy.isin(simultaniousInput, selectedInputs[j]))[0]
    plt.plot(principalComponents[sampleLabel==curData, i], principalComponents[sampleLabel==curData, i+1], 'o', color=colors[j])
    plt.xlim(minAndMax)
    plt.ylim(minAndMax)
    plt.xlabel(PCAString(i+1, projection))
    plt.ylabel(PCAString(i+2, projection))
plt.legend(selectedInputs, frameon=False)


#%%
plt.figure()
plt.rcParams["figure.figsize"] = (3,3)
model = LinearRegression()
kf = KFold(n_splits=20)

results = numpy.zeros((len(simultaniousInput), 3))
allX = data.numpy()
allY = Ypredict

X = allX[sampleLabel==0,:]
Y = allY[sampleLabel==0,:]
referenceModel = model.fit(X, Y)
for i in range(len(simultaniousInput)):
    X = allX[sampleLabel==i,:]
    Y = allY[sampleLabel==i,:]
    Yhat = referenceModel.predict(X)
    r, p = pearsonr(Y.flatten(), Yhat.flatten())
    results[i, 0] = r

for i in range(len(simultaniousInput)):
    X = allX[sampleLabel==i,:]
    Y = allY[sampleLabel==i,:]
    bestModel = model.fit(X, Y)
    Yhat =  bestModel.predict(X)
    r, p = pearsonr(Y.flatten(), Yhat.flatten())
    results[i, 1] = r

for i in range(len(simultaniousInput)):
    X = allX[sampleLabel==i,:]
    Y = allY[sampleLabel==i,:]
    Yhat = numpy.zeros(Y.shape)
    for train_index, test_index in kf.split(X):
        bestModel = model.fit(X[train_index,:], Y[train_index,:])
        Yhat[test_index,:] = bestModel.predict(X[test_index,:])
    r, p = pearsonr(Y.flatten(), Yhat.flatten())
    results[i, 2] = r

df = pandas.DataFrame((results[:,[0, 2]]), columns=['Extrapolation', 'Best fit'], index=simultaniousInput)
plt.plot(simultaniousInput, df['Extrapolation'])
#plt.plot(simultaniousInput, results[:,1])
plt.plot(simultaniousInput,  df['Best fit'])

#plt.plot(simultaniousInput, results[:,2])
plt.ylim([0, 1])
plt.xlim([1, 10])
plt.xticks(numpy.arange(1, 11))
plt.xlabel('Simultanious Input')
plt.ylabel('Correlation')
plt.legend(['Extrapolation', 'Best fit'], frameon=False)  #'Best fit',
plt.savefig(folder + 'D.svg')
df.to_csv(folder + 'D.tsv', sep='\t')
