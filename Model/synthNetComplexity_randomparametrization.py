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


folder = 'figures/SI Figure 7/'

#%%
N=100
simultaniousInput = 5
data = torch.zeros(N, len(inName), dtype=torch.double)

for i in range(N):
    data[i, numpy.random.randint(0, len(inName), simultaniousInput)] = torch.rand(simultaniousInput, dtype=torch.double)

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
    correlation = numpy.mean(numpy.abs(numpy.corrcoef(y.detach().numpy()))) #Can result in nan, could use nan mean instead
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
plt.savefig(folder + 'correlationRandomParameters.svg')
df.to_csv(folder + 'correlationRandomParameters.tsv', sep='\t')

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



