import torch
import numpy
import matplotlib.pyplot as plt
import bionetwork
import plotting
import pandas
from scipy.stats import pearsonr
import seaborn as sns
import copy
from matplotlib import cm
from matplotlib.colors import ListedColormap

def loadModel(refModel, fileName):
    #work around to copy weights and bias values 
    #because class structure has been updated since the run
    curModel = torch.load(fileName)
    model = copy.deepcopy(refModel)
    model.network.weights.data = curModel.network.weights.data.clone()
    model.network.bias.data = curModel.network.bias.data.clone()
    return model

def hashArray(array, hashMap):
    outputArray = array.copy()
    for i in range(len(array)):
        outputArray[i] = hashMap[array[i]]
    return outputArray

N = 1000
simultaniousInput = 5
inputAmplitude = 3
projectionAmplitude = 1.2

#Load network
networkList, nodeNames, modeOfAction = bionetwork.loadNetwork('data/KEGGnet-Model.tsv')
annotation = pandas.read_csv('data/KEGGnet-Annotation.tsv', sep='\t')
uniprot2gene = dict(zip(annotation['code'], annotation['name']))


dataConditions = pandas.read_csv('synthNetScreen/conditions.tsv', sep='\t')
criterion = torch.nn.MSELoss(reduction='mean')


#Subset input and output to intersecting nodes
inName = annotation.loc[annotation['ligand'],'code'].values
outName = annotation.loc[annotation['TF'],'code'].values
inName = numpy.intersect1d(nodeNames, inName)
outName = numpy.intersect1d(nodeNames, outName)
inNameGene = [uniprot2gene[x] for x in inName]
outNameGene = [uniprot2gene[x] for x in outName]
internalNodes = numpy.logical_not(numpy.logical_or(numpy.isin(nodeNames, inName), numpy.isin(nodeNames, outName)))


bionetParams = bionetwork.trainingParameters(iterations = 150, clipping=1, leak=0.01)

model = bionetwork.model(networkList, nodeNames, modeOfAction, inputAmplitude, projectionAmplitude, inName, outName, bionetParams)
model.inputLayer.weights.requires_grad = False
model.projectionLayer.weights.requires_grad = False
model.network.preScaleWeights()

parameterizedModel = bionetwork.model(networkList, nodeNames, modeOfAction, inputAmplitude, projectionAmplitude, inName, outName, bionetParams)
parameterizedModel = bionetwork.loadParam('synthNetScreen/equationParams.txt', parameterizedModel, nodeNames)


#Generate test data
Xtest = torch.zeros(N, len(inName), dtype=torch.double)
for i in range(N): #skip 0 to include a ctrl sample i.e. zero input
    Xtest[i, numpy.random.randint(0, len(inName), simultaniousInput)] = torch.rand(simultaniousInput, dtype=torch.double)
Ytest, YfullRef = parameterizedModel(Xtest)
Ytest = Ytest.detach()

#%%
nExperiments = dataConditions.shape[0]

sampleCorrelations = numpy.zeros((nExperiments, 2))

# trainFit = numpy.zeros(len(conditionOrder))
# testFit = numpy.zeros(len(conditionOrder))

for i in range(nExperiments):
    curModel = loadModel(model, 'synthNetScreen/model_' + str(i) + '.pt')
    curX = torch.load('synthNetScreen/X_' + str(i) + '.pt')

    #Train
    Yhat, YhatFull = curModel(curX)
    Y, _ = parameterizedModel(curX)
    r, p = pearsonr(Yhat.detach().flatten(), Y.detach().flatten())
    sampleCorrelations[i, 0] = r

    #Test
    Yhat, YhatFull = curModel(Xtest)
    r, p = pearsonr(Yhat.detach().flatten(), Ytest.flatten())
    sampleCorrelations[i, 1] = r

#Predicting the average
Yhat = torch.mean(Ytest, axis=0).reshape(1, -1).repeat(Ytest.shape[0], 1)
r, p = pearsonr(Yhat.detach().flatten(), Ytest.flatten())
predictingAverageTF = r


#samplePrediction = plotting.calculateCorrelations(referenceY.T, predictionY.T)


#%%
plt.rcParams["figure.figsize"] = (3,3)
allLiganValues = numpy.unique(dataConditions['Ligands'].values)
allDataValues = numpy.unique(dataConditions['DataSize'].values)
dataPositions = list(range(len(allDataValues)))
dataSizeLookup = dict(zip(allDataValues, dataPositions))

for i in range(len(allLiganValues)):
    conditionFilter = dataConditions['Ligands'].values == allLiganValues[i]
    dataSize = dataConditions.loc[conditionFilter, 'DataSize'].values
    positions = hashArray(dataSize, dataSizeLookup)
    correlation = sampleCorrelations[conditionFilter,1]
    plt.plot(positions, correlation)

#positions = hashArray(dataConditions['DataSize'].values, dataSizeLookup)
#plt.scatter(positions, sampleCorrelations[:,0], color='gray')

plt.plot([min(dataPositions), max(dataPositions)], [predictingAverageTF, predictingAverageTF], 'k--')


legendValues = numpy.append(allLiganValues.astype('object'), ['Average TF'])
plt.legend(legendValues, frameon=False)
plt.ylim([0, 1])
plt.xlim([0, max(dataPositions)])
plt.gca().set_xticklabels(dataPositions)
plt.gca().set_xticklabels(allDataValues)
plt.xlabel('Train data size')
plt.ylabel('Test correlation')

#%%
curModel = loadModel(model, 'synthNetScreen/model_14.pt')
curX = torch.load('synthNetScreen/X_14.pt')

plt.figure()
finalWeights = curModel.network.weights.detach().numpy()
finalBias = curModel.network.bias.detach().numpy().flatten()
trueWeights = parameterizedModel.network.weights.detach().numpy()
trueBias = parameterizedModel.network.bias.detach().numpy().flatten()

r1, p1 = pearsonr(finalWeights, trueWeights)
r2, p2 = pearsonr(finalBias, trueBias)
r3, p3 = pearsonr(numpy.abs(finalWeights), numpy.abs(trueWeights))
print('Abs correlation', r3, p3)

plt.axhline(0, color='black', label='_nolegend_')
plt.axvline(0, color='black', label='_nolegend_')
plt.scatter(finalWeights, trueWeights, alpha=0.5)
plt.scatter(finalBias, trueBias, alpha=0.5)
plt.plot([-6, 6], [-6, 6], 'black', label='_nolegend_')
plt.xlabel('fit')
plt.ylabel('reference')
plt.legend(['Weight r {:.2f}'.format(r1), 'Bias r {:.2f}'.format(r2)], frameon=False)
plt.gca().axis('equal')


#%%
#Train
Yhat, _ = curModel(curX)
Y, _ = parameterizedModel(curX)


#Test
YhatTest, YhatFull = curModel(Xtest)

#srlevel = bionetwork.getAllSpectralRadius(curModel, YhatFull)

plt.rcParams["figure.figsize"] = (4,3)
plt.figure()
A = YhatTest.detach().numpy().flatten()
B = Ytest.detach().numpy().flatten()
df = pandas.DataFrame((A, B), index = ['Model', 'Reference']).T
df.to_csv('figures/syntheticNet/syntheticModelVsPredict.tsv', sep='\t')

#%%

blues = cm.get_cmap('Blues', 256)
newcolors = blues(numpy.linspace(0, 1, 256))
newcolors = newcolors[75:, :]
blues = ListedColormap(newcolors)


df = pandas.read_csv('figures/syntheticNet/syntheticModelVsPredict.tsv', sep='\t', index_col=0)
axisScale = 100

counts, rangeX, rangeY = numpy.histogram2d(df['Model'].values, df['Reference'].values, bins=axisScale, range=[[0, 1], [0, 1]])
counts_transformed = numpy.log10(counts+1)
ax = sns.heatmap(counts_transformed.T, mask=counts_transformed==0, vmin=0, cbar_kws={'label': 'log10(#preditions + 1)'}, cmap=blues) #cmap="Blues", 
ax.invert_yaxis()
for _, spine in ax.spines.items():
    spine.set_visible(True)
#sns.histplot(df, x="Model", y="Reference", bins=100, cbar=True, cbar_kws={'label': 'number of preditions'}, vmax=50)
ax.axis('equal')
plt.xlabel('fit.')
plt.ylabel('ref.')
plt.gca().set_xticks(numpy.linspace(0, axisScale, 5))
plt.gca().set_yticks(numpy.linspace(0, axisScale, 5))
plt.gca().set_xlim([0, axisScale])
plt.gca().set_ylim([0, axisScale])
plt.gca().set_xticklabels(numpy.linspace(0, 1, 5), rotation = 0)
plt.gca().set_yticklabels(numpy.linspace(0, 1, 5))
plt.gca().set_xlabel('Model')
plt.gca().set_xlabel('Reference')
r, p = pearsonr(df['Model'].values, df['Reference'].values)
plt.text(0, axisScale *0.9, 'r {:.2f}'.format(r))


plt.savefig('figures/syntheticNet/syntheticModelVsPredict.svg')

#%%

plt.figure()
plt.rcParams["figure.figsize"] = (3,3)
plt.figure()
TFCorrelation = plotting.calculateCorrelations(Ytest, YhatTest)
plt.hist(TFCorrelation)
plt.xlim([0, 1])
plt.ylabel('number of TFs')
plt.xlabel('correlation')
print('Lowest TF correlation', min(TFCorrelation))

plt.figure()
conditionCorrelations = plotting.calculateCorrelations(Ytest.T, YhatTest.T)
plt.hist(conditionCorrelations)
plt.xlim([0, 1])
plt.ylabel('number of conditions')
plt.xlabel('correlation')
print('Lowest Condition correlation', min(conditionCorrelations))

plt.figure()
stateRef = YfullRef[:,internalNodes].detach()
state = YhatFull[:,internalNodes].detach()
stateCorrelations = plotting.calculateCorrelations(stateRef, state)
plt.hist(stateCorrelations, 20)
plt.xlabel('correlation')
plt.ylabel('number of state variables')

#%%
# plt.figure()
# A = Yhat.detach().numpy()
# B = Y.detach().numpy()
# plt.scatter(A, B, alpha=0.1, color='gray')
# r1, p1 = pearsonr(A.flatten(), B.flatten())

# A = YhatTest.detach().numpy()
# B = Ytest.detach().numpy()
# plt.scatter(A, B, alpha=0.1)
# r2, p2 = pearsonr(A.flatten(), B.flatten())

# plt.gca().axis('equal')
# plt.gca().set_xticks([0, 0.5, 1])
# plt.gca().set_yticks([0, 0.5, 1])

# plt.xlabel('Model')
# plt.ylabel('Reference')
# leg = plt.legend(['Train r {:.2f}'.format(r1), 'Test r {:.2f}'.format(r2)], frameon=False)
# for lh in leg.legendHandles:
#     lh.set_alpha(1)
# plotting.lineOfIdentity()


# plt.rcParams["figure.figsize"] = (6,6)
# plt.figure()
# stateRef = YfullRef[0:Ntests,internalNodes].detach()
# state = YhatFull[0:Ntests,internalNodes].detach()
# for i in range(state.shape[1]):
#     A = stateRef[:,i].detach().numpy()
#     B = state[:,i].detach().numpy()
#     order = numpy.argsort(A)
#     plt.plot(A[order], B[order], 'o-', color='black', alpha=0.1)

# r, p = pearsonr(state.flatten(), stateRef.flatten())


# plotting.lineOfIdentity()
# plt.xlabel('Fit')
# plt.ylabel('Reference')
# plt.gca().axis('equal')
# plt.xlim([0, 1])
# plt.ylim([0, 1])



#%%
replicates = 10
simultaniousInput = 5

results = numpy.zeros((replicates, 3))
resultConditions = numpy.array(['Average TF', 'Only Reg.', 'Scrambled Y'])


for i in range(replicates):
    Xtest = torch.zeros(N, len(inName), dtype=torch.double)
    for j in range(N): #skip 0 to include a ctrl sample i.e. zero input
        Xtest[j, numpy.random.randint(0, len(inName), simultaniousInput)] = torch.rand(simultaniousInput, dtype=torch.double)
    Ytest, YfullRef = parameterizedModel(Xtest)
    Ytest = Ytest.detach()

    #Predicting the average
    Yhat = torch.mean(Ytest, axis=0).reshape(1, -1).repeat(Ytest.shape[0], 1)
    r, p = pearsonr(Yhat.detach().flatten(), Ytest.flatten())
    results[i,0] = r

    #Only regularization
    curModel = torch.load('synthNetScreen/onlyRegularization.pt')
    Yhat, YhatFull = curModel(Xtest)
    r, p = pearsonr(Yhat.detach().flatten(), Ytest.flatten())
    results[i,1] = r

    #Only regularization
    curModel = torch.load('synthNetScreen/scrambledY.pt')
    Yhat, YhatFull = curModel(Xtest)
    r, p = pearsonr(Yhat.detach().flatten(), Ytest.flatten())
    results[i,2] = r

#%%
plt.rcParams["figure.figsize"] = (3,3)
plt.figure()
df = pandas.DataFrame(results, columns=resultConditions)
sns.boxplot(data=df)  #orient="h",
plt.ylabel('Correlation')
plt.ylim([0, 0.5])
plt.gca().set_xticklabels(plt.gca().get_xticklabels(), rotation=20)
