import torch
import numpy
import matplotlib.pyplot as plt
import bionetwork
import pandas
from scipy.stats import pearsonr
import seaborn as sns
import copy
import matplotlib.patches as patches
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
curModel = loadModel(model, 'synthNetScreen/model_14.pt')
curX = torch.load('synthNetScreen/X_14.pt')

#Train
Yhat, _ = curModel(curX)
Y, _ = parameterizedModel(curX)

#Test
YhatTest, YhatFull = curModel(Xtest)

#%%
#Knock out test
plt.rcParams["figure.figsize"] = (5, 5)
knockOutLevel = -5

# inputX[:,internalNodes] = 0
# filterInput = torch.sum(inputX,dim=1)<0
# filterInput[0] = True
# inputX = inputX[filterInput,:]

nrOfKO = 100
nrInternalNodes = sum(internalNodes)
samplesToKO = numpy.random.permutation(curX.shape[0])[0:nrOfKO]
resultsRef = numpy.zeros((nrOfKO, nrInternalNodes, Y.shape[1]))
resultsModel  = numpy.zeros((nrOfKO, nrInternalNodes, Y.shape[1]))
stateVariableRef = numpy.zeros((nrOfKO, nrInternalNodes))
stateVariableModel = numpy.zeros((nrOfKO, nrInternalNodes))

for i in range(nrOfKO):
    inputX = curModel.inputLayer(curX[samplesToKO[i],:].reshape(1,-1)).detach()
    inputX = inputX.repeat(sum(internalNodes)+1, 1)
    inputX[1:inputX.shape[0], :] = inputX[1:inputX.shape[0], :]  + knockOutLevel * torch.eye(len(nodeNames))[internalNodes,:]
    KOFullRef = parameterizedModel.network(inputX)
    KOFull = curModel.network(inputX)

    KORef = parameterizedModel.projectionLayer(KOFullRef).detach()
    KO = curModel.projectionLayer(KOFull).detach()
    resultsRef[i, :, :] = (KORef[1:, :] - KORef[0, :]).numpy()
    resultsModel[i, :, :] = (KO[1:, :] - KO[0, :]).numpy()
    stateVariableRef[i,:] = KOFullRef[0, internalNodes].detach().numpy()
    stateVariableModel[i,:] = KOFull[0, internalNodes].detach().numpy()


#%%
df = pandas.DataFrame((resultsModel.flatten(), resultsRef.flatten()), index = ['Model', 'Reference']).T
df.to_csv('figures/syntheticNet/syntheticKO.tsv', sep='\t') 

#%%
plt.rcParams["figure.figsize"] = (4,3)
plt.figure()
df = pandas.read_csv('figures/syntheticNet/syntheticKO.tsv', sep='\t', index_col=0)

blues = cm.get_cmap('Blues', 256)
newcolors = blues(numpy.linspace(0, 1, 256))
newcolors = newcolors[50:, :]
blues = ListedColormap(newcolors)


axisScale = 100
counts, rangeX, rangeY = numpy.histogram2d(df['Model'].values, df['Reference'].values, bins=axisScale, range=[[-1, 1], [-1, 1]])
counts_transformed = numpy.log10(counts+1)
ax = sns.heatmap(counts_transformed.T, mask=counts_transformed==0, vmin=0, cbar_kws={'label': 'log10(#preditions + 1)'}, cmap=blues)
ax.invert_yaxis()
for _, spine in ax.spines.items():
    spine.set_visible(True)
#sns.histplot(df, x="Model", y="Reference", bins=100, cbar=True, log_scale=False, cbar_kws={'label': 'number of preditions'}, vmax=100)
#plt.scatter(resultsModel.flatten(), resultsRef.flatten(), alpha=0.05)
plt.gca().axis('equal')
plt.xlabel('model $\Delta$TF')
plt.ylabel('reference $\Delta$TF')
plt.gca().set_xticks(numpy.linspace(0, axisScale, 5))
plt.gca().set_yticks(numpy.linspace(0, axisScale, 5))
plt.gca().set_xticklabels([-1,-0.5,0,0.5,1], rotation = 0)
plt.gca().set_yticklabels([-1,-0.5,0,0.5,1])
plt.gca().add_patch(patches.Polygon(axisScale * numpy.array([[0, 0.5], [0, 1], [0.5, 1]]), closed=True, fill=True, alpha=0.5, color='#CCCCCC'))
plt.gca().add_patch(patches.Polygon(axisScale * numpy.array([[0.5, 0], [1, 0], [1, 0.5]]), closed=True, fill=True, alpha=0.5, color='#CCCCCC'))

r, p = pearsonr(df['Model'].values, df['Reference'].values)
plt.text(0, axisScale * 0.9, 'r {:.2f}'.format(r))
plt.gca().axhline(y=axisScale *0.5, color='k', alpha=0.3)
plt.gca().axvline(x=axisScale *0.5, color='k', alpha=0.3)

plt.savefig('figures/syntheticNet/syntheticKO.svg')
