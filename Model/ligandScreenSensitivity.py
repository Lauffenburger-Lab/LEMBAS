import torch
import numpy
import matplotlib.pyplot as plt
import bionetwork
import pandas
import seaborn as sns

def generatePerturbe(Yin, refFullRaw, level):
    nrKO = Yin.shape[1]
    inputKO = Yin.repeat(nrKO, 1)
    perturb = numpy.eye(nrKO)
    numpy.fill_diagonal(perturb, refFullRaw.flatten() * level)
    inputKO = inputKO + torch.tensor(perturb)
    return inputKO

def runSensitivity(curModel, inputX, perturbeLevel):
    curModel.network.A.data = curModel.network.weights.data
    Yin = curModel.inputLayer(inputX).detach()
    refFull = curModel.network(Yin)
    bIn = curModel.network.bias.detach().numpy() + Yin.T.detach().numpy()
    refFullRaw = curModel.network.A.dot(refFull.T.detach().numpy()) + bIn
    ref = curModel.projectionLayer(refFull).detach()
        
    inputKO = generatePerturbe(Yin, refFullRaw, perturbeLevel)
    
    KOFull = curModel.network(inputKO)
    KO = curModel.projectionLayer(KOFull).detach()
    stateVariableModel = KO.detach().numpy()
    stateVariableReference = ref.detach().numpy()   
    return stateVariableModel, stateVariableReference

#Load network
networkList, nodeNames, modeOfAction = bionetwork.loadNetwork('data/ligandScreen-Model.tsv')
annotation = pandas.read_csv('data/ligandScreen-Annotation.tsv', sep='\t')
uniprot2gene = dict(zip(annotation['code'], annotation['name']))
ligandInput = pandas.read_csv('data/ligandScreen-Ligands.tsv', sep='\t', low_memory=False, index_col=0)
TFOutput = pandas.read_csv('data/ligandScreen-TFs.tsv', sep='\t', low_memory=False, index_col=0)

CVconditions = pandas.read_csv('CVligandScreen/conditions.tsv', sep='\t')
criterion = torch.nn.MSELoss(reduction='mean')

# def loadModel(refModel, fileName):
#     #work around to copy weights and bias values 
#     #because class structure has been updated since the run
#     curModel = torch.load(fileName)
#     model = copy.deepcopy(refModel)
#     model.network.weights.data = curModel.network.weights.data.clone()
#     model.network.bias.data = curModel.network.bias.data.clone()
#     model.inputLayer.weights.data = curModel.inputLayer.weights.data.clone()
#     model.projectionLayer.weights.data = curModel.projectionLayer.weights.data.clone()
#     model.network.param = curModel.network.parameters.copy()
#     return model


#Subset input and output to intersecting nodes
inName = ligandInput.columns.values
outName = TFOutput.columns.values
inName = numpy.intersect1d(nodeNames, inName)
outName = numpy.intersect1d(nodeNames, outName)
inNameGene = [uniprot2gene[x] for x in inName]
outNameGene = [uniprot2gene[x] for x in outName]
nodeNameGene = numpy.array([uniprot2gene[x] for x in nodeNames])
internalNodes = numpy.logical_not(numpy.logical_or(numpy.isin(nodeNames, inName), numpy.isin(nodeNames, outName)))

ligandInput = ligandInput.loc[:,inName]
TFOutput = TFOutput.loc[:,outName]

sampleName = ligandInput.index.values
X = torch.tensor(ligandInput.values.copy(), dtype=torch.double)
Y = torch.tensor(TFOutput.values, dtype=torch.double)

perturbeLevel = 0.001

foldLookup = dict(zip(CVconditions['Index'].values, CVconditions['Condition'].values))


#%%
#Knock out test
nrFolds = len(numpy.unique(CVconditions['Index'].values))
nrKO = len(nodeNames)
nrConditions = len(sampleName)
nrTF = Y.shape[1]

stateVariableModel = numpy.zeros((nrFolds, nrConditions, nrKO, nrTF))
stateVariableReference = numpy.zeros((nrFolds, nrConditions, nrTF))

for i in range(0, nrFolds): 
    print('Simulating sensitivity fold nr: ', i)    
    curModel = torch.load('CVligandScreen/MML_model_' + str(i) + '.pt')
    currentConditions = CVconditions['Condition'].values[CVconditions['Index'] == i]
    currentMissing = numpy.argwhere(numpy.isin(sampleName, currentConditions)).flatten()
    for j in range(0, nrConditions):
        print('', sampleName[j])
        if numpy.isin(j, currentMissing):
            stateVariableModel[i, j, :, :] = numpy.nan
            stateVariableReference[i, j, :] = numpy.nan            
        else:
            inputX = X[j,:].reshape(1,-1)
            modelOutput, refOutput = runSensitivity(curModel, inputX, perturbeLevel)
            stateVariableModel[i, j, :, :] = modelOutput
            stateVariableReference[i, j, :] = refOutput
stateVariableModel = numpy.swapaxes(stateVariableModel, 2, 3)

#%%
ChangeInQ = stateVariableModel.copy()

zeroFilter = numpy.abs(stateVariableReference)>0.01
zeroFilter = zeroFilter.astype(int)
for i in range(nrKO):
    ChangeInQ[:, :, :, i]  = (stateVariableModel[:, :, :, i] - stateVariableReference)/stateVariableReference
    ChangeInQ[:, :, :, i] = zeroFilter * ChangeInQ[:, :, :, i]
ChangeInP = perturbeLevel
elasticity = ChangeInQ/ChangeInP

medianElasticity = numpy.nanmedian(elasticity, axis=0)
#numpy.save('results/elasticity.npy', medianElasticity)


#%%
plt.rcParams["figure.figsize"] = (6, 3)
plt.figure()
internalNodeName = numpy.array(nodeNameGene)[internalNodes]
medianElasticity = numpy.load('results/elasticity.npy')

meanInternal = medianElasticity[:,:,internalNodes]
maxInternal = numpy.max(meanInternal, axis=0)
minInternal = numpy.min(meanInternal, axis=0)
minMaxInternal = numpy.where(maxInternal>-minInternal, maxInternal, minInternal)
#meanCondition = numpy.mean(meanInternal, axis=0)

#superElastic = numpy.max(numpy.abs(meanCondition), axis=0)>1
superElastic = numpy.max(numpy.abs(minMaxInternal), axis=0)>0.5
print(sum(superElastic))
df = pandas.DataFrame(minMaxInternal, index=outNameGene, columns=internalNodeName)
#df.to_csv('results/elasticity.tsv', sep='\t')
df = df.loc[:,superElastic]
#sns.clustermap(df, cmap='RdBu_r', center=0, vmin=-1, vmax=1, yticklabels=True, figsize=(8,10), dendrogram_ratio=0.15, cbar_pos=(0.02, 0.02, 0.05, 0.1))
sns.clustermap(df, cmap='RdBu_r', center=0, vmin=-1, vmax=1, figsize=(6,3), cbar_pos=(0.05, 0.05, 0.03, 0.13), dendrogram_ratio=0.15)
plt.savefig('figures/ligand screen KO/sensitivity.svg')




