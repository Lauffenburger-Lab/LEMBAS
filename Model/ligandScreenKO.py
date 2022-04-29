import torch
import numpy
import matplotlib.pyplot as plt
import bionetwork
import pandas
import seaborn as sns

def generateKO(Yin, knockOutLevel):
    nrKO = Yin.shape[1]
    inputKO = Yin.repeat(nrKO, 1)
    inputKO = inputKO + knockOutLevel * torch.eye(nrKO)
    return inputKO

def runKO(curModel, inputX, knockOutLevel):
    ref, _ = curModel(inputX)
    
    Yin = curModel.inputLayer(inputX).detach()
    inputKO = generateKO(Yin, knockOutLevel)
    
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
ligandInput = ligandInput.loc[:,inName]
TFOutput = TFOutput.loc[:,outName]

sampleName = ligandInput.index.values
X = torch.tensor(ligandInput.values.copy(), dtype=torch.double)
Y = torch.tensor(TFOutput.values, dtype=torch.double)



knockOutLevel = -3
knockInLevel = 3
condition = 'PBS-BSA_LPS'
TF = 'RELA'

foldLookup = dict(zip(CVconditions['Condition'].values, CVconditions['Index'].values))
#conditionNr = numpy.argwhere(numpy.isin(testedConditions, condition)).flatten()
#skipFolds = [foldLookup['PBS-BSA'], foldLookup['PBS-BSA_LPS']]
skipFolds = []

conditionNr = numpy.argwhere(sampleName==condition).flatten()

bionetParams = bionetwork.trainingParameters(iterations = 150, clipping=1, leak=0.01)
model = bionetwork.model(networkList, nodeNames, modeOfAction, 3, 1.2, inName, outName, bionetParams)

folder = 'figures/Figure 6/'

#%%
#Knock out test
nrFolds = len(numpy.unique(CVconditions['Index'].values))
nrKO = len(nodeNames)
nrTF = Y.shape[1]

stateVariableModel = numpy.zeros((nrFolds, nrKO, nrTF))
stateVariableReference = numpy.zeros((nrFolds, nrTF))

stateVariableModelKnockIn = numpy.zeros((nrFolds, nrKO, nrTF))
#stateVariableReferenceKnockIn = numpy.zeros((nrFolds, nrTF))

inputX = X[conditionNr,:].reshape(1,-1)

for i in range(nrFolds):
    print('Simulating KO and knock-in fold nr: ', i)
    curModel = torch.load('CVligandScreen/MML_model_' + str(i) + '.pt')
    modelOutput, refOutput = runKO(curModel, inputX, knockOutLevel)
    stateVariableModel[i,:,:] = modelOutput
    stateVariableReference[i,:] = refOutput

    modelOutput, refOutput = runKO(curModel, inputX, knockInLevel)
    stateVariableModelKnockIn[i,:,:] = modelOutput
    #stateVariableReferenceKnockIn[i,:] = refOutput

#plt.scatter(stateVariableReferenceKnockIn.flatten(), stateVariableReference.flatten())


numpy.save(folder + 'Reference', stateVariableReference)
numpy.save(folder + 'KO', stateVariableModel)
numpy.save(folder + 'KI', stateVariableModelKnockIn)
#numpy.save('Reference', stateVariableModel)


#%%
#DisplayResults
topNr = 10
nodeName = numpy.array(nodeNames)
plt.rcParams["figure.figsize"] = (3, 1.2)
conditionFilter = numpy.isin(numpy.arange(nrFolds), skipFolds)==False


TFNr = numpy.argwhere(numpy.isin(outNameGene, TF)).flatten()
delta = stateVariableModel[:, :, TFNr].squeeze() - stateVariableReference[:, TFNr]
delta = delta[conditionFilter,:]

meanDelta = numpy.median(delta, axis=0)
deltaSign = numpy.sign(meanDelta)
deltaVal = numpy.abs(meanDelta)
error = numpy.std(delta, axis=0)

order = numpy.flip(numpy.argsort(deltaVal))
order = order[0:topNr]
#plt.bar(numpy.arange(topNr), deltaVal[order])
#plt.bar(numpy.arange(topNr), deltaVal[order], yerr=error[order], align='center', ecolor='black', capsize=2)


df = pandas.DataFrame(numpy.abs(delta), columns=nodeNameGene)
df = df.iloc[:, order]

signColor = numpy.zeros((len(order), 3))
signColor[deltaSign[order] == 1, :] = [0.8, 0.14, 0.14]
signColor[deltaSign[order] == -1, :] = [0.12, 0.47, 0.71]


ax = sns.boxplot(data=df, showfliers=False)   #color='#1f77b4',
for i in range(len(ax.artists)):
 ax.artists[i].set_facecolor(signColor[i,:])

sns.stripplot(data=df, color='black', marker="$\circ$", facecolors="none", s=7)
#plt.gca().set_xticks(numpy.arange(topNr))
#plt.gca().set_xticklabels(nodeNameGene[order])
plt.gca().set_ylabel('|$\Delta$'+TF+'|')
plt.xticks(rotation = 90)
plt.ylim([0, 1.1])
plt.title(condition)
print(nodeName[order])
print(deltaSign[order])
plt.savefig(folder + 'E.svg')
df.to_csv(folder + 'E.tsv', sep='\t')

#%%
plt.rcParams["figure.figsize"] = (2, 1.2)
plt.figure()

topNr = 5
delta = stateVariableModelKnockIn[:, :, TFNr].squeeze() - stateVariableReference[:, TFNr]
delta = delta[conditionFilter,:]

meanDelta = numpy.median(delta, axis=0)
deltaSign = numpy.sign(meanDelta)
deltaVal = numpy.abs(meanDelta)
error = numpy.std(delta, axis=0)

order = numpy.flip(numpy.argsort(deltaVal))
order = order[0:topNr]
#plt.bar(numpy.arange(topNr), deltaVal[order])
#plt.bar(numpy.arange(topNr), deltaVal[order], yerr=error[order], align='center', ecolor='black', capsize=2)


df = pandas.DataFrame(numpy.abs(delta), columns=nodeNameGene)
df = df.iloc[:, order]

signColor = numpy.zeros((len(order), 3))
signColor[deltaSign[order] == 1, :] = [0.8, 0.14, 0.14]
signColor[deltaSign[order] == -1, :] = [0.12, 0.47, 0.71]


ax = sns.boxplot(data=df, showfliers=False)   #color='#1f77b4',
for i in range(len(ax.artists)):
 ax.artists[i].set_facecolor(signColor[i,:])

sns.stripplot(data=df, color='black', marker="$\circ$", facecolors="none", s=7)
#plt.gca().set_xticks(numpy.arange(topNr))
#plt.gca().set_xticklabels(nodeNameGene[order])
plt.gca().set_ylabel('|$\Delta$'+TF+'|')
plt.xticks(rotation = 90)
plt.ylim([0, 0.4])
plt.title(condition)
print(nodeName[order])
print(deltaSign[order])
plt.savefig(folder + 'F.svg')
df.to_csv(folder + 'F.tsv', sep='\t')