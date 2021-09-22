import torch
import numpy
import matplotlib.pyplot as plt
import bionetwork
import pandas
import seaborn as sns

#Load network
networkList, nodeNames, modeOfAction = bionetwork.loadNetwork('data/macrophage-Model.tsv')
annotation = pandas.read_csv('data/macrophage-Annotation.tsv', sep='\t')
uniprot2gene = dict(zip(annotation['code'], annotation['name']))
ligandInput = pandas.read_csv('data/macrophage-Ligands.tsv', sep='\t', low_memory=False, index_col=0)
TFOutput = pandas.read_csv('data/macrophage-TFs.tsv', sep='\t', low_memory=False, index_col=0)

CVconditions = pandas.read_csv('CVmacrophage/conditions.tsv', sep='\t')
criterion = torch.nn.MSELoss(reduction='mean')

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



knockOutLevel = -5
condition = 'TNFa'
TF = 'RELA'
topNr = 10


testedConditions = CVconditions['Condition'].values
dictionary = dict(zip(sampleName, list(range(len(sampleName)))))
conditionOrder = numpy.array([dictionary[x] for x in testedConditions])
conditionNr = numpy.argwhere(numpy.isin(testedConditions, condition)).flatten()


referenceX = X[conditionOrder,:]
referenceY = Y[conditionOrder,:]


#%%
#Knock out test
curModel = torch.load('CVmacrophage/model_0.pt')
nrConditions = len(testedConditions)
nrKO = len(nodeNames)
nrTF = Y.shape[1]

stateVariableModel = numpy.zeros((nrConditions, nrKO, nrTF))
stateVariableReference = numpy.zeros((nrConditions, nrTF))

inputX = referenceX[conditionNr,:].reshape(1,-1)
inputKO = curModel.inputLayer(inputX).detach()
inputKO = inputKO.repeat(nrKO, 1)
inputKO = inputKO + knockOutLevel * torch.eye(nrKO)

for i in range(nrConditions):
    curModel = torch.load('CVmacrophage/model_' + str(i) + '.pt')
    ref, _ = curModel(inputX)
    KOFull = curModel.network(inputKO)
    KO = curModel.projectionLayer(KOFull).detach()
    stateVariableModel[i,:,:] = KO.detach().numpy()
    stateVariableReference[i,:] = ref.detach().numpy()


#%%
#DisplayResults
nodeName = numpy.array(nodeNames)
plt.rcParams["figure.figsize"] = (3,1)
conditionFilter = numpy.isin(numpy.arange(nrConditions), conditionNr)==False


TFNr = numpy.argwhere(numpy.isin(outNameGene, TF)).flatten()
delta = stateVariableModel[:, :, TFNr].squeeze() - stateVariableReference[:, TFNr]
delta = delta[conditionFilter,:]

meanDelta = numpy.mean(delta, axis=0)
deltaSign = numpy.sign(meanDelta)
deltaVal = numpy.abs(meanDelta)
error = numpy.std(numpy.abs(delta), axis=0)

order = numpy.flip(numpy.argsort(deltaVal))
order = order[0:topNr]
#plt.bar(numpy.arange(topNr), deltaVal[order])
#plt.bar(numpy.arange(topNr), deltaVal[order], yerr=error[order], align='center', ecolor='black', capsize=2)

df = pandas.DataFrame(numpy.abs(delta), columns=nodeNameGene)
df = df.iloc[:, order]
sns.boxplot(data=df, color='#1f77b4', showfliers=False)
sns.stripplot(data=df, color='black', marker="$\circ$", facecolors="none", s=7)
#plt.gca().set_xticks(numpy.arange(topNr))
#plt.gca().set_xticklabels(nodeNameGene[order])
plt.gca().set_ylabel('|$\Delta$'+TF+'|')
plt.xticks(rotation = 90)
plt.ylim([0, 1])
plt.title(condition)
plt.show()
print(nodeName[order])
print(deltaSign[order])

