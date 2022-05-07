import torch
import numpy
import matplotlib.pyplot as plt
import bionetwork
import pandas
from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
import umap

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


folder = 'figures/SI Figure 9/'

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

plt.savefig(folder + 'PCA.svg')

sampleInputs = numpy.array([simultaniousInput[int(x)] for x in sampleLabel])
data = numpy.concatenate((principalComponents[:,0:2], sampleInputs.reshape(-1,1)), axis=1)
df = pandas.DataFrame(data, columns=['PC 1', 'PC 2', 'Condition'])
df['Condition'] = df['Condition'].astype(int)
df = df.loc[numpy.isin(df['Condition'], selectedInputs),:]
df.to_csv(folder + 'PCA.tsv', sep='\t')
