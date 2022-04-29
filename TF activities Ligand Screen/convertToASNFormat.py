import numpy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def updateIndexName(df, dictionary):
    allIndex = df.index.values
    for i in range(len(allIndex)):
        if allIndex[i] in dictionary:
            allIndex[i] = dictionary[allIndex[i]]
    df.index = allIndex
    return df

#%%
stimName = 'LPS'
controlName = 'PBS-BSA'
countCutOf = 1
#rangeCutOff = 0.1 #at least one condition where it is within X from the extreemes
consistencyCutOf = 0.2 #Median STD within a condition

plt.rcParams["figure.figsize"] = (5, 5)
dorotheaData = pd.read_csv('results/dorothea.tsv', sep='\t')
allTFs = pd.read_csv('data/tfList.tsv', sep='\t').values.flatten()
dorotheaData = dorotheaData.loc[allTFs,:]



#dropConditions = ['IL1RN', 'IL33']
dropConditions = []


ligandMapping = pd.read_csv('data/ligandMap.tsv', sep='\t')
ligand2id = dict(zip(ligandMapping['Name'], ligandMapping['Code']))

uniprot = pd.read_csv('data/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab', sep='\t')
gene2uniprot = dict(zip(uniprot['Gene names  (primary )'], uniprot['Entry']))

metaData = pd.read_csv('filtered/metaData.tsv', sep='\t')
metaData.index = metaData['uniqueId']
metaData = metaData.loc[dorotheaData.columns.values,:]


t = 1
dorotheaData = 1/(1 + numpy.exp(-t * dorotheaData))
#plt.hist(dorotheaData.values.flatten())
#plt.figure()
#dorotheaData = customSigmodal(dorotheaData)

stim = numpy.where(metaData['stim'], '_' + stimName, '')
conditionId = metaData['ligand'].values + stim

allConditions, counts = numpy.unique(conditionId, return_counts=True)
allConditions = allConditions[counts >= countCutOf]
allConditions = numpy.setdiff1d(allConditions, dropConditions)


allTFs = dorotheaData.index.values

outputs = numpy.zeros((len(allTFs), len(allConditions)))
outputStd = numpy.zeros((len(allTFs), len(allConditions)))
outputCount = numpy.zeros(len(allConditions), dtype=int)
for i in range(len(allConditions)):
    affectedSamples = allConditions[i].split('_')
    affectedLigand = numpy.isin(metaData['ligand'].values, affectedSamples[0])
    stimState = len(affectedSamples) == 2
    affectedStim = metaData['stim'].values == stimState
    affectedFilter = numpy.logical_and(affectedLigand, affectedStim)
    selectedConditions = metaData.index.values[affectedFilter]
    curData = dorotheaData.loc[:,selectedConditions].values
    outputs[:, i] = numpy.mean(curData, axis=1)
    outputStd[:, i] = numpy.std(curData, axis=1)
    outputCount[i] = curData.shape[1]

print(outputs.shape)


print(allConditions[numpy.argsort(numpy.median(outputStd, axis=0))])
print(numpy.sort(numpy.median(outputStd, axis=0)))

signalConsistency = numpy.percentile(outputStd, 75, axis=1) < consistencyCutOf
#signalConsistency = numpy.median(outputStd, axis=1) < consistencyCutOf
inconsistentTFs = allTFs[signalConsistency==False]
print(inconsistentTFs)

conditionsPlusN = numpy.array(allConditions.copy(), dtype=object)
for i in range(len(conditionsPlusN)):
    conditionsPlusN[i] = '{:s} (N={:d})'.format(allConditions[i], outputCount[i])

plt.rcParams["figure.figsize"] = (7,20)
df = pd.DataFrame(outputStd.T, columns = allTFs, index=conditionsPlusN)
order = numpy.argsort(numpy.percentile(df.values, 75, axis=0))
df = df.iloc[:,order]
ax = sns.boxplot(data=df, orient='h', showfliers=False)
ax = sns.stripplot(data=df, orient='h', color = 'black')
leftRight = plt.ylim()
plt.plot([consistencyCutOf, consistencyCutOf], [leftRight[0], leftRight[1]], color='black')
ax.set_title('TF consistency')
plt.xlabel('STD')
plt.savefig("figures/TFSTD.svg") 

plt.figure()
order = numpy.argsort(numpy.percentile(df.values, 75, axis=1))
df = df.iloc[order,:]
ax = sns.boxplot(data=df.T, orient='h', showfliers=False)
ax = sns.stripplot(data=df.T, orient='h', color = 'black')
ax.set_title('Condition consistency')
plt.xlabel('STD')
plt.savefig("figures/ConditionSTD.svg") 


df = pd.DataFrame(outputs, index = allTFs, columns = allConditions)

qualityCriteria = signalConsistency
df = df.loc[qualityCriteria,:].copy()
#df = logIt(df)

# signalRange = numpy.logical_and(numpy.max(df, axis=1)>(1-rangeCutOff), numpy.min(df, axis=1)<rangeCutOff)
# qualityCriteria = signalRange
# print(df.index[qualityCriteria==False])
# df = df.loc[qualityCriteria,:].copy()


#df = customSigmodal(df)

#h = sns.clustermap(df, cmap='RdBu_r', vmin=0, vmax=1)
#sns.set(font_scale=0.7)
folder = '../model/figures/Figure 6/'
h = sns.clustermap(df.T, cmap='RdBu_r', vmin=0, vmax=1, xticklabels=True, yticklabels=True, figsize=(15, 20), dendrogram_ratio=0.08, cbar_pos=(0.02, 0.02, 0.02, 0.08))
h.ax_heatmap.set_xticklabels(h.ax_heatmap.get_xmajorticklabels(), fontsize = 14)
h.ax_heatmap.set_yticklabels(h.ax_heatmap.get_ymajorticklabels(), fontsize = 14)
plt.savefig(folder + "B.svg")
h.data2d.to_csv(folder + 'B.tsv', sep='\t')

df = updateIndexName(df, gene2uniprot)
df = df.T
df.to_csv('results/ligandScreen-TFs.tsv', sep='\t')

