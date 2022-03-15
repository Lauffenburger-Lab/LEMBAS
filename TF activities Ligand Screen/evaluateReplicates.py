import numpy
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def updateIndexName(df, dictionary):
    allIndex = df.index.values
    for i in range(len(allIndex)):
        if allIndex[i] in dictionary:
            allIndex[i] = dictionary[allIndex[i]]
    df.index = allIndex
    return df

def getMeanCorrelation(data):
    N = 0
    corrValue = 0
    for j in range(data.shape[1]):
        for k in range(j+1, data.shape[1]):
            r, p = pearsonr(data[:, j], data[:, k])
            corrValue+=r
            N+=1
    meanCorr = corrValue/N
    return meanCorr

#%%
stimName = 'LPS'
controlName = 'PBS-BSA'
correlationCutOf = 0.5 

plt.rcParams["figure.figsize"] = (5, 5)
dorotheaData = pd.read_csv('results/dorothea.tsv', sep='\t')

selectedTFs = pd.read_csv('results/ligandScreen-TFs.tsv', sep='\t', index_col=0)
selectedTFs = selectedTFs.columns.values


ligandMapping = pd.read_csv('data/ligandMap.tsv', sep='\t')
ligand2id = dict(zip(ligandMapping['Name'], ligandMapping['Code']))

uniprot = pd.read_csv('data/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab', sep='\t')
gene2uniprot = dict(zip(uniprot['Gene names  (primary )'], uniprot['Entry']))
uniprot2gene = dict(zip(uniprot['Entry'], uniprot['Gene names  (primary )']))


selectedTFs = numpy.array([uniprot2gene[X] for X in selectedTFs])


metaData = pd.read_csv('filtered/metaData.tsv', sep='\t')
metaData.index = metaData['uniqueId']
metaData = metaData.loc[dorotheaData.columns.values,:]

stim = numpy.where(metaData['stim'], '_' + stimName, '')
conditionId = metaData['ligand'].values + stim

allConditions, counts = numpy.unique(conditionId, return_counts=True)


dorotheaData = 1/(1 + numpy.exp(-1 * dorotheaData))
dorotheaData = dorotheaData.loc[selectedTFs,:]

allTFs = dorotheaData.index.values
correlationLevel = numpy.zeros(len(allConditions))


for i in range(len(allConditions)):
    affectedSamples = allConditions[i].split('_')
    affectedLigand = numpy.isin(metaData['ligand'].values, affectedSamples[0])
    stimState = len(affectedSamples) == 2
    affectedStim = metaData['stim'].values == stimState
    affectedFilter = numpy.logical_and(affectedLigand, affectedStim)
    selectedConditions = metaData.index.values[affectedFilter]
    curData = dorotheaData.loc[:,selectedConditions].values    
    if curData.shape[1]> 1:
        correlationLevel[i] = getMeanCorrelation(curData)

        if correlationLevel[i]>correlationCutOf:    
            for j in range(curData.shape[1]):
                for k in range(j+1, curData.shape[1]):
                    plt.scatter(curData[:, j], curData[:, k], color = [0.5,0.5,0.5], alpha=0.05)         
        else:
            print(i, affectedSamples, correlationLevel[i])

    #else:
    #    print(i, affectedSamples, '(No replicates)')
    
    
plt.figure()
plt.hist(correlationLevel)
plt.xlabel('Mean correlation between replicates')
plt.ylabel('#conditions')
print(numpy.mean(correlationLevel[correlationLevel>0]))

