import numpy
import pandas as pd

def updateNames(nameList, dictionary):
    for i in range(len(nameList)):
        if nameList[i] in dictionary:
            nameList[i] = dictionary[nameList[i]]
    return nameList

#%%
stimName = 'LPS'
control = 'PBS-BSA'

metaData = pd.read_csv('filtered/metaData.tsv', sep='\t')
metaData.index = metaData['uniqueId']
dorotheaData = pd.read_csv('results/ligandScreen-TFs.tsv', sep='\t', index_col=0)
ligandMapping = pd.read_csv('data/ligandMap.tsv', sep='\t')
ligand2id = dict(zip(ligandMapping['Name'], ligandMapping['Code']))

allConditions = dorotheaData.index.values

allLigands = numpy.unique(metaData['ligand'].values)
allLigands = allLigands[allLigands!=control]
allLigands = numpy.concatenate((allLigands, [stimName]))

inputs = numpy.zeros((len(allConditions), len(allLigands)))

for i in range(len(allConditions)):
    affectedLigands = allConditions[i].split('_')
    affectedFilter = numpy.isin(allLigands, affectedLigands)
    inputs[i, affectedFilter] = 1

df = pd.DataFrame(inputs, index = allConditions, columns = allLigands)

df.columns = updateNames(df.columns.values, ligand2id)
df = df.astype(int)

#Remove unused ligands
unusedLigands = numpy.sum(df,axis=0)==0
print('Removed unused ligands, ', allLigands[unusedLigands])


df.to_csv('results/ligandScreen-Ligands.tsv', sep='\t')