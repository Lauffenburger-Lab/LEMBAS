import numpy
import pandas as pd

#%%

dorotheaData = pd.read_csv('results/macrophage-TFs.tsv', sep='\t', index_col=0)
ligandMapping = pd.read_csv('annotation/macrophage-ligandMap.tsv', sep='\t')

allLigands = numpy.unique(ligandMapping['Source'])
ligandNames = [x.replace('Ligand_', '') for x in allLigands]


allConditions = dorotheaData.index.values

inputs = numpy.zeros((len(allConditions), len(allLigands)))

for i in range(len(allConditions)):
    affectedLigands = allConditions[i].split('+')
    affectedFilter = numpy.isin(ligandNames, affectedLigands)
    inputs[i, affectedFilter] = 1

df = pd.DataFrame(inputs, index = allConditions, columns = allLigands)

df = df.astype(int)

#Remove unused ligands
unusedLigands = numpy.sum(df,axis=0)==0
print('Removed unused ligands, ', allLigands[unusedLigands])


df.to_csv('results/macrophage-Ligands.tsv', sep='\t')