import torch
import pandas
import numpy

ligandInput = pandas.read_csv('data/ligandScreen-Ligands.tsv', sep='\t', low_memory=False, index_col=0)
inName = ligandInput.columns.values
ligandInput = ligandInput.loc[:,inName]
sampleName = numpy.array(ligandInput.index.values)
X = torch.tensor(ligandInput.values.copy(), dtype=torch.double)


#Split data
samplesPerFold = 3

multiLigand = numpy.sum(ligandInput.values, axis=0)>1
multiLigand[numpy.isin(inName, ['CHEBI16412'])] = False #cannot directly select LPS because this is the stimulation
selectedLigands = numpy.argwhere(multiLigand).flatten()

LPSfirst = numpy.random.rand(len(selectedLigands))>0.5
ligandOrder = numpy.random.permutation(len(selectedLigands)).flatten()
ligandOrder2 = numpy.random.permutation(len(selectedLigands)).flatten()


sampleIndex = numpy.zeros(len(ligandOrder) * 2, dtype=int)
sampleFold = numpy.zeros(len(ligandOrder) * 2, dtype=int)
currentFold = 0

lpsLookup = (ligandInput['CHEBI16412'] == 1).values
ligandLookup = selectedLigands[ligandOrder]
k = 0
for i in range(len(ligandOrder)):
    lpsCondition = lpsLookup == LPSfirst[ligandOrder[i]]
    ligandCondition = (ligandInput.iloc[:,ligandLookup[i]] == 1).values
    sampleIndex[k] = numpy.argwhere(numpy.logical_and(lpsCondition, ligandCondition)).flatten()
    sampleFold[k] = currentFold
    if k%samplesPerFold==(samplesPerFold-1):
        currentFold+=1
    k+=1

ligandLookup = selectedLigands[ligandOrder2]
for i in range(len(ligandOrder)):
    lpsCondition = lpsLookup != LPSfirst[ligandOrder2[i]]
    ligandCondition = (ligandInput.iloc[:,ligandLookup[i]] == 1).values
    sampleIndex[k] = numpy.argwhere(numpy.logical_and(lpsCondition, ligandCondition)).flatten()
    sampleFold[k] = currentFold
    if k%samplesPerFold==(samplesPerFold-1):
        currentFold+=1
    k+=1

cvConditions = sampleName[sampleIndex]

pd = pandas.DataFrame((sampleFold, cvConditions), index = ['Index', 'Condition']).T
pd.to_csv('CVligandScreen/conditions.tsv', sep='\t', index=False)

