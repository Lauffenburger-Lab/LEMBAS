import torch
import pandas
import numpy

ligandInput = pandas.read_csv('data/macrophage-Ligands.tsv', sep='\t', low_memory=False, index_col=0)
inName = ligandInput.columns.values
ligandInput = ligandInput.loc[:,inName]
sampleName = ligandInput.index.values
X = torch.tensor(ligandInput.values.copy(), dtype=torch.double)

#Split data
#TODO split
singleConditionLigands = torch.sum(X,dim=0) == 1
singleConditions = torch.sum(X[:,singleConditionLigands],dim=1)>0

cvConditions = sampleName[singleConditions == False]
cvIndex = numpy.array(range(len(cvConditions)))

pd = pandas.DataFrame((cvIndex, cvConditions), index = ['Index', 'Condition']).T
pd.to_csv('CVmacrophage/conditions.tsv', sep='\t', index=False)


