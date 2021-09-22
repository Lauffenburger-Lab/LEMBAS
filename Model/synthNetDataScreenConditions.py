import pandas
import numpy

simulataniousLigands = numpy.array([2, 3, 5])
trainingSamples = numpy.array([10, 50, 100, 400, 800])

allConditions = numpy.zeros((len(simulataniousLigands)*len(trainingSamples), 3), dtype=int)
k=0

for i in range(len(simulataniousLigands)):
    for j in range(len(trainingSamples)):
        allConditions[k, 0] = k
        allConditions[k, 1] = simulataniousLigands[i]
        allConditions[k, 2] = trainingSamples[j]
        k+=1

pd = pandas.DataFrame(allConditions, columns = ['Index', 'Ligands', 'DataSize'])
pd.to_csv('synthNetScreen/conditions.tsv', sep='\t', index=False)


