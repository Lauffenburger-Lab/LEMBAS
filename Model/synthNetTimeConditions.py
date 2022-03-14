import pandas
import numpy

trainingSamples = numpy.array([50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])

allConditions = numpy.zeros((len(trainingSamples), 2), dtype=int)

for i in range(len(trainingSamples)):
    allConditions[i, 0] = i
    allConditions[i, 1] = trainingSamples[i]


pd = pandas.DataFrame(allConditions, columns = ['Index', 'DataSize'])
pd.to_csv('synthNetTime/conditions.tsv', sep='\t', index=False)


