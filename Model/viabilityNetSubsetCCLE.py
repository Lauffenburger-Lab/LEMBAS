import numpy
import bionetwork
import pandas

#Load network
networkList, nodeNames, modeOfAction = bionetwork.loadNetwork('viabilityModel/ligandScreen-Model.tsv')
annotation = pandas.read_csv('viabilityModel/ligandScreen-Annotation.tsv', sep='\t')
uniprot2gene = dict(zip(annotation['code'], annotation['name']))
nodeNamesGene = numpy.array([uniprot2gene[x] for x in nodeNames])

#Load cell line data
cellLineLevels = pandas.read_csv('viabilityModel/cellLineRKPMZscored.tsv', sep='\t', low_memory=False, index_col=0)
missingValues = numpy.setdiff1d(nodeNamesGene, cellLineLevels.index.values)
#Zero padding:
df = pandas.DataFrame(numpy.zeros((len(missingValues), cellLineLevels.shape[1])), index=missingValues, columns=cellLineLevels.columns)
cellLineLevels = cellLineLevels.append(df)
cellLineLevels = cellLineLevels.loc[nodeNamesGene,:]


cellLineLevels.to_csv('viabilityModel/cellLineRKPMZscored_subset.tsv', sep='\t')