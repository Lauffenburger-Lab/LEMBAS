import pandas
import numpy

cellLineList = pandas.read_csv('cellLine.tsv', sep='\t', low_memory=False, index_col=0)
cellLineList = cellLineList.columns

cellLineData = pandas.read_csv('data CCLE/CCLE_RNAseq_genes_rpkm_20180929.gct ', sep='\t', low_memory=False, index_col=1, skiprows=2)


cellLineName = cellLineData.columns.values

for i in range(1, len(cellLineName)):
    cellLineName[i] = cellLineName[i].split('_')[0]


cellLineData.columns = cellLineName
#overlap = numpy.isin(cellLineName, cellLineList)

subsetdata = cellLineData.loc[:,cellLineList]

subsetLogp1 = numpy.log(subsetdata.values.copy()+1)
subsetLogp1STD = numpy.std(subsetLogp1, axis=1).reshape(-1,1)
subsetLogp1STD[subsetLogp1STD==0] = 1 #No standard dev -> ignore
subsetLogp1Mean = numpy.mean(subsetLogp1, axis=1).reshape(-1,1)
subsetLogp1Zscore = (subsetLogp1-subsetLogp1Mean)/subsetLogp1STD
subsetdata = pandas.DataFrame(subsetLogp1Zscore, index=subsetdata.index, columns=subsetdata.columns)

#sum duplicates
subsetdata = subsetdata.groupby(subsetdata.index).sum()

subsetdata.to_csv('cellLineRKPMZscored.tsv', sep='\t')
