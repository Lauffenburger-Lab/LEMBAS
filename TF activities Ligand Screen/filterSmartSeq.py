import numpy
import pandas
import matplotlib.pyplot as plt

#Hyper parameters
countCutOf = 2e6
geneCutOf = 5000

ensembl2geneData = pandas.read_csv('filtered/ensemblTable.tsv', sep='\t', low_memory=False)
ensembl2gene = dict(zip(ensembl2geneData['Source'], ensembl2geneData['external_gene_name']))

counts = pandas.read_csv('data/counts.tsv', sep='\t', low_memory=False, index_col=0)


metaData = pandas.read_csv('data/complete_meta.csv', sep=',', low_memory=False)
metaData = metaData.loc[:, ['plate_sample_id', 'stim', 'ligand', 'plate',  'donor', 'sample_id']]
metaData['condition'] = metaData['ligand'] + numpy.where(metaData['stim'], '_L', '')

#Fix a glitched metaData
metaData.loc[numpy.isin(metaData['plate_sample_id'], 'R2_2_B01'), 'donor'] = 'B'
metaData.loc[numpy.isin(metaData['plate_sample_id'], 'R2_2_C42_L'), 'donor'] = 'C'



metaId = metaData['plate_sample_id'].values
metaIdShort = metaId.copy()
for i in range(len(metaIdShort)):
    metaIdShort[i] = metaIdShort[i].replace('_', '')
    if len(metaIdShort[i]) == 6:
        metaIdShort[i] = metaIdShort[i] + 'V'
metaData.index = metaIdShort


plate = metaData['plate'].values.copy()
replicate = plate.copy()
for i in range(len(replicate)):
    curStr = plate[i].split('_')
    plate[i] = curStr[0]
    replicate[i] = curStr[1]
metaData['sample_plate'] = plate
metaData['replicate'] = replicate
metaData['plate_donor'] = metaData['sample_plate'] + '_' + metaData['donor']

keepIds = numpy.full(len(metaIdShort), True)

#Remove data without meta data
missMatchCols = numpy.setdiff1d(counts.columns.values, metaIdShort)
print('ID not in meta data:', missMatchCols)
counts = counts.drop(columns = missMatchCols)

#Remove data with low counts
totalCounts = numpy.sum(counts, axis=0)
plt.hist(totalCounts, 50)
plt.title('Total counts')
countFilter = totalCounts<countCutOf
print('Filter out low counts', sum(countFilter))
keepIds[countFilter] = False

#Remove data with few expressed genes
geneCounts = numpy.sum(counts>1, axis=0)
plt.figure()
plt.hist(geneCounts[keepIds], 50)
plt.title('Total genes')
countFilter = geneCounts<geneCutOf
print('Filter out low nr of expressed genes', sum(countFilter))
keepIds[countFilter] = False

#Subset
counts = counts.loc[:, keepIds].copy()
metaData = metaData.loc[counts.columns.values,:].copy()


#Colapse technical replicates
countValues = counts.values
allConditons = numpy.unique(metaData['condition'].values)
allPlates = numpy.unique(metaData['plate_donor'].values)

joinedCount = numpy.empty((counts.shape[0],0))
joinedMeta = metaData[0:0].copy()

for i in range(len(allPlates)):
    plateFilter = metaData['plate_donor'].values == allPlates[i]
    for j in range(len(allConditons)):
        conditionFilter = metaData['condition'].values == allConditons[j]   
        joinFilter = numpy.logical_and(plateFilter, conditionFilter)
        if sum(joinFilter)>0:
            firstIndex = numpy.argwhere(joinFilter)[0]
            addedReplicates = numpy.sum(countValues[:,joinFilter], axis=1).reshape(-1,1)
            joinedCount = numpy.append(joinedCount, addedReplicates, axis=1)
            joinedMeta = joinedMeta.append(metaData.iloc[firstIndex,:].copy())
       


joinedMeta['uniqueId'] = joinedMeta['sample_plate'] + '_' + joinedMeta['sample_id']
joinedMeta = joinedMeta[['uniqueId', 'condition', 'ligand', 'stim', 'donor', 'sample_plate']]
joinedMeta.to_csv('filtered/metaData.tsv', sep='\t', index=False)


joinedCount = pandas.DataFrame(joinedCount, index = counts.index, columns=joinedMeta['uniqueId'].copy())
#Load Gene names
fullNames = joinedCount.index.values.copy()
#fullNames = numpy.array([x.split('.')[0] for x in fullNames])

translatedGenes = numpy.array([ensembl2gene[x] for x in fullNames])
vals, counts = numpy.unique(translatedGenes, return_counts=True)
duplicatedNames = vals[counts>1]
duplicateFilter = numpy.isin(translatedGenes, duplicatedNames)
translatedGenes[duplicateFilter] = joinedCount.index.values[duplicateFilter]
joinedCount.index = translatedGenes
joinedCount.to_csv('filtered/counts.tsv', sep='\t')