import numpy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def updateColumnName(df, dictionary):
    allIndex = df.columns.values
    for i in range(len(allIndex)):
        if allIndex[i] in dictionary:
            allIndex[i] = dictionary[allIndex[i]]
    df.columns = allIndex
    return df


def logIt(data):
    data = -numpy.log10(1/data -1)
    return data

#x = numpy.linspace(0,1,1000)


#%%
controlName = 'CON'
consistencyCutOf = 0.2 #75th percentile of STD below this level

plt.rcParams["figure.figsize"] = (5, 5)
dorotheaData = pd.read_csv('results/macrophage_DoRothEA.txt', sep='\t', index_col=0)
dorotheaData = dorotheaData.T

uniprot = pd.read_csv('annotation/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab', sep='\t')
gene2uniprot = dict(zip(uniprot['Gene names  (primary )'], uniprot['Entry']))

metaData = pd.read_csv('results/macrophageKey.txt', sep='\t')
dorotheaData = dorotheaData.loc[metaData['Condition'],:]

t = 1
dorotheaData = 1/(1 + numpy.exp(-t * dorotheaData))

conditionId = metaData['Stimulation']
allConditionList = [x.split('_')[0] for x in conditionId]
allConditions = numpy.unique(allConditionList)
allTFs = dorotheaData.columns.values


outputs = numpy.zeros((len(allConditions), len(allTFs)))
outputStd = numpy.zeros((len(allConditions), len(allTFs)))
outputCount = numpy.zeros(len(allConditions))

for i in range(len(allConditions)):
    selectedConditions = numpy.isin(allConditionList, allConditions[i])
    curData = dorotheaData.loc[selectedConditions,:].values
    outputs[i,:] = numpy.mean(curData, axis=0)
    outputStd[i,:] = numpy.std(curData, axis=0)
    outputCount[i] = curData.shape[0]


signalConsistency = numpy.percentile(outputStd, 75, axis=0) < consistencyCutOf
inconsistentTFs = allTFs[signalConsistency==False]
print(inconsistentTFs)
plt.rcParams["figure.figsize"] = (7,20)
df = pd.DataFrame(outputStd, columns = allTFs, index=allConditions)
order = numpy.argsort(numpy.percentile(df.values, 75, axis=0))
df = df.iloc[:,order]
ax = sns.boxplot(data=df, orient='h', showfliers=False)
ax = sns.stripplot(data=df, orient='h', color = 'black')
leftRight = plt.ylim()
plt.plot([consistencyCutOf, consistencyCutOf], [leftRight[0], leftRight[1]], color='black')
ax.set_title('TF consistency')
plt.xlabel('STD')
plt.savefig("figures/TFSTD.svg")   

plt.figure()
order = numpy.argsort(numpy.percentile(df.values, 75, axis=1))
df = df.iloc[order,:]
ax = sns.boxplot(data=df.T, orient='h', showfliers=False)
ax = sns.stripplot(data=df.T, orient='h', color = 'black')
ax.set_title('Condition consistency')
plt.xlabel('STD')
plt.savefig("figures/ConditionSTD.svg")   

# plt.rcParams["figure.figsize"] = (5,1)
# plt.figure()
# order = numpy.argsort(df['STAT2'].values)
# df = df.iloc[order,:]
# plt.bar(numpy.arange(df.shape[0]), df['STAT2'])
# plt.xticks(numpy.arange(df.shape[0]), labels=df.index.values, rotation=90)
# plt.ylim([0, 0.5])
# plt.title('STAT2')
# plt.ylabel('STD')

#plt.gca().set_xticklabels(plt.gca().get_xticklabels(), rotation=90)

df = pd.DataFrame(outputs, index = allConditions, columns = allTFs)
qualityCriteria = signalConsistency
df = df.loc[:,qualityCriteria].copy()

#df = logIt(df)

#h = sns.clustermap(df, cmap='RdBu_r', vmin=0, vmax=1)
#sns.set(font_scale=0.7)
h = sns.clustermap(df, cmap='RdBu_r', vmin=0, vmax=1, yticklabels=True, xticklabels=True)


plt.rcParams["figure.figsize"] = (7,7)
plt.figure()
plt.figure()
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df.values.T)
sc = plt.scatter(principalComponents[:,0], principalComponents[:,1])
#plt.colorbar(sc)
for i in range(len(df.columns)):
    plt.text(principalComponents[i,0], principalComponents[i,1], df.columns[i])
plt.xlabel(pca.explained_variance_ratio_[0])
plt.ylabel(pca.explained_variance_ratio_[1])


plt.figure()
A = pca.components_[0,:]
B = pca.components_[1,:]
plt.scatter(A, B)
for i in range(len(df.index)):
    plt.text(A[i], B[i], df.index[i])


df = updateColumnName(df, gene2uniprot)
df.to_csv('results/macrophage-TFs.tsv', sep='\t')
#print(allTFs[signalRange<rangeCutOff])
plt.figure()
plt.hist(outputCount)
plt.xlabel('Sample count')
plt.ylabel('Instances')

