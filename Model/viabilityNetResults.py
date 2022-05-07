import numpy
import matplotlib.pyplot as plt
import pandas
from scipy.stats import pearsonr
import seaborn as sns
import plotting
import torch

#Load experimental data
viabilitydata = pandas.read_csv('viabilityModel/viability.tsv', sep='\t', low_memory=False, index_col=0)
drugData = pandas.read_csv('viabilityModel/drug.tsv', sep='\t', low_memory=False, index_col=0)
drugNames = drugData.columns.values.copy()
drugConcentrations = numpy.unique(drugData.values)
drugConcentrations = drugConcentrations[drugConcentrations!=0]  #only use conditions with drugs

folder = 'figures/SI Figure 18/'

#%%%
plt.rcParams["figure.figsize"] = (3,3)
plt.figure()
regression = pandas.read_csv('viabilityModel/CV/fit_ref.tsv', sep='\t', low_memory=False, index_col=0)
Yhat = regression['viability'].values
Y = viabilitydata['viability'].values

plt.scatter(Yhat, Y, color=[0.5,0.5,0.5], alpha=0.1)
plotting.lineOfIdentity()
plotting.addCorrelation(torch.tensor(Yhat), torch.tensor(Y))
plt.xlabel('Fit')
plt.ylabel('Data')
plt.gca().axis('equal')
#plt.savefig("figures/viability/train.svg")
df = pandas.DataFrame((Yhat, Y), index=['Train', 'Data']).T
plt.savefig(folder + 'B.svg')
df.to_csv(folder + 'B.tsv', sep='\t')


#%%
plt.figure()

#load full model
referenceViability = pandas.read_csv('viabilityModel/CV/fit_ref.tsv', sep='\t', low_memory=False, index_col=0)
performance = numpy.zeros((len(drugConcentrations), len(drugNames))) 
performanceAllDrugs = numpy.zeros(len(drugConcentrations))

predictedViability = referenceViability['viability'].values
matchingData = viabilitydata.loc[referenceViability.index,:]
matchingData = matchingData['viability'].values.copy()
matchingConditions = drugData.loc[referenceViability.index,:]

for i in range(len(drugConcentrations)):
    curConcentration = drugConcentrations[i]
    for j in range(len(drugNames)):
        dataFilter = matchingConditions.iloc[:,j] == curConcentration
        r, p = pearsonr(predictedViability[dataFilter], matchingData[dataFilter])
        performance[i, j] = r
        
    dataFilter = numpy.any(matchingConditions == curConcentration, axis=1)
    r, p = pearsonr(predictedViability[dataFilter], matchingData[dataFilter])
    performanceAllDrugs[i] = r
    
r, p = pearsonr(predictedViability, matchingData)
generalPerformance = r
print(generalPerformance)

plt.rcParams["figure.figsize"] = (4,3)
#Make some plots
X = 10**drugConcentrations/1e3

for i in range(len(drugNames)):
    plt.plot(X, performance[:, i])
plt.plot(X, performanceAllDrugs, 'k-')

legendValues = numpy.append(drugNames, 'All')
plt.legend(legendValues, prop={'size': 7}, frameon=False) #ncol=2,
plt.ylabel('correlation')
plt.xlabel('concentration')
plt.xscale('log', base=10)
plt.ylim([-0.2, 1])
plt.plot(X[[0, -1]], [generalPerformance, generalPerformance], 'k--')

#this type of plot is perhaps not so informative, it shows how well predictions rank in each category
#if there is not much difference within a category then rank is not particularly important



#%%

plt.figure()
CVtest = pandas.read_csv('viabilityModel/CVtest.tsv', sep='\t', low_memory=False, index_col=0)
allFolds = numpy.unique(CVtest['testfold'].values)
df = pandas.DataFrame(columns=['drug', 'fold', 'source', 'correlation'])


drugPressence = drugData.values>0
drugList = numpy.array([drugNames[X][0] for X in drugPressence])
drugList = pandas.DataFrame(drugList, index=drugData.index, columns=['drug'])


source = ['fit', 'prediction']
labels = ['train', 'test']
for curSource in source:
    for i in range(len(allFolds)):
        referenceViability = pandas.read_csv('viabilityModel/CV/' + curSource + '_' + str(allFolds[i]) + '.tsv', sep='\t', low_memory=False, index_col=0)
        predictedViability = referenceViability['viability'].values
        matchingData = viabilitydata.loc[referenceViability.index,:]
        matchingData = matchingData['viability'].values.copy()
        matchingConditions = drugList.loc[referenceViability.index,'drug'].values
        
        for j in range(len(drugNames)):
            dataFilter = matchingConditions == drugNames[j]
            r, p = pearsonr(predictedViability[dataFilter], matchingData[dataFilter])
            dfRow = {'correlation': r, 'fold': allFolds[i], 'drug': drugNames[j], 'source': curSource}
            df = df.append(dfRow, ignore_index = True)
             
        r, p = pearsonr(predictedViability, matchingData)
        dfRow = {'correlation': r, 'fold': allFolds[i], 'drug': 'All', 'source': curSource}
        df = df.append(dfRow, ignore_index = True)        


dfAll = df.loc[df['drug']=='All',:].copy()
dfDrug = df.loc[df['drug']!='All',:].copy()

results = numpy.zeros((2,2))
for i in range(2):
    curData = dfAll.loc[dfAll['source']==source[i], 'correlation']
    results[i, 0] = numpy.mean(curData)
    results[i, 1] = numpy.std(curData)
print(results)

plt.rcParams["figure.figsize"] = (2,3)
plt.figure()
g = sns.boxplot(x="source", y="correlation", data=dfAll, showfliers = False, linewidth=0.5)
ax = sns.stripplot(x="source", y="correlation", data=dfAll, color=".25", dodge=True, size=4)
ax.set_xticklabels(labels)
ax.set(xlabel=None)
plt.ylim([0, 1])
plt.text(-0.5, results[0, 0]-0.4, '{:.2f}±{:.2f}'.format(results[0, 0], results[0, 1]))
plt.text(0.5, results[1, 0]-0.4, '{:.2f}±{:.2f}'.format(results[1, 0], results[1, 1]))
#plt.savefig("figures/viability/test.svg")   
df = pandas.DataFrame((Yhat, Y), index=['Train', 'Data']).T
plt.savefig(folder + 'C.svg')
dfAll.to_csv(folder + 'C.tsv', sep='\t')

      
#%%         
plt.rcParams["figure.figsize"] = (6,3)
#Drug figure
plt.figure()
g = sns.boxplot(x="drug", y="correlation", hue='source', data=dfDrug, showfliers = False)
handles = g.legend_.legendHandles
ax = sns.stripplot(x="drug", y="correlation", hue='source', data=dfDrug, color=".25", dodge=True, size=4)
plt.ylim([0, 1])
plt.legend(handles, labels, frameon=False)
plt.xticks(rotation=45)