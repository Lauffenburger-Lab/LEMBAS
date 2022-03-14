import torch
import numpy
import bionetwork
import plotting
import pandas
import argparse
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


#Get data number
testCondtions = pandas.read_csv('synthNetTime/conditions.tsv', sep='\t', low_memory=False)
nExperiments = testCondtions.shape[0]
dataSize = testCondtions['DataSize'].values

plt.rcParams["figure.figsize"] = (3,3)
timeToSample = numpy.zeros((nExperiments, 2))

for i in range(nExperiments):
    curResults = pandas.read_csv('synthNetTime/generalResults_' + str(i) + '.tsv', low_memory=False, sep='\t')
    curResults['deltaT'] =  curResults['stop'] - curResults['start']
    curResults = curResults.loc[curResults['index']<10000]
    timeToSample[i, 0] = numpy.mean(curResults['deltaT'])
    timeToSample[i, 1] = numpy.std(curResults['deltaT'])

plt.errorbar(dataSize, timeToSample[:,0], timeToSample[:,1], marker = 'o', capsize=3)
plt.xscale('log', base=10)
plt.ylim(bottom=0)
plt.xlabel('number of samples')
plt.ylabel('time per batch [s]')
plt.savefig("figures/samplesVsTime/timePerBatch.svg")


plt.figure()
for i in range(nExperiments):
    curResults = pandas.read_csv('synthNetTime/targetedEvaluation_' + str(i) + '.tsv', low_memory=False, sep='\t')
    plt.plot(curResults['iteration'], curResults['correlation'])
plt.legend(testCondtions['DataSize'], prop={'size': 7}, frameon=False)
plt.xlabel('iterations')
plt.ylim([0, 1])
plt.ylabel('correlation')


plt.figure()
for i in range(nExperiments):
    curResults = pandas.read_csv('synthNetTime/targetedEvaluation_' + str(i) + '.tsv', low_memory=False, sep='\t')
    plt.plot(curResults['iteration'], curResults['mse'])
plt.legend(testCondtions['DataSize'], prop={'size': 7}, ncol=2, frameon=False)
plt.yscale('log', base=10)
plt.xlabel('iterations')
#plt.ylim([0, 1])
plt.ylabel('mse')

plt.figure()
for i in range(nExperiments):
    curResults = pandas.read_csv('synthNetTime/generalResults_' + str(i) + '.tsv', low_memory=False, sep='\t')
    plt.plot(curResults['index'], plotting.movingaverage(curResults['mse'],50))
plt.legend(testCondtions['DataSize'], prop={'size': 7}, ncol=2, frameon=False)
plt.yscale('log', base=10)
plt.xlabel('iterations')
#plt.ylim([0, 1])
plt.ylabel('mse')

# curResults = pandas.read_csv('synthNetTime/targetedEvaluation_0.tsv', low_memory=False, sep='\t')
# iterations = curResults['iteration'].values
# corrResults = numpy.zeros((len(dataSize), len(iterations)))

# for i in range(nExperiments):
#     curResults = pandas.read_csv('synthNetTime/targetedEvaluation_' + str(i) + '.tsv', low_memory=False, sep='\t')
#     corrResults[i, :] = curResults['mse']
#     print(max(curResults['correlation']))

# plt.figure()
# df = pandas.DataFrame(corrResults, index=dataSize, columns=iterations)
# sns.heatmap(df, norm=LogNorm())

fitnessLevels = [0.75, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95]
results = numpy.zeros((nExperiments , len(fitnessLevels)))

for i in range(nExperiments):
    curResults = pandas.read_csv('synthNetTime/targetedEvaluation_' + str(i) + '.tsv', low_memory=False, sep='\t')
    curIterations = curResults['iteration'].values
    for j in range(len(fitnessLevels)):
        indexWithCompletedTarget = numpy.argwhere(curResults['correlation'].values>fitnessLevels[j]).flatten()
        if numpy.shape(indexWithCompletedTarget)[0] == 0:
            results[i, j] = numpy.nan   
        else:
            fitnessGoal = numpy.min(indexWithCompletedTarget)
            results[i, j] = curIterations[fitnessGoal] 

print('Increase in time at r=0.90', results[-1,1]/results[0,1])

# plt.rcParams["figure.figsize"] = (6,6)
# plt.figure()
# for i in range(len(fitnessLevels)):
#     plt.subplot(2, 2, i+1)
#     plt.plot(dataSize, results[:,i], 'o-', markersize=4)
#     plt.xscale('log', base=10)
#     plt.xlabel('data size')
#     plt.ylim(bottom=0)
#     plt.ylabel('iterations')
#     plt.title(fitnessLevels[i])
# plt.tight_layout()

plt.rcParams["figure.figsize"] = (4,4)
plt.figure()
for i in range(len(fitnessLevels)):
    plt.plot(dataSize, results[:,i], 'o-', markersize=4)
plt.xscale('log', base=10)
plt.xlabel('data size')
plt.ylim(bottom=0)
plt.ylabel('iterations')
plt.legend(fitnessLevels, frameon=False)
plt.savefig("figures/samplesVsTime/iterationsPerFitness.svg")

#%%

