import torch
import numpy
import matplotlib.pyplot as plt
import bionetwork
import plotting
import pandas
from scipy.stats import pearsonr
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LinearRegression

#Load network
networkList, nodeNames, modeOfAction = bionetwork.loadNetwork('data/ligandScreen-Model.tsv')
annotation = pandas.read_csv('data/ligandScreen-Annotation.tsv', sep='\t')
uniprot2gene = dict(zip(annotation['code'], annotation['name']))
ligandInput = pandas.read_csv('data/ligandScreen-Ligands.tsv', sep='\t', low_memory=False, index_col=0)
TFOutput = pandas.read_csv('data/ligandScreen-TFs.tsv', sep='\t', low_memory=False, index_col=0)

CVconditions = pandas.read_csv('CVligandScreen/conditions.tsv', sep='\t')
criterion = torch.nn.MSELoss(reduction='mean')


#Subset input and output to intersecting nodes
inName = ligandInput.columns.values
outName = TFOutput.columns.values
inName = numpy.intersect1d(nodeNames, inName)
outName = numpy.intersect1d(nodeNames, outName)
inNameGene = [uniprot2gene[x] for x in inName]
outNameGene = [uniprot2gene[x] for x in outName]
ligandInput = ligandInput.loc[:,inName]
TFOutput = TFOutput.loc[:,outName]

sampleName = ligandInput.index.values
X = torch.tensor(ligandInput.values.copy(), dtype=torch.double)
Y = torch.tensor(TFOutput.values, dtype=torch.double)


#%%
testedConditions = CVconditions['Condition'].values
dictionary = dict(zip(sampleName, list(range(len(sampleName)))))
conditionOrder = numpy.array([dictionary[x] for x in testedConditions])
folds = CVconditions.Index.values
foldNames = numpy.unique(folds)
Nfolds = len(foldNames)
selectedSamples = sampleName[conditionOrder]


referenceX = X[conditionOrder,:]
referenceY = Y[conditionOrder,:]
predictionY = torch.zeros(referenceY.shape).double()

resultsY = torch.zeros(Nfolds, referenceY.shape[0],  referenceY.shape[1])

sampleCorrelations = numpy.zeros((Nfolds, referenceY.shape[0]))
tfCorrelations = numpy.zeros((Nfolds, referenceY.shape[1]))
tfCorrelationPredict = numpy.zeros((Nfolds, referenceY.shape[1]))

samplePrediction = numpy.zeros(referenceY.shape[0])


# trainFit = numpy.zeros(len(conditionOrder))
# testFit = numpy.zeros(len(conditionOrder))

for i in range(Nfolds):
    curModel = torch.load('CVligandScreen/MML_model_' + str(i) + '.pt')
    Yhat, YhatFull = curModel(referenceX)
    resultsY[i,:,:] = Yhat
    predictionMap= folds==i
    trainMap = numpy.logical_not(predictionMap)
    predictionY[predictionMap, :] = Yhat[predictionMap, :]

    # trainFit[i] = criterion(Yhat[trainMap,:], referenceY[trainMap,:]).item()
    # testFit[i] = criterion(Yhat[predictionMap,:], referenceY[predictionMap,:]).item() #might need reshaping

    for j in range(len(conditionOrder)):
        r, p = pearsonr(Yhat[j, :].detach().numpy(), referenceY[j, :].numpy())
        if numpy.isnan(r):
            r = 0
        if predictionMap[j]:
            sampleCorrelations[i, j] = numpy.NaN
        else:
            sampleCorrelations[i, j] = r

    for j in range(Y.shape[1]):
        A = Yhat[:, j].detach().numpy()
        B = referenceY[:, j].numpy()
        A = numpy.delete(A, predictionMap)
        B = numpy.delete(B, predictionMap)
        r, p = pearsonr(A, B)
        if numpy.isnan(r):
            r = 0
        tfCorrelations[i, j] = r

    # for j in range(Y.shape[1]):
    #     A = Yhat[:, j].detach().numpy()
    #     B = referenceY[:, j].numpy()
    #     A = numpy.delete(A, numpy.logical_not(predictionMap))
    #     B = numpy.delete(B, numpy.logical_not(predictionMap))
    #     r, p = pearsonr(A, B)
    #     if numpy.isnan(r):
    #         r = 0
    #     tfCorrelationPredict[i, j] = r



samplePrediction = plotting.calculateCorrelations(referenceY.T, predictionY.T)
predictionCorrelations = plotting.calculateCorrelations(referenceY, predictionY)



scrambledY = torch.zeros(referenceY.shape, dtype=referenceY.dtype)

for i in range(Nfolds):
    curModel = torch.load('CVligandScreen/MML_sramble_model_' + str(i) + '.pt')
    Yhat, YhatFull = curModel(referenceX)
    predictionMap= folds==i
    trainMap = numpy.logical_not(predictionMap)
    scrambledY[predictionMap, :] = Yhat[predictionMap, :]


samplePredictionScrambled = plotting.calculateCorrelations(referenceY.T, scrambledY.T)        
predictionCorrelationsScrambled = plotting.calculateCorrelations(referenceY, scrambledY)


reluY = torch.zeros(referenceY.shape, dtype=referenceY.dtype)

for i in range(Nfolds):
    curModel = torch.load('CVligandScreen/leakyRelu_model_' + str(i) + '.pt')
    Yhat, YhatFull = curModel(referenceX)
    predictionMap= folds==i
    trainMap = numpy.logical_not(predictionMap)
    reluY[predictionMap, :] = Yhat[predictionMap, :]

samplePredictionRelu = plotting.calculateCorrelations(referenceY.T, scrambledY.T)        
predictionCorrelationsRelu = plotting.calculateCorrelations(referenceY, scrambledY)


#curModel = torch.load('CVmacrophage/model_scramble.pt')
#scrambledY, YhatFull = curModel(referenceX)
#scrambledCorrelation = plotting.calculateCorrelations(referenceY, scrambledY)
#scrambledConditions = plotting.calculateCorrelations(referenceY.T, scrambledY.T)


#%%
plt.rcParams["figure.figsize"] = (3,3)

plt.figure()
A = predictionY.detach().numpy()
B = referenceY.detach().numpy()
# A = A[:, failedTfs==False]
# B = B[:, failedTfs==False]
plt.scatter(A, B, alpha=0.02)
r, p = pearsonr(A.flatten(), B.flatten())

# A = predictionY[:, failedTfs].detach().numpy()
# B = referenceY[:, failedTfs].detach().numpy()
# plt.scatter(A, B, alpha=0.1, color='red')
plt.xlabel('Prediction')
plt.ylabel('Data')
plt.gca().axis('equal')
plt.gca().set_xticks([0, 0.5, 1])
plt.gca().set_yticks([0, 0.5, 1])
plotting.lineOfIdentity()
plt.text(0, 0.9, 'r {:.2f}\np {:.2e}'.format(r, p))
plt.savefig("figures/ligand screen CV/testPerformance.svg")   

# axisScale = 40
# counts, rangeX, rangeY = numpy.histogram2d(A.flatten(), B.flatten(), bins=axisScale, range=[[0, 1], [0, 1]])
# counts_transformed = numpy.log10(counts+1)
# ax = sns.heatmap(counts_transformed.T, mask=counts_transformed==0, vmin=0, cmap="Blues", cbar_kws={'label': 'log10(#preditions + 1)'})
# ax.invert_yaxis()
# for _, spine in ax.spines.items():
#     spine.set_visible(True)
# #sns.histplot(df, x="Model", y="Reference", bins=100, cbar=True, cbar_kws={'label': 'number of preditions'}, vmax=50)
# ax.axis('equal')
# plt.xlabel('fit.')
# plt.ylabel('ref.')
# plt.gca().set_xticks(numpy.linspace(0, axisScale, 5))
# plt.gca().set_yticks(numpy.linspace(0, axisScale, 5))
# plt.gca().set_xlim([0, axisScale])
# plt.gca().set_ylim([0, axisScale])
# plt.gca().set_xticklabels(numpy.linspace(0, 1, 5), rotation = 0)
# plt.gca().set_yticklabels(numpy.linspace(0, 1, 5))
# plt.gca().set_xlabel('Model')
# plt.gca().set_xlabel('Reference')
# r, p = pearsonr(A.flatten(), B.flatten())
# plt.text(0, axisScale *0.9, 'r {:.2f}'.format(r))
# plotting.lineOfIdentity()

# plt.figure()
# plt.scatter(trainFit, testFit)
# plt.xlim(left=0)
# plt.ylim(bottom=0)

#sampleCorrelations =

plt.figure()
trainCorrelations = numpy.mean(tfCorrelations, axis=0)
plt.scatter(trainCorrelations,predictionCorrelations, alpha=0.5)
#plt.scatter(scrambledCorrelation, trainCorrelations)

failedTFCutof = 0.3
failedTfs = predictionCorrelations<failedTFCutof

for i in range(len(trainCorrelations)):
    if predictionCorrelations[i]< failedTFCutof:
        plt.text(trainCorrelations[i], predictionCorrelations[i], outNameGene[i])
plt.xlabel('Train')
plt.ylabel('Test')
plt.xlim(right=1)
#plotting.lineOfIdentity()
plt.axis('equal')
plt.xlim([0, 1])
plt.ylim([0, 1])
r, p = pearsonr(trainCorrelations, predictionCorrelations)
plt.text(0.7, 0.8, 'r={:.1f}'.format(r))
reg = LinearRegression().fit(predictionCorrelations.reshape(-1, 1), trainCorrelations.reshape(-1, 1))
Y = numpy.array([0, 1])
X = reg.predict(Y.reshape(-1, 1))
plt.plot(X, Y, color = 'tab:orange')
plt.savefig("figures/ligand screen CV/trainVSTestTF.svg")   

# plt.figure()
# plt.scatter(scrambledCorrelation, predictionCorrelations)
# plt.xlabel('Scrambled Y')
# plt.ylabel('Cross validation')


plt.rcParams["figure.figsize"] = (4, 15)
# plt.figure()
# df = pandas.DataFrame(tfCorrelations, columns=outNameGene, index=testedConditions)
# sns.boxplot(data=df, orient="h", color='grey')
# Ylocation = numpy.array(range(len(predictionCorrelations)))
# plt.scatter(predictionCorrelations, Ylocation)
# plt.scatter(scrambledCorrelation, Ylocation)
plt.figure()
sampleOrder = numpy.flip(numpy.argsort(predictionCorrelations))
df = pandas.DataFrame(tfCorrelations, columns=outNameGene.copy(), index=foldNames)
df = df.iloc[:,sampleOrder]
df = pandas.melt(df)
df = df.dropna()
sns.boxplot(x='value', y='variable', data=df, color='grey')

# df = pandas.DataFrame(tfCorrelationPredict, columns=outNameGene.copy(), index=foldNames)
# df = df.iloc[:,sampleOrder]
# df = pandas.melt(df)
# df = df.dropna()
# sns.boxplot(x='value', y='variable', data=df, color='blue')

Ylocation = numpy.array(range(len(predictionCorrelations)))
plt.scatter(predictionCorrelations[sampleOrder], Ylocation)
#plt.scatter(predictionCorrelationsScrambled[sampleOrder], Ylocation, alpha=0.2)
# plt.scatter(scrambledConditions, Ylocation)
# plt.plot([0, 0], [0, len(samplePrediction)], 'k')
# U1, p = mannwhitneyu(samplePrediction, scrambledConditions)
# plt.text(-0.3, 1, 'p {:.2e}'.format(p))
plt.ylabel('TF')
plt.xlabel('Correlation')
plt.xlim(right=1)
plt.xlim([0, 1])
plt.savefig("figures/ligand screen CV/correlationsTF.svg")   


plt.rcParams["figure.figsize"] = (4, 18)
plt.figure()
sampleOrder = numpy.flip(numpy.argsort(samplePrediction))
df = pandas.DataFrame(sampleCorrelations, columns=testedConditions.copy(), index=foldNames)
df = df.iloc[:,sampleOrder]
df = pandas.melt(df)
df = df.dropna()
sns.boxplot(x='value', y='variable', data=df, color='grey')
Ylocation = numpy.array(range(len(samplePrediction)))
plt.scatter(samplePrediction[sampleOrder], Ylocation)
plt.xlim([-1, 1])
plt.savefig("figures/ligand screen CV/correlationsConditions.svg")   
#plt.scatter(samplePredictionScrambled[sampleOrder], Ylocation, alpha=0.2)


#colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#Ylocation = [0, len(samplePrediction)]





#plt.scatter(scrambledConditions, Ylocation)
#plt.plot([0, 0], [0, len(samplePrediction)], 'k')
#U1, p = mannwhitneyu(samplePrediction, scrambledConditions)
#plt.text(-0.3, 1, 'p {:.2e}'.format(p))
plt.ylabel('Condition')
plt.xlabel('Correlation')
plt.xlim(right=1)

plt.rcParams["figure.figsize"] = (5, 5)
plt.figure()
results = numpy.zeros((Nfolds, 4))
for i in range(Nfolds):
    predictionMap= folds==i
    trainMap = numpy.logical_not(predictionMap)
    
    #Train
    r, p = pearsonr(resultsY[i, trainMap,:].detach().numpy().flatten(), referenceY[trainMap,:].numpy().flatten())
    results[i, 0] = r
    
    #Test
    r, p = pearsonr(resultsY[i, predictionMap,:].detach().numpy().flatten(), referenceY[predictionMap,:].numpy().flatten())
    results[i, 1] = r    
    
    #Scrambled
    r, p = pearsonr(scrambledY[predictionMap,:].detach().numpy().flatten(), referenceY[predictionMap,:].numpy().flatten())
    results[i, 2] = r    

    #leaky relu
    r, p = pearsonr(reluY[predictionMap,:].detach().numpy().flatten(), referenceY[predictionMap,:].numpy().flatten())
    results[i, 3] = r         
             
    

df = pandas.DataFrame(results, columns = ['Train', 'Test', 'Scramble', 'Relu'])
sns.boxplot(data=df)
plt.ylabel('correlation')
plt.plot([-0.5, 3.5], [0, 0], color='k')
plt.ylim([-1, 1])

#r = numpy.mean(samplePrediction)
#plt.plot([r, r], Ylocation, color=colors[0], alpha=0.5)
#r = numpy.mean(samplePredictionScrabled)
#plt.plot([r, r], Ylocation, color=colors[1], alpha=0.5)

U1, p = mannwhitneyu(results[:,1], results[:,2])
plt.text(1, -0.5, 'p={:.2e}'.format(p))

U1, p = mannwhitneyu(results[:,1], results[:,3])
print('Compared to ReLU', p)

meanR = numpy.mean(results, axis=0)
stdR = numpy.std(results, axis=0)
for i in range(len(meanR)):
    plt.text(i-0.3, -0.85, '{:.2f}±{:.2f}'.format(meanR[i], stdR[i]))
plt.savefig("figures/ligand screen CV/compareCV.svg")   

# plt.rcParams["figure.figsize"] = (5, 5)
# plt.figure()
# meanCorrelationTrain = numpy.nanmean(sampleCorrelations, axis=0)
# plt.scatter(meanCorrelationTrain, samplePrediction)

# failedConditionCutof = 0.5

# for i in range(len(meanCorrelationTrain)):
#     if samplePrediction[i]< failedConditionCutof:
#         plt.text(meanCorrelationTrain[i], samplePrediction[i], testedConditions[i])
# plt.xlabel('Train')
# plt.ylabel('Test')
# plt.xlim(right=1)
# plotting.lineOfIdentity()



#plt.legend(['CV', 'Scrambled Y'])

# plt.rcParams["figure.figsize"] = (10,10)
# plt.figure()
# rank = plotting.compareAllTFs(Yhat, Y, outNameGene)

# plt.figure()
# rank = plotting.compareAllTFs(Yhat.T, Y.T, sampleName)


# plotting.compareDataAndModel(X.detach(), Y.detach(), Yhat.detach(), sampleName, outNameGene)

# plt.figure()
# pca = PCA(n_components=4)
# principalComponents = pca.fit_transform(referenceY.detach().numpy())
# sc = plt.scatter(principalComponents[:,1], principalComponents[:,2])
# #plt.colorbar(sc)
# for i in range(len(testedConditions)):
#     plt.text(principalComponents[i,1], principalComponents[i,2], testedConditions[i])
# plt.xlabel(pca.explained_variance_ratio_[1])
# plt.ylabel(pca.explained_variance_ratio_[2])

