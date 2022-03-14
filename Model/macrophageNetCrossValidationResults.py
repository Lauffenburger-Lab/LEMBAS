import torch
import numpy
import matplotlib.pyplot as plt
import bionetwork
import plotting
import pandas
from scipy.stats import pearsonr
import seaborn as sns
from scipy.stats import mannwhitneyu
import copy

def loadModel(refModel, fileName):
    #work around to copy weights and bias values 
    #because class structure has been updated since the run
    curModel = torch.load(fileName)
    model = copy.deepcopy(refModel)
    model.network.weights.data = curModel.network.weights.data.clone()
    model.network.bias.data = curModel.network.bias.data.clone()
    model.inputLayer.weights.data = curModel.inputLayer.weights.data.clone()
    model.projectionLayer.weights.data = curModel.projectionLayer.weights.data.clone()
    model.network.param = curModel.network.parameters.copy()
    return model

#Load network
networkList, nodeNames, modeOfAction = bionetwork.loadNetwork('data/macrophage-Model.tsv')
annotation = pandas.read_csv('data/macrophage-Annotation.tsv', sep='\t')
uniprot2gene = dict(zip(annotation['code'], annotation['name']))
ligandInput = pandas.read_csv('data/macrophage-Ligands.tsv', sep='\t', low_memory=False, index_col=0)
TFOutput = pandas.read_csv('data/macrophage-TFs.tsv', sep='\t', low_memory=False, index_col=0)

CVconditions = pandas.read_csv('CVmacrophage/conditions.tsv', sep='\t')
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

bionetParams = bionetwork.trainingParameters(iterations = 150, clipping=1, leak=0.01)
model = bionetwork.model(networkList, nodeNames, modeOfAction, 3, 1.2, inName, outName, bionetParams)

#%%
testedConditions = CVconditions['Condition'].values
dictionary = dict(zip(sampleName, list(range(len(sampleName)))))
conditionOrder = numpy.array([dictionary[x] for x in testedConditions])


referenceX = X[conditionOrder,:]
referenceY = Y[conditionOrder,:]
predictionY = torch.zeros(referenceY.shape)

sampleCorrelations = numpy.zeros((referenceY.shape[0], referenceY.shape[0]))
tfCorrelations = numpy.zeros(referenceY.shape)
samplePrediction = numpy.zeros(referenceY.shape[0])


trainFit = numpy.zeros(len(conditionOrder))
testFit = numpy.zeros(len(conditionOrder))

for i in range(len(conditionOrder)):
    curModel = loadModel(model, 'CVmacrophage/model_' + str(i) + '.pt')
    Yhat, YhatFull = curModel(referenceX)
    predictionY[i,:] = Yhat[i,:]
    trainMap = numpy.array(range(len(conditionOrder))) !=i
    trainFit[i] = criterion(Yhat[trainMap,:], referenceY[trainMap,:]).item()
    testFit[i] = criterion(Yhat[i,:].view(1, Yhat.shape[1]), referenceY[i,:].view(1, Yhat.shape[1])).item() #might need reshaping

    for j in range(len(conditionOrder)):
        r, p = pearsonr(Yhat[j, :].detach().numpy(), referenceY[j, :].numpy())
        if numpy.isnan(r):
            r = 0

        if i==j:
            sampleCorrelations[i,j] = numpy.NaN
            samplePrediction[i] = r
        else:
            sampleCorrelations[i,j] = r

    for j in range(Y.shape[1]):
        A = Yhat[:, j].detach().numpy()
        B = referenceY[:, j].numpy()
        A = numpy.delete(A, i)
        B = numpy.delete(B, i)
        r, p = pearsonr(A, B)
        if numpy.isnan(r):
            r = 0
        tfCorrelations[i, j] = r



scrambledY = torch.zeros(referenceY.shape)
sampleCorrelationsScrabled = numpy.zeros((referenceY.shape[0], referenceY.shape[0]))
tfCorrelationsScrabled = numpy.zeros(referenceY.shape)
samplePredictionScrabled = numpy.zeros(referenceY.shape[0])

trainFitScrambled = numpy.zeros(len(conditionOrder))
testFitScrambled = numpy.zeros(len(conditionOrder))


for i in range(len(conditionOrder)):
    curModel = loadModel(model, 'CVmacrophage/scrambled_' + str(i) + '.pt')
    Yhat, YhatFull = curModel(referenceX)
    scrambledY[i,:] = Yhat[i,:]
    trainMap = numpy.array(range(len(conditionOrder))) !=i
    trainFitScrambled[i] = criterion(Yhat[trainMap,:], referenceY[trainMap,:]).item()
    testFitScrambled[i] = criterion(Yhat[i,:].view(1, Yhat.shape[1]), referenceY[i,:].view(1, Yhat.shape[1])).item() #might need reshaping

    for j in range(len(conditionOrder)):
        r, p = pearsonr(Yhat[j, :].detach().numpy(), referenceY[j, :].numpy())
        if numpy.isnan(r):
            r = 0

        if i==j:
            sampleCorrelationsScrabled[i,j] = numpy.NaN
            samplePredictionScrabled[i] = r
        else:
            sampleCorrelationsScrabled[i,j] = r

    for j in range(Y.shape[1]):
        A = Yhat[:, j].detach().numpy()
        B = referenceY[:, j].numpy()
        A = numpy.delete(A, i)
        B = numpy.delete(B, i)
        r, p = pearsonr(A, B)
        if numpy.isnan(r):
            r = 0
        tfCorrelationsScrabled[i, j] = r



predictionCorrelations = plotting.calculateCorrelations(referenceY, predictionY)
predictionCorrelationsScrambled = plotting.calculateCorrelations(referenceY, scrambledY)



failedTFCutof = 0.4
failedTfs = numpy.mean(tfCorrelations, axis=0)<failedTFCutof


#curModel = torch.load('CVmacrophage/model_scramble.pt')
#scrambledY, YhatFull = curModel(referenceX)
#scrambledCorrelation = plotting.calculateCorrelations(referenceY, scrambledY)
#scrambledConditions = plotting.calculateCorrelations(referenceY.T, scrambledY.T)


curModel = loadModel(model, 'CVmacrophage/model_reference.pt')
fullModelY, YhatFull = curModel(X)
#fullCorrelation = plotting.calculateCorrelations(referenceY, scrambledY)
#fullConditions = plotting.calculateCorrelations(referenceY.T, scrambledY.T)

#%%
plt.rcParams["figure.figsize"] = (15,15)
plt.figure()
plotting.displayData(Y, sampleName, outNameGene)
plt.savefig("figures/literature CV/heatmap.svg")   

plt.rcParams["figure.figsize"] = (3,3)
plt.figure()


averagePrediction = torch.mean(referenceY, axis=0).repeat(referenceY.shape[0],1)
r, p = pearsonr(referenceY.numpy().flatten(), averagePrediction.numpy().flatten())
predictingAverage = r


plt.figure()
A = fullModelY.detach().numpy()
B = Y.detach().numpy()
# A = A[:, failedTfs==False]
# B = B[:, failedTfs==False]
plt.scatter(A, B, alpha=0.2, color='gray')
r, p = pearsonr(A.flatten(), B.flatten())
plt.xlabel('Train')
plt.ylabel('Data')
plt.gca().axis('equal')
plt.gca().set_xticks([0, 0.5, 1])
plt.gca().set_yticks([0, 0.5, 1])
plotting.lineOfIdentity()
plt.text(0, 0.9, 'r {:.2f}'.format(r))
plt.savefig("figures/literature CV/train.svg")   

plt.figure()
A = predictionY.detach().numpy()
B = referenceY.detach().numpy()
# A = A[:, failedTfs==False]
# B = B[:, failedTfs==False]
plt.scatter(A, B, alpha=0.2)
r, p = pearsonr(A.flatten(), B.flatten())

# A = predictionY[:, failedTfs].detach().numpy()
# B = referenceY[:, failedTfs].detach().numpy()
# plt.scatter(A, B, alpha=0.1, color='red')
plt.xlabel('Test (LOOCV)')
plt.ylabel('Data')
plt.gca().axis('equal')
plt.gca().set_xticks([0, 0.5, 1])
plt.gca().set_yticks([0, 0.5, 1])
plotting.lineOfIdentity()
plt.text(0, 0.9, 'r {:.2f}\np {:.2e}'.format(r, p))
plt.savefig("figures/literature CV/test.svg")   

plt.figure()
A = scrambledY.detach().numpy()
B = referenceY.detach().numpy()
# A = A[:, failedTfs==False]
# B = B[:, failedTfs==False]
plt.scatter(A, B, alpha=0.2, color='gray')
r, p = pearsonr(A.flatten(), B.flatten())

# A = predictionY[:, failedTfs].detach().numpy()
# B = referenceY[:, failedTfs].detach().numpy()
# plt.scatter(A, B, alpha=0.1, color='red')
plt.xlabel('Test (Scrambled Y)')
plt.ylabel('Data')
plt.gca().axis('equal')
plt.gca().set_xticks([0, 0.5, 1])
plt.gca().set_yticks([0, 0.5, 1])
plotting.lineOfIdentity()
plt.text(0, 0.9, 'r {:.2f}\np {:.2e}'.format(r, p))

# plt.figure()
# plt.scatter(trainFit, testFit)
# plt.xlim(left=0)
# plt.ylim(bottom=0)

#sampleCorrelations =

plt.figure()
trainCorrelations = numpy.mean(tfCorrelations, axis=0)
plt.scatter(trainCorrelations, predictionCorrelations, alpha=0.5)
#plt.scatter(scrambledCorrelation, trainCorrelations)

for i in range(len(trainCorrelations)):
    if trainCorrelations[i]< failedTFCutof:
        plt.text(trainCorrelations[i], predictionCorrelations[i], outNameGene[i], rotation=45)
plt.xlabel('Correlation Train')
plt.ylabel('Correlation Test')
plt.xlim(right=1)
plt.ylim(top=1)


plt.figure()
trainCorrelations = numpy.mean(tfCorrelationsScrabled, axis=0)
plt.scatter(predictionCorrelationsScrambled, trainCorrelations, alpha=0.5)
#plt.scatter(scrambledCorrelation, trainCorrelations)

for i in range(len(trainCorrelations)):
    if trainCorrelations[i]< failedTFCutof:
        plt.text(predictionCorrelationsScrambled[i], trainCorrelations[i], outNameGene[i])
plt.xlabel('Test TF Correlation')
plt.ylabel('Train TF Correlation')
plt.xlim(right=1)

# plt.figure()
# plt.scatter(scrambledCorrelation, predictionCorrelations)
# plt.xlabel('Scrambled Y')
# plt.ylabel('Cross validation')


plt.rcParams["figure.figsize"] = (6, 3)
# plt.figure()
# df = pandas.DataFrame(tfCorrelations, columns=outNameGene, index=testedConditions)
# sns.boxplot(data=df, orient="h", color='grey')
# Ylocation = numpy.array(range(len(predictionCorrelations)))
# plt.scatter(predictionCorrelations, Ylocation)
# plt.scatter(scrambledCorrelation, Ylocation)

plt.figure()
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

#defineOrder = numpy.argsort(samplePrediction)
defineOrder = numpy.flip(numpy.argsort(samplePrediction))
df = pandas.DataFrame(sampleCorrelations, columns=testedConditions, index=testedConditions)
df = df.iloc[:,defineOrder]
sns.boxplot(data=df, color='#BBBBBB')  #orient="h",
plt.ylabel('Correlation')
plt.gca().set_xticklabels(plt.gca().get_xticklabels(), rotation=90)
#plt.gca().yaxis.tick_right()
#plt.gca().yaxis.set_label_position("right")


Xinterval = numpy.array([0, len(samplePrediction)])-0.5
Xlocation = numpy.array(range(len(samplePrediction)))

plt.scatter(Xlocation, samplePrediction[defineOrder], color=colors[0])
plt.scatter(Xlocation, samplePredictionScrabled[defineOrder], color=colors[1])
plt.plot(Xinterval, [0, 0], 'k')

r = numpy.mean(samplePrediction)
plt.plot(Xinterval, [r, r], color=colors[0], alpha=0.5)
plt.text(18, r, '{:.2f}'.format(r))

r = numpy.mean(samplePredictionScrabled)
plt.plot(Xinterval, [r, r], color=colors[1], alpha=0.5)
plt.text(18, r, '{:.2f}'.format(r))

plt.plot(Xinterval, [predictingAverage, predictingAverage], 'k--')

U1, p = mannwhitneyu(samplePrediction, samplePredictionScrabled)
plt.text(0, r, 'p {:.2e}'.format(p))
plt.xlim(Xinterval)
plt.savefig("figures/literature CV/CVconditions.svg")   

#plt.legend(['CV', 'Scrambled Y'])

# plt.rcParams["figure.figsize"] = (10,10)
# plt.figure()
# rank = plotting.compareAllTFs(Yhat, Y, outNameGene)

# plt.figure()
# rank = plotting.compareAllTFs(Yhat.T, Y.T, sampleName)


# plotting.compareDataAndModel(X.detach(), Y.detach(), Yhat.detach(), sampleName, outNameGene)

