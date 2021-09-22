import numpy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import scipy.stats
from scipy.stats import pearsonr
import time

def getR2(yhat, y):
    y_mean_line = numpy.mean(y) * numpy.ones(y.shape[0])
    squared_error_regr = numpy.sum((y - yhat) ** 2)
    squared_error_y_mean = numpy.sum((y - y_mean_line) ** 2)
    return 1 - (squared_error_regr/squared_error_y_mean)

def movingaverage(values, w):
    moving_averages = numpy.zeros(values.shape)
    stepsBefore = numpy.ceil(w/2)
    stepsAfter = numpy.ceil(w/2)
    for i in range(values.shape[0]):
        start = numpy.max((i-stepsBefore, 0)).astype(int)
        stop = numpy.min((i+stepsAfter, values.shape[0])).astype(int)
        moving_averages[i] = numpy.mean(values[start:stop])
    return moving_averages


def lineOfIdentity():
    xBounds = plt.xlim()
    yBounds = plt.ylim()
    minLevel = numpy.min([xBounds[0], yBounds[0]])
    maxLevel = numpy.max([xBounds[1], yBounds[1]])
    plt.plot([minLevel, maxLevel], [minLevel, maxLevel], 'black', label='_nolegend_')
    plt.text(maxLevel, maxLevel*0.75, 'X=Y ', ha='right', va='top')

def addCorrelation(YHat, Y):

    r, p = pearsonr(YHat.detach().flatten(), Y.detach().flatten())
    plt.text(0, 0.9, 'r {:.2f}'.format(r))

def plotComparison(YHat, Y, YtestHat, Ytest):
    YHat = YHat.detach().numpy()
    YtestHat = YtestHat.detach().numpy()
    Y = Y.detach().numpy()
    #YHat = numpy.clip(YHat, 0, 1)
    #YtestHat = numpy.clip(YtestHat, 0, 1)

#    plt.plot([0, 1.1], [0, 1.1], 'black', label='_nolegend_')
#    plt.plot([0, 1], [0, 1], 'black', )
    lineOfIdentity()
    plt.scatter(YHat, Y, alpha=0.1, color=[0.5, 0.5, 0.5])
    plt.scatter(YtestHat, Ytest, alpha=0.3)
    plt.xlabel('Fit')
    plt.ylabel('Experimental data')
    plt.gca().axis('equal')
#    plt.gca().set_xticks([0,0.5,1])
#    plt.gca().set_yticks([0,0.5,1])

    r, p = pearsonr(YHat.flatten(), Y.flatten())
    rTest, pTest = pearsonr(YtestHat.flatten(), Ytest.flatten())
    plt.text(0, 0.8, 'Train: r {:.2f}\nTest: r {:.2f}, p {:.2e}'.format(r, rTest, pTest))

def contourPlot(X, Y, Z):
    df = pd.DataFrame.from_dict(numpy.array([X.numpy().flatten(), Y.numpy().flatten(), Z.numpy().flatten()]).T)
    df.columns = ['X_value','Y_value','Z_value']
    #df = df.round(1)
    pivotted = df.pivot('Y_value','X_value','Z_value')
    ax = sns.heatmap(pivotted, cmap='gray', vmin=0, vmax=1, square=True)
    ax.invert_yaxis()
    ax.set_ylabel('')
    ax.set_xlabel('')

    fmt = '{:0.1f}'
    xticklabels = []
    for item in ax.get_xticklabels():
        item.set_text(fmt.format(float(item.get_text())))
        xticklabels += [item]
    yticklabels = []
    for item in ax.get_yticklabels():
        item.set_text(fmt.format(float(item.get_text())))
        yticklabels += [item]
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

def shadePlot(X, Y, STD):
    plt.plot(X, Y)
    curColor = plt.gca().lines[-1].get_color()
    plt.fill_between(X, Y - STD, Y + STD, color=curColor, alpha=0.2)


def errorAndDistance(signalDistance, fitDistance, signalDistanceTest, fitDistanceTest, trainName, testName):
    plt.scatter(signalDistance, fitDistance)
    plt.scatter(signalDistanceTest, fitDistanceTest)
    for i in range(len(trainName)):
        plt.text(signalDistance[i], fitDistance[i], trainName[i])
    for i in range(len(testName)):
        plt.text(signalDistanceTest[i], fitDistanceTest[i], testName[i])
    plt.ylim(bottom=0)
    plt.xlabel('Distance from ctrl')
    plt.ylabel('Error')

def errorAndSR(sr, fitDistance, srTest, fitDistanceTest, trainName, testName, spectralCapacity):
    # plt.subplot(1 ,3, 1)
    # plt.scatter(sr[:,0].flatten(), fitDistance)
    # plt.scatter(srTest[:,0].flatten(), fitDistanceTest)
    # boundY = plt.ylim()
    # plt.plot([spectralCapacity, spectralCapacity], [0, boundY[1]])

    # for i in range(len(trainName)):
    #     plt.text(sr[i,0], fitDistance[i], trainName[i])
    # for i in range(len(testName)):
    #     plt.text(srTest[i,0], fitDistanceTest[i], testName[i])
    # plt.ylim(bottom=0)
    # plt.xlabel('Spectral radius')
    # plt.ylabel('Error')

    # plt.subplot(1,3,2)
    # plt.scatter(sr[:,1].flatten(), fitDistance)
    # plt.scatter(srTest[:,1].flatten(), fitDistanceTest)
    # boundY = plt.ylim()
    # plt.plot([spectralCapacity, spectralCapacity], [0, boundY[1]])

    # for i in range(len(trainName)):
    #     plt.text(sr[i,1], fitDistance[i], trainName[i])
    # for i in range(len(testName)):
    #     plt.text(srTest[i,1], fitDistanceTest[i], testName[i])
    # plt.ylim(bottom=0)
    # plt.xlabel('Spectral radius')
    # plt.ylabel('Error')

    # plt.subplot(1, 3, 3)
    plt.scatter(sr[:, 0].flatten(), sr[:, 1].flatten())
    plt.scatter(srTest[:, 0].flatten(), srTest[:, 1].flatten())

    for i in range(len(trainName)):
        plt.text(sr[i,0], sr[i,1], trainName[i])
    for i in range(len(testName)):
        plt.text(srTest[i,0], srTest[i,1], testName[i])

    plt.xlabel('Spectral radius F')
    plt.ylabel('Spectral radius B')



def allCorrelations(YhatFull, Y, nodeNames, outName, uniprot2gene, cutOf):
#    nodeNamesGene = [uniprot2gene[x] for x in nodeNames]
    nodeNamesGene = nodeNames
    outNameGene = [uniprot2gene[x] for x in outName]
    pearson = numpy.corrcoef(YhatFull.detach().numpy().T, Y.detach().numpy().T)
    pearson = pearson[0:YhatFull.shape[1],:]
    pearson = pearson[:,YhatFull.shape[1]:]
    print(pearson.shape)
    df = pd.DataFrame(pearson, index=nodeNamesGene, columns=outNameGene)
    df.round(3)
    sns.clustermap(df, cmap='RdBu_r', vmin=-1, vmax=1)

def plotAllIncommingStates(YhatFull, Y, networkList, nodeNames, outname, uniprot2gene, node):
    YhatFull = YhatFull.detach().numpy()
    maxNeighbors = 9
    curNode = numpy.argwhere(numpy.isin(nodeNames, node))[0]
    incommingNodes = networkList[1, networkList[0,:]==curNode]
    Yhat = YhatFull[:,curNode]
    X = YhatFull[:,incommingNodes]

    curNode = numpy.argwhere(numpy.isin(outname, node))[0]
    Ydata = Y[:,curNode]
    nodesToPlot = min(maxNeighbors, len(incommingNodes))
    for i in range(nodesToPlot):
        plt.subplot(3, 3, i+1)
        plt.scatter(X[:,i], Yhat)
        if len(curNode)>0:
            plt.scatter(X[:,i], Ydata)
        curName = nodeNames[incommingNodes[i]]
        plt.xlabel(curName + '(' + uniprot2gene[curName] + ')')
        plt.xlim([-0.1, 1])
        plt.ylim([-0.1, 1])
    plt.suptitle(node)

def initProgressObject(maxIter):
    stats = {}
    stats['startTime'] = time.time()
    stats['endTime'] = 0
    stats['loss'] = float('nan')*numpy.ones(maxIter)
    stats['lossSTD'] = float('nan')*numpy.ones(maxIter)
    stats['eig'] = float('nan')*numpy.ones(maxIter)
    stats['eigSTD'] = float('nan')*numpy.ones(maxIter)

    stats['test'] = float('nan')*numpy.ones(maxIter)
    stats['rate'] = float('nan')*numpy.ones(maxIter)
    stats['violations'] = float('nan')*numpy.ones(maxIter)

    return stats

def finishProgress(stats):
    stats['endTime'] = time.time()
    print('Time:',  stats['endTime']- stats['startTime'])
    return stats

def storeProgress(stats, e, loss=None, eig=None, lr=None, violations=None, test=None):
    if loss != None:
        stats['loss'][e] = numpy.mean(numpy.array(loss))
        stats['lossSTD'][e] = numpy.std(numpy.array(loss))

    if eig != None:
        stats['eig'][e] = numpy.mean(numpy.array(eig))
        stats['eigSTD'][e] = numpy.std(numpy.array(eig))

    if lr != None:
        stats['rate'][e] = lr

    if violations != None:
        stats['violations'][e] = violations

    if test != None:
        stats['test'][e] = test

    return stats


def printStats(e, stats):
    outString = 'i={:.0f}'.format(e)
    if numpy.isnan(stats['loss'][e]) == False:
        outString += ', l={:.5f}'.format(stats['loss'][e])
    if numpy.isnan(stats['test'][e]) == False:
        outString += ', t={:.5f}'.format(stats['test'][e])
    if numpy.isnan(stats['eig'][e]) == False:
        outString += ', s={:.3f}'.format(stats['eig'][e])
    if numpy.isnan(stats['rate'][e]) == False:
        outString += ', r={:.5f}'.format(stats['rate'][e])
    if numpy.isnan(stats['violations'][e]) == False:
        outString += ', v={:.0f}'.format(stats['violations'][e])
    print(outString)



def plotTrainingProgress(stats, mLoss, N, semiLog = False):

    T = numpy.array(range(stats['loss'].shape[0]))


    plt.subplot(2, 2, 1)
    curT = T[numpy.isnan(stats['loss']) == False]
    curE = stats['loss'][numpy.isnan(stats['loss']) == False]
    Tm = movingaverage(curT, N)
    E = movingaverage(curE, N)
    if semiLog:
        plt.semilogy(Tm, E, color='black')
    else:
        plt.plot(Tm, E, color='black')
    plt.plot([0, len(Tm)], numpy.array([1, 1])*min(curE))
    plt.plot([0, len(Tm)], numpy.array([1, 1])*mLoss[0])
    plt.plot([0, len(Tm)], numpy.array([1, 1])*mLoss[1])
    plt.xlim([0, len(Tm)])

    plt.ylim(bottom=0)
    plt.title('Loss')
    plt.text(0.8, 0.9, 'e = {:.3f}'.format(curE[-1]), ha='center', va='center', transform=plt.gca().transAxes)

    plt.subplot(2, 2, 2)
    curT = T[numpy.isnan(stats['test']) == False]
    curE = stats['test'][numpy.isnan(stats['test']) == False]
    if len(curT)>0:
        if semiLog:
            plt.semilogy(curT, curE, color='black')
        else:
            plt.plot(curT, curE, color='black')
        plt.plot([0, curT[-1]], numpy.array([1, 1])*min(curE))
        plt.plot([0, curT[-1]], numpy.array([1, 1])*mLoss[0])
        plt.plot([0, curT[-1]], numpy.array([1, 1])*mLoss[1])
        plt.ylim(bottom=0)
        plt.title('Test Loss')
        plt.text(0.8, 0.9, 't = {:.3f}'.format(curE[-1]), ha='center', va='center', transform=plt.gca().transAxes)
        plt.xlim([0, len(Tm)])

    plt.subplot(2, 2, 3)
    plt.plot(T, stats['rate'], color='black')
    plt.ylim(bottom=0)
    plt.title('learning rate')
    plt.xlim([0, len(Tm)])

    plt.subplot(2, 2, 4)
    curT = T[numpy.isnan(stats['eig']) == False]
    curE = stats['eig'][numpy.isnan(stats['eig']) == False]
    plt.plot(curT, curE, color='black')
    plt.plot([0, len(T)], [1, 1])
    plt.ylim(bottom=0)
    plt.title('spectral radius')
    plt.xlim([0, len(Tm)])

    # plt.figure()
    # plt.plot(storeTest[storeTest!=1])
    # plt.ylim(bottom=0)


    plt.tight_layout()


def displayData(Y, sampleName, outName):
    df = pd.DataFrame(Y.T.numpy())
    df.columns = sampleName
    df.index = outName
    df.round(3)
    sns.clustermap(df, cmap='RdBu_r', vmin=0, vmax=1)
    plt.title('data')

def compareDataAndModel(X, Y, Yhat, sampleName, outName):
    Yhat[Yhat<0] = 0
    Yhat[Yhat>1] = 1
    Yhat = Yhat.detach().T.numpy()

    df = pd.DataFrame(Y.T.numpy())
    df.columns = sampleName
    df.index = outName
    df.round(3)

    clustergrid = sns.clustermap(df,  cmap='RdBu_r', vmin=0, vmax=1)
    plt.title('data')
    rowOrder = clustergrid.data2d.index
    colOrder = clustergrid.data2d.columns

    df = pd.DataFrame(Yhat)
    df.columns = sampleName
    df.index = outName
    df.round(3)
    df = df.loc[:, colOrder]
    df = df.loc[rowOrder, :]
    plt.figure()
    sns.heatmap(df,  cmap='RdBu_r', vmin=0, vmax=1)
    plt.title('model')


def compareValues(Yhat, Y):
    #plt.figure()
    Y = Y.flatten().numpy()
    Yhat = Yhat.flatten().numpy()
    Yhat[Yhat<0] = 0
    Yhat[Yhat>1] = 1

    plt.scatter(Yhat, Y, color='black', alpha=0.2, edgecolors=None)
    plt.xlabel('model')
    plt.ylabel('data')
    plt.plot([0, 1], [0, 1])

    #pearson = stats.pearsonr(Yhat, Y)
    #R2Value = pearson[0]**2
    R2Value = getR2(Yhat, Y)
    (R, p) = scipy.stats.pearsonr(Yhat, Y)
    r2 = "R2 {:.3f}\nR {:.3f}\np {:.2e}".format(R2Value, R, p)
    plt.text(0, 1, r2, verticalalignment='top')


def compareTrainAndTest(Yhat, Y, YtestHat, Ytest):
    Yhat[Yhat<0] = 0
    Yhat[Yhat>1] = 1
    plt.subplot(1, 2, 1)
    compareValues(Yhat, Y)
    plt.title('train')

    YtestHat[YtestHat<0] = 0
    YtestHat[YtestHat>1] = 1
    plt.subplot(1, 2, 2)
    plt.title('test')
    compareValues(YtestHat, Ytest)
    plt.tight_layout()


def compareAllTFs(Yhat, Y, outputNames):
    Yhat = Yhat.detach().numpy()
    Y = Y.detach().numpy()
    Yhat[Yhat<0] = 0
    Yhat[Yhat>1] = 1
    outputNames = numpy.array(outputNames)

    # for i in range(Y.shape[1]):
    #     plt.subplot(5, 5, i+1)
    #     plt.plot(Yhat[:, i], Y[:, i], 'o', color='black')
    #     plt.plot([0, 1], [0, 1])
    #     plt.title(outputNames[i])
    #plt.tight_layout()

    result = numpy.zeros(Y.shape[1])
    for i in range(Y.shape[1]):
        r, p = pearsonr(Yhat[:, i], Y[:, i])
        #result[i] = getR2(Yhat[:, i].numpy(), Y[:, i].numpy())
        if numpy.isnan(r):
            r, p = pearsonr(Yhat[:, i]+numpy.random.randn(Yhat.shape[0])*1e-8, Y[:, i])
        result[i] = r

    order = numpy.argsort(result)
    plt.barh(outputNames[order], result[order])
    print(outputNames[order])
    plt.plot([1, 1], [0, len(outputNames)])
    plt.xlabel('Correlation Fit')
    return result


def calculateCorrelations(Y, Yhat):
    Yhat = Yhat.detach().numpy()
    Y = Y.detach().numpy()
    result = numpy.zeros(Yhat.shape[1])
    for i in range(len(result)):
        r, p = pearsonr(Yhat[:, i], Y[:, i])
        if numpy.isnan(r):
            r = 0
        result[i] = r
    return result

def compareTFcorrelations(Yhat, Y, YtestHat, Ytest, outputNames):
    Yhat = torch.clamp(Yhat, 0, 1)
    YtestHat = torch.clamp(YtestHat, 0, 1)

    outputNames = numpy.array(outputNames)

    # for i in range(Y.shape[1]):
    #     plt.subplot(5, 5, i+1)
    #     plt.plot(Yhat[:, i], Y[:, i], 'o', color='black')
    #     plt.plot([0, 1], [0, 1])
    #     plt.title(outputNames[i])
    #plt.tight_layout()

    resultTrain = calculateCorrelations(Y, Yhat)
    resultTest = calculateCorrelations(Ytest, YtestHat)

    plt.scatter(resultTest, resultTrain)
    plt.plot([-1, 1], [-1, 1], color=[0, 0, 0])
    plt.xlabel('Test corelation')
    plt.ylabel('Train correlation')


    r, p = pearsonr(resultTest, resultTrain)
    plt.text(-0.5, 0.8, 'r {:.2f} \np {:.2e}'.format(r, p))

    for i in range(len(outputNames)):
        plt.text(resultTest[i], resultTrain[i], outputNames[i])


def plotHeatmap(Y, names):
    df = pd.DataFrame(Y.T)
    #df.columns = sampleName
    df.index = names
    df.round(3)
    sns.clustermap(df, cmap='gray', vmin=0, vmax=1)


def plotHistogram(vals, tresh):
    valFilter = torch.abs(vals)>tresh
    plt.hist(vals[valFilter], 100)
    plt.text(0, 10, 'non-zeros {}'.format(torch.sum(valFilter).item()))


def showTrainingProgress(data):
    T = numpy.array(range(data.shape[0]))
    data = data.detach().numpy()
    plt.subplot(4, 3, 1)
    plt.plot(T, numpy.mean(data, axis=1))
    plt.title('mean')

    plt.subplot(4, 3, 2)
    plt.plot(T, numpy.std(data, axis=1))
    plt.title('std')

    plt.subplot(4, 3, 3)
    plt.plot(T, numpy.median(data, axis=1))
    plt.title('median')

    plt.subplot(4, 3, 4)
    plt.plot(T, numpy.mean(numpy.abs(data), axis=1))
    plt.title('abs mean')

    plt.subplot(4, 3, 5)
    plt.plot(T, numpy.std(numpy.abs(data), axis=1))
    plt.title('abs std')

    plt.subplot(4, 3, 6)
    plt.plot(T, numpy.median(numpy.abs(data), axis=1))
    plt.title('abs median')

    nonZeroFilter = numpy.abs(data)>0.001

    meanNonZero = numpy.zeros(data.shape[0])
    stdNonZero = numpy.zeros(data.shape[0])
    medianNonZero = numpy.zeros(data.shape[0])

    meanNonZeroAbs = numpy.zeros(data.shape[0])
    stdNonZeroAbs = numpy.zeros(data.shape[0])
    medianNonZeroAbs = numpy.zeros(data.shape[0])


    for i in range(len(meanNonZero)):
        meanNonZero[i] = numpy.mean(data[i, nonZeroFilter[i,:]])
        stdNonZero[i] = numpy.std(data[i, nonZeroFilter[i,:]])
        medianNonZero[i] = numpy.median(data[i, nonZeroFilter[i,:]])

        meanNonZeroAbs[i] = numpy.mean(numpy.abs(data[i, nonZeroFilter[i,:]]))
        stdNonZeroAbs[i] = numpy.std(numpy.abs(data[i, nonZeroFilter[i,:]]))
        medianNonZeroAbs[i] = numpy.median(numpy.abs(data[i, nonZeroFilter[i,:]]))

    plt.subplot(4, 3, 7)
    plt.plot(T, meanNonZero)
    plt.title('mean non-zeros')

    plt.subplot(4, 3, 8)
    plt.plot(T, stdNonZero)
    plt.title('std non-zeros')

    plt.subplot(4, 3, 9)
    plt.plot(T, medianNonZero)
    plt.title('median non-zeros')

    plt.subplot(4, 3, 10)
    plt.plot(T, meanNonZeroAbs)
    plt.title('mean non-zeros')

    plt.subplot(4, 3, 11)
    plt.plot(T, stdNonZeroAbs)
    plt.title('std non-zeros')

    plt.subplot(4, 3, 12)
    plt.plot(T, medianNonZeroAbs)
    plt.title('median non-zeros')

    plt.tight_layout()

    plt.figure()
    plt.plot(T, numpy.sum(nonZeroFilter, axis = 1))
    plt.ylim(bottom=0)
