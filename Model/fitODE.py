import numpy
from scipy import optimize
import matplotlib.pyplot as plt
import torch
import pandas
import seaborn as sns
from scipy import interpolate
import time
from scipy.stats import pearsonr
from scipy.integrate import solve_ivp


def interpolateData(df, resolution, normalize):
    epsilon=0
    df = df.copy()
    df.columns = df.columns.astype('float')
    df.index = df.index.astype('float')
    X = df.index
    Y = df.columns
    points = (X, Y)
    x, y = numpy.meshgrid(X,Y)
    z = df.values

    outputGrid = numpy.linspace(epsilon, 1-epsilon, resolution)
    xR, yR = numpy.meshgrid(outputGrid, outputGrid, indexing='ij')

    f = interpolate.interpn(points, z, (xR, yR), method='linear')
    if normalize:
        f = f/numpy.max(f)

    torchX = numpy.array((xR.flatten(), yR.flatten())).T
    torchY = numpy.array(f.flatten()).reshape(-1,1)
    return torchX, torchY


def independentActivation(A, B, x, k):
    #states = 2
    def r1(A, B, k, x):
        return k[0] * A * x[1]
    def r2(A, B, k, x):
        return k[1] * x[0]
    def r3(A, B, k, x):
        return k[2] * B * x[1]

    dx = numpy.zeros(2)
    dx[0] = r1(A, B, k, x) + r3(A, B, k, x) - r2(A, B, k, x)
    dx[1] = r2(A, B, k, x) - r1(A, B, k, x) - r3(A, B, k, x)
    return dx

def independentDeactivation(A, B, x, k):
    #states = 2
    def r1(A, B, k, x):
        return k[0] * A * x[1]
    def r2(A, B, k, x):
        return k[1] * x[0]
    def r3(A, B, k, x):
        return k[2] * B * x[1]

    dx = numpy.zeros(2)
    dx[0] = r1(A, B, k, x) - r3(A, B, k, x) - r2(A, B, k, x)
    dx[1] = r2(A, B, k, x) + r3(A, B, k, x) - r1(A, B, k, x)
    return dx

def cooperativeActivation(A, B, x, k):
    #states = 2
    def r1(A, B, k, x):
        return k[0] * A * B * x[1]
    def r2(A, B, k, x):
        return k[1] * x[0]

    dx = numpy.zeros(2)
    dx[0] = r1(A, B, k, x) - r2(A, B, k, x)
    dx[1] = r2(A, B, k, x) - r1(A, B, k, x)
    return dx

def competitiveInhibition(A, B, x, k):
    #states = 3
    def r1(A, B, k, x):
        return k[0] * A * x[1]
    def r2(A, B, k, x):
        return k[1] * x[0]
    def r3(A, B, k, x):
        return k[2] * B * x[1]
    def r4(A, B, k, x):
        return k[3] * x[2]

    dx = numpy.zeros(3)
    dx[0] = r1(A, B, k, x) - r2(A, B, k, x)
    dx[1] = r2(A, B, k, x) + r4(A, B, k, x) - r1(A, B, k, x) - r3(A, B, k, x)
    dx[2] = r3(A, B, k, x) - r4(A, B, k, x)
    return dx

def sampleSteadyStates(diffFun, n, X, k):
    results = numpy.zeros((len(X)))
    tspan = [0, 1000]
    x0 = 1/n * numpy.ones(n)

    for i in range(len(results)):
        fun = lambda t, y: diffFun(X[i, 0], X[i, 1], y, k)
        res = solve_ivp(fun, tspan, x0)
        results[i] = res.y[0,-1]

    results[results<0] = 0  #constrain to [0 - 1]
    return results

def interfaceIndependentActivation(X, a, b, c):
    k = numpy.array([a, b, c])
    Y = sampleSteadyStates(independentActivation, 2, X.T, k)
    return Y.flatten()

def interfaceIndependentDeactivation(X, a, b, c):
    k = numpy.array([a, b, c])
    Y = sampleSteadyStates(independentDeactivation, 2, X.T, k)
    return Y.flatten()

def interfaceCooperativeActivation(X, a, b):
    k = numpy.array([a, b])
    Y = sampleSteadyStates(cooperativeActivation, 2, X.T, k)
    return Y.flatten()

def interfaceCompetitiveInhibition(X, a, b, c, d):
    k = numpy.array([a, b, c, d])
    Y = sampleSteadyStates(competitiveInhibition, 3, X.T, k)
    return Y.flatten()



folder = 'data/evaluatedODE'
ODEfiles = ['independentActivation.tsv',
            'independentDeActivation.tsv',
            'cooperativeActivation.tsv',
            'competitiveInhibition.tsv']
resultFolder = 'figures/SI Figure 1/'

numberOfParameters = [3, 3, 2, 4]

ODENames = ['A', 'I', 'CA', 'CI']

#k = numpy.array([0.2, 0.1, 0.2])
#k = numpy.array([0.2, 0.1, 0.2])
#k = numpy.array([0.5, 0.1])
#k = numpy.array([0.5, 0.1, 2, 0.1])

normalizeData = False
replicates = 3
trainResolution = 7
testResolution = 20


#%%
allData = pandas.DataFrame(columns=['Data', 'Function', 'Rep', 'Type', 'Value'])

start = time.time()
i = 0
for i in range(i, len(ODEfiles)):
    print('Data', ODEfiles[i])
    df = pandas.read_csv(folder + '/' + ODEfiles[i], sep='\t', low_memory=False, index_col=0)
    X, Y = interpolateData(df, trainResolution, normalizeData)
    Xtest, Ytest = interpolateData(df, testResolution, normalizeData)

    for j in range(len(ODEfiles)):
        if j == 0:
            curFunction = interfaceIndependentActivation
        elif j==1:
            curFunction = interfaceIndependentDeactivation
        elif j==2:
            curFunction = interfaceCooperativeActivation
        elif j==3:
            curFunction = interfaceCompetitiveInhibition
        print('Function', ODENames[j])
            
        for l in range(replicates):
            print('Replicate', l)
            p0=numpy.random.rand(numberOfParameters[j])
            param = optimize.curve_fit(curFunction, xdata = X.T, ydata = Y.flatten(), p0=p0, bounds=(0, 1))[0]
            print(param)
            YtestHat = curFunction(Xtest.T, *list(param))
            r, p = pearsonr(YtestHat.flatten(), Ytest.flatten())    
            print('', r)           
            allData = allData.append({'Data': ODEfiles[i], 'Function': ODENames[j], 'Rep': l, 'Type': 'r', 'Value':r}, ignore_index=True)
            for v in range(len(param)):
                allData = allData.append({'Data': ODEfiles[i], 'Function': ODENames[j], 'Rep': l, 'Type': 'Param ' + str(v+1), 'Value':param[v]}, ignore_index=True)

print('Time:', time.time()-start)

allData.to_csv(resultFolder + 'ODEdataVsFunction.tsv', sep='\t', index=False)

#%%
allData = pandas.read_csv(resultFolder + 'ODEdataVsFunction.tsv', sep='\t')

resultData = allData.loc[allData['Type']=='r', :].copy()
results = numpy.zeros((len(ODEfiles), len(ODEfiles),  replicates))
for i in range(len(ODEfiles)):
    fileFilter = (ODEfiles[i] == resultData['Data'].values)
    for j in range(len(ODENames)):
        functionFilter = (ODENames[j] == resultData['Function'].values)
        curFilter = numpy.logical_and(fileFilter, functionFilter)
        results[i, j, :] = resultData.loc[curFilter,:]['Value'].values



meanResults = numpy.mean(results, axis =2)
stdResults = numpy.std(results, axis =2)



df = pandas.DataFrame(meanResults, index=ODENames, columns = ODENames)

plt.rcParams["figure.figsize"] = (4,3)
plt.figure()
h = sns.heatmap(df, annot=True, fmt="2.3f", cmap='gray', vmin=0, vmax=1, square=True)
plt.savefig(resultFolder + 'B.svg')
df.to_csv(resultFolder + 'B.tsv', sep='\t')

