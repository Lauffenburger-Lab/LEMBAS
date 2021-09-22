import numpy
import pandas
import seaborn as sns
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def independentActivation(A, B, x):
    #states = 2
    k = numpy.array([0.2, 0.1, 0.2])
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

def independentDeactivation(A, B, x):
    #states = 2
    k = numpy.array([0.2, 0.1, 0.2])
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

def cooperativeActivation(A, B, x):
    #states = 2
    k = numpy.array([0.5, 0.1])
    def r1(A, B, k, x):
        return k[0] * A * B * x[1]
    def r2(A, B, k, x):
        return k[1] * x[0]

    dx = numpy.zeros(2)
    dx[0] = r1(A, B, k, x) - r2(A, B, k, x)
    dx[1] = r2(A, B, k, x) - r1(A, B, k, x)
    return dx

def competitiveInhibition(A, B, x):
    #states = 3
    k = numpy.array([0.5, 0.1, 2, 0.1])
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

def nonCompetitiveInhibtion(A, B, x):
    #states = 3
    k = numpy.array([0.5, 0.1, 2, 0.1, 0.1, 0.1]) #k[5]<k[0], k[4] interesting but not important for this application
    def r1(A, B, k, x):
        return k[0] * A * x[1]
    def r2(A, B, k, x):
        return k[1] * x[0]
    def r3(A, B, k, x):
        return k[2] * B * x[1]
    def r4(A, B, k, x):
        return k[3] * x[2]
    def r5(A, B, k, x):
        return k[4] * B * x[0]
    def r6(A, B, k, x):
        return k[5] * A * x[2]

    dx = numpy.zeros(3)
    dx[0] = r1(A, B, k, x) + r6(A, B, k, x) - r2(A, B, k, x) - r5(A, B, k, x)
    dx[1] = r2(A, B, k, x) + r4(A, B, k, x) - r1(A, B, k, x) - r3(A, B, k, x)
    dx[2] = r3(A, B, k, x) + r5(A, B, k, x) - r4(A, B, k, x) - r6(A, B, k, x)
    return dx

def advancedCooperativeActivation(A, B, x):
    #states = 2
    k = numpy.array([0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1])
    def r1(A, B, k, x):
        return k[0] * A * x[1]
    def r2(A, B, k, x):
        return k[1] * x[2]
    def r3(A, B, k, x):
        return k[2] * B * x[2]
    def r4(A, B, k, x):
        return k[3] * x[0]
    def r5(A, B, k, x):
        return k[4] * B * x[1]
    def r6(A, B, k, x):
        return k[5] * x[3]
    def r7(A, B, k, x):
        return k[6] * A * x[3]
    def r8(A, B, k, x):
        return k[7] * x[0]

    dx = numpy.zeros(4)
    dx[0] = r3(A, B, k, x) - r4(A, B, k, x) + r7(A, B, k, x) - r8(A, B, k, x)
    dx[1] = r2(A, B, k, x) - r1(A, B, k, x) + r6(A, B, k, x) - r5(A, B, k, x)
    dx[2] = r1(A, B, k, x) - r2(A, B, k, x) + r4(A, B, k, x) - r3(A, B, k, x)
    dx[3] = r5(A, B, k, x) - r6(A, B, k, x) + r8(A, B, k, x) - r7(A, B, k, x)
    return dx



def displayResult(df, normalize=False):
    df.columns = df.columns.values.round(3)
    df.index = df.index.values.round(3)
    if normalize:
        df = df/numpy.max(df.values)
    sns.heatmap(df, cmap='gray', vmin=0, vmax=1)

def displaySimulation(diffFun, n, values):
    tspan = [0, 100]
    x0 = 1/n * numpy.ones(n) #+ 0.0001 * numpy.random.rand(n)

    fun = lambda t, y: diffFun(values[0], values[1], y)
    res = solve_ivp(fun, tspan, x0)
    t = res.t
    y = res.y[0,:]
    plt.plot(t, y)
    plt.ylim([0, 1])
    return res



def sampleAllSteadyStates(diffFun, n, resolution):
    epsilon = 0
    values = numpy.linspace(epsilon, 1-epsilon, resolution)
    results = numpy.zeros((resolution, resolution))
    tspan = [0, 1000]
    x0 = 1/n * numpy.ones(n)

    for i in range(resolution):
        for j in range(resolution):
            fun = lambda t, y: diffFun(values[i], values[j], y)
            res = solve_ivp(fun, tspan, x0)
            results[i,j] = res.y[0,-1]

    results[results<0] = 0

    df = pandas.DataFrame(results, index=values.copy(), columns=values.copy())
    return df

folder = 'data/evaluatedODE'
resolution = 50

plt.rcParams["figure.figsize"] = (3,3)

df = sampleAllSteadyStates(independentActivation, 2, resolution)
df.to_csv(folder+'/independentActivation.tsv', sep='\t')

df = sampleAllSteadyStates(independentDeactivation, 2, resolution)
df.to_csv(folder+'/independentDeActivation.tsv', sep='\t')

df = sampleAllSteadyStates(cooperativeActivation, 2, resolution)
df.to_csv(folder+'/cooperativeActivation.tsv', sep='\t')
#displayResult(df)


df = sampleAllSteadyStates(advancedCooperativeActivation, 4, resolution)
df.to_csv(folder+'/advancedCooperativeActivation.tsv', sep='\t')
displayResult(df)

df = sampleAllSteadyStates(competitiveInhibition, 3, resolution)
df.to_csv(folder+'/competitiveInhibition.tsv', sep='\t')

df = sampleAllSteadyStates(nonCompetitiveInhibtion, 3, resolution)
df.to_csv(folder+'/nonCompetitiveInhibtion.tsv', sep='\t')

#displayResult(df)
#res = displaySimulation(advancedCooperativeActivation, 2, [0.1, 0.8])
