import numpy
import pandas
import seaborn as sns
from scipy import interpolate
import matplotlib.pyplot as plt

def displayResult(df, normalize):

    df.columns = df.columns.values.round(2)
    df.index = df.index.values.round(2)
    if normalize:
        df = df/numpy.max(df.values)
    ax = sns.heatmap(df, cmap='gray', vmin=0, vmax=1, linewidths=0.1, cbar=False)
    ax.invert_yaxis()
    # ax.set_ylabel('')
    # ax.set_xlabel('')
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
    plt.gca().axis('equal')

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.xticks([0,5,10], labels=['0', '0.5', '1'])
    plt.yticks([0,5,10], labels=['0', '0.5', '1'])
    plt.gca().set_xticks(list(range(0,11)), minor=True)
    plt.gca().set_yticks(list(range(0,11)), minor=True)

    for _, spine in plt.gca().spines.items():
        spine.set_visible(True)

    #plt.gca().xaxis.grid(True, 'both', linewidth=1, color=[0,0,0])
    #plt.gca().yaxis.grid(True, 'both', linewidth=1, color=[0,0,0])
    plt.ylim([0, 10])
    plt.xlim([0, 10])



def interpolateData(resolution, df):
    epsilon=0
    df.columns = df.columns.astype('float')
    df.index = df.index.astype('float')
    X = df.index
    Y = df.columns
    points = (X, Y)
    x, y = numpy.meshgrid(X,Y)
    z = df.values

    outputGrid = numpy.linspace(epsilon, 1-epsilon, resolution)
    xR, yR = numpy.meshgrid(outputGrid, outputGrid, indexing='ij',)

    f = interpolate.interpn(points, z, (xR, yR), method='linear')
    df = pandas.DataFrame(f, index=outputGrid.copy(), columns=outputGrid.copy())
    return df

folder = 'data/evaluatedODE'


ODEfiles = ['independentActivation.tsv',
            'independentDeActivation.tsv',
            'cooperativeActivation.tsv',
            'competitiveInhibition.tsv']
            #nonCompetitiveInhibtion.tsv

for file in ODEfiles:
    plt.figure()
    df = pandas.read_csv(folder + '/' + file, sep='\t', low_memory=False, index_col=0)
    downSampled = interpolateData(10, df)
    displayResult(downSampled, True)





