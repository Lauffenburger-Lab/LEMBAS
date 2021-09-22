import pandas as pd
import numpy
import networkx

def getConnectionToTF(model, allTFs, affectedNodes):
    g = networkx.from_pandas_edgelist(model, 'source', 'target', create_using=networkx.DiGraph())
    allNodes = numpy.array(list(g.nodes))
    includedTFs = numpy.intersect1d(allTFs, allNodes)

    connectedToTF = numpy.isin(affectedNodes, includedTFs)
    for i in range(len(affectedNodes)):
        if affectedNodes[i] in allNodes:
            for tf in includedTFs:
                if networkx.algorithms.shortest_paths.generic.has_path(g, affectedNodes[i], tf):
                    connectedToTF[i] = True
                    break

    return connectedToTF

def getConnectionToLigand(model, allLigands, affectedNodes):
    g = networkx.from_pandas_edgelist(model, 'target', 'source', create_using=networkx.DiGraph())
    allNodes = numpy.array(list(g.nodes))
    includedLigands = numpy.intersect1d(allLigands, allNodes)
    connectedToLigand = numpy.isin(affectedNodes, allLigands)
    for i in range(len(affectedNodes)):
        if affectedNodes[i] in allNodes:
            for ligand in includedLigands:
                if networkx.algorithms.shortest_paths.generic.has_path(g, affectedNodes[i], ligand):
                    connectedToLigand[i] = True
                    break
    return connectedToLigand

def trimDeadEnds(model, allTFs, allLigands):
    allNodes = numpy.union1d(model['source'].values, model['target'].values)

    connectedToTF = getConnectionToTF(model, allTFs, allNodes)
    connectedToLigand = getConnectionToLigand(model, allLigands, allNodes)
    connectedToBoth = numpy.logical_and(connectedToTF, connectedToLigand)
    disconectedNodes = allNodes[connectedToBoth == False]

    dissconectedSources = numpy.isin(model.source.values, disconectedNodes)
    disconnectedTargets = numpy.isin(model.target.values, disconectedNodes)
    disconnectedEdges = numpy.logical_or(dissconectedSources, disconnectedTargets)
    model = model.loc[disconnectedEdges==False,:]
    return model

def trimSelfConnecting(model, allTFs, allLigands):
    lastSize = numpy.inf
    curentSize = model.shape[0]
    while curentSize<lastSize:
        lastSize = curentSize
        sources, counts = numpy.unique(model['source'], return_counts=True)
        onlyOneInput = sources[counts == 1]
        targets, counts = numpy.unique(model['target'], return_counts=True)
        onlyOneOutput = targets[counts == 1]
        overlap = numpy.intersect1d(onlyOneInput, onlyOneOutput)
        #Exclude ligands and TFs
        overlap = numpy.setdiff1d(overlap, allLigands)
        overlap = numpy.setdiff1d(overlap, allTFs)
        selfLoop = numpy.full(len(overlap), False, dtype=bool)
        for i in range(len(selfLoop)):
            curSource = model.loc[numpy.isin(model['target'], overlap[i]), 'source'].values[0]
            curTarget = model.loc[numpy.isin(model['source'], overlap[i]), 'target'].values[0]
            selfLoop[i] = curSource == curTarget

        affectedProteins = overlap[selfLoop]

        affectedInteractions = numpy.logical_or(numpy.isin(model['source'], affectedProteins), numpy.isin(model['target'], affectedProteins))
        model = model.loc[affectedInteractions==False,:]
        curentSize = model.shape[0]

    return model

def subsetOnSource(df, coreSources):
    dfFilter = numpy.full(df.shape[0], False, dtype=bool)
    for i in range(len(dfFilter)):
        dfFilter[i] = len(numpy.intersect1d(df.iloc[i,:]['sources'].split(';'), coreSources))>0
    df = df.loc[dfFilter,:].copy()
    return df


coreSources = ['KEGG', 'InnateDB']


#Load and trim PKN
PKN = pd.read_csv('model/PKN.tsv', sep='\t', low_memory=False)
PKNFull = PKN.copy()
PKN = subsetOnSource(PKN, coreSources)
#knownSign = numpy.logical_or(PKN['stimulation'].values, PKN['inhibition'].values)
#PKN = PKN.loc[knownSign,:]

RL = pd.read_csv('experiment/macrophage-ligandMap.tsv', sep='\t', low_memory=False)
inferedReceptors = numpy.intersect1d(RL['target'].values, PKNFull['source'].values)
allLigands = numpy.unique(RL['source'])

TFgene = pd.read_csv('experiment/macrophage-TFs.tsv', sep='\t', low_memory=False, index_col=0)
allTFs = numpy.intersect1d(TFgene.columns.values, PKN['target'].values)

#%%
#Attach detached ligands
# ligandConnection = getConnectionToTF(PKN, allTFs, inferedReceptors)
# detachedLigands = inferedReceptors[ligandConnection == False]
# print('Detached ligands', detachedLigands, len(detachedLigands))

# nodesInCore = numpy.unique(PKN['source'].values)

# g = networkx.from_pandas_edgelist(PKNFull, 'source', 'target', create_using=networkx.DiGraph())
# allNodes = numpy.array(list(g.nodes))

# allPaths = []
# for i in range(len(detachedLigands)):
#     if detachedLigands[i] in allNodes:
#         path = []
#         pathLength = []
#         for coreNode in nodesInCore:
#             if coreNode != detachedLigands[i]:
#                 if networkx.has_path(g, detachedLigands[i], coreNode):
#                     curPath = numpy.array(networkx.shortest_path(g, detachedLigands[i], coreNode))
#                     path.append(curPath)
#                     pathLength.append(len(curPath))
#         pathLength = numpy.array(pathLength)
#         shortestPathLength = min(pathLength)

#         path = numpy.array(path, dtype=object)
#         path = path[pathLength == shortestPathLength]

#         for j in range(len(path)):
#             curPath = path[j]
#             for k in range(shortestPathLength-1):
#                 allPaths.append(curPath[k] + '_' + curPath[k+1])
#     else:
#         print('Ligand not in extended network', detachedLigands[i])

# includedInteractions = numpy.unique(allPaths)
# fullOmniIds = PKNFull['source'] + '_' + PKNFull['target']
# subset = PKNFull.loc[numpy.isin(fullOmniIds, includedInteractions),:]

# PKN = pd.concat((PKN, subset), axis=0)
# PKN = PKN.drop_duplicates()


#%%

PKN = pd.concat((PKN, RL), axis=0)

PKN = trimDeadEnds(PKN, allTFs, allLigands)
PKN = trimSelfConnecting(PKN, allTFs, allLigands)

ligandConnection = getConnectionToTF(PKN, allTFs, allLigands)
detachedLigands = allLigands[ligandConnection == False]
print('Detached ligands', detachedLigands, len(detachedLigands))


allLigands = numpy.intersect1d(allLigands, PKN['source'])
allTFs = numpy.intersect1d(allTFs, PKN['target'])


PKN.to_csv('macrophage-Model.tsv', sep='\t', index=False)


#Build annotation file
uniprot = pd.read_csv('annotation/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab', sep='\t', low_memory=False)
uniprot = uniprot.loc[:, ['Entry', 'Gene names  (primary )']].values
nodeNames = numpy.union1d(PKN['source'], PKN['target'])
annotation = uniprot[numpy.isin(uniprot[:,0], nodeNames),:]
missingNodes = nodeNames[numpy.isin(nodeNames, uniprot[:,0])==False]
missingNodeAnotation = numpy.array([missingNodes, missingNodes]).T
annotation = numpy.concatenate((annotation, missingNodeAnotation))
annotation = pd.DataFrame(annotation, columns=['code', 'name'])
annotation = annotation.drop_duplicates(subset='code', keep='first')
annotation['name'] = annotation['name'].str.replace('; ','/')
annotation['TF'] = numpy.isin(annotation['code'], allTFs)
annotation['ligand'] = numpy.isin(annotation['code'], allLigands)
annotation.to_csv('macrophage-Annotation.tsv', sep='\t', index=False)

