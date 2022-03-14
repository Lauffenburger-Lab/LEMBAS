import pandas as pd
import numpy
import networkx

def getConnectionToTF(model, allTFs, affectedNodes):
    g = networkx.from_pandas_edgelist(model, 'source', 'target', create_using=networkx.DiGraph())
    allNodes = numpy.array(list(g.nodes))
    includedTFs = numpy.intersect1d(allTFs, allNodes)

    connectedToTF = numpy.isin(affectedNodes, includedTFs)
    for i in range(len(affectedNodes)):
        if numpy.isin(affectedNodes[i], allNodes):
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
        if numpy.isin(affectedNodes[i], allNodes):
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


coreSources = ['KEGG', 'SIGNOR', 'HuGeSiM']  # , , 'SignaLink3' 'InnateDB', 


uniprot = pd.read_csv('annotation/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab', sep='\t', low_memory=False)

ligandList = pd.read_csv('experiment/ligandScreen-Ligands.tsv', sep='\t', low_memory=False, index_col=0)
allLigands = numpy.unique(ligandList.columns.values)
TFList = pd.read_csv('experiment/ligandScreen-TFs.tsv', sep='\t', low_memory=False, index_col=0)
allTFs = numpy.unique(TFList.columns.values)

#Load and trim PKN
PKN = pd.read_csv('model/PKN.tsv', sep='\t', low_memory=False)
#knownMOA = numpy.logical_or(PKN['stimulation'], PKN['inhibition'])
#PKN = PKN.loc[knownMOA, :]

allModelTargets = numpy.unique(PKN['target'].values)
overlapTF = numpy.intersect1d(allModelTargets, allTFs)

coreFilter = numpy.full(PKN.shape[0], False, dtype=bool)
for i in range(len(coreFilter)):
    coreFilter[i] = len(numpy.intersect1d(PKN.iloc[i,:]['sources'].split(';'), coreSources))>0
coreModel = PKN.loc[coreFilter,:]

#Load and trim R-L network
RL = pd.read_csv('model/RL.tsv', sep='\t', low_memory=False)
RL = RL.loc[numpy.isin(RL['source'].values, allLigands),:]


#%%
#Add ligands
enhancedModel = pd.concat((RL, coreModel), axis=0)
enhancedModel = enhancedModel.drop_duplicates()
enhancedModel = trimDeadEnds(enhancedModel, allTFs, allLigands)
#enhancedModel = trimSelfConnecting(enhancedModel, allTFs, allLigands)

#remove receptor-ligand back reactions
affectedInteractions = numpy.isin(enhancedModel['target'], allLigands)
enhancedModel = enhancedModel.loc[affectedInteractions==False,:]
print('Removed Receptor -> Ligand back reactions', sum(affectedInteractions))

#Count detached ligands
ligandConnection = getConnectionToTF(enhancedModel, allTFs, allLigands)
detachedLigands = allLigands[ligandConnection == False]
print('Detached ligands', detachedLigands, len(detachedLigands))


#%%
#Remove TFs with only one interaction
# TFBefore = numpy.intersect1d(allTFs, enhancedModel['target'])

# TFIn = enhancedModel.loc[numpy.isin(enhancedModel['target'], TFBefore),:]
# weakTFs, counts = numpy.unique(TFIn['target'], return_counts=True)
# weakTFs = weakTFs[counts==1]
# print('Candidate weak TFs', weakTFs, len(weakTFs))

# allTFs = numpy.setdiff1d(TFBefore, weakTFs)
# enhancedModel = trimDeadEnds(enhancedModel, allTFs, allLigands)
# allTFs = numpy.intersect1d(TFBefore, enhancedModel['target'])
# removedTfs = numpy.setdiff1d(TFBefore, allTFs)
# print('Removing weak TFs', removedTfs, len(removedTfs))

enhancedModel = trimDeadEnds(enhancedModel, allTFs, allLigands)

enhancedModel.to_csv('ligandScreen-Model.tsv', sep='\t', index=False)

#Build annotation file
compoundMap = pd.read_csv('annotation/ligandMap.tsv', sep='\t', low_memory=False)
uniprot = uniprot.loc[:, ['Entry', 'Gene names  (primary )']].values
compoundMap = compoundMap.loc[:, ['Code', 'Name']].values
annotation = numpy.concatenate((uniprot, compoundMap))
nodeNames = numpy.union1d(enhancedModel['source'], enhancedModel['target'])
annotation = annotation[numpy.isin(annotation[:,0], nodeNames),:]
# missingNodes = nodeNames[numpy.isin(nodeNames, uniprot[:,0])==False]
# print(missingNodes)
annotation = pd.DataFrame(annotation, columns=['code', 'name'])
annotation = annotation.drop_duplicates(subset='code', keep='first')
annotation['TF'] = numpy.isin(annotation['code'], allTFs)
annotation['ligand'] = numpy.isin(annotation['code'], allLigands)
annotation.to_csv('ligandScreen-Annotation.tsv', sep='\t', index=False)


#Print manual anotation
dependsOnManual = numpy.full((enhancedModel.shape[0]), False)
for i in range(len(dependsOnManual)):
    curSources = enhancedModel.iloc[i,:]['sources'].split(';')
    if bool(numpy.isin('HuGeSiM', curSources)) == True:
        if numpy.any(numpy.isin(['KEGG', 'SignaLink3', 'InnateDB', 'SIGNOR'], curSources))==False:
            dependsOnManual[i] = True
            
enhancedModel.loc[dependsOnManual,:].to_csv('ligandScreen-Manual.tsv', sep='\t', index=False)        


print('Interactions:', enhancedModel.shape[0], 'Nodes', annotation.shape[0], 'Ligands', sum(annotation['ligand']), 'TFs', sum(annotation['TF']))
