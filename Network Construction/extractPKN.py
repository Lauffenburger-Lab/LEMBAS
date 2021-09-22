import pandas as pd
import numpy
def contains(haystack, needles):
    result = numpy.full(len(haystack), False, dtype=bool)
    for curNeedle in needles:
        result = numpy.logical_or(result, [curNeedle in x for x in haystack])
    return result

def mergeContent(df):
    A = df.iloc[0,:]
    B = df.iloc[1,:]
    result = A.copy()
    result['stimulation'] = numpy.logical_or(A['stimulation'], B['stimulation'])
    result['inhibition'] = numpy.logical_or(A['inhibition'], B['inhibition'])
    mergedSources = numpy.unique(A['sources'].split(';') + B['sources'].split(';'))
    result['sources']  = ';'.join(mergedSources)
    mergedReferences = numpy.unique(A['references'].split(';') + B['references'].split(';'))
    result['references']  = ';'.join(mergedReferences)
    return result


human = 9606
trustedSource = numpy.array(['KEGG',
             'Macrophage',
             'InnateDB',
             'NetPath',
             'SignaLink3',
             'SIGNOR',
             'HPMR',
             'HuGeSiM' #manual curation will be marked with this id
             ])

#trustedReferences = numpy.array(['SIGNOR:31160049', 'SIGNOR:17145764'])

omnipath = pd.read_csv('annotation/omnipath_webservice_interactions__recent.tsv', sep='\t', low_memory=False)
humanFilter = omnipath['ncbi_tax_id_target'] == human
omnipath = omnipath.loc[humanFilter, :]

#Only in omnipath
omnipathFilter = omnipath['omnipath'].values
omnipath =  omnipath.loc[omnipathFilter, :]

#Do not include ligand-receptor interactions here
#LRFilter = omnipath['ligrecextra'].values == False
#omnipath =  omnipath.loc[LRFilter, :]

#Subset to relevant info
relevantInformation = ['source', 'target', 'consensus_direction',  'consensus_stimulation', 'consensus_inhibition', 'sources', 'references']
omnipath = omnipath[relevantInformation]
omnipath = omnipath.rename(columns={'consensus_direction': 'direction', 'consensus_stimulation': 'stimulation', 'consensus_inhibition': 'inhibition'})
omnipath[['references']] = omnipath[['references']].astype(str)

#Remove interactions without reference
referenceFilter = omnipath['references']=='nan'
omnipath = omnipath.loc[referenceFilter==False, :]

#Add interactions
currationAdd = pd.read_csv('curation/PKN/add.tsv', sep='\t', low_memory=False)
for i in range(currationAdd.shape[0]):
    curSource = currationAdd.iloc[i,:]['source']
    curTarget = currationAdd.iloc[i,:]['target']
    inList = numpy.logical_and(numpy.isin(omnipath['source'], curSource), numpy.isin(omnipath['target'], curTarget))
    if sum(inList) == 0:  #add new
        omnipath = omnipath.append(currationAdd.iloc[i,:])
    else:         #add reference
        #Note, does not check for consistency of sign etc
        omnipath.loc[inList,'sources'] = omnipath.loc[inList,'sources'] + ';' + currationAdd.iloc[i,:]['sources']
        omnipath.loc[inList,'references'] = omnipath.loc[inList,'references'] + ';' + currationAdd.iloc[i,:]['references']

#Remove interactions
currationRemove = pd.read_csv('curation/PKN/remove.tsv', sep='\t', low_memory=False)
for i in range(currationRemove.shape[0]):
    curSource = currationRemove.iloc[i,:]['source']
    curTarget = currationRemove.iloc[i,:]['target']
    inList = numpy.logical_and(numpy.isin(omnipath['source'], curSource), numpy.isin(omnipath['target'], curTarget))
    if sum(inList)>0:
        omnipath = omnipath.loc[inList==False,:]
    else:
        print('No match for remove', currationRemove.iloc[i,:])

#Edit interactions
currationEdit = pd.read_csv('curation/PKN/edit.tsv', sep='\t', low_memory=False)
for i in range(currationEdit.shape[0]):
    curSource = currationEdit.iloc[i,:]['source']
    curTarget = currationEdit.iloc[i,:]['target']
    inList = numpy.logical_and(numpy.isin(omnipath['source'], curSource), numpy.isin(omnipath['target'], curTarget))
    #Note, should add a reference for the change
    if sum(inList)>0:
        if currationEdit.iloc[i,:]['action'] == 'set_stimulation':
            omnipath.loc[inList,'stimulation'] = currationEdit.iloc[i,:]['value']
        elif currationEdit.iloc[i,:]['action'] == 'set_inhibition':
            omnipath.loc[inList,'inhibition'] = currationEdit.iloc[i,:]['value']
        elif currationEdit.iloc[i,:]['action'] == 'reverse_direction':
            omnipath.loc[inList,'source'] = currationEdit.iloc[i,:]['target']
            omnipath.loc[inList,'target'] = currationEdit.iloc[i,:]['source']
        elif currationEdit.iloc[i,:]['action'] == 'set_direction':
            omnipath.loc[inList,'direction'] = currationEdit.iloc[i,:]['value']
    else:
        print('No match for edit', currationEdit.iloc[i,:])


#Remove interactions with same source and target
sameSourceAndTargetFilter = omnipath['source'] == omnipath['target']
print('Removed interactions with same source and target', sum(sameSourceAndTargetFilter))
omnipath =  omnipath.loc[sameSourceAndTargetFilter==False, :]


#Duplicate reversible
reversibleFilter = omnipath['direction'] == 0
revOmni = omnipath.loc[reversibleFilter,:].copy()
revOmni = revOmni.rename(columns={'source': 'target', 'target': 'source'})
omnipath = pd.concat([omnipath.copy(), revOmni])
omnipath['direction'] = 1

#Merge Duplicates
interactionIds = omnipath['source'] + omnipath['target']
uniqueIds, counts = numpy.unique(interactionIds, return_counts=True)
duplicates = uniqueIds[counts>1]
for i in range(len(duplicates)):
    affected = numpy.isin(interactionIds, duplicates[i])
    data = mergeContent(omnipath.loc[affected,:])
    curIndex = numpy.argwhere(affected).flatten()
    for k in curIndex:
        omnipath.iloc[k,:] = data

omnipath = omnipath.drop_duplicates()

#Ensure integers
omnipath[['direction', 'stimulation', 'inhibition']] = omnipath[['direction', 'stimulation', 'inhibition']].astype(int)



#Store full network
omnipath.to_csv('model/pknFull.tsv', sep='\t', index=False)

#Keep only trusted sources or references
# pknFilter = numpy.full(omnipath.shape[0], False, dtype=bool)
# for i in range(len(pknFilter)):
#     S = len(numpy.intersect1d(omnipath.iloc[i,:]['sources'].split(';'), trustedSource))>0
#     #R = len(numpy.intersect1d(omnipath.iloc[i,:]['references'].split(';'), trustedReferences))>0
#     pknFilter[i] = S
#     #pknFilter[i] = numpy.logical_or(S, R)

# omnipath =  omnipath.loc[pknFilter, :]

#Keep only trusted sources
pknFilter = numpy.full(omnipath.shape[0], False, dtype=bool)
confidenceValues = numpy.unique(omnipath.loc[:, 'sources'])
for src in trustedSource:
    curValues = confidenceValues[[src in x for x in confidenceValues]]
    pknFilter = numpy.logical_or(pknFilter, numpy.isin(omnipath['sources'], curValues))
omnipath =  omnipath.loc[pknFilter, :]


#Drop interactions allready in RL net
RL = pd.read_csv('model/RL.tsv', sep='\t', low_memory=False)
interactionOmnipath = omnipath['source'] + omnipath['target']
interactionRL = RL['source'] + RL['target']
overlappingInteractions = numpy.isin(interactionOmnipath, interactionRL)
omnipath = omnipath.loc[overlappingInteractions==False, :]


#Resolve paradoxes, interaction can not be both stimulation and inhibition
paradox = numpy.logical_and(omnipath['stimulation'], omnipath['inhibition'])
omnipath.loc[paradox,['stimulation', 'inhibition']] = 0

#Subset to only uniprot proteins
uniprot = pd.read_csv('annotation/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab', sep='\t', low_memory=False)
uniprot = uniprot['Entry'].values
uniprotFilter = numpy.logical_and(numpy.isin(omnipath['source'].values, uniprot), numpy.isin(omnipath['target'].values, uniprot))
omnipath = omnipath.loc[uniprotFilter, :]

omnipath.to_csv('model/pkn.tsv', sep='\t', index=False)


