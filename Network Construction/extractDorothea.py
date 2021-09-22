import pandas as pd
import numpy

trustedSource = ['A', 'B']
human = 9606
omnipath = pd.read_csv('annotation/omnipath_webservice_interactions__recent.tsv', sep='\t', low_memory=False)
humanFilter = omnipath['ncbi_tax_id_target'] == human
omnipath = omnipath.loc[humanFilter, :]


#Only in dorothea
omnipathFilter = omnipath['dorothea'].values
omnipath = omnipath.loc[omnipathFilter, :]

#only curated
omnipathFilter = omnipath['dorothea_curated'].values == 'True'
omnipath = omnipath.loc[omnipathFilter, :]



#Subset to relevant info
relevantInformation = ['source', 'target', 'consensus_direction', 'consensus_stimulation', 'consensus_inhibition',  'dorothea_level', 'sources', 'references']
omnipath = omnipath[relevantInformation]
omnipath = omnipath.rename(columns={'consensus_direction': 'direction', 'consensus_stimulation': 'stimulation', 'consensus_inhibition': 'inhibition', 'dorothea_level' : 'confidence'})
omnipath[['references']] = omnipath[['references']].astype(str)

#Resolve paradoxes, interaction can not be both stimulation and inhibition
paradox = numpy.logical_and(omnipath['stimulation'], omnipath['inhibition'])
omnipath = omnipath.loc[paradox==False, :]

#Keep only highest confidence identification
pknFilter = numpy.full(omnipath.shape[0], False, dtype=bool)
confidenceValues = numpy.unique(omnipath.loc[:, 'confidence'])
for src in trustedSource:
    curValues = confidenceValues[[src in x for x in confidenceValues]]
    confidenceFilter = numpy.isin(omnipath['confidence'], curValues)
    omnipath.loc[confidenceFilter, 'confidence'] = src
    pknFilter = numpy.logical_or(pknFilter, confidenceFilter)
omnipath =  omnipath.loc[pknFilter, :]

omnipath.to_csv('model/regulon.tsv', sep='\t', index=False)


