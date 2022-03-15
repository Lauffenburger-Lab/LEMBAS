import numpy
import pandas
import h5py


f = h5py.File('data/Smartseq2_Multisample_human_single_2021-12-15T21-07-41.loom', 'r')
cellIds = numpy.array(f['col_attrs']['CellID'])
ensambleId = numpy.array(f['row_attrs']['ensembl_ids'])
geneName = numpy.array(f['row_attrs']['gene_names'])
countTable = numpy.array(f['layers']['estimated_counts'])


f.close()

df = pandas.DataFrame(countTable, index=ensambleId, columns=cellIds)
df.to_csv('data/counts.tsv', sep='\t', index=True)


#'col_attrs'
#'CellID'

#layers
#estimated_counts
