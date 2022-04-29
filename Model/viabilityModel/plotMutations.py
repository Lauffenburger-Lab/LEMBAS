import pandas as pd
import seaborn as sns
import numpy



mutationMap = pd.read_csv('mutations.tsv', sep='\t', low_memory=False, index_col=0)

simplifiedIndex = mutationMap.index.copy()
simplifiedIndex = [x.replace('MUT_MutAA[', '').replace(']','') for x in simplifiedIndex]
simplifiedIndex = [x.split('-')[1] + '-' + x.split('-')[0] for x in simplifiedIndex]

mutationMap.index = simplifiedIndex

blank = numpy.sum(mutationMap.values, axis=1)==0
mutationMap = mutationMap.loc[blank==False, :]

cg = sns.clustermap(mutationMap, cmap='gray_r', figsize=(6,5))
cg.ax_row_dendrogram.set_visible(False)
cg.ax_col_dendrogram.set_visible(False)