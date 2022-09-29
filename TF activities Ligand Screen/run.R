library("dorothea")
library("DESeq2")
library("limma")

setwd("C:/work/publications/technologyPaper/Artificial-Signaling-Network/TF activities Ligand Screen")
##
countData = read.table('filtered/counts.tsv', sep ='\t', stringsAsFactors = FALSE, header=TRUE, row.names=1)
metaData = read.table('filtered/metadata.tsv', sep ='\t', stringsAsFactors = FALSE, header=TRUE)

colData = data.frame(metaData$library.name)
colData$condition = factor(metaData$lps.stimulation)
colData$donor = factor(metaData$donor)
colData$plate = factor(metaData$plate)
#colData$class = factor(metaData$class)
colData$ligand = factor(metaData$ligand)
rownames(colData) = metaData$library.name
colData = colData[colnames(countData), ]

mm = model.matrix(~condition + ligand, colData)

ddsTxi <- DESeqDataSetFromMatrix(countData=round(countData), colData=colData, design=mm)
ddsTxi = estimateSizeFactors(ddsTxi)

inputCounts = counts(ddsTxi, normalized=TRUE)
write.table(inputCounts, file = 'results/counts.tsv', sep = "\t", row.names = TRUE, quote = FALSE)




#rld <- vst(ddsTxi, blind=TRUE)
#plotPCA(rld, intgroup="condition", ntop = 1000)


#a
rld <- vst(ddsTxi, blind=TRUE)
#plotPCA(rld, intgroup="condition", ntop = 5000)
#plotPCA(rld, intgroup="donor", ntop = 5000)
#plotPCA(rld, intgroup="plate", ntop = 5000)
#title('Stimulation')


#plotPCA(rld, intgroup="plate", ntop = 5000)

# 
# dds = DESeq(ddsTxi)
# res = results(ddsTxi)
# resOrdered <- res[order(res$padj),]
# resOrdered$padj[is.na(resOrdered$padj)] = 1
# sig = resOrdered[resOrdered$padj<0.01,]
#ddsTxi <- normTransform(ddsTxi)




#

meanFactor = mean(colSums(inputCounts))/1e6
#input = inputCounts
input = assay(rld)

#input = 1e6 * t(t(inputCounts)/colSums(inputCounts))

method = 'center'
LogTransform = TRUE
minNrOfGenes = 10
medianTFCutof = 0
regressOutBatch = TRUE
#meanCountCutOf = 1 * meanFactor



#filter = rowMeans(inputCounts)>meanCountCutOf
#input = input[filter,]


# if (LogTransform){
#   input = log(input+1)
# }

if(method == 'center'){
  data.standardized = input-rowMeans(input)
  #data.standardized = t(apply(input, 1, scale))
  #colnames(data.standardized) = colnames(input)
}


 if (regressOutBatch){
#   #correct for plate and donor effects
#   #data.standardized = limma::removeBatchEffect(data.standardized, batch=colData$plate, batch2=colData$donor, design=mm)
   data.standardized = limma::removeBatchEffect(data.standardized, batch=colData$donor, design=mm)
#   #data.standardized = ComBat(dat=input, batch=colData$plate, mod=mm)
 }


# Load TF regulon genesets

# dorotheaData = read.table('data/annotation/dorotheaSelection.tsv', sep = "\t", header=TRUE) 


dorotheaData = read.table('data/annotation/dorothea.tsv', sep = "\t", header=TRUE)
#from file for reproducibility, for latest regulon: data(dorothea_hs, package = "dorothea")
#write.table(dorothea_hs, file = 'data/annotation/dorothea.tsv', sep = "\t", row.names = FALSE, quote = FALSE)
print(paste('All confidence levels:', paste(sort(unique(dorotheaData$confidence)), collapse = ', ')))

# Keep TFs with high confidence
confidenceFilter = is.element(dorotheaData$confidence, c('A', 'B'))
dorotheaData = dorotheaData[confidenceFilter,]


# Remove TFs that are not expressed
expressedGenes = rowMedians(inputCounts)>medianTFCutof
expressedGeneNames = rownames(inputCounts)[expressedGenes]
expressedFilter = dorotheaData[,'tf'] %in% expressedGeneNames
dorotheaData  = dorotheaData[expressedFilter,]

#Remove Interactions without data
#includedFilter = dorotheaData$target %in% rownames(data.standardized)
#dorotheaData = dorotheaData[includedFilter,]

print(paste('Selected confidence levels:', paste(sort(unique(dorotheaData$confidence)), collapse = ', ')))
print(paste('Number of TFs ', length(unique(dorotheaData$tf))))
print(paste('Number of Targets ', length(unique(dorotheaData$target))))


settings = list(verbose = TRUE, minsize = minNrOfGenes)
tf_activities <- run_viper(data.standardized, dorotheaData, options =  settings)
write.table(tf_activities, file = 'results/dorothea.tsv', sep = "\t", row.names = TRUE, quote = FALSE)



