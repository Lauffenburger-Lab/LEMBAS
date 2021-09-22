inputFile = "raw/jointData.txt"
outputFile = "results/macrophage_DoRothEA.txt"
minNrOfGenes = 5

# Load requeired packages
library("dorothea")

dorotheaData = read.table('annotation/dorothea.tsv', sep = "\t", header=TRUE)
confidenceFilter = is.element(dorotheaData$confidence, c('A', 'B'))
dorotheaData = dorotheaData[confidenceFilter,]

E = read.table(inputFile, sep="\t", header=TRUE, row.names = 1)
E.standardized = E-rowMeans(E)
colnames(E.standardized) = colnames(E)

# Estimate TF activities
settings = list(verbose = TRUE, minsize = minNrOfGenes)
TF_activities = run_viper(E.standardized, dorotheaData, options =  settings)

# Save results
write.table(TF_activities, file = outputFile, quote=FALSE, sep = "\t", row.names = TRUE, col.names = NA)

