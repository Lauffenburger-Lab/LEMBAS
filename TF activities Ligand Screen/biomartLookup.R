require("biomaRt")

counts = read.table('data/counts.tsv', sep ='\t', stringsAsFactors = FALSE, header=TRUE, row.names=1)
originalNames = rownames(counts)
geneNames <- gsub("\\.[0-9]*$", "", originalNames)


mart <- useMart("ENSEMBL_MART_ENSEMBL")
mart <- useDataset("hsapiens_gene_ensembl", mart)

annotLookup <- getBM(
  mart=mart,
  attributes=c("ensembl_gene_id", "gene_biotype", "external_gene_name"),
  filter="ensembl_gene_id",
  values=geneNames,
  uniqueRows=TRUE)

missingDataFilter = annotLookup$external_gene_name == ''
annotLookup$external_gene_name[missingDataFilter] = annotLookup$ensembl_gene_id[missingDataFilter]
rownames(annotLookup) = annotLookup$ensembl_gene_id

allData = annotLookup[geneNames,]
allData$Source=originalNames

missingDataFilter = is.na(allData$external_gene_name)
allData[missingDataFilter, 'external_gene_name'] = allData[missingDataFilter,'Source']



write.table(allData, file ='filtered/ensemblTable.tsv', sep = "\t", row.names = FALSE, quote = FALSE)
