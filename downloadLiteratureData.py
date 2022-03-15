import urllib.request
import shutil
import ssl
import gzip

def gunzip(inFile, outFile):
    with gzip.open(inFile, 'rb') as f_in:
        with open(outFile, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)    

print('Testing download...')
url = 'https://archive.omnipathdb.org/README.txt'
urllib.request.urlretrieve(url, 'Network Construction/annotation/test.tsv')

print('Downloading OmniPath...')
url = 'https://archive.omnipathdb.org/omnipath_webservice_interactions__recent.tsv'
urllib.request.urlretrieve(url, 'Network Construction/annotation/omnipath_webservice_interactions__recent.tsv')


print('Downloading transcriptomics data...')
url = 'https://www.ebi.ac.uk/arrayexpress/files/E-GEOD-46903/E-GEOD-46903.processed.1.zip'
urllib.request.urlretrieve(url, 'TF activities/raw/E-GEOD-46903.processed.1.zip')

url = 'https://www.ebi.ac.uk/arrayexpress/files/E-GEOD-46903/E-GEOD-46903.idf.txt'
urllib.request.urlretrieve(url, 'TF activities/raw/E-GEOD-46903.idf.txt')

url = 'https://www.ebi.ac.uk/arrayexpress/files/E-GEOD-46903/E-GEOD-46903.sdrf.txt'
urllib.request.urlretrieve(url, 'TF activities/raw/E-GEOD-46903.sdrf.txt')

url = 'https://www.ebi.ac.uk/arrayexpress/files/A-MEXP-1171/A-MEXP-1171.adf.txt'
urllib.request.urlretrieve(url, 'TF activities/raw/A-MEXP-1171.adf.txt')


print('Downloading Cell line data')
url = 'https://depmap.org/portal/download/api/download?file_name=ccle%2Fccle_2019%2FCCLE_RNAseq_genes_rpkm_20180929.gct.gz&bucket=depmap-external-downloads'
ssl._create_default_https_context = ssl._create_unverified_context
urllib.request.urlretrieve(url, 'Viability data/data CCLE/CCLE_RNAseq_genes_rpkm_20180929.gct.gz')

print('Downloading Fröhlich et al 2018 data')
url = 'https://zenodo.org/record/1472794/files/supplement_code.zip?download=1'
urllib.request.urlretrieve(url, 'Viability data/processed data/supplement_code.zip')


print('Unziping data')
shutil.unpack_archive('TF activities/raw/E-GEOD-46903.processed.1.zip', 'TF activities/raw/E-GEOD-46903.processed.1')
gunzip('Viability data/data CCLE/CCLE_RNAseq_genes_rpkm_20180929.gct.gz', 'Viability data/data CCLE/CCLE_RNAseq_genes_rpkm_20180929.gct')
shutil.unpack_archive('Viability data/processed data/supplement_code.zip', 'Viability data/processed data/')

print('Moving files')
shutil.copyfile('Viability data/processed data/supplement_code/results/preprocessed.mat', 'Viability data/preprocessed.mat')
shutil.copyfile('Viability data/processed data/supplement_code/results/mutations.mat', 'Viability data/mutations.mat')


