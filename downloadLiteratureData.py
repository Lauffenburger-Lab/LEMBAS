import urllib.request
import shutil

# print('Testing download...')
# url = 'https://archive.omnipathdb.org/README.txt'
# urllib.request.urlretrieve(url, 'Network Construction/annotation/test.tsv')

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


print('Unziping data')
shutil.unpack_archive('TF activities/raw/E-GEOD-46903.processed.1.zip', 'TF activities/raw/E-GEOD-46903.processed.1')