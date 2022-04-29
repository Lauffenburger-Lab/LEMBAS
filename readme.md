# LEMBAS - Large-scale knowledge-EMBedded Artificial Signaling-networks

- LEMBAS enables genome-scale simulation of intracellular signaling.

- Abstract:

  Mammalian cells adapt their functional state in response to external signals in form of ligands that bind
  receptors on the cell-surface. Mechanistically, this involves signal-processing through a complex network 
  of molecular interactions that govern transcription factor activity patterns. Computer simulations of the 
  information flow through this network could help predict cellular responses in health and disease. Here we 
  develop a recurrent neural network framework constrained by prior knowledge of the signaling network with 
  ligand-concentrations as input and transcription factor -activity as output. Applied to synthetic data, it
  predicts unseen test-data (Pearson correlation r=0.98) and the effects of gene knockouts (r=0.8). We 
  stimulate macrophages with 59 different ligands, with and without addition of lipopolysaccharide, and 
  collect transcriptomics data. The framework predicts this data under cross-validation (r=0.8) and knockout
  simulations suggest a role for RIPK1 in modulating the lipopolysaccharide response. This work demonstrates
  the feasibility of genome-scale simulations of intracellular signaling.

- Reference: -

- Pubmed ID: -

- Last update: 2022-04-29

This repository is administered by @avlant.


## Installation

### Required Software:

* **python**_**3.7.10**
  * **matplotlib_3.3.4**
  * **networkx_2.5** (only required for network reconstruction)
  * **numpy_1.20.2**
  * **pandas_1.1.3**
  * **pytorch_1.6.0**
  * **scipy_1.6.2**
  * **seaborn_0.11.0**
* **Matlab_R_2020_a** (only required for transcriptomics data assembly, network visualization and complexity tests)
* **R_4.0.3** (only required for TF activity estimates)
  * **dorothea_1.0.1**
  * **DESeq2_1.28.1**
  * **limma_3.44.3**

This is the tested environment, but program may run with other specifications.

Analysis was carried out under Windows 10 with python run under anaconda with the spyder 3.7 IDE.

### Download data
* Run downloadLiteratureData.py to download OmniPath and transcriptomics data (only required for model construction and TF activity predictions).

## Run

Run the following files to reproduce the figures. Note that some processes are stochastic so slight deviations are expected.

- **Figure 1** (<10 min)
  - **a**, Model/displayODEvalues.py (Model/calculateODEvalues.py to generate the data)
  - **b**, Model/showActivationFunction.py
  - **c**, Model/ODEsimulation.py
- **Figure 2** (<10 min)
  - **a, b, c**, (illustrations)
  - **d, e, f, g**, Model/toyNetRecurrent.py
- **Figure 3** (<10 min)
  - **a**, (illustration)
  - **b, c, d**, Model/testEigenvectorRegularization.py
- **Figure 4** (<10 min, ~2 h to generate parametrization)
  - **a**, Network Construction/drawNetwork.m
  - **b**, **c, d**, Model/synthNetComplexity.py (Model/generateSynthNet.py to generate the parameterization)
- **Figure 5** (<10 min, ~2 days to run the full scan)
  - **a**, **b** Model/synthNetDataScreenResults.py (For generating data Model/synthNetDataScreen.py executed on cluster with conditions specified by Model/synthNetDataScreenConditions.py and macrophageNet.py)
  - **c**, Model/synthNetKO.py
- **Figure 6** (to display results <10 min, to train models ~2 h for each CV fold)
  - **a** (Illustration)
  - **b** TF activities Ligand Screen/convertToASNFormat.py (TF activities Ligand Screen/run.R to generate TF data from count matrix)
  - **c**, **d** Model/ligandScreenCrossValidationResults.py (For training models Model/ligandScreenCrossValidation.py executed on a cluster after conditions specified by Model/ligandScreenExtractConditions.py
  - **e, f** ligandScreenKO.py
  - **g**  ligandScreenSensitivity.py


------

- **Figure S1** (~10 min)

  - **a**, (illustration)
  - **b**, fitODE.py
  - **c**, Model/ODEsimulation.py

- **Figure S2** (<2 min)
  - Model/testAutograd.py

- **Figure S3** (<10 min)

  - **a**, complexity tests/profileTime.m
  - **b**, complexity tests/profileTimeBackprop.m

- **Figure S4** (<10 min per test)

  - **a**, Model/toyNetRecurrent.py (change parameter testSelection to the adversarial test of choice)
  - **d, e, f, g**, Model/pipeNet.py

- **Figure S5** (<10 min)

  - **a**, (illustration)
  - **b, c, d**, Model/toyNetNFKB.py

- **Figure S6** (<10 min)

  - **a, b**, Model/testEigenvectorDerivative.py

- **Figure S7** (<10 min)

  - Model/synthNetComplexity.py 

- **Figure S8** (<10 min)

  - Model/generateSynthToy.py

- **Figure S9** (<10 min)

  - Model/synthNetComplexity.py 

- **Figure S10** (~10 min)

  - **a**, Model/testBPspeed.py
  - **b, c,** Model/synthNetTimeResults.py (generate conditions with Model/synthNetTimeConditions.py and deploy them with Model/synthNetTimeScreen.py).

- **Figure S11** (2 h) 

  - **a**, Model/synthNetNoRegularization.py
  - **b, c, d**, **e,** Model/synthNetDataScreenResults.py

- **Figure S12** (<10 min)

  - **a, b,c** Model/synthNetPredictMissingInteractions_Posthoc.py

- **Figure S13** (<10 min)

  - **a, b**, TF activities/convertToASNFormat.py

- **Figure S14** (<10 min)

  - TF activities/convertToASNFormat.py

- **Figure S15**  (<10 min, ~2 h for CV and scrambled CV)

  - **a,b,c**, Model/macrophageNetCrossValidationResults.py (For generating data Model/macrophageNetCrossValidation.py and Model/macrophageNetCrossValidationScramble.py executed on cluster with Model/runSlurmMacrophageScramble.sh and Model/runSlurmMacrophage.sh after conditions specified by Model/macrophageNetExtractConditions.py and and Model/macrophageNet.py for reference data)

- **Figure S16** (<10 min)

  - TF activities Ligand Screen/convertToASNFormat.py

- **Figure S17** (<10 min)

  - **a,b,c** Model/ligandScreenCrossValidationResults.py

- **Figure S18** (<10 min, 15 min per CV fold)

  - TF activities Ligand Screen/convertToASNFormat.py

  

------

**For generating the TF output data and ligand input data for literature data(~30 min)**

1. Extract metadata on relevant conditions, TF activities/parseCondition.m 
2. Assemble and filter transcriptomics data from files, TF activities/raw/joinData.m
3. Generate the TF activities with Dorothea, TF activities/macrophage.R
4. Convert into ASN output format, TF activities/convertToASNFormat.py
5. Generate matching ASN input file, TF activities/constructASNLigandMatrix.py

**For generating the TF output data and ligand input data for experimental data(~30 min)**

1. Extract estimated_counts from loom object, TF activities Ligand Screen/extractSmartSeq.py
2. Assemble and filter transcriptomics data from extracted counts, TF activities Ligand Screen/filterSmartSeq.py
3. Generate the TF activities with Dorothea, TF activities/runR.R
4. Convert into ASN output format, TF activities/convertToASNFormat.py
5. Generate matching ASN input file, TF activities/constructASNLigandMatrix.py

**For reconstructing the signaling models (~15 min)**

1. Extract TFs from Omnipath, Network Construction/constructASNLigandMatrix.py
2. Extract Ligand-Receptor from Omnipath, Network Construction/extractRL.py
3. Extract raw network from Omnipath, Network Construction/extractPKN.py
4. Reduce network to KEGG interactions, Network Construction/trimKeggModel.py
5. Reduce network to KEGG+InnateDB, Network Construction/trimMacrophageModel.py
5. Reduce network to KEGG+SIGNOR, Network Construction/trimLigandModel.py

## Run on your data

1. Generate TF activity output data and Ligand input data as in the example above.
2. Construct a data specific model as in the example above.
3. Train a predictive model as in the example above.
