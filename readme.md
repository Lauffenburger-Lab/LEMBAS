# Artificial Signaling Network (ASN)

- The Artificial Signaling Network enables genome-scale simulation of intracellular signaling.

- Abstract:

  Mammalian cells adapt their functional state in response to external signals in form
  of ligands that bind receptors on the cell-surface. Mechanistically, this involves
  signal-processing through a complex network of molecular interactions that govern
  transcription factor (TF) activity patterns. Computer simulations of the information
  flow through this network could help predict cellular responses in health and disease.
  Here we develop a recurrent neural network constrained by prior knowledge of the
  signaling network with ligand concentrations as input, TF activity as output and 
  signaling molecules as hidden nodes. Simulations are assumed to reach steady state,
  and we regularize the parameters to enforce this. Using synthetic data, we train
  models that generalize to unseen data and predict the effects of gene knockouts. We 
  also fit models to a small experimental data set from literature and confirm the 
  predictions using cross validation. This demonstrates the feasibility of simulating
  intracellular signaling at the genome-scale.

- Reference: -

- Pubmed ID: -

- Last update: 2021-09-14

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
* **Matlab_R_2020_a** (only required for transcriptomics data assembly and network visualization)
* **R_4.0.3** (only required for TF activity estimates)

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
  - **a**, (illustration)
  - **b**, **c**, **d** Model/synthNetDataScreenResults.py (For generating data Model/synthNetDataScreen.py executed on cluster with Model/runSlurmSynthScreen.sh after conditions specified by Model/synthNetDataScreenConditions.py and macrophageNet.py)
  - **e**, (illustration)
  - **f**, Model/synthNetKO.py

- **Figure 6** (<10 min, ~2 h each for CV and  scrambled CV)
  - **a, b, c, d** Model/macrophageNetCrossValidationResults.py (For generating data Model/macrophageNetCrossValidation.py and Model/macrophageNetCrossValidationScramble.py executed on cluster with Model/runSlurmMacrophageScramble.sh and Model/runSlurmMacrophage.sh after conditions specified by Model/macrophageNetExtractConditions.py and and Model/macrophageNet.py for reference data)
  - **e**, Model/macrophageNetKO.py

------

- **Figure S1** (~10 min)
  - **a**, (illustration)
  - **b**, Model/ODEsimulation.py
- **Figure S2** (<2 min)
  - Model/testAutograd.py
- **Figure S3** (<10 min per test)
  - **a**, Model/toyNetRecurrent.py (change parameter testSelection to the adversarial test of choice)
  - **d, e, f, g**, Model/pipeNet.py
- **Figure S4** (<10 min)
  - **a, b**, Model/testEigenvectorDerivative.py
- **Figure S5** (~20 min)
  - Model/testBPspeed.py
- **Figure S6** (2 h)
  - **a**, Model/synthNetNoRegularization.py
  - **b, c, d**, Model/synthNetDataScreenResults.py

- **Figure S7** (<10 min)
  - **a, b**, TF activities/convertToASNFormat.py

------

**For generating the TF output data and ligand input data (~30 min)**

1. Extract metadata on relevant conditions, TF activities/parseCondition.m 
2. Assemble and filter transcriptomics data from files, TF activities/raw/joinData.m
3. Generate the TF activities with Dorothea, TF activities/macrophage.R
4. Convert into ASN output format, TF activities/convertToASNFormat.py
5. Generate matching ASN input file, TF activities/constructASNLigandMatrix.py

**For reconstructing the signaling models (~15 min)**

1. Extract TFs from Omnipath, Network Construction/constructASNLigandMatrix.py
2. Extract Ligand-Receptor from Omnipath, Network Construction/extractRL.py
3. Extract raw network from Omnipath, Network Construction/extractPKN.py
4. Reduce network to KEGG interactions, Network Construction/trimKeggModel.py
5. Reduce network to KEGG+InnateDB, Network Construction/trimMacrophageModel.py

## Run on your data

1. Generate TF activity output data and Ligand input data as in the example above.
2. Construct a data specific model as in the example above.
3. Train a predictive model as in the example above.
