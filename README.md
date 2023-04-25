FedSTL Implementation 
===============

This document serves as a step-by-step guide for running the FedSTL codebase. 

### TODO:
- Fix dataloader, check dataloader length. 
- Fix data normalization.

### 1. Dependencies
- Python 3.9 and python-dev packages are required. 
- The following additional packages are needed to run this repository: 
matplotlib==3.5.1, numpy==1.21.6, pandas==1.3.5, parsimonious==0.10.0, scikit_learn==1.2.0, scipy==1.10.0, singledispatch==3.7.0, torch==1.11.0, tqdm==4.64.0


### 2. Data Generation 
- Please refer to [SUMO tutorials](https://sumo.dlr.de/docs/Tutorials/index.html) for quick installation guides. 
- Once SUMO is successfully installed, please make sure that the SUMO binary is located at `/usr/local/bin/sumo-gui/` for Ubuntu system. 
- First, follow the instruction at [OSMWebWizard](https://sumo.dlr.de/docs/Tutorials/OSMWebWizard.html) to generate SUMO demand and network files.  
- Then, run the script `control_sumo.py` to generate the dataset. The saved file path can be defined at line 22. 

### 3. Data Preprocessing 
- The script `dataset.py` includes the code for preprocessing the FHWA dataset. One can use the command `python3.9 dataset.py` to generate the training dataset. 
- The script `sumo_dataset.py` includes the code for preprocessing the SUMO dataset. Please make sure that the data path is correctly defined. 

### 4. Specification Inference 
- The folder telex includes the code needed for specification inference, where `scorer.py` includes STL metrics and the implementations of Equation 3 defined in the text. 
Additionally, `synth.py` includes the code for generating specifications from STL templates. 

### 5. Network Training and Evaluation 
- To train the FedAvg model on SUMO data, use the following command: 
    ```
    python3.9 fed_train.py --mode train --dataset sumo --client 100 --cluster 0 --frac 0.1
    ```
    Optional parameters:\
    `--mode` Select the mode from these options: `train`, `train-logic`, `eval`, `eval-sumo`\
    `--dataset` Use either sumo or fhwa\
    `--client` The number of participating clients \
    `--cluster` The total number of clusters \
    `--frac` The client participation rate\
    `--model` RNN, GRU, or LSTM backbone\
    `--epoch` Number of total communication rounds\
    `--batch_size` The batch size\
    `--max_lr` Maximum learning rate\

- Similarly, to run the IFCA model on SUMO dataset with 5 clusters, run the following command: 
    ```
    python3.9 fed_train.py --mode train --dataset sumo --client 100 --cluster 5 --frac 0.1
    ```

- Finally, to run FedSTL on SUMO dataset with 5 clusters and multivariate correlation specifications, run the following command: 
    ```
    python3.9 fed_train.py --mode train-logic --dataset sumo --client 100 --cluster 5 --frac 0.1 --property_type corr --client_iter 15
    ```

- To evaluation, run either `python3.9 fed_train.py --mode eval` or `python3.9 fed_train.py --mode eval-sumo`. 

### 6. Note
- We include a sample of the dataset due to file size limitations of the supplementary material and the double blinded review policy. 
However, the full dataset can be shared upon request. 
