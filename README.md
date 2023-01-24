1. To construct dataset, use dataset.py.
2. To train LSTM model, use train.py
3. To train LSTM model with logic, use train.py --model train-logic
4. To construct fed dataset, use dataset.py load_fed_dataset.


FedSTL Implementation 
===============

This document serves as a step-by-step guide for running the FedSTL codebase. 

### 1. Dependencies
- Python 3.9 and python-dev packages are required. 
- The following additional packages are needed to run this repository: 
matplotlib==3.5.1, numpy==1.21.6, pandas==1.3.5, parsimonious==0.10.0, scikit_learn==1.2.0, scipy==1.10.0, singledispatch==3.7.0, torch==1.11.0, tqdm==4.64.0

### 2. Data Generation 
- Please refer to [SUMO tutorials](https://sumo.dlr.de/docs/Tutorials/index.html) for quick installation guides. 
- Once SUMO is installed, please make sure the SUMO binary is located at `/usr/local/bin/sumo-gui/` for Ubuntu systems. 
- First, follow the instruction at [OSMWebWizard](https://sumo.dlr.de/docs/Tutorials/OSMWebWizard.html) to generate SUMO demand and network files.  
- Then, run the script `control_sumo.py` to generate the dataset. The saved file path can be defined at line 22. 

### 3. Data Preprocessing 
- The script `dataset.py` includes the code for preprocessing the FHWA dataset. One can use the command `python3.9 dataset.py` to generate the splitted dataset. 
- The script `sumo_dataset.py` includes the code for preprocessing the SUMO dataset. Please make sure that the data path is correctly defined. 

### 4. Specification Inference 
- The folder telex include the code needed for specification inference, where `scorer.py` include STL metrics and the implementations of Equation 3 defined in the text. 
Additionally, `synth.py` include the code for generating specifications from STL templates. 
