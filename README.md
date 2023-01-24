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
- Once SUMO is successfully installed, please make sure that the SUMO binary is located at `/usr/local/bin/sumo-gui/` for Ubuntu system. 
- First, follow the instruction at [OSMWebWizard](https://sumo.dlr.de/docs/Tutorials/OSMWebWizard.html) to generate SUMO demand and network files.  
- Then, run the script `control_sumo.py` to generate the dataset. The saved file path can be defined at ln 22. 

### 3. Data Preprocessing 
- Once CARLA is installed and working. Create a new folder `data_generation` under the directory `carla/PythonAPI/`. Then add the files in this repo to the new folder. 

- Use the command 
    ```
    cd carla/Unreal/CarlaUE4 &&
    ~/UnrealEngine_4.24/Engine/Binaries/Linux/UE4Editor "$PWD/CarlaUE4.uproject" -opengl
    ```
    to launch a new CARLA server, then select a town you would like to use. Currently, we support Town 3, Town 4, and Town 5. 

- Run the following command to spwan NPCs controlled by SUMO: 
    ```
    cd carla/Co-Simulation/Sumo/ &&
    python3 spawn_npc_sumo.py --tls-manager carla --sumo-gui
    ```
    Use the optional parameter `-n, --number-of-vehicles` (default: 10) to change the number of vehicles you would like to spawn. 

- Use the following command to start recording:
    ```
    cd carla/PythonAPI/data_generation/ &&
	python3 recording.py
    ```
    Optional parameters:\
    `-t` (default: 5) Town number\
    `-v` (default: 500) Number of vehicles

    By default, the log is stored under `carla/Unreal/CarlaUE4/Saved/`.

- Once the log is recorded, run the script `carla/PythonAPI/data_generation/find_group.py` to replay and generate a group.txt that stores the ego vehicle IDs. 

    Optional parameters:\
    `-n` (default: 1) Intersection number\
    `-v` (default: 500) Number of vehicles\
    `-t` (default: 5) Town number

    The intersection numbers can be found at `carla/PythonAPI/data_generation/town_maps`. 

- To generate one scene, use the script `carla/PythonAPI/data_generation/carScene_data_generation.py`. 

    Optional parameters:\
    `-r` (default: 1) Line number in `group.txt`\
    `-f` (default: sumo-carla-town5-500.log) Recorded file name\
    `-t` (default: 5) Town number
