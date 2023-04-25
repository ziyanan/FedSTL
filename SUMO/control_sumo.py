import os, sys
import traci
import traci.constants as tc
import sumolib
import random
import numpy as np
random.seed(0) 

# stdoutOrigin=sys.stdout
# sys.stdout = open("/hdd/SUMO_dataset/log.txt", "a")

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


sumoBinary = "/usr/local/bin/sumo-gui"
sumoCmd = [sumoBinary, "-c", "/home/theuser/sumo/tools/2022-12-31-12-37-53/osm.sumocfg"]
net = sumolib.net.readNet('/home/theuser/sumo/tools/2022-12-31-12-37-53/osm.net.xml')
dataset_path = "/hdd/SUMO_dataset/"


traci.start(sumoCmd)
all_edges = traci.edge.getIDList()

print(len(all_edges))


# init loop
step = 0
while step < 200:
    traci.simulationStep()
    step += 1


select_edges = []
for e in all_edges:
    if traci.edge.getLastStepVehicleNumber(e) > 5:
        select_edges.append(e)


print(len(select_edges))


select_edges = random.sample(select_edges, k=100)
select_files = [dataset_path+str(e)+".npy" for e in select_edges]



record_data = {}
for e in select_edges:
    record_data[e] = []



try:
    while step < 3600:
        traci.simulationStep()
        print("time", step)
        
        for e in select_edges:
            record_data[e].append(
                [   
                    traci.edge.getLastStepVehicleNumber(e),
                    traci.edge.getLastStepOccupancy(e),
                    traci.edge.getLastStepMeanSpeed(e),
                    traci.edge.getCO2Emission(e),
                    traci.edge.getFuelConsumption(e),
                    traci.edge.getNoiseEmission(e),
                ])
        step += 1

    traci.close(False)

except KeyboardInterrupt:
    for idx, e in enumerate(select_edges):
        np.save(select_files[idx], np.array(record_data[e]))
else:
    for idx, e in enumerate(select_edges):
        np.save(select_files[idx], np.array(record_data[e]))


print("Done.")




# sys.stdout.close()
# sys.stdout=stdoutOrigin