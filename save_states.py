import os
from copy import deepcopy

def save_states(id, inputs, outputs, states, idx):
    os.makedirs(f"./States",exist_ok=True)
    info_filepath = f"./States/states{id}.txt"
    f2 = open(info_filepath, "w")

    rounded_states = deepcopy(states)
    for i in range(len(states)):
        for j in range(len(states[i])):
            rounded_states[i][j] = round(states[i][j], 3)
    

    for i in range(len(inputs)):
        f2.write(f'{inputs[i]}    -->    {outputs[i]}\n\n')
        for y in range(len(inputs[i])+1):
            if y == 0:
                f2.write(f'espison   -->  h_{idx[i][y]} = {rounded_states[idx[i]][y]}\n')
            else:
                f2.write(f'{inputs[i][y-1]}   -->  h_{idx[i][y]} = {list(rounded_states[idx[i]][y])}\n')
        f2.write('\n\n\n')
    f2.close()