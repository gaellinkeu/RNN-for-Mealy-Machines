def load_data(filepath):

    inputs = []
    outputs = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    #print(lines[:3])
    #max_length = 0

    for line in lines:
        res = ""
        isInput = True
        for symbol in line:
            if symbol in [',', '\n']:
                if isInput:
                    inputs.append(res)
                    #max_length = len(res) if len(res) > max_length else max_length
                    res = ""
                    isInput = not isInput
                    continue
                else:
                    outputs.append(res)
            res += symbol
        #print(line)
    max_length = len(max(inputs, key=len))

    return inputs, outputs, max_length

