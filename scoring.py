# The accuracy of the whole dataset
def score_whole_words(machine, inputs, outputs, type = 'Mealy machine'):
    acc = 0
    for word, y in zip(inputs, outputs):
        acc += (machine.return_output(word) == y)
    print(f'\n Uncorrect output sequences for the {type}: {len(inputs) - acc} ({100 - acc / len(inputs) * 100})%')
    return (acc / len(inputs) * 100)

# The accuracy of the dataset based on each prefix of each input sequence
def score_all_prefixes(mealy, inputs, outputs, type = 'Mealy machine'):
    
    score , count = 0, 0
    for i in range(len(inputs)):
        output = mealy.return_output(inputs[i])
        scores = [outputs[i][j] == output[j] for j in range(len(output))]
        for x in scores:
            if x == False:
                break
            score += 1
            
        count += len(inputs[i])
    print(f'\n Uncorrect output symbol for the {type}: {count - score} ({100 - score/count * 100})%')
    return score/count * 100