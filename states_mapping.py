import numpy as np

def map_states(trie, rnn_states, inputs, outputs, mask, hidden_size):

    idx = [trie.return_states(sent) for sent in inputs]
    n_states = len(trie.states)
    states = np.zeros((n_states, hidden_size))
    states_mask = np.zeros(n_states)

    outputs_ = np.array([np.array(list(x)) for x in outputs])
    for i, _r in enumerate(rnn_states):
        states[idx[i]] = _r[mask[i]]
        states_mask[idx[i]] = outputs[i][mask[i]]

    return states, states_mask