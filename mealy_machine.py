from copy import deepcopy
class Mealy(object):

    def __init__(self, id, root, nodes, arcs):
        # nodes = [0,1,2,...]
        # arcs = [(0,a,1,1), ...]
        self.id = id
        self.root =  root
        self.nodes = nodes
        self.transitions = [list(x) for x in arcs]
        self.inputAlphabet = []
        self.outputAlphabet = []
        self.states_to_merge = []
        aplh = []
        for x in self.transitions:
            if x[0] == 0:
                if x[1] not in aplh:
                    aplh.append(x[1])
            if x[1] not in self.inputAlphabet:
                self.inputAlphabet.append(x[1])
            if x[2] not in self.outputAlphabet:
                self.outputAlphabet.append(x[2])
        print(f'\n\n ************************** Le char du state 0 sont {aplh}\n\n')
    
    def output(self, initial_state, input_char):
        for x in self.transitions:
            if x[0] == initial_state and x[1] ==  input_char:
                return (x[2], x[3])
        return None

    def getInpOut(self, node):
        inp_out = []
        for x in self.transitions:
            if x[0] == node:
                inp_out.append([x[1],x[2]])
        
        return inp_out

    # get the output of the machine given a word
    def return_output(self, word):
        # we consider that the word comes without the bos sign
        output = ''
        idx = self.root
        for i in range(len(word)):
           if self.output(idx, word[i]) == None:
            print(f'There\'s no transitions from {idx} with {word[i]}')
            break
           output += self.output(idx, word[i])[0]
           idx = self.output(idx, word[i])[1]
        return output
    
    # get the trace of the machine given a word
    def return_states(self, word):
        
        # we consider that the word comes without the bos sign
        # for a word abba we have [0,1,2,3,4] for example
        idx = [self.root]
        for i in range(len(word)):
           if self.output(idx[i], word[i]) == None:
            print(f'There\'s no transitions from {idx[i]} with {word[i]}')
            break
           idx.append(self.output(idx[i], word[i])[1])
        return idx
    
    
    
    def print(self, num_transitions = 10, all=False):
        print(f'-- The initial state is {self.root}')
        print(f'-- The amount of states is {len(self.nodes)}')
        #print("Different states of the Tree: ")
        #for i in self.nodes:
        #    print(f'ID: {i}\tHidden value: {0}')
        if all:
            num_transitions = len(self.transitions)

        print(f'-- The amount of Transitions is {len(self.transitions)}')
        print(f"\nFirst {num_transitions} over {len(self.transitions)} Transitions of the FSM")
        if len(self.transitions) <= num_transitions :
            for transition in self.transitions:
                print(f'-> {transition[0]} --> {transition[1]}/{transition[2]} --> {transition[3]}')
        else:
            for i, transition in enumerate(self.transitions):
                print(f'-> {transition[0]} --> {transition[1]}/{transition[2]} --> {transition[3]}')
                if i == 9:
                    break

        
    def removeDuplicate(self):
        add = True
        states = []
        for x in self.nodes:
            if x not in states:
                states.append(x)
        
        self.nodes = deepcopy(states)

        transitions = [self.transitions[0]]
        for x in self.transitions:
            for y in transitions:
                if x == y:
                    add = False
            if add:
                transitions.append(x)
            add = True
        self.transitions = deepcopy(transitions)
        
        nodes = []
        for x in self.transitions:
            if x[0] not in nodes:
                nodes.append(x[0])
            if x[3] not in nodes:
                nodes.append(x[3])

        self.nodes = deepcopy(nodes)
        
    def merge_states(self, state1, state2):
        
        
        k = self.merging(state1, state2)

        
        self.removeDuplicate()
        #self.print()


    def merging(self, state1, state2, real_merging = True):

       
        if ((state1, state2) not in self.states_to_merge) and ((state2, state1) not in self.states_to_merge):
            self.states_to_merge.append((state1, state2))
        else:
            return 0
        
        if state1 not in self.nodes or state2 not in self.nodes:
            return 0
        if real_merging:
            pass
            #print(f'\nThe real merging of two states {state1} and {state2}')
        else:
            pass
            #print(f'The submerging of two states {state1} and {state2}')
        submerge = []
        if state1 == state2:
            return 0
        if (state1 not in self.nodes or state2 not in self.nodes):
            return 1
        
        

        non_determinism = False
        for i in range(len(self.transitions)):
            for j in range(len(self.transitions)):
                if(i == j):
                    continue
                if self.transitions[i][0] == state1 and self.transitions[j][0] == state2:
                    if self.transitions[i][1] == self.transitions[j][1] and self.transitions[i][2] != self.transitions[j][2]:
                        non_determinism = True

        if non_determinism:
            return 0

        for i in range(len(self.transitions)):
            for j in range(len(self.transitions)):
                if(i == j):
                    continue

                # merge the children of the two mergable states
                if self.transitions[i][0] == state1 and self.transitions[j][0] == state2:
                    if self.transitions[i][1:3] == self.transitions[j][1:3]:
                        if self.transitions[i][3] in (state1, state2) and self.transitions[j][3] in (state1, state2):
                            submerge.append((self.transitions[i][3], self.transitions[j][3]))
                        else: 
                            #print(f'\n The two SUB states are {self.transitions[i][3]} and {self.transitions[j][3]}\n')
                            self.merging(self.transitions[i][3], self.transitions[j][3], False)
                    

        
        for x in submerge:
            for i in range(len(self.transitions)):
                if self.transitions[i][0] == x[1]:
                    self.transitions[i][0] = x[0]
                if self.transitions[i][3] == x[1]:
                    self.transitions[i][3] = x[0]

        
        for i in range(len(self.transitions)):
            if self.transitions[i][0] == state2:
                self.transitions[i][0] = state1
            if self.transitions[i][3] == state2:
                self.transitions[i][3] = state1
        
        # If the merged is the root
        if self.root == state2:
            self.root = state1


        return 0