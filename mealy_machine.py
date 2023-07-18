import os
from copy import deepcopy

class Mealy(object):

    def __init__(self, id, root, nodes, arcs):
        # nodes = [0,1,2,...]
        # arcs = [(0,a,1,1), ...]
        self.id = id
        self.root =  root
        self.states = nodes
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
            #print(f'There\'s no transitions from {idx} with {word[i]}')
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
            #print(f'There\'s no transitions from {idx[i]} with {word[i]}')
            break
           idx.append(self.output(idx[i], word[i])[1])
        return idx
    
    
    
    def print(self, print_all=False):
        print("\n********************* The extracted Mealy Machine *********************\n")
        print(f'-- The initial state is {self.root}')
        print(f'-- The amount of states is {len(self.states)}')
        #print("Different states of the Tree: ")
        #for i in self.states:
        #    print(f'ID: {i}\tHidden value: {0}')
        num_transitions = 10
        if print_all:
            num_transitions = len(self.transitions)
            

        print(f'-- The amount of Transitions is {len(self.transitions)}')
        print(f"\nFirst {num_transitions} over {len(self.transitions)} Transitions of the FSM")
        for i, transition in enumerate(self.transitions):
            print(f'-> {transition[0]} --> {transition[1]}/{transition[2]} --> {transition[3]}')
            if i==(num_transitions-1):
                break
                

        
    def removeDuplicate(self):
        add = True
        states = []
        for x in self.states:
            if x not in states:
                states.append(x)
        
        self.states = deepcopy(states)

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

        self.states = deepcopy(nodes)
        
    def merge_states(self, state1, state2, similarity_matrix):
        
        all_merge, correct_merge = 0, 0

        all_merge, correct_merge = self.merging(state1, state2, similarity_matrix)
        self.removeDuplicate()
        #self.print()
        return all_merge, correct_merge


    def merging(self, state1, state2, similarity_matrix, real_merging = True):
        
        x, y = 0, 0
        if ((state1, state2) not in self.states_to_merge) and ((state2, state1) not in self.states_to_merge):
            self.states_to_merge.append((state1, state2))
        else:
            return 0, 0
        
        if state1 not in self.states or state2 not in self.states:
            return 0, 0
        if real_merging:
            pass
            #print(f'\nThe real merging of two states {state1} and {state2}')
        else:
            pass
            #print(f'The submerging of two states {state1} and {state2}')
       
        if state1 == state2:
            return 0, 0
        if (state1 not in self.states or state2 not in self.states):
            return 0, 0
        
        non_determinism = False
        for i in range(len(self.transitions)):
            for j in range(len(self.transitions)):
                if(i == j):
                    continue
                if self.transitions[i][0] == state1 and self.transitions[j][0] == state2:
                    if self.transitions[i][1] == self.transitions[j][1] and self.transitions[i][2] != self.transitions[j][2]:
                        non_determinism = True

        if non_determinism:
            return 0, 0
        
        correct_sub = 0
        res = 1

        if state1 > state2:
            if similarity_matrix[state1][state2]:
                correct_sub += 1
        else:
            if similarity_matrix[state2][state1]:
                correct_sub += 1
        
        
        

        for i in range(len(self.transitions)):
            for j in range(len(self.transitions)):
                if(i == j):
                    continue

                if self.transitions[i][0] == state1 and self.transitions[j][0] == state2:
                    if self.transitions[i][1:3] == self.transitions[j][1:3]:
                        if self.transitions[i][3] in (state1, state2) and self.transitions[j][3] in (state1, state2):
                           pass
                        else:
                            #print(f'\n The two SUB states are {self.transitions[i][3]} and {self.transitions[j][3]}\n')
                            
                            x, y = self.merging(self.transitions[i][3], self.transitions[j][3], similarity_matrix, False)
                            res += x
                            correct_sub += y

        for i in range(len(self.transitions)):
            if self.transitions[i][0] == state2:
                self.transitions[i][0] = state1
            if self.transitions[i][3] == state2:
                self.transitions[i][3] = state1
        
        # If the merged is the root
        if self.root == state2:
            self.root = state1

        return res, correct_sub
    
    def toDot(self):
        rst = 'digraph fsm {'
        for x in self.states:
            if self.root == x:
                rst += f'\n\ts_{x} [root=true]'
            else:
                rst += f'\n\ts_{x}'

      	rst += "\n\tqi [shape = point]"
      	rst += f'\n\tqi -> s_{self._initial.id}'
        
        for x in self.transitions:
            rst += f'\n\ts_{x[0]} -> s_{x[3]} [label="{x[1]}/{x[2]}"]'

        rst += '\n}'
        return rst 
    
    def save(self):
        os.makedirs(f"./FSMs_extracted",exist_ok=True)
        f1 = open(f"./FSMs_extracted/fsm{self.id}.txt", "w")

        f1.write(f'{self.id}\n')
        f1.write(f'{self.states}\n')
        f1.write(f'{self.transitions}\n')