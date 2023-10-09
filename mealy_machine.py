import os
import sys
from copy import deepcopy


def get_index(states, state):
    for i in range(len(states)):
        if state in states[i]:
            return i
    return len(states)

def allzeros(list_):
    for x in list_:
        if x != 0:
            return False
    return True

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
    
    # print the fsm
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
        else :
            if num_transitions > len(self.transitions):
                num_transitions = len(self.transitions)
            
        print(f'-- The amount of Transitions is {len(self.transitions)}')
        print(f"\nFirst {num_transitions} over {len(self.transitions)} Transitions of the FSM")
        for i, transition in enumerate(self.transitions[:num_transitions]):
            print(f'-> {transition[0]} --> {transition[1]}/{transition[2]} --> {transition[3]}')

    # Delete double transition or states          
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

    # Merge preparation
    def merge_states(self, state1, state2, similarity_matrix = None):
        
        all_merge, correct_merge = 0, 0

        all_merge, correct_merge = self.merging(state1, state2, similarity_matrix)
        self.removeDuplicate()
        self.states_to_merge.clear()
        #self.print()
        return all_merge, correct_merge

    # Merging operation
    def merging(self, state1, state2, similarity_matrix = None, real_merging = True):
        
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
        
        non_determinism = False
        for i in range(len(self.transitions)):
            for j in range(len(self.transitions)):
                if self.transitions[i][0] == state1 and self.transitions[j][0] == state2:
                    if self.transitions[i][1] == self.transitions[j][1] and self.transitions[i][2] != self.transitions[j][2]:
                        #(f'\n{i}')
                        #print(self.transitions[i])
                        #print(self.transitions[j])
                        #print(f'{j}\n')
                        non_determinism = True
                        break

        if non_determinism:
            print('Cette Machine n\'est pas déterministe\n')
            return 0, 0
        
        correct_sub = 0
        res = 1
        if similarity_matrix != None:
            if state1 > state2:
                if similarity_matrix[state1][state2]:
                    correct_sub += 1
            else:
                if similarity_matrix[state2][state1]:
                    correct_sub += 1
        

        for i in range(len(self.transitions)):
            for j in range(len(self.transitions)):
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
    
    def determinize(self):
        # Pretty much the same as is_state_deterministic
        # But here we combine the 
        transitions = deepcopy(self.transitions)
        _continue = True
        while _continue:
            _continue = False
            transitions = deepcopy(self.transitions)
            states_to_merge = []
            for i in range(len(transitions)):
                for j in range(i):
                    if transitions[i][:3] == transitions[j][:3]:
                        #print(f'{transitions[i][0]}')
                        _continue = True
                        #print(f'First transition {transitions[i]}')
                        #print(f'Second transition {transitions[j]}')
                        #print(f'States {self.states}')
                        if transitions[i][3] in self.states and transitions[j][3] in self.states:
                            #print(f'{transitions[i][3]}\t{transitions[j][3]}')
                            merge_amount, _ = self.merge_states(transitions[i][3], transitions[j][3])
                            if merge_amount == 0:
                                return False
                                
                        #print(self.states)
                        #self.print()
        return True

    def minimize(self):
        states = self.states
        transitions = self.transitions
        min_states = []
        last_state_transitions = []
        for i in range(len(states)-1):
            min_states.append([])
            for j in range(len(states)):
                if j <= i:
                    continue
                #print(f'{states[i]}\t{states[j]}')
                check = {}
                notequiv = False
                for p in range(len(transitions)):
                    # sauvegarde des transitions du dernier états puisqu'il n'est pas représenté dans le tableau
                    if transitions[p][0] == states[-1]:
                        last_state_transitions.append(transitions[p])
                    if transitions[p][0] in (states[i],states[j]):
                        if transitions[p][1] in list(check.keys()):
                            for x in check[transitions[p][1]]:
                                if x[0] != transitions[p][2]:
                                    notequiv = True
                                    break
                            if transitions[p][2] not in list(check[transitions[p][1]].keys()):
                                check[transitions[p][1]][transitions[p][2]] = []
                            check[transitions[p][1]][transitions[p][2]].append(states.index(transitions[p][3]))
                        else:
                            check[transitions[p][1]] = {}
                            check[transitions[p][1]][transitions[p][2]] = [states.index(transitions[p][3])]
                 
                if notequiv:
                    min_states[i].append(0)
                else:
                    #print(check)
                    #{'b': {'1': [194, 83]}, 'a': {'0': [318, 318]}}
                    # if one state has less than two 2 transitions(one for b and one ofr a)
                    #print(check)
                    for (p, q) in list(check.items()):
                        if len(list(q.values())[0]) < len(self.inputAlphabet):
                            check[p][list(q.keys())[0]] = check[p][list(q.keys())[0]]*len(self.inputAlphabet)
                    
                    values = list(check.values())
                   
                    # if len(check) == 1:
                    #     key = list(check.keys())[0]
                    #     for x in transitions:
                    #         if x[0] in (self.states[i], self.states[j]) and x[1] !=key:
                    #             d = {f'{x[2]}': [states.index(x[3])]*len(self.inputAlphabet)}
                    #             values += [d]
                    min_states[i] += [[list(x.values())[0] for x in values]]
        #min_states = deepcopy(min_states1)
        still_check = True
        while still_check:
            still_check = False
            for i in range(len(min_states)):
                for j in range(len(min_states[i])):
                    if type(min_states[i][j]) == list:
                        for p in min_states[i][j]:
                            if len(p) < len(self.inputAlphabet):
                                continue
                            s1 = min(p[0],p[1])
                            s2 = max(p[0],p[1])
                            if s1 == s2:
                                continue
                            if min_states[s1][s2-s1-1] == 0:
                                still_check = True
                                min_states[i][j] = 0
                                break
        
        macro_states = [] # states = [[0,3,4],[1,6],[5]]
        one_last_state = True
        for i in range(len(min_states)):
            if allzeros(min_states[i]) and get_index(macro_states, i) == len(macro_states):
                macro_states.append([i])
                # print(macro_states)
                continue
            for j in range(len(min_states[i])):
                # print('Equivalent')
                if type(min_states[i][j]) == list:
                    if len(min_states[i][j]) < len(self.inputAlphabet):
                        # print("\n\n I'm here \n\n")
                        continue
                    # Check if the last state (j) is mergeable with another state (i)
                    if j+i+1 == len(states)-1:
                        one_last_state = False

                    # Cluster the resembling states
                    idx = get_index(macro_states, i)
                    if idx == len(macro_states):
                        macro_states.append([i])
                        # print(macro_states)
                    if get_index(macro_states, j+i+1) != len(macro_states):
                        continue
                    if j+i+1 not in macro_states[idx]:
                        macro_states[idx].append(j+i+1)
                        # print(macro_states)

        # for i in range(len(macro_states)):
        #     print(f'\n{i}\n')
        #     rer = []
        #     for x in macro_states[i]:
        #         rer.append(self.states[x])
        #     print(f'{rer}\n')

        if one_last_state:
            macro_states.append([len(states)-1])

        new_transitions_ = deepcopy(transitions)
        new_states = list(range(len(macro_states)))
        for i in range(len(new_transitions_)):
            for j in range(len(new_transitions_[i])):
                if j in (0,3):
                    new_transitions_[i][j] = get_index(macro_states, states.index(new_transitions_[i][j]))
        new_transitions = []
        for x in new_transitions_:
            if x not in new_transitions:
                new_transitions.append(x)
        new_root = get_index(macro_states, states.index(self.root))
        self.root = new_root
        self.states = new_states
        self.transitions = new_transitions
        #self = Mealy(self.id, new_root, new_states, new_transitions)
     
    def toDot(self):
        rst = 'digraph fsm {'
        for x in self.states:
            if self.root == x:
                rst += f'\n\ts_{x} [root=true]'
            else:
                rst += f'\n\ts_{x}'

        rst += "\n\tqi [shape = point]"
        rst += f'\n\tqi -> s_{self.root}'
        
        for x in self.transitions:
            rst += f'\n\ts_{x[0]} -> s_{x[3]} [label="{x[1]}/{x[2]}"]'

        rst += '\n}'
        return rst 
    
    def save(self, filepath, extra=None):
        os.makedirs(filepath,exist_ok=True)
        if extra != None:
            f1 = open(filepath + f"/fsm{self.id}_{extra}.txt", "w")
        else:
            f1 = open(filepath + f"/fsm{self.id}.txt", "w")

        f1.write(f'{self.id}\n')
        f1.write(f'{self.states}\n')
        f1.write(f'{self.transitions}\n')

    def is_output_deterministic(self):
        # We check if every couple (inp_state, inp) corresponds to
        # one and only one out symbol 
        for i in range(len(self.transitions)):
            for j in range(i):
                if self.transitions[i][:2] == self.transitions[j][:2]:
                    if self.transitions[i][2] != self.transitions[j][2]:
                        return False
        return True
    
    def is_state_deterministic(self):
        # We check if every state in the fsm doesn't have two transition with the
        # (inp/out) couple but to two different target states
        #st = [{'a':[2], 'b':[0]}, {'a':[1], 'b':[3]}]
        st = []
        ISD = True
        for _ in range(len(self.states)):
            st.append(dict())

        for i in range(len(self.transitions)):
            #print(self.transitions[i])
            idx = self.states.index(self.transitions[i][0])
            if self.transitions[i][1] not in list(st[idx].keys()):
                st[idx][self.transitions[i][1]] = []
            else:
                ISD = False
            if self.transitions[i][3] not in st[idx][self.transitions[i][1]]:
                st[idx][self.transitions[i][1]].append(self.transitions[i][3])
            #print(st)
        return ISD, st

    def is_complete(self):
        complete = []
        for _ in self.states:
            complete.append([])
        for x in self.transitions:
            idx = self.states.index(x[0])
            if x[1] not in complete[idx]:
                complete[idx].append(x[1])
        for x in complete:
            if len(x) < len(self.inputAlphabet):
                return False
        return True
    
    def final_merges(self, states_to_merges):
        # Last determinize
        # We asume her that if a state conducts to two different states
        # couple (inp/out), then the 2 target states have to 
        # be merged
        #states_to_merges = [[[0,1,2], [3, 0, 1]],[[2]]]
        already_merged = []
        for y in states_to_merges:
            #print(y)
            for x in y:
                if len(x) == len(self.inputAlphabet):
                    if (x[0], x[1]) not in already_merged and (x[1], x[0]) not in already_merged:
                        #print((x[0], x[1]))
                        merge_amount, _ = self.merge_states(x[0], x[1])
                        if merge_amount == 0:
                            return False
                        already_merged.append((x[0],x[1]))
                if len(x) > len(self.inputAlphabet):
                    first = x[0]
                    for i in range(len(x)-1):
                        if (first, x[i+1]) not in already_merged and (x[i+1], first) not in already_merged:
                            #print((first, x[i+1]))
                            merge_amount, _ = self.merge_states(first, x[i+1])
                            if merge_amount == 0:
                                return False
                            already_merged.append((first,x[i+1]))
        return True
            