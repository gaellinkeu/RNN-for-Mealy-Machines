from copy import deepcopy


#from utils import merging_checking

class TrieTransition:
    """A transition in the trie structure"""

    def __init__(self, src, tgt, input=None, output=None, id=-1):
        self.src = src
        self.tgt = tgt
        self._input = input
        self._output = output
        self._id = id

    def getCharOutput(self):
        return self._output
    
    def getStateOutput(self):
        return self.tgt
    
    def serialize(self):
        return (self.src, self._input, self._output, self.tgt)

class TrieNode:
    """A node in the trie structure"""

    def __init__(self, id, char="", label=None):
        
        self.id = id
        self._outTr = {}
        self._inTr = {}
        self.children = {}
        self.char = char
        self.label = label
  
    
    def addOutTr(self, transition:TrieTransition) :
      if not(transition.getInput() in self._outTr.keys()) :
         self._outTr[transition.getInput()] = {}
      if not(transition.getOutput() in self._outTr[transition.getInput()].keys()) :
         self._outTr[transition.getInput()][transition.getOutput()] = []
      self._outTr[transition.getInput()][transition.getOutput()].append(transition)
   
    def addInTr(self, transition:TrieTransition) :
      if not(transition.getInput() in self._inTr.keys()) :
         self._inTr[transition.getInput()] = {}
      if not(transition.getOutput() in self._inTr[transition.getInput()].keys()) :
         self._inTr[transition.getInput()][transition.getOutput()] = []
      self._inTr[transition.getInput()][transition.getOutput()].append(transition)
   



class Trie(object):

    def __init__(self, corpus, labels):
        """
        The trie has at least the root node.
        The root node does not store any character
        """

        self.root = TrieNode(0)
        self.state_count = 0
        self.transition_count= 0
        self.transitions = []
        self.inputVocabulary = []
        self.outputVocabulary = []
        self.arcs = []
        self.states = {0: self.root} # The initial state is the state 0

        # self.states = {0: node0, 1: node1, ...}
        # self.transitions = {transition0, transition1, ...}

        for i , (word, label) in enumerate(zip(corpus, labels)):
            self.insert(word, label, i)

        print(f'\nThe number of states of the Trie: {len(self.states)}\n')

        self.dfs(self.root)
        self.print()

        #self.states = [[i, False] for i in range(self.count + 1)]

        #self.dfs(self.root)

    def insert(self, word, label, i=0):
        """Insert a word into the trie"""
        
        node = self.root
    
        # Loop through each character in the word
        # Check if there is no child containing the character, create a new child for the current node
        for i, char in enumerate(word):

            new_transition = False
            if char not in self.inputVocabulary:
                self.inputVocabulary.append(char)
            if label[i] not in self.outputVocabulary:
                self.outputVocabulary.append(label[i])

            """
            if char in node.children:
                node = node.children[char]
            else:
                self.count += 1
                new_node = TrieNode(self.count, char)
                node.children[char] = new_node
                node = new_node
            """

            if char not in node._outTr.keys():
                node._outTr[char] = {}
                new_transition = True
            
            if label[i] not in node._outTr[char].keys():
                self.state_count += 1
                new_node = TrieNode(self.state_count, char, label[i])
                node._outTr[char][label[i]] = []
                new_node._inTr[char] = {}
                #new_node._inTr[char][label[i]] = [node.id]
                #node._outTr[char][label[i]].append(new_node.id)
                node.children[char] = (label[i], new_node.id)
                self.states[new_node.id] = new_node

            else:
                new_node = self.states[node.children[char][1]]
            

            if new_transition:
                transition = TrieTransition(node, new_node, char, label[i], self.transition_count)
                self.transitions.append(transition)
                self.transition_count += 1

            node = new_node

    # Store all trie transition in our Tree
    def dfs(self, node):
        """Depth-first transversal of the trie"""

        for i in node.children.keys():
            self.arcs.append((node.id, self.states[node.children[i][1]].char, self.states[node.children[i][1]].label, node.children[i][1]))
            self.dfs(self.states[node.children[i][1]])

    # Print the details of our Tree
    def print(self):
        print("\n********************* Prefix Tree of Mealy Machine corpus **********************\n\n")
        print(f'Number of states: {self.state_count + 1}')
        print(f'Number of transitions: {self.transition_count}')
        print(f'Initial state: {self.root.id}')
        print(f'Input vocabulary: {str(self.inputVocabulary)}')
        print(f'Output vocabulary: {str(self.outputVocabulary)}\n')

        """print("Different states of the Tree: ")
        for i in list(self.states.keys()):
            print(f'ID: {self.states[i].id}')"""

        
        if len(self.transitions) <= 10:
            print(f"\nFirst {len(self.transitions)} transitions of the Tree: ")
            for transition in self.transitions:
                print(f'-> {transition.src.id} --> {transition._input}/{transition._output} --> {transition.tgt.id}')
        else:
            print(f"\nFirst 10 over {len(self.transitions)} transitions of the Tree: ")
            for i, transition in enumerate(self.transitions):
                print(f'-> {transition.src.id} --> {transition._input}/{transition._output} --> {transition.tgt.id}')
                if i == 9:
                    break
        print("\n********************************************************************************\n\n")
              
    
    """def merging_state(self, k):
        # k is the similarity treshold
        for i in len(self.states.keys()):
            for j in len(self.states.keys()):

                if i == j:
                    continue
                merge = merging_checking(self.states[i], self.states[j], k)  # check if the two states are mergable

                if not merge:
                    continue"""

    def return_states(self, word):
        node = self.root
        trace = [node]
        for i,char in enumerate(word):
            node = self.states[node.children[char][1]]
            trace.append(node) 

        return trace
    
                



                

