# Original author : Omer Nguena Timo
# Modified by: Gaël Linkeu

from state import State
from transition import Transition
import os

class FSM :
   def __init__(self, initState=None, data=None) :
      self._initial = initState
      self._statesById = dict() 
      self._inputSet =set()
      self._outputSet =set()
      self._transitionsById =dict()
   
   def nextStateId(self) -> int :
      return len(self._statesById.keys())
   
   def nextTransitionId(self) -> int :
      return len(self._transitionsById.keys())
   
   
   def getInitialSate(self) -> State :
      return self._initial

   def addState(self, state: State) -> State :
      id = self.nextStateId()
      self._statesById[id] = state
      state.setID(id)
      if (self._initial==None) :
         self._initial= state 
      return state 
   
   def getState(self,id:int) -> State :
      return self._statesById[id]
   
   def addTransition(self, idSrc, idTgt, input, output) -> Transition:
      srcState = self.getState(idSrc)
      tgtState = self.getState(idTgt)
      if (srcState!=None and tgtState!=None and input!=None and output!=None) :
         transition = Transition(srcState, tgtState, input, output)
         srcState.addOutTr(transition)
         tgtState.addInTr(transition)
         id = self.nextTransitionId()
         self._transitionsById[id] = transition
         transition.setID(id)
         self._inputSet.add(input)
         self._outputSet.add(output)
         return transition
      return None

   def nbTransitions(self):
      return len(self._transitionsById.keys())
   
   def min_words(self, remaining_states, current_state):
      stop = False
      if len(remaining_states) == 0:
         stop = True
      state = current_state
      inputs_set = []
      outputs_set = []
      alphabet = 0
      transitions = list(self._transitionsById.values())
      for x in transitions:
         if x._src._id == state:
            alphabet += 1
            if stop:
               inputs_set.append(x._input)
               outputs_set.append(x._output)
            else:
               input_char = x._input
               output_char = x._output
               if x._tgt._id in remaining_states:
                  input_words, output_words = self.min_words(x._tgt._id, remaining_states.remove(x._tgt._id))
                  inputs_set.append([input_char+p for p in input_words])
                  outputs_set.append([output_char+p for p in output_words])

            if alphabet > len(self._inputSet):
               print(f'The state {state} has more output transitions than the alphabet lenght')
      return inputs_set, outputs_set
   
   def __str__(self) -> str:
      pass
  
   def toDot(self) -> str :
      rst =""
      rst+= f'digraph fsm' + "{"
      for cle in self._statesById.keys() :
         rst += "\n\t" + self._statesById[cle].toDot()
      rst += "\n\tqi [shape = point]"
      rst += f'\n\tqi -> s_{self._initial.id}'
      
      for cle in self._transitionsById.keys() :
         rst += "\n\t" + self._transitionsById[cle].toDot()
      rst+="\n}"
      return rst  
   
   def produceOutput(self, input) -> str:

      currentState = self._initial
      output = ""
      # i for input_symbol
      # o for output_symbol
      for i in input:
         o = list(self._statesById[currentState.getID()]._outTr[i].keys())[0]
         currentState = self._statesById[currentState.getID()]._outTr[i][o][0]._tgt
         output += o
      return output 

   def print(self):
      print(f'-- The initial state is {self._initial._id}')
      print(f'-- The amount of states is {len(self._statesById)}')
      print(f'-- The amount of Transitions is {len(self._transitionsById)}')
      
      transitions = list(self._transitionsById.values())
      for x in transitions:
         print(f'-> {x._src._id} --> {x._input}/{x._output} --> {x._tgt._id}')

   def save(self, id=0, times=0):
      os.makedirs(f"./FSMs",exist_ok=True)
      f1 = open(f"./FSMs/fsm{id}_{times}.txt", "w")
      f1.write(f'{id}\n')

      states = []
      transitions = []
      for i in self._statesById.keys():
         states.append(self._statesById[i]._label)
      for i in self._transitionsById.keys():
         t = self._transitionsById[i]
         transitions.append(f'({t._src._label}, {t._input}, {t._output}, {t._tgt._label})')

      f1.write(f'{states}\n')
      f1.write(f'{transitions}\n')
      f1.close()

      os.makedirs(f"./FSMs_visuals",exist_ok=True)
      f1 = open(f"./FSMs_visuals/fsm{id}_{times}.dot", "w")
      f1.write(self.toDot())
      f1.close()


   
