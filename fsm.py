# Initial author : Omer Nguena Timo
# Modified by: Gael Linkeu

from state import State
from transition import Transition

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
   
   def __str__(self) -> str:
      pass
  
   def toDot(self) -> str :
      rst =""
      rst+= f'digraph fsm' + "{"
      for cle in self._statesById.keys() :
         rst += "\n\t" + self._statesById[cle].toDot()
      
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



   
