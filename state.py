import random
from transition import Transition
class State :
   def __init__(self, label=None, id = -1, accepting=False):
      self._label = label
      self._id = id
      self._accepting=accepting
      self._outTr = {}
      self._inTr = {}

   def setID(self, id) :
      self._id = id

   def getID(self) :
      return self._id
   
   def getLabel(self) :
      return self._label 
   
   def isAccepting(self) :
      return self._accepting
   
   def addOutTr(self, transition:Transition) :
      if not(transition.getInput() in self._outTr.keys()) :
         self._outTr[transition.getInput()] = {}
      if not(transition.getOutput() in self._outTr[transition.getInput()].keys()) :
         self._outTr[transition.getInput()][transition.getOutput()] = []
      self._outTr[transition.getInput()][transition.getOutput()].append(transition)
   
   def addInTr(self, transition:Transition) :
      if not(transition.getInput() in self._inTr.keys()) :
         self._inTr[transition.getInput()] = {}
      if not(transition.getOutput() in self._inTr[transition.getInput()].keys()) :
         self._inTr[transition.getInput()][transition.getOutput()] = []
      self._inTr[transition.getInput()][transition.getOutput()].append(transition)
   
   def __str__(self) :
      if (self.isAccepting()) :
         return  f'c{self._label}[{self._id}]'
      else :
         return  f'f{self._label}[{self._id}]'
   
   def print(self):
      print(f'\nLes transitions sortantes du state {self._label}:\n')
      rst_=""
      for i in self._outTr.keys():
         rst_ += f'{i} : '
         for j in self._outTr[i].keys():
            rst = rst_
            rst += f'{j} '
            #rst += f'{self._outTr[i][j]}'
            for x in self._outTr[i][j]:
               x.print()
            #print(f'La transition: {self._outTr[i][j]}')
            print(rst)
         rst_ = ""
      
      print(f'\nLes transitions Entrantes du state {self._label}:\n')
      rst_=""
      for i in self._inTr.keys():
         rst_ += f'{i} : '
         for j in self._inTr[i].keys():
            rst = rst_
            rst += f'{j} '
            #rst += f'{self._inTr[i][j]}'
            for x in self._inTr[i][j]:
               x.print()
            print(rst)
         rst_ = ""

      print("\n\n")
   
   def toDot(self) :
      if (self.isAccepting()) :
         return f's_{self._id} [label="{self._label}" shape="square"]'
      else : 
         return f's_{self._id} [label="{self._label}" shape="circle"]'
      
   
   def defineTransitionOn(self, label):
      return  label in self._outTr.keys()
