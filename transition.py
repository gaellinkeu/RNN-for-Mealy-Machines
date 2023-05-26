# author : Omer Nguena Timo
# version : v0
# use : research only

import random


class Transition :
    def __init__(self, src, tgt, input=None, output=None, id=-1) -> None:
        self._src = src 
        self._tgt = tgt 
        self._input = input
        self._output = output
        self._id = id
    
    def setID(self, id) :
      self._id = id

    def getID(self) :
      return self._id
    
    def getInput(self) :
       return self._input
    
    def getOutput(self) :
       return self._output
    
    def __str__(self) -> str:
        rst = "\n\t" + f"s_{self._src.getID()} -> s_{self._tgt.getID()} "
        rst+= f'[label="{self._input}/{self._output}"]'
        return rst
    
    def toDot(self) -> str :
        rst = "\n\t" + f"s_{self._src.getID()} -> s_{self._tgt.getID()} "
        rst+= f'[label="{self._input}/{self._output}"]'
        return rst
    
    def print(self):
      print(f'({self._src._label}, {self._input}, {self._output}, {self._tgt._label})')
   
    

