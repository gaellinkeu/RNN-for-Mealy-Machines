
from state import State
from fsm import FSM

def buildExampleFsm0() -> FSM :
   fsm = FSM()
   states=[State(0), State(1)]
   transitions = [(0, 0, "b", "1"), (0, 1, "a", "1"), (1, 1, "a", "0"), (1, 0, "b", "0")]
   for state in states :
      fsm.addState(state)
   for transition in transitions:
      fsm.addTransition(transition[0], transition[1], transition[2], transition[3])

   return fsm

def buildExampleFsm1() -> FSM :
   fsm = FSM()
   states=[State(0), State(1), State(2)]
   transitions = [(0, 1, "b", "1"), 
                  (0, 0, "a", "0"), 
                  (1, 1, "a", "1"), 
                  (1, 2, "b", "0"), 
                  (2, 0, "b", "0"), 
                  (2, 1, "a", "1")]
   for state in states :
      fsm.addState(state)
   for transition in transitions:
      fsm.addTransition(transition[0], transition[1], transition[2], transition[3])

   return fsm

def buildExampleFsm2() -> FSM :
   fsm = FSM()
   states=[State(0), State(1), State(2), State(3)]
   transitions = [(0, 1, "b", "1"),
                (0, 3, "a", "1"),
                (1, 1, "a", "1"),
                (1, 2, "b", "1"),
                (2, 0, "a", "0"),
                (2, 3, "b", "0"),
                (3, 1, "a", "0"),
                (3, 3, "b", "0")]
   for state in states :
      fsm.addState(state)
   for transition in transitions:
       fsm.addTransition(transition[0], transition[1], transition[2], transition[3])

   return fsm

def buildExampleFsm3() -> FSM :
   fsm = FSM()
   states=[State(0), State(1), State(2), State(3), State(4)]
   transitions = [(0, 0, "a", "1"),
                (0, 3, "b", "0"),
                (1, 1, "a", "0"),
                (1, 0, "b", "1"),
                (2, 4, "a", "1"),
                (2, 4, "b", "0"),
                (3, 4, "a", "1"),
                (3, 3, "b", "1"),
                (4, 2, "a", "0"),
                (4, 3, "b", "0")]
   for state in states :
      fsm.addState(state)
   for transition in transitions:
       fsm.addTransition(transition[0], transition[1], transition[2], transition[3])

   return fsm

def buildExampleFsm4() -> FSM :
   fsm = FSM()
   states=[State(0), State(1), State(2), State(3), State(4), State(5)]
   transitions = [(0, 2, "a", "0"),
                (0, 5, "b", "1"),
                (1, 1, "a", "1"),
                (1, 5, "b", "0"),
                (2, 4, "a", "0"),
                (2, 5, "b", "0"),
                (3, 2, "a", "0"),
                (3, 0, "b", "0"),
                (4, 4, "a", "0"),
                (4, 1, "b", "0"),
                (5, 3, "a", "0"),
                (5, 4, "b", "1")]
   for state in states :
      fsm.addState(state)
   for transition in transitions:
       fsm.addTransition(transition[0], transition[1], transition[2], transition[3])

   return fsm

def buildExampleFsm5() -> FSM :
   fsm = FSM()
   states=[State(0), State(1), State(2), State(3), State(4), State(5), State(6)]
   transitions = [(0, 2, "a", "1"),
                (0, 6, "b", "1"),
                (1, 4, "a", "0"),
                (1, 0, "b", "1"),
                (2, 4, "a", "1"),
                (2, 6, "b", "0"),
                (3, 1, "a", "0"),
                (3, 3, "b", "1"),
                (4, 5, "a", "1"),
                (4, 1, "b", "1"),
                (5, 0, "a", "1"),
                (5, 1, "b", "1"),
                (6, 4, "a", "1"),
                (6, 2, "b", "0")]
   for state in states :
      fsm.addState(state)
   for transition in transitions:
       fsm.addTransition(transition[0], transition[1], transition[2], transition[3])

   return fsm

def buildExampleFsm6() -> FSM :
   fsm = FSM()
   states=[State(0), State(1), State(2), State(3), State(4), State(5), State(6), State(7)]
   transitions = [(0, 1, "a", "1"),
                (0, 2, "b", "0"),
                (1, 0, "a", "1"),
                (1, 3, "b", "0"),
                (2, 0, "a", "1"),
                (2, 7, "b", "0"),
                (3, 7, "a", "1"),
                (3, 7, "b", "1"),
                (4, 6, "a", "1"),
                (4, 7, "b", "1"),
                (5, 1, "a", "0"),
                (5, 2, "b", "1"),
                (6, 6, "a", "1"),
                (6, 6, "b", "1"),
                (7, 0, "a", "0"),
                (7, 7, "b", "0")]
   for state in states :
      fsm.addState(state)
   for transition in transitions:
       fsm.addTransition(transition[0], transition[1], transition[2], transition[3])

   return fsm

def buildExampleFsm7() -> FSM :
   fsm = FSM()
   states=[State(0), State(1), State(2), State(3), State(4), State(5), State(6), State(7), State(8)]
   transitions = [(0, 3, "a", "0"),
                (0, 2, "b", "0"),
                (1, 2, "a", "0"),
                (1, 7, "b", "0"),
                (2, 5, "a", "0"),
                (2, 7, "b", "1"),
                (3, 5, "a", "1"),
                (3, 0, "b", "1"),
                (4, 6, "a", "0"),
                (4, 8, "b", "1"),
                (5, 0, "a", "1"),
                (5, 0, "b", "1"),
                (6, 4, "a", "0"),
                (6, 6, "b", "0"),
                (7, 2, "a", "1"),
                (7, 2, "b", "0"),
                (8, 3, "a", "0"),
                (8, 7, "b", "0")]
   for state in states :
      fsm.addState(state)
   for transition in transitions:
       fsm.addTransition(transition[0], transition[1], transition[2], transition[3])

   return fsm

def buildExampleFsm8() -> FSM :
   fsm = FSM()
   states=[State(0), State(1), State(2), State(3), State(4), State(5), State(6), State(7), State(8), State(9)]
   transitions = [(0, 2, "a", "0"),
                (0, 5, "b", "0"),
                (1, 2, "a", "1"),
                (1, 0, "b", "1"),
                (2, 0, "a", "0"),
                (2, 7, "b", "0"),
                (3, 8, "a", "1"),
                (3, 8, "b", "0"),
                (4, 7, "a", "1"),
                (4, 0, "b", "1"),
                (5, 0, "a", "0"),
                (5, 2, "b", "0"),
                (6, 4, "a", "1"),
                (6, 8, "b", "0"),
                (7, 5, "a", "1"),
                (7, 2, "b", "0"),
                (8, 4, "a", "0"),
                (8, 9, "b", "0"),
                (9, 7, "a", "0"),
                (9, 9, "b", "0")]
   for state in states :
      fsm.addState(state)
   for transition in transitions:
       fsm.addTransition(transition[0], transition[1], transition[2], transition[3])

   return fsm

def buildExampleFsm9() -> FSM :
   fsm = FSM()
   states=[State(0), State(1), State(2), State(3), State(4), State(5), State(6), State(7), State(8), State(9), State(10)]
   transitions = [(0, 7, "a", "0"),
                (0, 5, "b", "1"),
                (1, 6, "a", "1"),
                (1, 9, "b", "0"),
                (2, 0, "a", "1"),
                (2, 2, "b", "0"),
                (3, 0, "a", "0"),
                (3, 3, "b", "0"),
                (4, 4, "a", "1"),
                (4, 8, "b", "1"),
                (5, 7, "a", "1"),
                (5, 7, "b", "1"),
                (6, 1, "a", "0"),
                (6, 1, "b", "0"),
                (7, 5, "a", "1"),
                (7, 4, "b", "0"),
                (8, 4, "a", "1"),
                (8, 8, "b", "1"),
                (9, 6, "a", "0"),
                (9, 1, "b", "1"),
                (10, 2, "a", "1"),
                (10, 5, "b", "0")]
   for state in states :
      fsm.addState(state)
   for transition in transitions:
       fsm.addTransition(transition[0], transition[1], transition[2], transition[3])

   return fsm