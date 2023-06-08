# extract-a-Mealy-Machine-from-RNNs
Our aim is to see a an RNN can mimic the behaviour of an RNN

We try to extract a Mealy Machine from an RNN by 
- Generating random inputs and getting their outputs from a pedefined Mealy Machine
- Training a many to many RNN to 100% accuracy
- Building a prefix tree from inputs and outputs 
- Mapping states of the RNN to the states of our prefix Tree
- Merging the states of our tree

Here we have a Mealy Machine well defined.
Then we try to compare this Mealy Machine to the first one from which we got the outputs.