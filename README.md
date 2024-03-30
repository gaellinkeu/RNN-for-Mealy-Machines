# Extract Mealy Machine from a RNN

An approach to get a Mealy Machine from a RNN that can imitate this later perfectly.

## Project description

Our aim is to see extract a Melay Machine from a trained RNN

We implement the **Prefix Tree states Merging** approach.

Extract a Mealy Machine from an RNN using the state merging follow this pattern 

1 - Generating random inputs and getting their outputs from a pedefined Mealy Machine
2 - Training a many to many RNN to 100% accuracy (or close to 100%)
3 - Building a prefix tree from inputs and outputs 
4 - Mapping states of the RNN got from predicting outputs to the states of the prefix Tree
5 - Merging the nodes (states) of the tree
6 - Minimize the tree (for better representation)

The final merged Tree is what we call the extracted Mealy Machine from the RNN. Some say that, this MM simplify all the processings done by the RNN to predict an output.


## Challenges

We faced many challenges when trying to address this extraction problem
1 - The algorithm complexity of the approach is exponential, so we had to used few data compared to what we had
2 - The similarity distances used here is not yet proven to be the best for this problem
3 - Define the best (optimal) merging policy that will keep the determinism of the tree whhile taking into account similarity between nodes based on RNNs mapped to them.

## Run

To run this project, you have to first make sure you have all the packages listed in the __requirements.txt__ file. and then execute the __main.py__ by running __
```bash
python main.py
```

## Notes

Being a first attempt of such a task on a many-to-many RNN, we experiment on a very trivial dataset.
The dataset is composed of input and output sequences. Inputs being constructed from an {a,b} alphabet and outputs on {0,1}.

## Authors ans acknowledgement

This project is a collaboration between
- **Gaël LINKEU** - University of Yaounde 1 
- **Omer Nguena** - Unuversité du Quebec en Outaouais 
- **Norbert TSOPZE** - University of Yaounde 1