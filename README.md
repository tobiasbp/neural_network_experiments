# Neural network experiments
A few simple experiments with neural networks (NN) in PyTorch.

# Run the examples
Descriptions of the examples in this repository.
## _binary_double.py_
A single layer neural network that learns to double the value of integers.
As this is a _bit shift left_, it's a very simple pattern.

### How it works
1. Takes an integer as input, represented as a binary vector (list of 0s and 1s)
2. Passes it through a single linear layer.
3. Outputs the doubled integer, also as a binary vector

The input uses 16 bits (can represent 0-65535), while the output uses 17 bits (needed because doubling can produce values up to 131070). During training, the network learns the weight matrix that transforms input bits to output bits, effectively learning the bit shift pattern.


# How to set up the virtual environment
Go to the root of the repository, and run these commands: 
1. Setup a virtual Python environment in the _.venv_ dir: `python -m venv .venv`
2. Activate the virtual environment:
   * Linux: `source .venv/bin/activate`
   * Windows: `.\.venv\Scripts\activate`
3. Install dependencies: `pip3 intstall -r requirements.txt`
4. Run a file: `python some_file.py`
5. When done, leave the virtual environment: `deactivate`
