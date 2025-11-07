import torch
import torch.nn.functional as F

'''
A simple neural network that learns to double integers.
The network takes an integer as input (represented as a binary vector),
and outputs the doubled integer (also as a binary vector).

To double the value of an integer, we can simply perform a bit shift to the left.
For example:
Input:  00000001 (1 in decimal)
Output: 00000010 (2 in decimal)

For this reason, the number of bits in the output is
one more than the number of bits in the input.
'''

# The number of bits in the input integer
INPUT_BITS = 16

# The number of bits in the output integer
OUTPUT_BITS = INPUT_BITS + 1

# The number of examples to train on 
DATASET_SIZE = 200

# The number of training iterations
EPOCHS = 400

# How much to change the gradients by when updating weights
LEARNING_RATE = 50.0

# Print the loss for each epoch during training
PRINT_TRAINING_PROGRESS = False

PRINT_PREDICTIONS = False

# Convert to string of bits. Least significant is rightmost
def int_to_bits(n, num_bits):
    return [int(b) for b in format(n, f'0{num_bits}b')]

# Convert from bits (list) to integer
def bits_to_int(bits):
    return sum(b << i for i, b in enumerate(reversed(bits)))

# Test the bit conversion
assert int_to_bits(1, 8) == [0, 0, 0, 0, 0, 0, 0, 1]
assert int_to_bits(5, 8) == [0, 0, 0, 0, 0, 1, 0, 1]
assert int_to_bits(255, 8) == [1, 1, 1, 1, 1, 1, 1, 1]
assert int_to_bits(255, 9) == [0, 1, 1, 1, 1, 1, 1, 1, 1]


assert bits_to_int([0, 0, 0, 0, 0, 0, 0, 1]) == 1
assert bits_to_int([0, 0, 0, 0, 0, 1, 0, 1]) == 5
assert bits_to_int([1, 1, 1, 1, 1, 1, 1, 1]) == 255

# The generator for random integers in the training set
g = torch.Generator().manual_seed(808303909)

# Build training set
X = [] # Inputs
Y = [] # Outputs

for _ in range(DATASET_SIZE):
    n = torch.randint(0, 2**INPUT_BITS, (1,), generator=g).item()
    # Append the input and output bit vectors
    X.append(int_to_bits(n, INPUT_BITS))
    Y.append(int_to_bits(2 * n, OUTPUT_BITS))


# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

# Setup the neural network

# INPUT_BITS inputs to OUTPUT_BITS neurons
W1 = torch.randn((INPUT_BITS, OUTPUT_BITS), generator=g, requires_grad=True)

# Bias for each output neuron
B1 = torch.randn((OUTPUT_BITS,), generator=g, requires_grad=True)

# Training loop
# In each iteration, we will use the entire dataset to train the model
for i in range(EPOCHS):
    # Unnormalized output of the model
    logits = X @ W1 + B1

    loss = F.binary_cross_entropy_with_logits(logits, Y)

    if PRINT_TRAINING_PROGRESS:
        print(f'Loss (Epoch {i}/{EPOCHS}): {loss.item()}')

    # set the gradients to zero
    W1.grad = None

    # Update gradients
    loss.backward()

    # Update weights by moving in the direction of negative gradient (gradient descent)
    W1.data += -LEARNING_RATE * W1.grad


# Use the trained model to make predictions
errors = 0
for i in range(100):
    n = torch.randint(0, 2**INPUT_BITS, (1,), generator=g).item()
    x_bits = int_to_bits(n, INPUT_BITS)
    x_tensor = torch.tensor([x_bits], dtype=torch.float32)

    logits = x_tensor @ W1 + B1
    probs = torch.sigmoid(logits)

    # Convert probabilities to bits (0 or 1)
    predicted_bits = (probs >= 0.5).int().squeeze().tolist()
    predicted_int = bits_to_int(predicted_bits)

    if 2 * n == predicted_int:
        match = "✅ Match"
    else:
        match = "❌ Mismatch"
        errors += 1

    if PRINT_PREDICTIONS:
        print(f"Input: {n}\t-> Predicted: {predicted_int}\t {match} ({2*n})")

print("Model loss:", loss.item())
print(f"Number of predictions: {i + 1}")
print(f"Number of wrong predictions: {errors}")
print(f"Prediction error: {100 * errors/100}%")