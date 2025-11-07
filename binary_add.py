import torch
import torch.nn.functional as F

'''
A two layer neural network that learns to add a constant value to an integer.
The network takes an integer as input (represented as a binary vector),
and outputs the integer plus the constant value (also as a binary vector).

The number of bits in the output needs to be greater than the number
of bits in the input to account for overflow when adding.
'''

# The number to add to the input integer to get the output integer
NUMBER_TO_ADD = 2

# The number of bits in the input integer
INPUT_BITS = 16

# The number of bits in the output integer
OUTPUT_BITS = INPUT_BITS + 1

# The number of neurons in the hidden layer
# Changing from 32 to 64 improved performance significantly
HIDDEN_LAYERS_SIZE = 64

# The number of examples to train on
DATASET_SIZE = 4000

# Batch size for training
BATCH_SIZE = 128

# The number of training iterations
EPOCHS = 4000

# How much to change the gradients by when updating weights
LEARNING_RATE = 0.6
# With shorter outputs
# 0.5 -> 0.02830662950873375 28% fejl
# 0.6 -> 0.0029239095747470856 5% fejl
# 1.0 -> 0.01887655071914196 15% fejl

# Print the loss for each epoch during training
PRINT_TRAINING_PROGRESS = True

PRINT_PREDICTIONS = True

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
    Y.append(int_to_bits(n + NUMBER_TO_ADD, OUTPUT_BITS))


# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

# Setup the neural network

# INPUT_BITS inputs to HIDDEN_LAYERS_SIZE neurons
W1 = torch.randn((INPUT_BITS, HIDDEN_LAYERS_SIZE), generator=g, requires_grad=True)

# Bias for first hidden layer
B1 = torch.randn((HIDDEN_LAYERS_SIZE,), generator=g, requires_grad=True)

# Second hidden layer
W2 = torch.randn((HIDDEN_LAYERS_SIZE, HIDDEN_LAYERS_SIZE), generator=g, requires_grad=True)
B2 = torch.randn((HIDDEN_LAYERS_SIZE,), generator=g, requires_grad=True)

# Output layer
W3 = torch.randn((HIDDEN_LAYERS_SIZE, OUTPUT_BITS), generator=g, requires_grad=True)
B3 = torch.randn((OUTPUT_BITS,), generator=g, requires_grad=True)

parameters = [W1, B1, W2, B2, W3, B3]

# Training loop
# In each epoch, we process the dataset in batches
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    num_batches = 0
    
    # Process dataset in batches
    for batch_start in range(0, DATASET_SIZE, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, DATASET_SIZE)
        X_batch = X[batch_start:batch_end]
        Y_batch = Y[batch_start:batch_end]
        
        # Forward pass
        hidden1 = torch.relu(X_batch @ W1 + B1)
        hidden2 = torch.relu(hidden1 @ W2 + B2)
        logits = hidden2 @ W3 + B3

        loss = F.binary_cross_entropy_with_logits(logits, Y_batch)
        epoch_loss += loss.item()
        num_batches += 1

        # set the gradients to zero
        for p in parameters:
            p.grad = None

        # Update gradients
        loss.backward()

        # Update weights by moving in the direction of negative gradient (gradient descent)
        for p in parameters:
            p.data += -LEARNING_RATE * p.grad

    if PRINT_TRAINING_PROGRESS:
        avg_loss = epoch_loss / num_batches
        print(f'Loss (Epoch {epoch+1}/{EPOCHS}):\t{avg_loss}')


# Use the trained model to make predictions
errors = 0
for i in range(100):
    n = torch.randint(0, 2**INPUT_BITS, (1,), generator=g).item()
    correct_int = n + NUMBER_TO_ADD
    x_bits = int_to_bits(n, INPUT_BITS)
    x_tensor = torch.tensor([x_bits], dtype=torch.float32)

    hidden1 = torch.relu(x_tensor @ W1 + B1)
    hidden2 = torch.relu(hidden1 @ W2 + B2)
    logits = hidden2 @ W3 + B3
    probs = torch.sigmoid(logits)

    # Convert probabilities to bits (0 or 1)
    predicted_bits = (probs >= 0.5).int().squeeze().tolist()
    predicted_int = bits_to_int(predicted_bits)

    if correct_int == predicted_int:
        match = "✅ Match"
    else:
        match = "❌ Mismatch"
        errors += 1

    if PRINT_PREDICTIONS:
        print(f"Input: {n}\t-> Predicted: {predicted_int}\t {match} ({correct_int})")

print("Model loss:", loss.item())
print(f"Number of predictions: {i + 1}")
print(f"Number of wrong predictions: {errors}")
print(f"Prediction error: {100 * errors/100}%")