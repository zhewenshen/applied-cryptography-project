import torch
import tenseal as ts
import random
from time import time
import numpy as np
from tqdm import tqdm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Importing libraries and setting up...")

torch.random.manual_seed(73)
random.seed(73)
np.random.seed(73)

print("Loading Iris dataset...")
iris = load_iris()
X, y = iris.data, iris.target

# Use only two classes for binary classification
X = X[y != 2]
y = y[y != 2]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).unsqueeze(1)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Display a sample of testing input and output labels
print("\nSample of testing data:")
for i in range(5):  # Show first 5 samples
    print(f"Input: {X_test[i].numpy()}")
    print(f"Label: {y_test[i].item()}")
    print()

print("Defining Logistic Regression model...")

class LR(torch.nn.Module):
    def __init__(self, n_features):
        super(LR, self).__init__()
        self.linear = torch.nn.Linear(n_features, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

n_features = X_train.shape[1]
model = LR(n_features)
optim = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = torch.nn.BCELoss()

EPOCHS = 5

print("Training client-side plaintext Logistic Regression model...")

def train_plaintext(model, optim, criterion, x, y, epochs=EPOCHS):
    for e in tqdm(range(1, epochs + 1), desc="Training epochs"):
        optim.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optim.step()
    return model

plaintext_model = train_plaintext(model, optim, criterion, X_train, y_train)

def accuracy(model, x, y):
    out = model(x)
    return ((out > 0.5).float() == y).float().mean().item()

plain_accuracy = accuracy(plaintext_model, X_test, y_test)
print(f"Client-side plaintext model accuracy: {plain_accuracy:.4f}")

print("Defining Encrypted Logistic Regression model...")

class EncryptedLR:
    
    def __init__(self, torch_lr):
        self.weight = torch_lr.linear.weight.data.tolist()[0]
        self.bias = torch_lr.linear.bias.data.tolist()
        # we accumulate gradients and counts the number of iterations
        self._delta_w = 0
        self._delta_b = 0
        self._count = 0
        
    def forward(self, enc_x):
        enc_out = enc_x.dot(self.weight) + self.bias
        enc_out = EncryptedLR.sigmoid(enc_out)
        return enc_out
    
    def backward(self, enc_x, enc_out, enc_y):
        out_minus_y = (enc_out - enc_y)
        self._delta_w += enc_x * out_minus_y
        self._delta_b += out_minus_y
        self._count += 1
        
    def update_parameters(self):
        if self._count == 0:
            raise RuntimeError("You should at least run one forward iteration")
        # update weights
        # We use a small regularization term to keep the output
        # of the linear layer in the range of the sigmoid approximation
        self.weight -= self._delta_w * (1 / self._count) + self.weight * 0.05
        self.bias -= self._delta_b * (1 / self._count)
        # reset gradient accumulators and iterations count
        self._delta_w = 0
        self._delta_b = 0
        self._count = 0
    
    @staticmethod
    def sigmoid(enc_x):
        # We use the polynomial approximation of degree 3
        # sigmoid(x) = 0.5 + 0.197 * x - 0.004 * x^3
        # from https://eprint.iacr.org/2018/462.pdf
        # which fits the function pretty well in the range [-5,5]
        return enc_x.polyval([0.5, 0.197, 0, -0.004])
    
    def plain_accuracy(self, x_test, y_test):
        # evaluate accuracy of the model on
        # the plain (x_test, y_test) dataset
        w = torch.tensor(self.weight)
        b = torch.tensor(self.bias)
        out = torch.sigmoid(x_test.matmul(w) + b).reshape(-1, 1)
        correct = torch.abs(y_test - out) < 0.5
        return correct.float().mean()    
    
    def encrypt(self, context):
        self.weight = ts.ckks_vector(context, self.weight)
        self.bias = ts.ckks_vector(context, self.bias)
        
    def decrypt(self):
        self.weight = self.weight.decrypt()
        self.bias = self.bias.decrypt()
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

print("Setting up TenSEAL context...")
poly_mod_degree = 8192
coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]
ctx_training = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
ctx_training.global_scale = 2 ** 21
ctx_training.generate_galois_keys()

print("Encrypting training data...")
enc_X_train = [ts.ckks_vector(ctx_training, x.tolist()) for x in tqdm(X_train, desc="Encrypting X_train")]
enc_y_train = [ts.ckks_vector(ctx_training, [y.item()]) for y in tqdm(y_train, desc="Encrypting y_train")]

eelr = EncryptedLR(LR(n_features))
initial_accuracy = eelr.plain_accuracy(X_test, y_test)
print(f"Initial encrypted model accuracy: {initial_accuracy:.4f}")

print("Training encrypted Logistic Regression model...")
times = []
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    eelr.encrypt(ctx_training)
    
    t_start = time()
    for enc_x, enc_y in tqdm(zip(enc_X_train, enc_y_train), total=len(enc_X_train), desc="Training"):
        enc_out = eelr.forward(enc_x)
        eelr.backward(enc_x, enc_out, enc_y)
    eelr.update_parameters()
    t_end = time()
    times.append(t_end - t_start)
    
    eelr.decrypt()
    accuracy = eelr.plain_accuracy(X_test, y_test)
    print(f"Accuracy at epoch #{epoch + 1} is {accuracy:.4f}")

print(f"\nAverage time per epoch: {int(sum(times) / len(times))} seconds")
print(f"Final encrypted model accuracy: {accuracy:.4f}")
print(f"Difference between client-side plaintext and encrypted accuracies: {abs(plain_accuracy - accuracy):.4f}")

# Display predictions for the first 5 test samples
print("\nPredictions for the first 5 test samples:")
w = torch.tensor(eelr.weight)
b = torch.tensor(eelr.bias)
for i in range(5):
    input_data = X_test[i]
    true_label = y_test[i].item()
    prediction = torch.sigmoid(input_data.dot(w) + b).item()
    predicted_class = 1 if prediction > 0.5 else 0
    print(f"Input: {input_data.numpy()}")
    print(f"True label: {true_label}")
    print(f"Predicted probability: {prediction:.4f}")
    print(f"Predicted class: {predicted_class}")
    print(f"Correct: {'Yes' if predicted_class == true_label else 'No'}")
    print()