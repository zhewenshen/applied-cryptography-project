import torch
import tenseal as ts
import random
from time import time
import numpy as np
from tqdm import tqdm
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Importing libraries and setting up...")

torch.random.manual_seed(73)
random.seed(73)
np.random.seed(73)

print("Loading California Housing dataset...")
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Use only the first 500 data points
X = X[:500]
y = y[:500]

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

print("Defining Linear Regression model...")


class LR(torch.nn.Module):
    def __init__(self, n_features):
        super(LR, self).__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)


n_features = X_train.shape[1]
model = LR(n_features)
optim = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

EPOCHS = 10

print("Training client-side plaintext Linear Regression model...")


def train_plaintext(model, optim, criterion, x, y, epochs=EPOCHS):
    for e in tqdm(range(1, epochs + 1), desc="Training epochs"):
        optim.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optim.step()
    return model


plaintext_model = train_plaintext(model, optim, criterion, X_train, y_train)


def mse(model, x, y):
    out = model(x)
    return ((y - out) ** 2).mean().item()


plain_mse = mse(plaintext_model, X_test, y_test)
print(f"Client-side plaintext model MSE: {plain_mse}")

print("Defining Encrypted Linear Regression model...")


class EncryptedLR:
    def __init__(self, torch_lr):
        self.weight = torch_lr.linear.weight.data.tolist()[0]
        self.bias = torch_lr.linear.bias.data.tolist()[0]
        self._delta_w = None
        self._delta_b = None
        self._count = 0

    def forward(self, enc_x):
        return enc_x.dot(self.weight) + self.bias

    def backward(self, enc_x, enc_out, enc_y):
        out_minus_y = (enc_out - enc_y)
        self._delta_w = enc_x * \
            out_minus_y if self._delta_w is None else self._delta_w + enc_x * out_minus_y
        self._delta_b = out_minus_y if self._delta_b is None else self._delta_b + out_minus_y
        self._count += 1

    def update_parameters(self):
        if self._count == 0:
            raise RuntimeError("You should at least run one forward iteration")
        lr = 0.01 / self._count
        self.weight -= lr * self._delta_w
        self.bias -= lr * self._delta_b
        self._delta_w = None
        self._delta_b = None
        self._count = 0

    def plain_mse(self, x_test, y_test):
        w = torch.tensor(self.weight.decrypt() if isinstance(
            self.weight, ts.CKKSVector) else self.weight)
        b = torch.tensor(self.bias.decrypt()[0] if isinstance(
            self.bias, ts.CKKSVector) else self.bias)
        out = x_test.matmul(w) + b
        return ((y_test - out) ** 2).mean().item()

    def encrypt(self, context):
        self.weight = ts.ckks_vector(context, self.weight)
        self.bias = ts.ckks_vector(context, [self.bias])

    def decrypt(self):
        if isinstance(self.weight, ts.CKKSVector):
            self.weight = self.weight.decrypt()
        if isinstance(self.bias, ts.CKKSVector):
            self.bias = self.bias.decrypt()[0]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


print("Setting up TenSEAL context...")
poly_mod_degree = 8192
coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]
ctx_training = ts.context(
    ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
ctx_training.global_scale = 2 ** 21
ctx_training.generate_galois_keys()

print("Encrypting training data...")
enc_X_train = [ts.ckks_vector(ctx_training, x.tolist())
               for x in tqdm(X_train, desc="Encrypting X_train")]
enc_y_train = [ts.ckks_vector(ctx_training, y.tolist())
               for y in tqdm(y_train, desc="Encrypting y_train")]

eelr = EncryptedLR(LR(n_features))
initial_mse = eelr.plain_mse(X_test, y_test)
print(f"Initial encrypted model MSE: {initial_mse}")

print("Training encrypted Linear Regression model...")
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
    mse = eelr.plain_mse(X_test, y_test)
    print(f"MSE at epoch #{epoch + 1} is {mse}")

print(f"\nAverage time per epoch: {int(sum(times) / len(times))} seconds")
print(f"Final encrypted model MSE: {mse}")
print(
    f"Difference between client-side plaintext and encrypted MSEs: {abs(plain_mse - mse)}")
