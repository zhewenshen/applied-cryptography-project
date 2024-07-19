import torch
import tenseal as ts
import random
from time import time
import numpy as np
from tqdm import tqdm
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

print("Importing libraries and setting up...")

torch.random.manual_seed(73)
random.seed(73)
np.random.seed(73)

print("Loading Boston Housing dataset...")
boston = load_boston()
X, y = boston.data, boston.target

print(f"Full dataset shape: X: {X.shape}, y: {y.shape}")

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to PyTorch tensors
X = torch.FloatTensor(X)
y = torch.FloatTensor(y).unsqueeze(1)

print(f"Final data shape: X: {X.shape}, y: {y.shape}")

print("Defining Linear Regression model...")

class LR(torch.nn.Module):
    def __init__(self, n_features):
        super(LR, self).__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)

n_features = X.shape[1]
model = LR(n_features)
optim = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

EPOCHS = 100
BATCH_SIZE = 32

print("Training client-side plaintext Linear Regression model...")

def train_plaintext(model, optim, criterion, x, y, epochs=EPOCHS, batch_size=BATCH_SIZE):
    for e in tqdm(range(1, epochs + 1), desc="Training epochs"):
        for i in range(0, len(x), batch_size):
            batch_x = x[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            optim.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optim.step()
    return model

plaintext_model = train_plaintext(model, optim, criterion, X, y)

def mse(model, x, y):
    out = model(x)
    return ((y - out) ** 2).mean().item()

plain_mse = mse(plaintext_model, X, y)
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
        return enc_x.mm(self.weight) + self.bias

    def backward(self, enc_x, enc_out, enc_y):
        out_minus_y = (enc_out - enc_y)
        self._delta_w = enc_x.transpose().mm(out_minus_y) if self._delta_w is None else self._delta_w + enc_x.transpose().mm(out_minus_y)
        self._delta_b = out_minus_y.sum() if self._delta_b is None else self._delta_b + out_minus_y.sum()
        self._count += enc_x.shape[0]

    def update_parameters(self):
        if self._count == 0:
            raise RuntimeError("You should at least run one forward iteration")
        lr = 0.01 / self._count
        self.weight -= lr * self._delta_w
        self.bias -= lr * self._delta_b
        self._delta_w = None
        self._delta_b = None
        self._count = 0

    def plain_mse(self, x, y):
        w = torch.tensor(self.weight.decrypt() if isinstance(self.weight, ts.CKKSTensor) else self.weight)
        b = torch.tensor(self.bias.decrypt() if isinstance(self.bias, ts.CKKSTensor) else self.bias)
        out = x.matmul(w) + b
        return ((y - out) ** 2).mean().item()

    def encrypt(self, context):
        self.weight = ts.ckks_tensor(context, self.weight)
        self.bias = ts.ckks_tensor(context, [self.bias])

    def decrypt(self):
        if isinstance(self.weight, ts.CKKSTensor):
            self.weight = self.weight.decrypt()
        if isinstance(self.bias, ts.CKKSTensor):
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
enc_X = ts.ckks_tensor(ctx_training, X)
enc_y = ts.ckks_tensor(ctx_training, y)

eelr = EncryptedLR(LR(n_features))
initial_mse = eelr.plain_mse(X, y)
print(f"Initial encrypted model MSE: {initial_mse}")

print("Training encrypted Linear Regression model...")
times = []
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    eelr.encrypt(ctx_training)

    t_start = time()
    for i in tqdm(range(0, len(X), BATCH_SIZE), desc="Training"):
        enc_x_batch = enc_X[i:i+BATCH_SIZE]
        enc_y_batch = enc_y[i:i+BATCH_SIZE]
        enc_out = eelr.forward(enc_x_batch)
        eelr.backward(enc_x_batch, enc_out, enc_y_batch)
    eelr.update_parameters()
    t_end = time()
    times.append(t_end - t_start)

    eelr.decrypt()
    mse = eelr.plain_mse(X, y)
    print(f"MSE at epoch #{epoch + 1} is {mse}")

print(f"\nAverage time per epoch: {int(sum(times) / len(times))} seconds")
print(f"Final encrypted model MSE: {mse}")
print(f"Difference between client-side plaintext and encrypted MSEs: {abs(plain_mse - mse)}")

# Show predictions for encrypted model
print("\nSample predictions from encrypted model:")
w = torch.tensor(eelr.weight)
b = torch.tensor(eelr.bias)
for i in range(5):  # Show first 5 samples
    input_features = X[i]
    true_price = y[i].item() * 1000
    predicted_price = (input_features.dot(w) + b).item() * 1000
    print(f"Input features: {input_features.numpy()}")
    print(f"True house price: ${true_price:.2f}")
    print(f"Predicted house price: ${predicted_price:.2f}")
    print(f"Absolute error: ${abs(true_price - predicted_price):.2f}")
    print()