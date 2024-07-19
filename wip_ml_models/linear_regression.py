import numpy as np
from Pyfhel import Pyfhel, PyCtxt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class EncryptedLR:
    def __init__(self, HE, n_features):
        self.HE = HE
        self.weight = np.random.randn(n_features)
        self.bias = np.random.randn()
        self._delta_w = None
        self._delta_b = None
        self._count = 0

    def forward(self, enc_x):
        enc_w = self.HE.encrypt(self.weight)
        enc_out = self.HE.dot(enc_x, enc_w)
        enc_out += self.HE.encrypt([self.bias])
        return enc_out

    def backward(self, enc_x, enc_out, enc_y):
        out_minus_y = enc_out - enc_y
        if self._delta_w is None:
            self._delta_w = self.HE.multiply(enc_x, out_minus_y)
        else:
            self._delta_w += self.HE.multiply(enc_x, out_minus_y)
        self._delta_b = out_minus_y if self._delta_b is None else self._delta_b + out_minus_y
        self._count += 1

    def update_parameters(self):
        if self._count == 0:
            raise RuntimeError("You should at least run one forward iteration")
        lr = 0.01 / self._count
        self.weight -= lr * self._delta_w.decrypt()
        self.bias -= lr * self._delta_b.decrypt()[0]
        self._delta_w = None
        self._delta_b = None
        self._count = 0

    def plain_mse(self, x_test, y_test):
        out = x_test.dot(self.weight) + self.bias
        return ((y_test - out) ** 2).mean()


housing = fetch_california_housing()
X, y = housing.data[:500], housing.target[:500]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

HE = Pyfhel()
HE.contextGen(scheme='ckks', n=8192, scale=2**40,
              qi_sizes=[40, 21, 21, 21, 21, 21, 21, 40])
HE.keyGen()
HE.relinKeyGen()
HE.rotateKeyGen()

enc_X_train = [HE.encrypt(x) for x in X_train]
enc_y_train = [HE.encrypt([y]) for y in y_train]

model = EncryptedLR(HE, X_train.shape[1])
EPOCHS = 10

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    for enc_x, enc_y in zip(enc_X_train, enc_y_train):
        enc_out = model.forward(enc_x)
        model.backward(enc_x, enc_out, enc_y)
    model.update_parameters()
    mse = model.plain_mse(X_test, y_test)
    print(f"MSE at epoch #{epoch + 1} is {mse}")

print(f"Final encrypted model MSE: {mse}")
