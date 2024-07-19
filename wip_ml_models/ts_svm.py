import tenseal as ts
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class EncryptedSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iterations=100):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.w = None
        self.b = None

    def fit(self, enc_X, enc_y):
        n_samples, n_features = len(enc_X), len(enc_X[0].decrypt())
        self.w = ts.ckks_vector(context, np.zeros(n_features))
        self.b = ts.ckks_vector(context, [0.0])

        for _ in range(self.n_iterations):
            for enc_xi, enc_yi in zip(enc_X, enc_y):
                condition = enc_yi * (enc_xi.dot(self.w) + self.b)
                if condition.decrypt()[0] >= 1:
                    self.w = self.w - self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w = self.w + self.lr * (enc_yi * enc_xi - 2 * self.lambda_param * self.w)
                    self.b = self.b + self.lr * enc_yi

    def predict(self, enc_X):
        approx_sign = lambda x: x.polyval([0, 1, 0, -4/27]) # approximation of sign function
        return [approx_sign(enc_xi.dot(self.w) + self.b) for enc_xi in enc_X]

# Generate a dataset
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X = StandardScaler().fit_transform(X)
y = np.where(y == 0, -1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create TenSEAL context
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[40, 21, 21, 21, 21, 21, 21, 40])
context.global_scale = 2**40
context.generate_galois_keys()

# Encrypt the data
enc_X_train = [ts.ckks_vector(context, xi) for xi in X_train]
enc_y_train = [ts.ckks_vector(context, [yi]) for yi in y_train]
enc_X_test = [ts.ckks_vector(context, xi) for xi in X_test]

# Train the model
svm = EncryptedSVM()
svm.fit(enc_X_train, enc_y_train)

# Make predictions
enc_predictions = svm.predict(enc_X_test)
predictions = [1 if p.decrypt()[0] > 0 else -1 for p in enc_predictions]

# Calculate accuracy
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy}")