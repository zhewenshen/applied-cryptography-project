import random
import tenseal as ts


class EncryptedLR:
    def __init__(self, n_features):
        self.weight = [random.uniform(-0.1, 0.1) for _ in range(n_features)]
        self.bias = random.uniform(-0.1, 0.1)
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
        lr = 0.01
        self.weight -= lr * self._delta_w
        self.bias -= lr * self._delta_b
        self._delta_w = None
        self._delta_b = None
        self._count = 0

    def get_parameters(self):
        return {
            'weight': self.weight.serialize(),
            'bias': self.bias.serialize()
        }

    def set_parameters(self, params):
        self.weight = ts.ckks_vector_from(params['context'], params['weight'])
        self.bias = ts.ckks_vector_from(params['context'], params['bias'])
