import random
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from client import Client
from server import Server
import pysodium as nacl
from utils import serialize, deserialize, dbg_break


def test_statistical_computations(client, server):
    datasets = [
        [random.uniform(0, 6453) for _ in range(347)],
        [random.uniform(0, 6453) for _ in range(443)],
        [random.uniform(0, 6453) for _ in range(42)]
    ]

    all_data = []

    for i, data in enumerate(datasets):
        print(f"\nDataset {i + 1}:")

        response = client.send_request(server, {
            'action': 'store',
            'key': f'dataset_{i}',
            'request_type': 'normal',
            'data': data,
            'size': len(data)
        })
        print(f"Store response: {response}")

        response = client.send_request(server, {
            'action': 'compute_average',
            'request_type': 'normal',
            'key': f'dataset_{i}'
        })
        print(f"Compute average response: {response}")
        print(f"Actual average: {np.mean(data)}")

        all_data.extend(data)

    keys = [f'dataset_{i}' for i in range(len(datasets))]
    response = client.send_request(server, {
        'action': 'compute_overall_average',
        'request_type': 'normal',
        'keys': keys
    })
    print(f"\nCompute overall average response: {response}")
    print(f"Actual overall average: {np.mean(all_data)}")

    response = client.send_request(server, {
        'action': 'compute_variance',
        'request_type': 'normal',
        'key': 'dataset_1'
    })
    print(f"\nCompute variance response: {response}")
    print(f"Actual variance: {np.var(datasets[1])}")


def test_machine_learning(client, server):
    california = fetch_california_housing()
    X, y = california.data, california.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=10, test_size=1, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model_key = 'california_model'
    client.test_data[model_key] = {'x': X_test, 'y': y_test}

    store_request = {
        'action': 'store_training_data',
        'request_type': 'ml',
        'key': model_key,
        'training_data': {'x': X_train.tolist(), 'y': y_train.tolist()}
    }

    response = client.send_request(server, store_request)
    print("\nStore training data response:", response)

    init_request = {
        'action': 'initialize_model',
        'request_type': 'ml',
        'key': model_key,
        'n_features': X_train.shape[1]
    }

    response = client.send_request(server, init_request)
    print("Initialize model response:", response)

    num_epochs = 2
    client.train_model(server, model_key, num_epochs)

    sample_test = X_test[0].tolist()
    predict_request = {
        'action': 'predict',
        'request_type': 'inference',
        'key': model_key,
        'inference_data': {'x': [sample_test]}
    }

    response = client.send_request(server, predict_request)
    print("Prediction response:", response)

    if response['status'] == 'success':
        print(f"Actual value: {y_test[0]}")


if __name__ == "__main__":
    # This is just a simulation of the protocol
    # In a real-world scenario, the client and server would be running on different machines and communicating over a secure channel
    # The ML model would need to be trained on a full sized dataset with many epochs
    # it is just a toy example here - hence inference is not accurate
    
    client = Client()
    server = Server()

    (client_pk, client_sk) = nacl.crypto_kx_keypair()
    (server_pk, server_sk) = nacl.crypto_kx_keypair()

    client.set_client_key_pair(client_pk, client_sk)
    client.set_server_pk(server_pk)

    server.set_server_key_pair(server_pk, server_sk)
    server.set_client_pk(client_pk)

    client.hello(server)

    print("Testing Statistical Computations:")
    test_statistical_computations(client, server)

    print("\nTesting Machine Learning:")
    test_machine_learning(client, server)
