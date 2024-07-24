import random
import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from client import Client
from server import Server


@pytest.fixture
def client_server():
    client = Client()
    server = Server()
    return client, server


def print_server_response(message):
    print(f"\n\t[SERVER] {message}")


def test_statistical_computations(client_server, capsys):
    client, server = client_server
    datasets = [
        [random.uniform(0, 100) for _ in range(347)],
        [random.uniform(0, 100) for _ in range(443)],
        [random.uniform(0, 100) for _ in range(42)]
    ]

    all_data = []

    for i, data in enumerate(datasets):
        response = client.send_request(server, {
            'action': 'store',
            'key': f'dataset_{i}',
            'request_type': 'normal',
            'data': data,
            'size': len(data)
        })
        print_server_response(f"Store response: {response}")
        assert response['status'] == 'success'

        response = client.send_request(server, {
            'action': 'compute_average',
            'request_type': 'normal',
            'key': f'dataset_{i}'
        })
        print_server_response(f"Compute average response: {response}")
        assert response['status'] == 'success'
        assert abs(response['result'] - np.mean(data)) <= 1

        all_data.extend(data)

    keys = [f'dataset_{i}' for i in range(len(datasets))]
    response = client.send_request(server, {
        'action': 'compute_overall_average',
        'request_type': 'normal',
        'keys': keys
    })
    print_server_response(f"Compute overall average response: {response}")
    assert response['status'] == 'success'
    assert abs(response['result'] - np.mean(all_data)) <= 1

    response = client.send_request(server, {
        'action': 'compute_variance',
        'request_type': 'normal',
        'key': 'dataset_1'
    })
    print_server_response(f"Compute variance response: {response}")
    assert response['status'] == 'success'
    assert abs(response['result'] - np.var(datasets[1])) <= 1

    response = client.send_request(server, {
        'action': 'sd',
        'request_type': 'normal',
        'key': 'dataset_2'
    })
    print_server_response(f"Compute standard deviation response: {response}")
    assert response['status'] == 'error'
    assert response['message'] == 'Not implemented'
    # assert abs(response['result'] - np.std(datasets[2])) <= 1


def test_machine_learning(client_server, capsys):
    client, server = client_server
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
    print_server_response(f"Store training data response: {response}")

    init_request = {
        'action': 'initialize_model',
        'request_type': 'ml',
        'key': model_key,
        'n_features': X_train.shape[1]
    }

    response = client.send_request(server, init_request)
    print_server_response(f"Initialize model response: {response}")

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
    print_server_response(f"Prediction response: {response}")
