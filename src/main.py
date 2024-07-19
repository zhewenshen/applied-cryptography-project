from client import Client
from server import Server
import numpy as np
import random

if __name__ == "__main__":
    client = Client()
    server = Server()

    datasets = [
        [random.uniform(0, 100) for _ in range(347)],
        [random.uniform(0, 100) for _ in range(443)],
        [random.uniform(0, 100) for _ in range(42)]
    ]

    all_data = []

    for i, data in enumerate(datasets):
        print(f"\nDataset {i + 1}:")
        # print(f"Original data: {data}")

        response = client.send_request(server, {
            'action': 'store',
            'key': f'dataset_{i}',
            'data': data,
            'size': len(data)
        })
        print(f"Store response: {response}")

        response = client.send_request(server, {
            'action': 'compute_average',
            'key': f'dataset_{i}'
        })
        print(f"Compute average response: {response}")
        print(f"Actual average: {np.mean(data)}")

        all_data.extend(data)

    keys = [f'dataset_{i}' for i in range(len(datasets))]
    response = client.send_request(server, {
        'action': 'compute_overall_average',
        'keys': keys
    })
    print(f"\nCompute overall average response: {response}")
    print(f"Actual overall average: {np.mean(all_data)}")