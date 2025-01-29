import multiprocessing
import subprocess
import sys
import itertools
import grpc

import fl_proto_pb2 as fl_pb2
import fl_proto_pb2_grpc as fl_pb2_grpc

def run_client(script_name, client_id, scenario, order_id):
    subprocess.run([
        sys.executable, script_name,
        "--client_id", client_id,
        "--scenario", scenario,
        "--order_id", str(order_id)
    ])

def reset_global_model():
    """
    Calls the RPC to reset the global model on the FedAvg server.
    """
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = fl_pb2_grpc.FederatedLearningServiceStub(channel)
        stub.ResetGlobalModel(fl_pb2.Empty())  # or google.protobuf.Empty() if using that

if __name__ == '__main__':
    scenario_set = ["HIGH", "HIGH", "HIGH", "HIGH", "HIGH"]
    client_script = "fashion_mnist_client_H.py"

    all_permutations = list(itertools.permutations([1, 2, 3, 4, 5]))

    for order_index, permutation in enumerate(all_permutations, start=1):
        print(f"\n=== Starting permutation #{order_index}: {permutation} ===")

        # 1) Reset the server's global model before launching clients for this permutation
        reset_global_model()

        processes = []
        for client_num in permutation:
            cid = f"client{client_num}"
            scenario_for_this_client = scenario_set[client_num - 1]

            p = multiprocessing.Process(
                target=run_client,
                args=(client_script, cid, scenario_for_this_client, order_index)
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        print(f"Completed permutation #{order_index}: {permutation}")
