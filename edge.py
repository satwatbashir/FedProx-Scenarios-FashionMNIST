import grpc
from concurrent import futures
import torch
import fl_proto_pb2 as fl_pb2
import fl_proto_pb2_grpc as fl_pb2_grpc
import logging
import io
from resnet_cifar10 import ResNetCIFAR10

class FederatedProxService(fl_pb2_grpc.FederatedLearningServiceServicer):
    def __init__(self, num_clients=5):
        super().__init__()
        self.global_model = None
        self.client_models = {}
        self.num_clients = num_clients
        self.current_round = 0
        self.received_updates = {}
        print("FedProx Server Initialized. Waiting for clients...")

    def RequestGlobalModel(self, request, context):
        if self.global_model is None:
            if request.metadata.data_source == 'Fashion-MNIST':
                model = FashionMNISTCNN()
                self.global_model = model.state_dict()
                print("Initialized new Fashion MNIST CNN (Global Model)")
          else:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                return fl_pb2.ModelResponse(model_data=b'', message="Unsupported data source")
        
        buffer = io.BytesIO()
        torch.save(self.global_model, buffer)
        return fl_pb2.ModelResponse(
            model_data=buffer.getvalue(),
            message=f"Global model sent (Round {self.current_round})",
            current_round=self.current_round  
        )

    def SendModelParameters(self, request, context):
        try:
            # Validate round synchronization
            client_round = request.metadata.current_round
            server_round = self.current_round
            
            if client_round != server_round:
                msg = f"Rejecting update: Client round {client_round} vs server round {server_round}"
                print(msg)
                return fl_pb2.Ack(message=msg)

            # Validate client ID format
            client_id = request.client_id
            if not (client_id.startswith("client") and client_id[6:].isdigit()):
                return fl_pb2.Ack(message="Invalid client ID format")

            # Validate data quality score
            if not (0 <= request.metadata.data_quality_score <= 1):
                return fl_pb2.Ack(message="Invalid data quality score (0-1 required)")
                
            samples = int(request.metadata.data_quality_score * 50000)
            if samples <= 0:
                return fl_pb2.Ack(message="Invalid sample count derived from data quality score")

            # Load model parameters safely
            try:
                client_state = torch.load(
                    io.BytesIO(request.model_data),
                    weights_only=True,
                    map_location="cpu"
                )
            except Exception as e:
                print(f"Error loading model from {client_id}: {str(e)}")
                return fl_pb2.Ack(message="Invalid model parameters")

            # Initialize round tracking if needed
            if server_round not in self.received_updates:
                self.received_updates[server_round] = {}

            # Prevent duplicate submissions
            if client_id in self.received_updates[server_round]:
                return fl_pb2.Ack(message="Already received your update for this round")

            # Store client update
            self.received_updates[server_round][client_id] = {
                "params": client_state,
                "samples": samples
            }
            print(f"[Round {server_round + 1}] Received update from {client_id} "
                  f"(samples: {samples}, progress: {len(self.received_updates[server_round])}/{self.num_clients})")

            # Check if ready for aggregation
            if len(self.received_updates[server_round]) < self.num_clients:
                remaining = self.num_clients - len(self.received_updates[server_round])
                return fl_pb2.Ack(message=f"Waiting for {remaining} more clients")

            # --- Perform aggregation ---
            print(f"\n[Round {server_round + 1}] Starting aggregation...")
            total_samples = sum(c["samples"] for c in self.received_updates[server_round].values())
            first_client = next(iter(self.received_updates[server_round].values()))["params"]
            
            aggregated_state = {}
            for param_name in first_client:
                aggregated_state[param_name] = torch.zeros_like(first_client[param_name], dtype=torch.float32)
                for client_data in self.received_updates[server_round].values():
                    weight = client_data["samples"] / total_samples
                    aggregated_state[param_name] += client_data["params"][param_name].float() * weight

            # Update global model
            self.global_model = aggregated_state
            
            # Cleanup and prepare for next round
            del self.received_updates[server_round]
            self.current_round += 1
            
            print(f"[Round {server_round + 1}] Aggregation complete. Moving to round {self.current_round + 1}")
            return fl_pb2.Ack(message=f"Aggregation successful (Round {server_round})")

        except Exception as e:
            print(f"Critical server error: {str(e)}")
            return fl_pb2.Ack(message=f"Server error: {str(e)}")
        
    def Heartbeat(self, request, context):
        return fl_pb2.Ack(message="Alive")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
        ('grpc.max_send_message_length', 256 * 1024 * 1024),
        ('grpc.max_receive_message_length', 256 * 1024 * 1024),
        ('grpc.http2.max_pings_without_data', 0),
        ('grpc.http2.max_concurrent_streams', 100)  # Added concurrency option

    ])
    fl_pb2_grpc.add_FederatedLearningServiceServicer_to_server(
        FederatedProxService(), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server running on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
