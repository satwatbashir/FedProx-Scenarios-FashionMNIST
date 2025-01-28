import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import grpc
import json
import datetime
import csv
import os
import time
import uuid
import io
import argparse
import random

import fl_proto_pb2 as fl_pb2
import fl_proto_pb2_grpc as fl_pb2_grpc

# -----------------------------
# Global Configurations
# -----------------------------
local_lr = 0.01
lr_decay = 0.995 
local_epochs = 5
target_accuracy = 90.0
momentum = 0.9
number_of_rounds = 100
mu = 1
base_data_source = 'Fashion-MNIST'
session_id = str(uuid.uuid4())

# -----------------------------
# Model Architecture
# -----------------------------
class FashionMNISTCNN(torch.nn.Module):
    def __init__(self):
        super(FashionMNISTCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout = torch.nn.Dropout(0.25)
        self.fc1 = torch.nn.Linear(1600, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# -----------------------------
# Utility Functions
# -----------------------------
def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def base_transform():
    return [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]

# -----------------------------
# Data Augmentation Transforms
# -----------------------------
def fs_low():
    return [transforms.RandomHorizontalFlip(p=0.3)]

def fs_medium():
    return [
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(p=0.5)
    ]

def fs_high():
    return [
        transforms.RandomAffine(degrees=15, translate=(0.2, 0.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
    ]

# -----------------------------
# Data Loading
# -----------------------------
def load_full_fashion_mnist(train=True):
    return datasets.FashionMNIST(root='./data', train=train, download=True)

def dirichlet_distribution(num_clients, num_classes, alpha):
    return np.random.dirichlet([alpha]*num_classes, num_clients)

def create_label_skew_indices(total_samples, client_index, alpha=0.5, num_clients=5):
    dataset = load_full_fashion_mnist(train=True)
    all_targets = np.array(dataset.targets)
    proportions = dirichlet_distribution(num_clients, 10, alpha)
    
    chosen_indices = []
    for class_idx in range(10):
        class_count = int(proportions[client_index][class_idx] * total_samples)
        class_indices = np.where(all_targets == class_idx)[0]
        chosen_indices.extend(np.random.choice(class_indices, class_count, replace=False))
    
    return chosen_indices

def build_hybrid_loader(client_id, scenario, order_id=0):
    c_num = int(client_id.split("client")[-1])
    scenario = scenario.upper()
    
    config = {
        'LOW': {'alpha': 0.8, 'quantity': [2000, 1600, 1400, 1200, 1000], 'transform': fs_low()},
        'MEDIUM': {'alpha': 0.5, 'quantity': [4000, 2000, 1000, 600, 300], 'transform': fs_medium()},
        'HIGH': {'alpha': 0.3, 'quantity': [5000, 3000, 1500, 800, 400], 'transform': fs_high()}
    }[scenario]

    indices = create_label_skew_indices(
        config['quantity'][c_num-1], 
        c_num-1, 
        config['alpha']
    )
    
    transform = transforms.Compose(config['transform'] + base_transform())
    dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform)
    train_loader = DataLoader(Subset(dataset, indices), batch_size=32, shuffle=True, num_workers=4)
    
    test_dataset = datasets.FashionMNIST(
        root='./data', train=False, 
        transform=transforms.Compose(base_transform())
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader

# -----------------------------
# Training Core
# -----------------------------
def fedprox_train_and_evaluate(model, global_params, device, train_loader, test_loader):
    optimizer = torch.optim.SGD(model.parameters(), lr=local_lr, momentum=momentum)
    global_params = {k: v.detach().clone().to(device) for k, v in global_params.items()}

    model.train()
    total_loss = 0.0
    for epoch in range(local_epochs):
        epoch_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            outputs = model(data)
            ce_loss = torch.nn.functional.cross_entropy(outputs, target)
            
            # FedProx regularization
            prox_term = 0.0
            for n, p in model.named_parameters():
                prox_term += torch.norm(p - global_params[n], p=2)**2
            
            loss = ce_loss + (mu / 2) * prox_term
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        total_loss += avg_epoch_loss
        print(f"Epoch {epoch+1}/{local_epochs} | Loss: {avg_epoch_loss:.4f}")

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / local_epochs
    return accuracy, avg_loss, 0.0

# -----------------------------
# gRPC Communication
# -----------------------------
def request_global_model(stub, client_id, session_id, metadata):
    try:
        response = stub.RequestGlobalModel(fl_pb2.ModelRequest(
            client_id=client_id,
            session_id=session_id,
            metadata=metadata
        ))
        return response.model_data
    except grpc.RpcError as e:
        print(f"[{client_id}] Model request failed: {e.code().name}")
        return None

def send_model_parameters(stub, model, accuracy, avg_loss, client_id, scenario, order_id, train_loader, round_num):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    
    metadata = fl_pb2.ModelMetadata(
        data_source=base_data_source,
        data_quality_score=len(train_loader.dataset)/60000,
        total_parameters=sum(p.numel() for p in model.parameters()),
        timestamp=datetime.datetime.now().isoformat(),
        model_performance={"accuracy": float(accuracy), "average_loss": float(avg_loss)},
        model_structure="FashionMNISTCNN",
        current_round=round_num
    )
    
    message = fl_pb2.ModelParameters(
        client_id=client_id,
        model_data=buffer.getvalue(),
        metadata=metadata,
        session_id=session_id
    )
    
    for attempt in range(5):
        try:
            ack = stub.SendModelParameters(message, timeout=30)
            print(f"Server ACK: {ack.message}")
            return True, ack
        except grpc.RpcError as e:
            wait_time = 2 ** attempt
            print(f"Attempt {attempt+1} failed: {e.code().name}")
            time.sleep(wait_time)
    print("Failed to send model parameters after 5 attempts.")
    return False, None

def simulate_network_parameters():
    return {
        'latency': float(np.random.uniform(50, 150)),
        'bandwidth': float(np.random.uniform(1, 10)),
        'reliability': float(np.random.uniform(0.8, 1.0)),
        'cpu_usage': float(np.random.uniform(30, 80)),
        'memory_consumption': float(np.random.uniform(200, 1000))
    }

# -----------------------------
# Metrics Handling
# -----------------------------
def save_training_metrics(client_id, scenario, order_id, round_num, accuracy, avg_loss):
    filename = f"training_metrics_fmnist_hybrid_{scenario.lower()}_{client_id}_order_{order_id}.csv"
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Round', 'Accuracy', 'Average Loss', 'Timestamp'])
        writer.writerow([
            round_num,
            f"{accuracy:.2f}%",
            f"{avg_loss:.4f}",
            datetime.datetime.now().isoformat()
        ])

# -----------------------------
# Main Execution
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="FedProx Fashion MNIST Client")
    parser.add_argument("--client_id", type=str, required=True)
    parser.add_argument("--scenario", type=str, required=True, choices=['LOW', 'MEDIUM', 'HIGH'])
    parser.add_argument("--order_id", type=int, required=True)
    args = parser.parse_args()

    set_random_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Client {args.client_id} starting on {device} ===")

    # Data loading
    train_loader, test_loader = build_hybrid_loader(args.client_id, args.scenario)
    
    # Model setup
    model = FashionMNISTCNN().to(device)
    
    # gRPC setup
    channel = grpc.insecure_channel('localhost:50051', options=[
        ('grpc.max_send_message_length', 256*1024*1024),
        ('grpc.max_receive_message_length', 256*1024*1024),
        ('grpc.http2.max_pings_without_data', 0)
    ])
    stub = fl_pb2_grpc.FederatedLearningServiceStub(channel)
    
    # Initial metadata
    initial_metadata = fl_pb2.ModelMetadata(
        data_source=base_data_source,
        data_quality_score=len(train_loader.dataset)/60000,
        total_parameters=sum(p.numel() for p in model.parameters()),
        timestamp=datetime.datetime.now().isoformat()
    )

    # Training loop
    client_round = 0
    while client_round < number_of_rounds:
        for attempt in range(5):
            try:
                response = stub.RequestGlobalModel(fl_pb2.ModelRequest(
                    client_id=args.client_id,
                    session_id=session_id,
                    metadata=initial_metadata
                ))
                server_round = response.current_round
                break
            except grpc.RpcError as e:
                if attempt == 4:
                    print("Critical: Failed to connect to server after 5 attempts")
                    return
                time.sleep(2 ** attempt)
                continue

        if server_round >= number_of_rounds:
            break

        if server_round > client_round:
            client_round = server_round
            continue

        if server_round < client_round:
            time.sleep(5)
            continue

        if response.model_data:
            try:
                global_state = torch.load(io.BytesIO(response.model_data), map_location=device)
                model.load_state_dict(global_state, strict=True)
            except Exception as e:
                print(f"Error loading global model: {str(e)}")
                continue

        print(f"\n=== Participating in Round {server_round + 1}/{number_of_rounds} ===")
        initial_global_params = {k: v.clone() for k, v in model.state_dict().items()}
        accuracy, avg_loss, _ = fedprox_train_and_evaluate(
            model, initial_global_params, device, train_loader, test_loader
        )
        print(f"\n=== Client {args.client_id} Results ===")
        print(f"Round {server_round} | Accuracy: {accuracy:.2f}% | Loss: {avg_loss:.4f}")

        save_training_metrics(
            client_id=args.client_id,
            scenario=args.scenario,
            order_id=args.order_id,
            round_num=server_round,
            accuracy=accuracy,
            avg_loss=avg_loss
        )

        success = False
        for send_attempt in range(5):
            try:
                send_success, ack = send_model_parameters(
                    stub=stub,
                    model=model,
                    accuracy=accuracy,
                    avg_loss=avg_loss,
                    client_id=args.client_id,
                    scenario=args.scenario,
                    order_id=args.order_id,
                    train_loader=train_loader,
                    round_num=server_round
                )
                if send_success and ack:
                    success = True
                    break
            except grpc.RpcError as e:
                print(f"Attempt {send_attempt+1} failed: {e.code().name}")
                time.sleep(2 ** send_attempt)

        if success:
            client_round += 1
            global local_lr
            local_lr *= lr_decay
            print(f"\n[Client {args.client_id}] LR decayed to: {local_lr:.4f}")
        else:
            print("Critical: Failed to send model after 5 attempts")
            return

    print(f"\n=== Client {args.client_id} completed all rounds ===")

if __name__ == "__main__":
    main()
