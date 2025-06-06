import argparse
import os
import torch
from torch_geometric.loader import DataLoader
from src.loadData import GraphDataset
from src.utils import set_seed
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm 
from sklearn.metrics import f1_score

from src.losses import GCODLoss
from src.models import GNN 
from src.utils import RandomEdgeDrop, GaussianEdgeNoise
# Set the random seed
set_seed()

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def train(data_loader, model, optimizer, criterion, device, save_checkpoints, checkpoint_path, current_epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in tqdm(data_loader, desc="Iterating training graphs", unit="batch"):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)

    # Save checkpoints if required
    if save_checkpoints:
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    return total_loss / len(data_loader),  correct / total

def evaluate(data_loader, model, device, calculate_accuracy=False, current_epoch=None):
            model.eval()
            correct = 0
            total = 0
            predictions = []
            true_labels = [] # To store true labels for F1 score
            total_loss = 0 # Aggiungi questa variabile
            criterion = torch.nn.CrossEntropyLoss() # Definizione del criterio di loss per la validazione

            desc = "Iterating eval graphs"
            if current_epoch is not None:
                desc = f"Epoch {current_epoch + 1} | {desc}"

            with torch.no_grad():
                for data in tqdm(data_loader, desc=desc, unit="batch"):
                    data = data.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1)

                    if calculate_accuracy:
                        # Calcola la loss per ogni batch
                        loss = criterion(output, data.y)
                        total_loss += loss.item() # Accumula la loss

                        correct += (pred == data.y).sum().item()
                        total += data.y.size(0)
                        true_labels.extend(data.y.cpu().numpy()) # Collect true labels
                        predictions.extend(pred.cpu().numpy()) # Collect predictions
                    else:
                        predictions.extend(pred.cpu().numpy())

            if calculate_accuracy:
                accuracy = correct / total
                # Calculate F1 score
                f1 = f1_score(true_labels, predictions, average='weighted') # Use 'weighted' for multi-class
                # Restituisci anche la validation loss media
                return total_loss / len(data_loader), accuracy, f1 # Return loss, accuracy, and F1 score
            return predictions

def save_predictions(predictions, test_path):
    script_dir = os.getcwd()
    submission_folder = os.path.join(script_dir, "submission")
    test_dir_name = os.path.basename(os.path.dirname(test_path))

    os.makedirs(submission_folder, exist_ok=True)

    output_csv_path = os.path.join(submission_folder, f"testset_{test_dir_name}.csv")

    test_graph_ids = list(range(len(predictions)))
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })

    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")


def plot_training_progress(train_losses, train_accuracies, output_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss", color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Training Accuracy", color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy per Epoch')

    # Save plots in the current directory
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progress.png"))
    plt.close()

def main(args):
    # Get the directory where the main script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    num_checkpoints = args.num_checkpoints if args.num_checkpoints else 3
    
    

    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    else:    
        raise ValueError('Invalid GNN type')
    
   

    num_classes = 6  # Replace with the actual number of classes in your dataset
    criterion = GCODLoss(num_classes=num_classes) # You can use noise_prob for q, or add a new argument

    # Identify dataset folder (A, B, C, or D)
    test_dir_name = os.path.basename(os.path.dirname(args.test_path))
    
    # Setup logging
    logs_folder = os.path.join(script_dir, "logs", test_dir_name)
    log_file = os.path.join(logs_folder, "training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())  # Console output as well


    # Define checkpoint path relative to the script's directory
    checkpoint_path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
    checkpoints_folder = os.path.join(script_dir, "checkpoints", test_dir_name)
    os.makedirs(checkpoints_folder, exist_ok=True)

    # Load pre-trained model for inference
    if os.path.exists(checkpoint_path) and not args.train_path:
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded best model from {checkpoint_path}")

    # Prepare test dataset and loader
    test_dataset = GraphDataset(args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # If train_path is provided, train the model
    if args.train_path:
        full_dataset = GraphDataset(args.train_path, transform=add_zeros)
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size


        generator = torch.Generator().manual_seed(12)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

        edge_drop_transform = RandomEdgeDrop(p=0.2)
        gaussian_noise_transform = GaussianEdgeNoise(std=0.08, p=0.8)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: Batch.from_data_list([train_transform(d) for d in batch]))
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)


    
        num_epochs = args.epochs
        best_val_accuracy = 0.0

        train_losses = []
        train_accuracies = []
        val_losses = [] # Aggiungi questa lista per salvare le validation losses
        val_accuracies = []

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002,weight_decay=1.27e-5)
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=5,
            min_lr=1e-6
        )
            
        if num_checkpoints > 1:
            checkpoint_intervals = [int((i + 1) * num_epochs / num_checkpoints) for i in range(num_checkpoints)]
        else:
            checkpoint_intervals = [num_epochs]
        best_val_accuracy = 0.0
        epochs_no_improve = 0
        early_stopping_patience = 15
        
        for epoch in range(num_epochs):
            train_loss, train_acc = train(
                train_loader, model, optimizer, criterion, device,
                save_checkpoints=(epoch + 1 in checkpoint_intervals),
                checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}"),
                current_epoch=epoch
            )

            # Cattura la validation loss, l'accuratezza e l'F1 score
            val_loss, val_acc, val_f1 = evaluate(val_loader, model, device, calculate_accuracy=True)

            # Stampa anche la validation loss
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss) # Aggiungi la validation loss alla lista
            val_accuracies.append(val_acc)
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Val F1: {val_f1:.4f}")

                # Early stopping logic
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                epochs_no_improve = 0  # Reset counter
                # Opzionalmente salva il modello qui se vuoi salvare il modello migliore finora
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Best model updated and saved at {checkpoint_path}") # Spostato all'esterno del ciclo se vuoi salvare solo alla fine

            else:
                epochs_no_improve += 1



            if epochs_no_improve == early_stopping_patience:
                print(f"Early stopping triggered after {early_stopping_patience} epochs without improvement.")
                break # Exit the training loop

        plot_training_progress(train_losses, train_accuracies, os.path.join(logs_folder, "plots"))

    # Generate predictions for the test set using the best model
    model.load_state_dict(torch.load(checkpoint_path))
    predictions = evaluate(test_loader, model, device, calculate_accuracy=False)
    save_predictions(predictions, args.test_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")
    parser.add_argument("--train_path", type=str, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--num_checkpoints", type=int, default=20, help="Number of checkpoints to save during training.")
    parser.add_argument('--device', type=int, default=1, help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin', help='GNN gin, gin-virtual, or gcn, or gcn-virtual, gat, graphsage (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=4, help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=480, help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 10)')
    
    args = parser.parse_args()
    main(args)
