import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # For progress bars
import wandb
import json
#################################################################################
from torch.utils.data import random_split, DataLoader


class HierarchicalCNN(nn.Module):
    def __init__(self):
        super(HierarchicalCNN, self).__init__()
        # Convolutional feature extraction layers
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second block
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third block
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth block
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Feature dimensions after convolutions: 512 x 2 x 2
        
        # Shared feature representation
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Superclass classification branch
        self.superclass_classifier = nn.Linear(512, 20)
        
        # Fine class classification branch
        self.fine_classifier = nn.Linear(512, 100)

        
    # Add this to HierarchicalCNN class
    def eval_mode(self, mode=True):
        """Toggle evaluation mode - when True, only fine class predictions are returned"""
        self.eval_only_fine = mode
        return self

    # Modify the forward method in HierarchicalCNN
    def forward(self, x):
        # Extract features
        features = self.features(x)
        
        # Get shared representation
        shared_features = self.shared_fc(features)
        
        # Predict superclass and fine class
        superclass_logits = self.superclass_classifier(shared_features)
        fine_class_logits = self.fine_classifier(shared_features)
        
        # During evaluation with existing code, return only fine class predictions
        if hasattr(self, 'eval_only_fine') and self.eval_only_fine and not self.training:
            return fine_class_logits
        
        return superclass_logits, fine_class_logits

##########################################################
## Train #######
#############################################################
def train_hierarchical(epoch, model, trainloader, optimizer, criterion, CONFIG):
    device = CONFIG["device"]
    model.train()
    running_loss = 0.0
    superclass_correct = 0
    fine_class_correct = 0
    total = 0
    
    # Weight for superclass vs fine-class loss
    superclass_weight = 0.3
    fine_class_weight = 0.7
    
    # Create loss function
    # criterion = nn.CrossEntropyLoss()
    
    # Get superclass mapping (you'll need to implement this)
    # Each CIFAR-100 fine class belongs to one of 20 superclasses
    superclass_mapping = get_superclass_mapping()  # See implementation below
    
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)
    
    for i, (inputs, fine_labels) in enumerate(progress_bar):
        inputs, fine_labels = inputs.to(device), fine_labels.to(device)
        
        # Convert fine labels to superclass labels using mapping
        superclass_labels = torch.tensor([superclass_mapping[label.item()] for label in fine_labels], 
                                         device=device)
        
        optimizer.zero_grad()
        
        # Forward pass
        superclass_outputs, fine_class_outputs = model(inputs)
        
        # Calculate losses
        superclass_loss = criterion(superclass_outputs, superclass_labels)
        fine_class_loss = criterion(fine_class_outputs, fine_labels)
        
        # Combined loss
        loss = superclass_weight * superclass_loss + fine_class_weight * fine_class_loss
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted_superclass = superclass_outputs.max(1)
        _, predicted_fine = fine_class_outputs.max(1)
        
        total += fine_labels.size(0)
        superclass_correct += predicted_superclass.eq(superclass_labels).sum().item()
        fine_class_correct += predicted_fine.eq(fine_labels).sum().item()
        
        running_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            "loss": running_loss / (i + 1), 
            "train_acc": 100. * superclass_correct / total,
            "fine_acc": 100. * fine_class_correct / total
        })
    
    train_loss = running_loss / len(trainloader)
    super_acc = 100. * superclass_correct / total
    fine_acc = 100. * fine_class_correct / total
    
    return train_loss, super_acc, fine_acc


# Helper function to get CIFAR-100 superclass mapping
def get_superclass_mapping():
    """
    Creates a mapping from CIFAR-100 fine labels (0-99) to their superclass labels (0-19)
    Returns a dictionary mapping fine_label -> superclass_label
    """
    # CIFAR-100 class hierarchy (simplified implementation)
    # This is the actual mapping from CIFAR-100
    superclass_ranges = {
        'aquatic mammals': [4, 30, 55, 72, 95],
        'fish': [1, 32, 67, 73, 91],
        'flowers': [54, 62, 70, 82, 92],
        'food containers': [9, 10, 16, 28, 61],
        'fruit and vegetables': [0, 51, 53, 57, 83],
        'household electrical devices': [22, 39, 40, 86, 87],
        'household furniture': [5, 20, 25, 84, 94],
        'insects': [6, 7, 14, 18, 24],
        'large carnivores': [3, 42, 43, 88, 97],
        'large man-made outdoor things': [12, 17, 37, 68, 76],
        'large natural outdoor scenes': [23, 33, 49, 60, 71],
        'large omnivores and herbivores': [15, 19, 21, 31, 38],
        'medium-sized mammals': [34, 63, 64, 66, 75],
        'non-insect invertebrates': [26, 45, 77, 79, 99],
        'people': [2, 11, 35, 46, 98],
        'reptiles': [27, 29, 44, 78, 93],
        'small mammals': [36, 50, 65, 74, 80],
        'trees': [47, 52, 56, 59, 96],
        'vehicles 1': [8, 13, 48, 58, 90],
        'vehicles 2': [41, 69, 81, 85, 89]
    }
    
    # Create the mapping
    mapping = {}
    for superclass_idx, (_, fine_labels) in enumerate(superclass_ranges.items()):
        for fine_label in fine_labels:
            mapping[fine_label] = superclass_idx
            
    return mapping
################################################################################
# Define a validation function
################################################################################
def validate_hierarchical(model, valloader, criterion, device):
    """Validate the hierarchical model"""
    # device = CONFIG["device"]
    model.eval()  # Set to evaluation
    running_loss = 0.0
    superclass_correct = 0
    fine_class_correct = 0
    total = 0
    
    # Weight for superclass vs fine-class loss
    superclass_weight = 0.3
    fine_class_weight = 0.7
    
    # Create loss function
    # criterion = nn.CrossEntropyLoss()- define criterion later on in code
    
    # Get superclass mapping
    superclass_mapping = get_superclass_mapping()
    
    with torch.no_grad():  # No need to track gradients
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)
        
        for i, (inputs, fine_labels) in enumerate(progress_bar):
            # Move inputs and labels to the target device
            inputs, fine_labels = inputs.to(device), fine_labels.to(device)
            
            # Convert fine labels to superclass labels using mapping
            superclass_labels = torch.tensor([superclass_mapping[label.item()] for label in fine_labels], 
                                           device=device)
            
            # Forward pass
            superclass_outputs, fine_class_outputs = model(inputs)
            
            # Calculate losses
            superclass_loss = criterion(superclass_outputs, superclass_labels)
            fine_class_loss = criterion(fine_class_outputs, fine_labels)
            
            # Combined loss
            loss = superclass_weight * superclass_loss + fine_class_weight * fine_class_loss
            
            running_loss += loss.item()
            
            # Get predictions
            _, predicted_superclass = superclass_outputs.max(1)
            _, predicted_fine = fine_class_outputs.max(1)
            
            total += fine_labels.size(0)
            superclass_correct += predicted_superclass.eq(superclass_labels).sum().item()
            fine_class_correct += predicted_fine.eq(fine_labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": running_loss / (i + 1),
                "super_acc": 100. * superclass_correct / total,
                "fine_acc": 100. * fine_class_correct / total
            })
    
    val_loss = running_loss / len(valloader)
    super_acc = 100. * superclass_correct / total
    fine_acc = 100. * fine_class_correct / total
    
    return val_loss, super_acc, fine_acc

    ############################################################################
    #    Configuration Dictionary (Modify as needed)
    ############################################################################
    # It's convenient to put all the configuration in a dictionary so that we have
    # one place to change the configuration.
    # It's also convenient to pass to our experiment tracking tool.
def main():

    CONFIG = {
        "model": "MyModel",   # Change name when using a different model
        "batch_size": 128, # run batch size finder to find optimal batch size
        "learning_rate": 0.0005,
        "epochs": 10,  # Train for longer in a real scenario
        "num_workers": 4, # Adjust based on your system
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",  # Make sure this directory exists
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge",
        "seed": 42,
        "superclass_weight": 0.3,  # New parameter for loss weighting
        "fine_class_weight": 0.7,  # New parameter for loss weighting
    }

    import pprint
    print("\nCONFIG Dictionary:")
    pprint.pprint(CONFIG)

    ############################################################################
    #      Data Transformation (Example - You might want to modify) 
    ############################################################################

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Example normalization
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    ############################################################################
    #       Data Loading
    ############################################################################

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)

    # Split train into train and validation (80/20 split)
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = random_split(trainset, [train_size, val_size])

    #efine loaders and test set
    trainloader = DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    valloader = DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    #  (Create validation and test loaders)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    
    ############################################################################
    #   Instantiate model and move to target device
    ############################################################################
    model = HierarchicalCNN()
    model = model.to(CONFIG["device"])   # move it to target device

    print("\nModel summary:")
    print(f"{model}\n")

    # The following code you can run once to find the batch size that gives you the fastest throughput.
    # You only have to do this once for each machine you use, then you can just
    # set it in CONFIG.
    SEARCH_BATCH_SIZES = False
    if SEARCH_BATCH_SIZES:
        from utils import find_optimal_batch_size
        print("Finding optimal batch size...")
        optimal_batch_size = find_optimal_batch_size(model, trainset, CONFIG["device"], CONFIG["num_workers"])
        CONFIG["batch_size"] = optimal_batch_size
        print(f"Using batch size: {CONFIG['batch_size']}")
    

    ############################################################################
    # Loss Function, Optimizer and optional learning rate scheduler
    ############################################################################
    # Cross Enropy Loss for image classification tasks
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Label smooting suggested from Claude
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)


    # Initialize wandb
    wandb.init(project="-sp25-ds542-challenge", config=CONFIG)
    wandb.watch(model)  # watch the model gradients

    ############################################################################
    # Training Loop 
    ############################################################################
    best_val_acc = 0.0

    for epoch in range(CONFIG["epochs"]):
        train_loss, train_super_acc, train_fine_acc = train_hierarchical(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_super_acc, val_fine_acc = validate_hierarchical(model, valloader, criterion, CONFIG["device"])
        scheduler.step(val_loss)

        # log to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_fine_acc,
            "val_loss": val_loss,
            "val_acc": val_fine_acc,
            "lr": optimizer.param_groups[0]["lr"] # Log learning rate
        })

        # Save best model
        if val_fine_acc > best_val_acc:
            best_val_acc = val_fine_acc
            torch.save(model.state_dict(), 'best_model.pth')
            wandb.save("best_model.pth") # Save to wandb as well

                # Print epoch results
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} - "
              f"Train: Loss {train_loss:.4f} | SuperAcc {train_super_acc:.2f}% | FineAcc {train_fine_acc:.2f}% | "
              f"Valid: Loss {val_loss:.4f} | SuperAcc {val_super_acc:.2f}% | FineAcc {val_fine_acc:.2f}%")
   
    wandb.finish()

    ############################################################################
    # Evaluation
    ############################################################################
    import eval_cifar100
    import eval_ood

    model.eval_mode(True)  # Ensure only fine class predictions
    model.eval()           # Switch to evaluation mode
    # --- Evaluation on Clean CIFAR-100 Test Set ---
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # --- Evaluation on OOD ---
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)

    # --- Create Submission File (OOD) ---
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood.csv", index=False)
    print("submission_ood.csv created successfully.")

if __name__ == '__main__':
    main()
