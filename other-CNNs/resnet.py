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
from torch.utils.data import random_split, DataLoader
import torchvision.models as models
import pprint

### Define a one epoch training function
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    """Train one epoch, e.g. all batches of one epoch."""
    device = CONFIG["device"]
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    # put the trainloader iterator in a tqdm so it can printprogress
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

    # iterate through all batches of one epoch
    for i, (inputs, labels) in enumerate(progress_bar):

        # move inputs and labels to the target device
        inputs, labels = inputs.to(device), labels.to(device)

        #reset gradients to zero - clears old gradients from the last step so the gradients don't accumulate
        optimizer.zero_grad()
        # FOrward pass to get predictions
        outputs = model(inputs)
        # Calculate loss function (compare predictions with true labels)
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()
        # Update weights - adjust model paramenters using computed gradients
        optimizer.step()

        # calculate loss
        running_loss += loss.item()
        # Get predicted class which is teh lass with the highest score
        _, predicted = outputs.max(1)

        total += labels.size(0) # Update total number of images
        correct += predicted.eq(labels).sum().item() # Count correct predictions

        # Update progress bar
        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

###-----------------------------------------------------------------------------------------------
def validate(model, valloader, criterion, device):
    """Validate the model"""
    model.eval() # Set to evaluation
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # No need to track gradients
        
        # Put the valloader iterator in tqdm to print progress
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)

        # Iterate throught the validation set
        for i, (inputs, labels) in enumerate(progress_bar):
            
            # move inputs and labels to the target device
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs) ##inference
            loss = criterion(outputs, labels)  #loss calculation

            running_loss += loss.item()  ### add loss from this sample
            _, predicted = outputs.max(1)   # predict the class

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({"loss": running_loss / (i+1), "acc": 100. * correct / total})

    val_loss = running_loss/len(valloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


### Early Stopping -- code from Claude + ChatGPT
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_model_state = None  # Store best model

    def __call__(self, val_loss, model):
        """Check if validation loss improved and decide whether to stop training."""
        if val_loss < self.best_loss - self.min_delta:
            print(f"Validation loss improved: {self.best_loss:.4f} → {val_loss:.4f}")
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict()  # Save best model
        else:
            self.counter += 1
            print(f"Early stopping counter: {self.counter}/{self.patience} (no improvement)")

            if self.counter >= self.patience:
                print(f"Early stopping triggered after {self.patience} epochs without improvement.")
                return True  # Stop training

        return False


### Main Training  Loop
def main():

    CONFIG = {
        "model": "MyModel",   # Change name when using a different model
        "batch_size": 128, # run batch size finder to find optimal batch size
        "learning_rate": 0.007,
        "epochs": 10,  # Train for longer in a real scenario
        "num_workers": 8, # Adjust based on your system
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",  # Make sure this directory exists
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge",
        "seed": 42,
    }

    # Load pretrained Densenet model
    model = models.resnet50(weights='IMAGENET1K_V1') # Try resnet18
    device = CONFIG["device"]
    model.to(device)  # Move model to the specified device

    criterion = nn.CrossEntropyLoss() # Cross Enropy Loss for image classification tasks
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"]) # Try out Adam optimizer
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1) # Add a scheduler   #optionally add a LR scheduler

    early_stopping = EarlyStopping(patience=3, min_delta=0.01)  # Adjust patience



    print("\nCONFIG Dictionary:")
    pprint.pprint(CONFIG)

    #      Data Transformation (MOdified for ResNet) Only augment training data, not test data
    transform_train = transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5), # Randomly flip images horizontally
        # transforms.RandomRotation(degrees=15), #Rotate image randomly 15 degrees
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),  # Advanced augmentation
        # ONly Need these two
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])



    #       Data Loading
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
    
    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])

        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        # **Early stopping check**
        if early_stopping(val_acc, model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

###    Instantiate model and move to target device

    # Make sure fully connectled layer fits CIFAR-100 imaging
    num_ftrs = model.fc.in_features
    # Make sure it fits 100 features for hte CIFAR-100 dataset
    model.fc = nn.Linear(num_ftrs, 100)
    
    # move it to target device
    model = model.to(CONFIG["device"])   

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
    


    

    # Initialize wandb
    wandb.init(project="-sp25-ds542-challenge", config=CONFIG)
    wandb.watch(model)  # watch the model gradients

### Training Loop 
    best_val_acc = 0.0

    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler.step()

        # log to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"] # Log learning rate
        })

        # Save the best model (based on validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth") # Save to wandb as well

    wandb.finish()



### Evaluation -- shouldn't have to change the following code
    import eval_cifar100
    import eval_ood

    # --- Evaluation on Clean CIFAR-100 Test Set ---
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # --- Evaluation on OOD ---
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)

    # --- Create Submission File (OOD) ---
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood1.csv", index=False)
    print("submission_ood.csv created successfully.")

if __name__ == '__main__':
    main()