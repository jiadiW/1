import torch
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader
import copy

def train_with_swa_and_early_stopping(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    max_steps=10000,
    swa_start_step=100,  # Start averaging after warmup
    val_every_n_steps=50,  # Validate every N steps
    patience_steps=200,  # Early stop if no improvement for N steps
):
    """
    Training with SWA and step-level early stopping.
    
    Key Logic:
    1. Update averaged model EVERY step after swa_start_step
    2. Validate using averaged model every val_every_n_steps
    3. Early stop based on averaged model performance
    4. Return the BEST averaged model (not the final one)
    """
    
    # Initialize SWA model
    swa_model = AveragedModel(model)
    
    # Tracking variables
    best_val_loss = float('inf')
    steps_without_improvement = 0
    global_step = 0
    best_swa_state = None  # Store best SWA model state
    best_model_step = 0
    
    model.train()
    
    for epoch in range(1000):  # Large number, will early stop
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # === STEP 1: Regular Training Update ===
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            global_step += 1
            
            # === STEP 2: Update Averaged Model ===
            if global_step >= swa_start_step:
                swa_model.update_parameters(model)
            
            # === STEP 3: Validation at Regular Intervals ===
            if global_step % val_every_n_steps == 0:
                # Use the AVERAGED model for validation
                eval_model = swa_model if global_step >= swa_start_step else model
                val_loss = validate(eval_model, val_loader, criterion, device)
                
                print(f"Step {global_step} | Val Loss: {val_loss:.4f} | Best: {best_val_loss:.4f}")
                
                # === STEP 4: Early Stopping Check ===
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    steps_without_improvement = 0
                    best_model_step = global_step
                    
                    # IMPORTANT: Save a COPY of the best model state
                    if global_step >= swa_start_step:
                        # Deep copy the SWA model's state
                        best_swa_state = copy.deepcopy(swa_model.module.state_dict())
                        print(f"  → New best SWA model saved!")
                    else:
                        # Save regular model state during warmup
                        best_swa_state = copy.deepcopy(model.state_dict())
                        print(f"  → New best model saved (warmup phase)!")
                    
                    # Optionally save to disk
                    torch.save({
                        'step': global_step,
                        'model_state_dict': best_swa_state,
                        'best_val_loss': best_val_loss,
                    }, 'best_swa_model.pth')
                else:
                    steps_without_improvement += val_every_n_steps
                
                # Early stop if patience exceeded
                if steps_without_improvement >= patience_steps:
                    print(f"\n{'='*60}")
                    print(f"Early stopping at step {global_step}")
                    print(f"Best validation loss: {best_val_loss:.4f} at step {best_model_step}")
                    print(f"{'='*60}\n")
                    
                    # Load the best model state into swa_model
                    swa_model.module.load_state_dict(best_swa_state)
                    return swa_model
                
                model.train()  # Back to training mode
            
            # === STEP 5: Max Steps Check ===
            if global_step >= max_steps:
                print(f"\n{'='*60}")
                print(f"Reached max steps: {max_steps}")
                print(f"Best validation loss: {best_val_loss:.4f} at step {best_model_step}")
                print(f"{'='*60}\n")
                
                # Load the best model state into swa_model
                if best_swa_state is not None:
                    swa_model.module.load_state_dict(best_swa_state)
                return swa_model if global_step >= swa_start_step else model
    
    # If loop completes naturally, return best model
    if best_swa_state is not None:
        swa_model.module.load_state_dict(best_swa_state)
    return swa_model if global_step >= swa_start_step else model


def validate(model, val_loader, criterion, device):
    """Validate the model and return average loss."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
    
    avg_loss = total_loss / len(val_loader.dataset)
    return avg_loss


# === USAGE EXAMPLE ===
if __name__ == "__main__":
    # Setup (replace with your actual model and data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Example model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 1)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # Dummy data loaders (replace with your actual data)
    train_dataset = torch.utils.data.TensorDataset(
        torch.randn(1000, 10), torch.randn(1000, 1)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.randn(200, 10), torch.randn(200, 1)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Train with SWA and early stopping
    final_model = train_with_swa_and_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        max_steps=10000,
        swa_start_step=100,  # Start averaging after 100 steps
        val_every_n_steps=50,  # Validate every 50 steps
        patience_steps=200  # Stop if no improvement for 200 steps
    )
    
    print("Training complete!")
    print("Returned model is the BEST SWA model based on validation loss")
