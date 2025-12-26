
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def save_results(save_dir, train_losses, train_accs, val_losses, val_accs, 
                test_labels, test_preds, class_names):
    """Save learning curves, confusion matrix, and training summary"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Learning curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training & Validation Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training & Validation Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Test Set Confusion Matrix')
    plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Training summary
    with open(f'{save_dir}/training_summary.txt', 'w') as f:
        f.write(f"Training Summary\n")
        f.write(f"================\n")
        f.write(f"Final Train Accuracy: {train_accs[-1]:.3f}\n")
        f.write(f"Final Val Accuracy: {val_accs[-1]:.3f}\n")
        f.write(f"Best Val Accuracy: {max(val_accs):.3f}\n")
        f.write(f"Test Accuracy: {accuracy_score(test_labels, test_preds):.3f}\n")
        f.write(f"Total Epochs: {len(train_accs)}\n")
        f.write(f"\nConfusion Matrix:\n{cm}\n")
        f.write(f"\nClassification Report:\n")
        f.write(classification_report(test_labels, test_preds, target_names=class_names))
    
    print(f"Results saved to {save_dir}/")
    
    
def save_model_info(model, save_dir):
    """Save detailed model information to file"""
    trainable, non_trainable, total = count_parameters(model)
    
    with open(f'{save_dir}/model_info.txt', 'w') as f:
        f.write("MODEL ARCHITECTURE & SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write("Model Architecture:\n")
        f.write("-" * 40 + "\n")
        f.write(str(model) + "\n\n")
        f.write("Parameter Summary:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Trainable parameters:     {trainable:,}\n")
        f.write(f"Non-trainable parameters: {non_trainable:,}\n")
        f.write(f"Total parameters:         {total:,}\n")
        f.write(f"Trainable percentage:     {trainable/total*100:.1f}%\n\n")
        
        backbone_params = sum(p.numel() for p in model.backbone.parameters())
        classifier_params = sum(p.numel() for p in model.classifier.parameters())
        f.write("Layer-wise Parameter Breakdown:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Backbone (ResNet50):      {backbone_params:,}\n")
        f.write(f"Custom Classifier:        {classifier_params:,}\n")
        
        memory_mb = (total * 4) / (1024**2)
        f.write(f"\nEstimated Model Size:     {memory_mb:.1f} MB\n")
    
    print(f"Model info saved to {save_dir}/model_info.txt")
    
import csv
import json
import os

def save_epoch_results(save_dir, train_losses, train_accs, val_losses, val_accs):
    """Save detailed epoch-by-epoch training results"""
    
    # Create results directory
    results_dir = os.path.join(save_dir, 'epoch_results')
    os.makedirs(results_dir, exist_ok=True)
    
    epochs = list(range(1, len(train_losses) + 1))
    
    # Save CSV
    with open(os.path.join(results_dir, 'epoch_results.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Train_Acc', 'Val_Loss', 'Val_Acc'])
        for i in range(len(epochs)):
            writer.writerow([epochs[i], f"{train_losses[i]:.6f}", f"{train_accs[i]:.6f}", 
                           f"{val_losses[i]:.6f}", f"{val_accs[i]:.6f}"])
    
    # Save JSON with summary
    json_data = {
        'epochs': epochs,
        'train_losses': [round(loss, 6) for loss in train_losses],
        'train_accuracies': [round(acc, 6) for acc in train_accs],
        'val_losses': [round(loss, 6) for loss in val_losses],
        'val_accuracies': [round(acc, 6) for acc in val_accs],
        'summary': {
            'total_epochs': len(epochs),
            'best_val_acc': round(max(val_accs), 6),
            'best_val_epoch': val_accs.index(max(val_accs)) + 1,
            'final_train_acc': round(train_accs[-1], 6),
            'final_val_acc': round(val_accs[-1], 6)
        }
    }
    
    with open(os.path.join(results_dir, 'epoch_results.json'), 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Epoch results saved to {results_dir}/")

def count_parameters(model):
    """Count trainable and non-trainable parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + non_trainable
    return trainable, non_trainable, total