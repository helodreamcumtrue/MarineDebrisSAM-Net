import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, loader, device, model_type='fusion', class_names=None):
    """
    Evaluates a PyTorch model and prints classification metrics and confusion matrix.
    """
    if class_names is None:
        class_names = ['Plastic', 'Glass', 'Metal', 'Cloth/Rope', 'Rubber', 'Paper']
        
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            orig_img, seg_img, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            
            if model_type == 'single_orig':
                outputs = model(orig_img)
            elif model_type == 'single_seg':
                outputs = model(seg_img)
            else:
                outputs = model(orig_img, seg_img)
                
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    label_indices = list(range(len(class_names)))
    
    print("\n--- Detailed Performance Metrics ---")
    print(classification_report(all_labels, all_preds, labels=label_indices, target_names=class_names, digits=4, zero_division=0))
    
    cm = confusion_matrix(all_labels, all_preds, labels=label_indices)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, linewidths=0.5, linecolor='white')
    plt.ylabel('Actual Class (Ground Truth)', fontweight='bold')
    plt.xlabel('Predicted Class', fontweight='bold')
    plt.title('Confusion Matrix', pad=15)
    plt.tight_layout()
    plt.show()
