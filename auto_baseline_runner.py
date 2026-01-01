"""
Automated Baseline Comparison Script for Chinese Minority Costume Classification

This script automatically runs all essential baselines and generates:
1. Training results for each model
2. Comparison tables (CSV, LaTeX)
3. Visualization plots
4. Classification reports
5. Comprehensive log files

Usage:
    python auto_baseline_runner.py --data_dir ./dataset --output_dir ./results

Estimated time: ~3-4 hours on GPU for all baselines
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import json
import os
import argparse
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ================== CONFIGURATION ==================
class Config:
    # Data
    IMG_SIZE = 224
    NUM_CLASSES = 5
    TRAIN_SPLIT = 0.8
    
    # Training
    BATCH_SIZE = 32
    MAX_EPOCHS = 100
    PATIENCE = 15
    NUM_WORKERS = 4
    
    # Optimizer
    WEIGHT_DECAY = 0.05
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths (will be set by argparse)
    DATA_DIR = 'path/to/your/dataset'
    OUTPUT_DIR = './baseline_results'

config = Config()

# ================== BASELINE MODEL DEFINITIONS ==================
BASELINE_MODELS = {
    # Essential baselines
    'ResNet50': {
        'model_name': 'resnet50',
        'lr': 1e-4,
        'description': 'Deep residual network (He et al., CVPR 2016)',
        'category': 'Traditional CNN',
        'params': '25M',
        'essential': True
    },
    'EfficientNet-B0': {
        'model_name': 'efficientnet_b0',
        'lr': 1e-4,
        'description': 'Efficient neural architecture (Tan & Le, ICML 2019)',
        'category': 'Modern CNN',
        'params': '5.3M',
        'essential': True
    },
    'ViT-Small': {
        'model_name': 'vit_small_patch16_224',
        'lr': 3e-4,
        'description': 'Vision Transformer (Dosovitskiy et al., ICLR 2021)',
        'category': 'Vision Transformer',
        'params': '22M',
        'essential': True
    },
    'DeiT-Tiny': {
        'model_name': 'deit_tiny_patch16_224',
        'lr': 3e-4,
        'description': 'Data-efficient Image Transformer (Touvron et al., ICML 2021)',
        'category': 'Vision Transformer',
        'params': '5.7M',
        'essential': True
    },
    'DeiT-Small': {
        'model_name': 'deit_small_patch16_224',
        'lr': 3e-4,
        'description': 'Data-efficient Image Transformer (Touvron et al., ICML 2021)',
        'category': 'Vision Transformer',
        'params': '22M',
        'essential': True
    },
    
    # Additional good baselines
    'VGG16': {
        'model_name': 'vgg16',
        'lr': 1e-4,
        'description': 'Very deep CNN (Simonyan & Zisserman, ICLR 2015)',
        'category': 'Traditional CNN',
        'params': '138M',
        'essential': False
    },
    'MobileNetV2': {
        'model_name': 'mobilenetv2_100',
        'lr': 1e-4,
        'description': 'Mobile-efficient CNN (Sandler et al., CVPR 2018)',
        'category': 'Modern CNN',
        'params': '3.5M',
        'essential': False
    },
    'Swin-Tiny': {
        'model_name': 'swin_tiny_patch4_window7_224',
        'lr': 3e-4,
        'description': 'Hierarchical Vision Transformer (Liu et al., ICCV 2021)',
        'category': 'Hybrid Architecture',
        'params': '28M',
        'essential': False
    },
    'ConvNeXt-Tiny': {
        'model_name': 'convnext_tiny',
        'lr': 1e-4,
        'description': 'Modernized ConvNet (Liu et al., CVPR 2022)',
        'category': 'Hybrid Architecture',
        'params': '28M',
        'essential': False
    }
}

# Ablation configurations
ABLATION_CONFIGS = {
    'DeiT-Small-NoUnfreeze': {
        'base_model': 'deit_small_patch16_224',
        'lr': 3e-4,
        'staged_unfreezing': False,
        'description': 'DeiT-Small without staged unfreezing'
    },
    'DeiT-Small-FromScratch': {
        'base_model': 'deit_small_patch16_224',
        'lr': 3e-4,
        'pretrained': False,
        'description': 'DeiT-Small trained from scratch (no pretrained weights)'
    },
    'DeiT-Small-NoLabelSmoothing': {
        'base_model': 'deit_small_patch16_224',
        'lr': 3e-4,
        'label_smoothing': 0.0,
        'description': 'DeiT-Small without label smoothing'
    }
}

# ================== DATA LOADING ==================
def get_data_loaders(data_dir, batch_size=32, num_workers=4):
    """Prepare train and validation data loaders with stratified split"""
    
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    
    # Transforms
    train_transforms = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    full_dataset = datasets.ImageFolder(data_dir)
    class_names = full_dataset.classes
    
    print(f"Total images: {len(full_dataset)}")
    print(f"Classes: {class_names}")
    
    # Check class distribution
    labels = [label for _, label in full_dataset.samples]
    class_counts = Counter(labels)
    print("\nClass distribution:")
    for i, name in enumerate(class_names):
        count = class_counts[i]
        print(f"  {name}: {count} images ({count/len(labels)*100:.1f}%)")
    
    # Stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-config.TRAIN_SPLIT, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    
    # Apply transforms
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms
    
    # Weighted sampling for balanced training
    train_labels = [labels[i] for i in train_idx]
    class_weights = {i: 1.0/class_counts[i] for i in range(len(class_names))}
    sample_weights = [class_weights[label] for label in train_labels]
    train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=num_workers, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True
    )
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print("="*70)
    
    return train_loader, val_loader, class_names

# ================== LOSS FUNCTIONS ==================
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prob = nn.functional.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prob).sum(dim=1).mean()
        return loss

# ================== TRAINING FUNCTIONS ==================
def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{running_loss/(pbar.n+1):.4f}', 
                         'acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation', leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return running_loss / len(loader), 100. * correct / total, all_preds, all_labels

# ================== MAIN TRAINING LOOP ==================
def train_model(model_name, model_config, train_loader, val_loader, class_names, 
                output_dir, run_ablation=False, ablation_config=None):
    """
    Train a single model and save results
    """
    print(f"\n{'='*70}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*70}")
    print(f"Description: {model_config.get('description', 'N/A')}")
    print(f"Learning Rate: {model_config['lr']}")
    print(f"Parameters: {model_config.get('params', 'N/A')}")
    
    # Create model
    try:
        if run_ablation and ablation_config:
            pretrained = ablation_config.get('pretrained', True)
            model = timm.create_model(
                ablation_config['base_model'], 
                pretrained=pretrained, 
                num_classes=config.NUM_CLASSES
            )
            print(f"Ablation: {ablation_config['description']}")
            print(f"Pretrained: {pretrained}")
        else:
            model = timm.create_model(
                model_config['model_name'], 
                pretrained=True, 
                num_classes=config.NUM_CLASSES
            )
        model = model.to(config.DEVICE)
    except Exception as e:
        print(f"ERROR: Failed to load model {model_name}: {e}")
        return None
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    label_smoothing = 0.1
    if run_ablation and ablation_config:
        label_smoothing = ablation_config.get('label_smoothing', 0.1)
    
    criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=model_config['lr'], 
                           weight_decay=config.WEIGHT_DECAY)
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    patience_counter = 0
    
    for epoch in range(config.MAX_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.MAX_EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, config.DEVICE)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_path = os.path.join(output_dir, f'{model_name}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_preds': val_preds,
                'val_labels': val_labels
            }, checkpoint_path)
            print(f"✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\n⏹ Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n{'='*70}")
    print(f"Training Complete: {model_name}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"{'='*70}")
    
    # Load best model for final evaluation


    try:
        checkpoint = torch.load(os.path.join(output_dir, f'{model_name}_best.pth'), weights_only=False)
    except TypeError:
    # For older PyTorch versions
        checkpoint = torch.load(os.path.join(output_dir, f'{model_name}_best.pth'))
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    _, final_acc, final_preds, final_labels = validate(model, val_loader, 
                                                        nn.CrossEntropyLoss(), config.DEVICE)
    
    # Classification report
    report = classification_report(final_labels, final_preds, 
                                   target_names=class_names, 
                                   output_dict=True, 
                                   zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(final_labels, final_preds)
    
    # Save results
    results = {
        'model_name': model_name,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'total_epochs': epoch + 1,
        'final_acc': final_acc,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'history': history,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'config': model_config
    }
    
    return results

# ================== VISUALIZATION ==================
def plot_results(all_results, output_dir, class_names):
    """Create comprehensive visualizations"""
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # 1. Accuracy comparison bar chart
    plt.figure(figsize=(14, 8))
    models = [r['model_name'] for r in all_results]
    accuracies = [r['best_val_acc'] for r in all_results]
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)[::-1]
    models = [models[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    
    colors = ['#e74c3c' if 'DeiT-Small' == m and 'No' not in m and 'From' not in m 
              else '#3498db' for m in models]
    
    bars = plt.barh(models, accuracies, color=colors)
    
    # Highlight best model
    best_idx = accuracies.index(max(accuracies))
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(2)
    
    plt.xlabel('Validation Accuracy (%)', fontsize=13, fontweight='bold')
    plt.title('Baseline Model Comparison: Chinese Minority Costume Classification', 
              fontsize=15, fontweight='bold', pad=20)
    plt.xlim([0, 100])
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (model, acc) in enumerate(zip(models, accuracies)):
        plt.text(acc + 1, i, f'{acc:.2f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), 
                dpi=300, bbox_inches='tight')
    print("✓ Saved: accuracy_comparison.png")
    plt.close()
    
    # 2. Training curves for top models
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, result in enumerate(all_results[:6]):  # Top 6 models
        ax = axes[idx]
        history = result['history']
        epochs = range(1, len(history['train_acc']) + 1)
        
        ax.plot(epochs, history['train_acc'], label='Train', linewidth=2)
        ax.plot(epochs, history['val_acc'], label='Val', linewidth=2)
        ax.set_title(f"{result['model_name']}\n(Best: {result['best_val_acc']:.2f}%)", 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), 
                dpi=300, bbox_inches='tight')
    print("✓ Saved: training_curves.png")
    plt.close()
    
    # 3. Confusion matrices for best model
    best_result = max(all_results, key=lambda x: x['best_val_acc'])
    cm = np.array(best_result['confusion_matrix'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f"Confusion Matrix: {best_result['model_name']}\n(Accuracy: {best_result['best_val_acc']:.2f}%)", 
              fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{best_result['model_name']}.png"), 
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved: confusion_matrix_{best_result['model_name']}.png")
    plt.close()
    
    # 4. Model size vs accuracy scatter plot
    plt.figure(figsize=(12, 8))
    
    sizes = []
    accs = []
    labels = []
    
    for result in all_results:
        param_str = result['config'].get('params', '0M')
        try:
            size = float(param_str.replace('M', ''))
            sizes.append(size)
            accs.append(result['best_val_acc'])
            labels.append(result['model_name'])
        except:
            continue
    
    plt.scatter(sizes, accs, s=200, alpha=0.6, c=range(len(sizes)), cmap='viridis')
    
    for i, label in enumerate(labels):
        plt.annotate(label, (sizes[i], accs[i]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=9, fontweight='bold')
    
    plt.xlabel('Model Size (Million Parameters)', fontsize=12, fontweight='bold')
    plt.ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Model Efficiency: Size vs Accuracy', fontsize=14, fontweight='bold', pad=15)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'size_vs_accuracy.png'), 
                dpi=300, bbox_inches='tight')
    print("✓ Saved: size_vs_accuracy.png")
    plt.close()

# ================== RESULTS EXPORT ==================
def export_results(all_results, output_dir, class_names):
    """Export results to CSV, JSON, and LaTeX"""
    
    print("\n" + "="*70)
    print("EXPORTING RESULTS")
    print("="*70)
    
    # 1. Summary CSV
    summary_data = []
    for result in all_results:
        summary_data.append({
            'Model': result['model_name'],
            'Category': result['config'].get('category', 'N/A'),
            'Parameters': result['config'].get('params', 'N/A'),
            'Best Val Acc (%)': f"{result['best_val_acc']:.2f}",
            'Best Epoch': result['best_epoch'],
            'Total Epochs': result['total_epochs']
        })
    
    df = pd.DataFrame(summary_data)
    df = df.sort_values('Best Val Acc (%)', ascending=False)
    df.to_csv(os.path.join(output_dir, 'results_summary.csv'), index=False)
    print("✓ Saved: results_summary.csv")
    
    # 2. Detailed results JSON
    with open(os.path.join(output_dir, 'detailed_results.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = []
        for r in all_results:
            r_copy = r.copy()
            if 'confusion_matrix' in r_copy:
                r_copy['confusion_matrix'] = [[int(x) for x in row] for row in r_copy['confusion_matrix']]
            json_results.append(r_copy)
        json.dump(json_results, f, indent=2)
    print("✓ Saved: detailed_results.json")
    
    # 3. LaTeX table
    latex_table = "\\begin{table}[h]\n\\centering\n\\caption{Baseline Model Comparison Results}\n"
    latex_table += "\\label{tab:baselines}\n\\begin{tabular}{lccc}\n\\hline\n"
    latex_table += "Model & Parameters & Accuracy (\\%) & Reference \\\\\n\\hline\n"
    
    for result in sorted(all_results, key=lambda x: x['best_val_acc'], reverse=True):
        model = result['model_name']
        params = result['config'].get('params', 'N/A')
        acc = f"{result['best_val_acc']:.2f}"
        
        # Extract reference from description
        desc = result['config'].get('description', '')
        if '(' in desc and ')' in desc:
            ref = desc[desc.find("(")+1:desc.find(")")]
        else:
            ref = "This work"
        
        if 'DeiT-Small' == model and 'No' not in model and 'From' not in model:
            latex_table += f"\\textbf{{{model}}} & {params} & \\textbf{{{acc}}} & {ref} \\\\\n"
        else:
            latex_table += f"{model} & {params} & {acc} & {ref} \\\\\n"
    
    latex_table += "\\hline\n\\end{tabular}\n\\end{table}"
    
    with open(os.path.join(output_dir, 'results_table.tex'), 'w') as f:
        f.write(latex_table)
    print("✓ Saved: results_table.tex")
    
    # 4. Per-class accuracy table
    best_result = max(all_results, key=lambda x: x['best_val_acc'])
    report = best_result['classification_report']
    
    per_class_data = []
    for class_name in class_names:
        if class_name in report:
            per_class_data.append({
                'Class': class_name,
                'Precision': f"{report[class_name]['precision']:.4f}",
                'Recall': f"{report[class_name]['recall']:.4f}",
                'F1-Score': f"{report[class_name]['f1-score']:.4f}",
                'Support': report[class_name]['support']
            })
    
    df_per_class = pd.DataFrame(per_class_data)
    df_per_class.to_csv(os.path.join(output_dir, 'per_class_results.csv'), index=False)
    print("✓ Saved: per_class_results.csv")
    
    # 5. Print summary to console
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(df.to_string(index=False))
    print("\n" + "="*70)

# ================== MAIN EXECUTION ==================
def main(args):
    """Main execution function"""
    
    # Set paths
    config.DATA_DIR = args.data_dir
    config.OUTPUT_DIR = args.output_dir
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(config.OUTPUT_DIR, f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    print("\n" + "="*70)
    print("AUTOMATED BASELINE COMPARISON")
    print("="*70)
    print(f"Data Directory: {config.DATA_DIR}")
    print(f"Output Directory: {config.OUTPUT_DIR}")
    print(f"Device: {config.DEVICE}")
    print(f"Run Essential Only: {args.essential_only}")
    print(f"Include Ablations: {args.ablations}")
    print("="*70)
    
    # Load data
    train_loader, val_loader, class_names = get_data_loaders(
        config.DATA_DIR, 
        batch_size=config.BATCH_SIZE, 
        num_workers=config.NUM_WORKERS
    )
    
    # Select models to run
    models_to_run = {}
    if args.essential_only:
        models_to_run = {k: v for k, v in BASELINE_MODELS.items() if v.get('essential', False)}
    else:
        models_to_run = BASELINE_MODELS
    
    print(f"\nRunning {len(models_to_run)} baseline models...")
    if args.ablations:
        print(f"Plus {len(ABLATION_CONFIGS)} ablation studies...")
    
    # Train all models
    all_results = []
    
    for model_name, model_config in models_to_run.items():
        try:
            result = train_model(
                model_name, 
                model_config, 
                train_loader, 
                val_loader, 
                class_names,
                config.OUTPUT_DIR
            )
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"ERROR training {model_name}: {e}")
            continue
    
    # Run ablation studies if requested
    if args.ablations:
        print("\n" + "="*70)
        print("RUNNING ABLATION STUDIES")
        print("="*70)
        
        for ablation_name, ablation_config in ABLATION_CONFIGS.items():
            try:
                result = train_model(
                    ablation_name,
                    {'lr': ablation_config['lr'], 
                     'description': ablation_config['description'],
                     'category': 'Ablation Study',
                     'params': '22M'},
                    train_loader,
                    val_loader,
                    class_names,
                    config.OUTPUT_DIR,
                    run_ablation=True,
                    ablation_config=ablation_config
                )
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f"ERROR in ablation {ablation_name}: {e}")
                continue
    
    # Generate visualizations and exports
    if all_results:
        plot_results(all_results, config.OUTPUT_DIR, class_names)
        export_results(all_results, config.OUTPUT_DIR, class_names)
        
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE!")
        print("="*70)
        print(f"Results saved to: {config.OUTPUT_DIR}")
        print(f"Total models trained: {len(all_results)}")
        
        best_model = max(all_results, key=lambda x: x['best_val_acc'])
        print(f"\nBest Model: {best_model['model_name']}")
        print(f"Best Accuracy: {best_model['best_val_acc']:.2f}%")
        print("="*70)
    else:
        print("\n⚠️ WARNING: No models were successfully trained!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automated Baseline Comparison for Image Classification')
    parser.add_argument('--data_dir', type=str, default='path/to/your/dataset',
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./baseline_results',
                       help='Path to output directory for results')
    parser.add_argument('--essential_only', action='store_true',
                       help='Run only essential baselines (faster)')
    parser.add_argument('--ablations', action='store_true',
                       help='Include ablation studies')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Update config with args
    config.BATCH_SIZE = args.batch_size
    config.MAX_EPOCHS = args.max_epochs
    config.PATIENCE = args.patience
    
    main(args)