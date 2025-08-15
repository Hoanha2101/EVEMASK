"""
Feature Extraction Model Benchmark Evaluation System - PyTorch Only.
This system provides comprehensive benchmarking capabilities for PyTorch 
feature extraction models in terms of recognition accuracy, embedding quality, and inference speed.

The benchmark evaluates:
- Recognition accuracy using representative vectors and cosine similarity
- Feature embedding quality through separation analysis
- Inference time analysis for PyTorch models
- Detailed performance analysis and visualization

Author: EVEMASK Team
Version: 1.0.0 - PyTorch Only
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tools.utils import *
import torch
import cv2
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
import pandas as pd
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class Network(nn.Module):
    """
    Feature extraction network based on ResNet50 backbone.
    
    This network extracts high-dimensional feature embeddings from input images
    using a pre-trained ResNet50 backbone followed by a custom fully connected layer.
    
    Args:
        emb_dim (int): Dimension of the output embedding vector (default: 128)
    """
    def __init__(self, emb_dim=128):
        super(Network, self).__init__()

        # Load pre-trained ResNet50 and remove the final classification layer
        base_model = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        
        # Custom fully connected layers for feature embedding
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),  # Reduce from ResNet50 output (2048) to 512
            nn.PReLU(),            # Parametric ReLU activation
            nn.Linear(512, emb_dim) # Final embedding dimension
        )

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input image tensor [batch_size, 3, height, width]
            
        Returns:
            torch.Tensor: Feature embedding vector [batch_size, emb_dim]
        """
        x = self.backbone(x)      # Extract features using ResNet50 backbone
        x = torch.flatten(x, 1)   # Flatten spatial dimensions
        x = self.fc(x)            # Project to embedding space
        return x

# Model Configuration and Loading
model = Network(224).to(device)  # Initialize model with 224-dimensional embeddings
checkpoint = torch.load("weights/pytorch/fe_v1.0.0.pt", map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set model to evaluation mode

# Dataset Configuration
folder_path = "DATASET/fe/"  # Path to feature extraction dataset
INPUT_SIZE = (224, 224)      # Input image size for the model

# Pre-allocated tensors for timing measurements
image_random_time = torch.rand(1, 3, 224, 224).cuda()  # PyTorch tensor for GPU timing

def preprocessing(image, half=False):
    """
    Preprocess input image for feature extraction model inference.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        half (bool): Whether to use half precision (float16) for processing
        
    Returns:
        torch.Tensor: Preprocessed tensor ready for model input
    """
    img_process = cv2.resize(image, (224, 224))  # Resize to model input size
    img_process = cv2.cvtColor(img_process, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img_process = img_process.astype(np.float32)  # Convert to float32
    
    # Convert to PyTorch tensor and move to GPU
    img_tensor = torch.from_numpy(img_process).permute(2, 0, 1).unsqueeze(0).cuda()
    if half:
        return img_tensor.half().contiguous()  # Convert to half precision
    return img_tensor

def fe_pytorch(input_img, time_infer=False):
    """
    Perform feature extraction inference using PyTorch model.
    
    Args:
        input_img (numpy.ndarray): Input image in BGR format
        time_infer (bool): If True, only measure inference time without processing
        
    Returns:
        torch.Tensor: Feature embedding vector
    """
    if time_infer:
        # Time inference only using pre-allocated tensor
        with torch.no_grad():
            output = model(image_random_time)
        return output
    else:
        # Full inference pipeline
        input_tensor = preprocessing(input_img)
        with torch.no_grad():
            output = model(input_tensor)
            return output

# Initialize data structures for storing feature vectors and metadata
VECTOR_DICT_PYTORCH = {}  # Dictionary to store PyTorch vectors by class
VECTOR_LIST_PYTORCH = []  # List to store all PyTorch vectors
IMAGE_PATHS_PYTORCH = []  # List to store PyTorch image paths
CLASS_LABELS_PYTORCH = [] # List to store PyTorch class labels

# Load and process images
CLASSES = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
print(f"Found {len(CLASSES)} classes: {CLASSES}")

for class_idx, class_name in enumerate(CLASSES):
    VECTOR_DICT_PYTORCH[class_name] = []
    
    class_path = os.path.join(folder_path, class_name)
    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    print(f"Processing class '{class_name}': {len(image_files)} images")
    
    for image_name in image_files:
        img_path = os.path.join(class_path, image_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        if img is None:
            continue
            
        # PyTorch inference
        output_pytorch = fe_pytorch(img)
        output_pytorch = output_pytorch.cpu().numpy().flatten()  # Flatten to 1D
        VECTOR_DICT_PYTORCH[class_name].append(output_pytorch)
        VECTOR_LIST_PYTORCH.append(output_pytorch)
        IMAGE_PATHS_PYTORCH.append(img_path)
        CLASS_LABELS_PYTORCH.append(class_idx)

def find_class_centroids(vector_dict, vector_list, class_names):
    """
    Find representative centroids for each class using KMeans clustering.
    
    This function ensures each class has exactly one representative vector by:
    1. Using KMeans clustering to find cluster centers
    2. Ensuring each class has exactly one centroid belonging to that class
    
    Args:
        vector_dict (dict): Dictionary containing vectors organized by class
        vector_list (list): List of all vectors
        class_names (list): List of class names
        
    Returns:
        tuple: (representative_indices, representative_vectors, vector_to_class)
            - representative_indices: Indices of representative vectors
            - representative_vectors: The representative vectors themselves
            - vector_to_class: Mapping from vector index to class name
    """
    X = np.vstack(vector_list)  # Stack all vectors into a matrix
    n_clusters = len(vector_dict)  # Number of clusters equals number of classes
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    
    # Create mapping from vector index to class
    vector_to_class = {}
    current_idx = 0
    for class_name in class_names:
        class_vectors = vector_dict[class_name]
        for _ in range(len(class_vectors)):
            vector_to_class[current_idx] = class_name
            current_idx += 1
    
    # Find representative vector for each class
    representative_indices = []
    representative_vectors = []
    used_classes = set()
    
    for i, center in enumerate(centroids):
        # Calculate distances from cluster center to all vectors
        distances = cdist([center], X, metric='euclidean')[0]
        sorted_indices = np.argsort(distances)
        
        # Find the closest vector that belongs to an unused class
        for idx in sorted_indices:
            vector_class = vector_to_class[idx]
            if vector_class not in used_classes:
                representative_indices.append(idx)
                representative_vectors.append(X[idx])
                used_classes.add(vector_class)
                break
    
    return representative_indices, representative_vectors, vector_to_class

# Find centroids for PyTorch
print("\n=== Finding PyTorch Centroids ===")
pytorch_repr_indices, pytorch_repr_vectors, pytorch_vector_to_class = find_class_centroids(
    VECTOR_DICT_PYTORCH, VECTOR_LIST_PYTORCH, CLASSES
)

print("\n=== PyTorch Representative Vectors Belong To Folders ===")
for i, idx in enumerate(pytorch_repr_indices):
    class_name = pytorch_vector_to_class[idx]
    print(f"Representative {i}: Folder = {class_name}")

def accuracy_calculator_fixed(vector_dict, repr_indices, repr_vectors, vector_to_class, class_names):
    """
    Calculate recognition accuracy using representative vectors and cosine similarity.
    
    This function evaluates the recognition performance by:
    1. Using representative vectors as class prototypes
    2. Computing cosine similarity between test vectors and prototypes
    3. Predicting class based on highest similarity
    4. Excluding representative vectors from testing to avoid bias
    
    Args:
        vector_dict (dict): Dictionary containing vectors organized by class
        repr_indices (list): Indices of representative vectors
        repr_vectors (list): Representative vectors for each class
        vector_to_class (dict): Mapping from vector index to class name
        class_names (list): List of class names
    
    Returns:
        tuple: (overall_accuracy, class_results)
            - overall_accuracy: Overall recognition accuracy
            - class_results: Detailed results for each class
    """
    
    # Create mapping from class name to representative vector
    class_to_repr = {}
    for i, repr_idx in enumerate(repr_indices):
        class_name = vector_to_class[repr_idx]
        class_to_repr[class_name] = repr_vectors[i]
    
    total_correct = 0
    total_samples = 0
    class_results = {class_name: {'correct': 0, 'total': 0, 'accuracy': 0.0} for class_name in class_names}
    
    # Iterate through all vectors (excluding representative vectors)
    vector_idx = 0
    for class_name, class_vectors in vector_dict.items():
        for vector in class_vectors:
            # Skip if this is a representative vector
            if vector_idx in repr_indices:
                vector_idx += 1
                continue
            
            # Calculate similarity with all representative vectors
            similarities = []
            for repr_class_name, repr_vector in class_to_repr.items():
                # Use cosine similarity
                cosine_sim = np.dot(vector, repr_vector) / (np.linalg.norm(vector) * np.linalg.norm(repr_vector))
                similarities.append((cosine_sim, repr_class_name))
            
            # Find class with highest similarity
            predicted_class = max(similarities, key=lambda x: x[0])[1]
            
            # Check prediction
            if predicted_class == class_name:
                total_correct += 1
                class_results[class_name]['correct'] += 1
            
            total_samples += 1
            class_results[class_name]['total'] += 1
            vector_idx += 1
    
    # Calculate overall accuracy
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    # Calculate accuracy for each class
    for class_name in class_names:
        if class_results[class_name]['total'] > 0:
            class_results[class_name]['accuracy'] = class_results[class_name]['correct'] / class_results[class_name]['total']
    
    return overall_accuracy, class_results

def print_detailed_accuracy(accuracy, class_results, model_name):
    """
    Print detailed recognition accuracy results.
    
    Args:
        accuracy (float): Overall recognition accuracy
        class_results (dict): Detailed results for each class
        model_name (str): Name of the model (PyTorch)
    """
    print(f"\n=== {model_name} Recognition Accuracy Results ===")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nPer-class Results:")
    print("-" * 50)
    
    for class_name, results in class_results.items():
        print(f"{class_name:20s}: {results['correct']:3d}/{results['total']:3d} = {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    
    # Calculate mean accuracy (macro average)
    mean_accuracy = np.mean([results['accuracy'] for results in class_results.values()])
    print("-" * 50)
    print(f"{'Mean Accuracy':20s}: {mean_accuracy:.4f} ({mean_accuracy*100:.2f}%)")

# Calculate accuracy for PyTorch
pytorch_accuracy, pytorch_class_results = accuracy_calculator_fixed(
    VECTOR_DICT_PYTORCH, pytorch_repr_indices, pytorch_repr_vectors, 
    pytorch_vector_to_class, CLASSES
)

# Print detailed results
print_detailed_accuracy(pytorch_accuracy, pytorch_class_results, "PyTorch")

# Speed benchmark
def benchmark_speed(num_iterations=100):
    """
    Benchmark inference speed for PyTorch model.
    
    Args:
        num_iterations (int): Number of iterations for timing measurement
    """
    print("\n" + "="*50)
    print("SPEED BENCHMARK - PYTORCH")
    print("="*50)
    
    # PyTorch timing
    pytorch_times = []
    for _ in range(num_iterations):
        start_time = time.time()
        _ = fe_pytorch(None, time_infer=True)
        pytorch_times.append(time.time() - start_time)
    
    pytorch_avg = np.mean(pytorch_times) * 1000  # Convert to ms
    pytorch_std = np.std(pytorch_times) * 1000
    pytorch_min = np.min(pytorch_times) * 1000
    pytorch_max = np.max(pytorch_times) * 1000
    
    print(f"PyTorch average inference time: {pytorch_avg:.2f} ms")
    print(f"PyTorch std inference time: {pytorch_std:.2f} ms")
    print(f"PyTorch min inference time: {pytorch_min:.2f} ms")
    print(f"PyTorch max inference time: {pytorch_max:.2f} ms")
    print(f"PyTorch FPS: {1000/pytorch_avg:.1f} frames/sec")
    
    return pytorch_times

def create_benchmark_table():
    """
    Create comprehensive benchmark table for PyTorch performance.
    
    This function generates detailed performance metrics including:
    - Recognition accuracy analysis
    - Inference time analysis
    - Per-class performance breakdown
    - Executive summary with recommendations
    
    Returns:
        tuple: (overall_df, class_df) - Overall and per-class performance dataframes
    """
    
    # Benchmark speed with more iterations for accurate measurement
    print("Running comprehensive speed benchmark...")
    pytorch_times = []
    num_iterations = 100
    
    # PyTorch timing
    for i in range(num_iterations):
        if i % 20 == 0:
            print(f"   PyTorch: {i}/{num_iterations}")
        start_time = time.time()
        _ = fe_pytorch(None, time_infer=True)
        pytorch_times.append(time.time() - start_time)
    
    # Calculate metrics
    pytorch_avg_time = np.mean(pytorch_times) * 1000  # ms
    pytorch_std_time = np.std(pytorch_times) * 1000
    pytorch_min_time = np.min(pytorch_times) * 1000
    pytorch_max_time = np.max(pytorch_times) * 1000
    pytorch_fps = 1000 / pytorch_avg_time
    
    # Overall performance table
    print("\n" + "="*80)
    print("PYTORCH BENCHMARK RESULTS")
    print("="*80)
    
    overall_data = {
        'Metric': [
            'Overall Accuracy (%)',
            'Mean Class Accuracy (%)', 
            'Avg Inference Time (ms)',
            'Std Inference Time (ms)',
            'Min Inference Time (ms)',
            'Max Inference Time (ms)',
            'FPS (Frames/sec)'
        ],
        'PyTorch': [
            f"{pytorch_accuracy*100:.2f}",
            f"{np.mean([r['accuracy'] for r in pytorch_class_results.values()])*100:.2f}",
            f"{pytorch_avg_time:.3f}",
            f"{pytorch_std_time:.3f}",
            f"{pytorch_min_time:.3f}",
            f"{pytorch_max_time:.3f}",
            f"{pytorch_fps:.1f}"
        ]
    }
    
    overall_df = pd.DataFrame(overall_data)
    print("\nOVERALL PERFORMANCE")
    print("-" * 80)
    print(overall_df.to_string(index=False))
    
    # Per-class detailed table
    class_data = {
        'Class': [],
        'PyTorch Acc (%)': [],
        'PyTorch Samples': [],
        'Status': []
    }
    
    for class_name in CLASSES:
        pytorch_acc = pytorch_class_results[class_name]['accuracy'] * 100
        pytorch_samples = pytorch_class_results[class_name]['total']
        
        class_data['Class'].append(class_name)
        class_data['PyTorch Acc (%)'].append(f"{pytorch_acc:.2f}")
        class_data['PyTorch Samples'].append(pytorch_samples)
        
        if pytorch_acc >= 90:
            status = "Excellent"
        elif pytorch_acc >= 80:
            status = "Good"
        elif pytorch_acc >= 70:
            status = "Fair"
        else:
            status = "Poor"
        class_data['Status'].append(status)
    
    class_df = pd.DataFrame(class_data)
    print("\nPER-CLASS ACCURACY ANALYSIS")
    print("-" * 80)
    print(class_df.to_string(index=False))
    
    # Executive Summary
    print("\n" + "="*60)
    print("EXECUTIVE SUMMARY")
    print("="*60)
    
    print(f"Model Performance:")
    print(f"   • Overall Accuracy: {pytorch_accuracy*100:.2f}%")
    print(f"   • Mean Class Accuracy: {np.mean([r['accuracy'] for r in pytorch_class_results.values()])*100:.2f}%")
    
    print(f"\nSpeed Performance:")
    print(f"   • Average Time: {pytorch_avg_time:.3f} ms/frame")
    print(f"   • Throughput: {pytorch_fps:.1f} FPS")
    print(f"   • Consistency: {pytorch_std_time:.3f} ms std")
    
    print(f"\nRecommendation:")
    if pytorch_accuracy >= 0.9:
        print("   PyTorch model shows EXCELLENT recognition performance")
    elif pytorch_accuracy >= 0.8:
        print("   PyTorch model shows GOOD recognition performance")
    elif pytorch_accuracy >= 0.7:
        print("   PyTorch model shows FAIR recognition performance")
    else:
        print("   PyTorch model needs improvement in recognition accuracy")
    
    if pytorch_fps >= 30:
        print("   PyTorch model shows EXCELLENT speed performance")
    elif pytorch_fps >= 20:
        print("   PyTorch model shows GOOD speed performance")
    elif pytorch_fps >= 10:
        print("   PyTorch model shows FAIR speed performance")
    else:
        print("   PyTorch model needs improvement in speed performance")
    
    return overall_df, class_df

# Save benchmark results to CSV
def save_benchmark_results_to_csv(overall_df, class_df, csv_dir):
    """
    Save benchmark results to CSV files.
    Args:
        overall_df: Overall performance comparison dataframe
        class_df: Per-class accuracy comparison dataframe  
        csv_dir: Directory to save CSV files
    """
    os.makedirs(csv_dir, exist_ok=True)
    
    # Save overall performance comparison
    overall_csv_path = os.path.join(csv_dir, 'pytorch_overall_performance.csv')
    overall_df.to_csv(overall_csv_path, index=False)
    
    # Save per-class accuracy comparison
    class_csv_path = os.path.join(csv_dir, 'pytorch_per_class_accuracy.csv')
    class_df.to_csv(class_csv_path, index=False)
    
    print(f"Overall performance saved to: {overall_csv_path}")
    print(f"Per-class accuracy saved to: {class_csv_path}")

# Run comparison
overall_df, class_df = create_benchmark_table()

# Save benchmark results to CSV
results_dir = os.path.join('benchmarks', 'results', 'fe_pytorch_benchmark')
save_benchmark_results_to_csv(overall_df, class_df, results_dir)

def visualize_embeddings():
    """
    Create t-SNE visualization of PyTorch embeddings.
    
    This function generates 2D t-SNE plots to visualize the distribution
    of feature embeddings from PyTorch model, showing how well they separate
    different classes in the embedding space.
    """
    try:
        print("\n=== Creating PyTorch Embeddings Visualization ===")
        
        # Combine all vectors for t-SNE
        pytorch_vectors = np.vstack(VECTOR_LIST_PYTORCH)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        pytorch_2d = tsne.fit_transform(pytorch_vectors)
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(CLASSES)))

        # PyTorch plot
        for i, class_name in enumerate(CLASSES):
            class_mask = np.array(CLASS_LABELS_PYTORCH) == i
            plt.scatter(pytorch_2d[class_mask, 0], pytorch_2d[class_mask, 1], 
                        c=[colors[i]], label=class_name, alpha=0.6)
        
        for idx in pytorch_repr_indices:
            plt.scatter(pytorch_2d[idx, 0], pytorch_2d[idx, 1], 
                        c='red', s=100, marker='x', linewidth=3, label='Representative Vectors' if idx == pytorch_repr_indices[0] else "")
        
        plt.title('PyTorch Feature Embeddings Visualization', fontsize=16, fontweight='bold')
        plt.xlabel('t-SNE 1', fontsize=12)
        plt.ylabel('t-SNE 2', fontsize=12)
        plt.legend(title="Classes", loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True, alpha=0.3)
        
        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, 'pytorch_embeddings_visualization.png'), dpi=300, bbox_inches='tight')
        plt.show()

    except Exception as e:
        print(f"Visualization failed: {e}")

def calculate_separation_scores(vector_dict, repr_indices, repr_vectors, vector_to_class, class_names, model_name):
    """
    Calculate separation scores for feature embeddings.
    
    In-class score: cosine similarity between the class representative vector
    and ALL other vectors in the same class (excluding the representative itself).
    Out-class score: cosine similarity between the class representative vector
    and ALL vectors from other classes.
    Separation score = in_class_score - out_class_score
    """

    # Map: class -> representative vector
    class_to_repr = {
        vector_to_class[repr_idx]: repr_vectors[i]
        for i, repr_idx in enumerate(repr_indices)
    }

    in_class_scores = []
    out_class_scores = []
    separation_scores = []

    for cls in class_names:
        repr_vec = class_to_repr[cls]

        # All vectors in the same class, excluding repr itself
        same_class_vectors = [
            vec for vec in vector_dict[cls]
            if not np.allclose(vec, repr_vec)
        ]

        # All vectors from other classes
        diff_class_vectors = [
            vec
            for other_cls in class_names if other_cls != cls
            for vec in vector_dict[other_cls]
        ]

        # In-class score
        in_class_cosines = [
            np.dot(repr_vec, vec) / (np.linalg.norm(repr_vec) * np.linalg.norm(vec))
            for vec in same_class_vectors
        ]
        in_class_score = np.mean(in_class_cosines) if in_class_cosines else 0.0

        # Out-class score
        out_class_cosines = [
            np.dot(repr_vec, vec) / (np.linalg.norm(repr_vec) * np.linalg.norm(vec))
            for vec in diff_class_vectors
        ]
        out_class_score = np.mean(out_class_cosines) if out_class_cosines else 0.0

        # Separation
        in_class_scores.append(in_class_score)
        out_class_scores.append(out_class_score)
        separation_scores.append(in_class_score - out_class_score)

    return in_class_scores, out_class_scores, separation_scores

def plot_separation_analysis():
    """
    Create separation analysis plots for PyTorch model.
    
    This function generates comprehensive visualization of separation scores:
    1. Individual separation performance for PyTorch model
    2. Detailed analysis showing in-class vs out-class similarity
    
    Returns:
        pytorch_separation: Separation scores for PyTorch model
    """
    
    # Calculate separation scores for PyTorch
    pytorch_in_class, pytorch_out_class, pytorch_separation = calculate_separation_scores(
        VECTOR_DICT_PYTORCH, pytorch_repr_indices, pytorch_repr_vectors, 
        pytorch_vector_to_class, CLASSES, "PyTorch"
    )
    
    # Create 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    x = np.arange(len(CLASSES))
    bar_width = 0.6
    
    # Colors
    pytorch_color = 'blue'
    trend_color = 'red'
    mean_color = 'green'
    
    # Calculate mean separation score
    pytorch_mean_sep = np.mean(pytorch_separation)
    
    # Plot 1: PyTorch Separation
    bars1 = axes[0].bar(x, pytorch_separation, width=bar_width, color=pytorch_color, 
                        alpha=0.7, label='Separation Score')
    
    # Add trend line
    axes[0].plot(x, pytorch_separation, color=trend_color, linewidth=2, 
                marker='o', markersize=6, label='Trend')
    
    # Add mean line (horizontal)
    axes[0].axhline(y=pytorch_mean_sep, color=mean_color, linestyle='--', 
                   linewidth=2, alpha=0.8, label=f'Mean: {pytorch_mean_sep:.4f}')
    
    axes[0].set_xlabel("Class Index", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Separation Score", fontsize=12, fontweight='bold')
    axes[0].set_title("PyTorch Model Separation Performance", fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"C{i}" for i in range(len(CLASSES))], rotation=45)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, pytorch_separation)):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Plot 2: In-class vs Out-class comparison
    x_pos = np.arange(len(CLASSES))
    width = 0.35
    
    bars2 = axes[1].bar(x_pos - width/2, pytorch_in_class, width, label='In-class Similarity', 
                        color='green', alpha=0.7)
    bars3 = axes[1].bar(x_pos + width/2, pytorch_out_class, width, label='Out-class Similarity', 
                        color='red', alpha=0.7)
    
    axes[1].set_xlabel("Class Index", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Cosine Similarity", fontsize=12, fontweight='bold')
    axes[1].set_title("In-class vs Out-class Similarity", fontsize=14, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f"C{i}" for i in range(len(CLASSES))], rotation=45)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars2, bars3]:
        for bar, val in zip(bars, pytorch_separation):
            axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.tight_layout()
    
    # Save combined plot
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'pytorch_separation_analysis.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Create individual separation plot
    plt.figure(figsize=(12, 6))
    
    bars_single = plt.bar(x, pytorch_separation, width=bar_width, color=pytorch_color, 
                         alpha=0.7, label='Separation Score', edgecolor='black', linewidth=0.5)
    
    # Trend line
    plt.plot(x, pytorch_separation, color=trend_color, linewidth=2, 
            marker='o', markersize=6, label='Trend')
    
    # Mean line
    plt.axhline(y=pytorch_mean_sep, color=mean_color, linestyle='--', 
               linewidth=2, alpha=0.8, label=f'Mean: {pytorch_mean_sep:.4f}')
    
    plt.xlabel("Class Index", fontsize=12, fontweight='bold')
    plt.ylabel("Separation Score", fontsize=12, fontweight='bold')
    plt.title("PyTorch Model Separation Performance Chart", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(x, [f"C{i}" for i in range(len(CLASSES))], rotation=45)
    
    # Add value annotations
    for i, (bar, val) in enumerate(zip(bars_single, pytorch_separation)):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, rotation=90)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'pytorch_separation_chart.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nPyTorch Model Statistics:")
    print(f"   Mean Separation Score: {pytorch_mean_sep:.6f}")
    print(f"   Standard Deviation:    {np.std(pytorch_separation):.6f}")
    print(f"   Min Separation:        {np.min(pytorch_separation):.6f}")
    print(f"   Max Separation:        {np.max(pytorch_separation):.6f}")
    print(f"   Range:                 {np.max(pytorch_separation) - np.min(pytorch_separation):.6f}")
    
    return pytorch_separation

# Save separation analysis results to CSV
def save_separation_results_to_csv(pytorch_separation, class_names, csv_dir):
    """
    Save separation analysis results to CSV file.
    Args:
        pytorch_separation: PyTorch separation scores
        class_names: List of class names
        csv_dir: Directory to save CSV file
    """
    os.makedirs(csv_dir, exist_ok=True)
    
    # Create separation results dataframe
    separation_data = {
        'Class': class_names,
        'PyTorch_Separation_Score': pytorch_separation
    }
    
    separation_df = pd.DataFrame(separation_data)
    
    # Save to CSV
    separation_csv_path = os.path.join(csv_dir, 'pytorch_separation_analysis.csv')
    separation_df.to_csv(separation_csv_path, index=False)
    
    print(f"Separation analysis saved to: {separation_csv_path}")
    
    # Save summary statistics
    summary_data = {
        'Metric': ['Mean', 'Std', 'Min', 'Max', 'Range'],
        'PyTorch': [
            np.mean(pytorch_separation),
            np.std(pytorch_separation),
            np.min(pytorch_separation),
            np.max(pytorch_separation),
            np.max(pytorch_separation) - np.min(pytorch_separation)
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(csv_dir, 'pytorch_separation_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    
    print(f"Separation summary saved to: {summary_csv_path}")

# Create visualizations
visualize_embeddings()

# Create separation analysis plots
pytorch_separation = plot_separation_analysis()

# Save separation analysis results to CSV
save_separation_results_to_csv(pytorch_separation, CLASSES, results_dir)

print(f"\nAll PyTorch benchmark results saved to: {results_dir}")
