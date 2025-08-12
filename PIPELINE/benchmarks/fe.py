"""
Feature Extraction Model Benchmark Evaluation System.
This system provides comprehensive benchmarking capabilities for comparing PyTorch and TensorRT 
feature extraction models in terms of recognition accuracy, embedding quality, and inference speed.

The benchmark evaluates:
- Recognition accuracy using representative vectors and cosine similarity
- Feature embedding quality through separation analysis
- Inference time comparison between PyTorch and TensorRT models
- Detailed performance analysis and visualization including t-SNE plots

Author: EVEMASK Team
Version: 1.0.0
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tools.utils import *
from src.models.initNet import net2
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
image_random_time_half = image_random_time.half().contiguous()  # TensorRT tensor for timing

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
        return img_tensor.half().contiguous()  # Convert to half precision for TensorRT
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
        
def fe_trt(input_img, time_infer=False):
    """
    Perform feature extraction inference using TensorRT model.
    
    Args:
        input_img (numpy.ndarray): Input image in BGR format
        time_infer (bool): If True, only measure inference time without processing
        
    Returns:
        numpy.ndarray or torch.Tensor: Feature embedding vector
    """
    if time_infer:
        # Time inference only using pre-allocated tensor
        output = net2.infer(image_random_time_half)
        return output
    else:
        # Full inference pipeline
        input_tensor = preprocessing(input_img, half=True)
        output = net2.infer(input_tensor)
        return output

# Initialize data structures for storing feature vectors and metadata
VECTOR_DICT_PYTORCH = {}  # Dictionary to store PyTorch vectors by class
VECTOR_LIST_PYTORCH = []  # List to store all PyTorch vectors
IMAGE_PATHS_PYTORCH = []  # List to store PyTorch image paths
CLASS_LABELS_PYTORCH = [] # List to store PyTorch class labels

VECTOR_DICT_TRT = {}      # Dictionary to store TensorRT vectors by class
VECTOR_LIST_TRT = []      # List to store all TensorRT vectors
IMAGE_PATHS_TRT = []      # List to store TensorRT image paths
CLASS_LABELS_TRT = []     # List to store TensorRT class labels

# Load and process images
CLASSES = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
print(f"Found {len(CLASSES)} classes: {CLASSES}")

for class_idx, class_name in enumerate(CLASSES):
    VECTOR_DICT_PYTORCH[class_name] = []
    VECTOR_DICT_TRT[class_name] = []
    
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
        
        # TensorRT inference
        output_trt = fe_trt(img)
        if isinstance(output_trt, torch.Tensor):
            output_trt = output_trt.cpu().numpy().flatten()
        else:
            output_trt = output_trt.flatten()
        VECTOR_DICT_TRT[class_name].append(output_trt)
        VECTOR_LIST_TRT.append(output_trt)
        IMAGE_PATHS_TRT.append(img_path)
        CLASS_LABELS_TRT.append(class_idx)

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

# Find centroids for TensorRT
print("\n=== Finding TensorRT Centroids ===")
trt_repr_indices, trt_repr_vectors, trt_vector_to_class = find_class_centroids(
    VECTOR_DICT_TRT, VECTOR_LIST_TRT, CLASSES
)

print("\n=== PyTorch Representative Vectors Belong To Folders ===")
for i, idx in enumerate(pytorch_repr_indices):
    class_name = pytorch_vector_to_class[idx]
    print(f"Representative {i}: Folder = {class_name}")

print("\n=== TensorRT Representative Vectors Belong To Folders ===")
for i, idx in enumerate(trt_repr_indices):
    class_name = trt_vector_to_class[idx]
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
        model_name (str): Name of the model (PyTorch or TensorRT)
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

# Calculate accuracy for TensorRT
trt_accuracy, trt_class_results = accuracy_calculator_fixed(
    VECTOR_DICT_TRT, trt_repr_indices, trt_repr_vectors, 
    trt_vector_to_class, CLASSES
)

# Print detailed results
print_detailed_accuracy(pytorch_accuracy, pytorch_class_results, "PyTorch")
print_detailed_accuracy(trt_accuracy, trt_class_results, "TensorRT")

# Compare results
def compare_results():
    """
    Compare PyTorch and TensorRT model performance comprehensively.
    
    This function provides detailed comparison of:
    - Recognition accuracy
    - Representative vector analysis
    - Vector similarity metrics
    - Overall performance statistics
    """
    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    
    print(f"Number of classes: {len(CLASSES)}")
    print(f"PyTorch total vectors: {len(VECTOR_LIST_PYTORCH)}")
    print(f"TensorRT total vectors: {len(VECTOR_LIST_TRT)}")
    
    print(f"\nPyTorch representative vectors: {len(pytorch_repr_vectors)}")
    print(f"TensorRT representative vectors: {len(trt_repr_vectors)}")
    
    # Compare accuracy
    print(f"\nAccuracy Comparison:")
    print(f"PyTorch Recognition Accuracy: {pytorch_accuracy:.4f} ({pytorch_accuracy*100:.2f}%)")
    print(f"TensorRT Recognition Accuracy: {trt_accuracy:.4f} ({trt_accuracy*100:.2f}%)")
    print(f"Accuracy Difference: {abs(pytorch_accuracy - trt_accuracy):.4f}")
    
    # Compare representative vectors of each class
    print("\n--- Class Representatives ---")
    for i, class_name in enumerate(CLASSES):
        pytorch_class = pytorch_vector_to_class[pytorch_repr_indices[i]]
        trt_class = trt_vector_to_class[trt_repr_indices[i]]
        
        # Calculate cosine similarity between PyTorch and TensorRT vectors
        pytorch_vec = pytorch_repr_vectors[i]
        trt_vec = trt_repr_vectors[i]
        
        cosine_sim = np.dot(pytorch_vec, trt_vec) / (np.linalg.norm(pytorch_vec) * np.linalg.norm(trt_vec))
        euclidean_dist = np.linalg.norm(pytorch_vec - trt_vec)
        
        print(f"Class {class_name}:")
        print(f"  PyTorch rep class: {pytorch_class}")
        print(f"  TensorRT rep class: {trt_class}")
        print(f"  Cosine similarity: {cosine_sim:.4f}")
        print(f"  Euclidean distance: {euclidean_dist:.4f}")
    
    # Overall statistics
    all_cosine_sims = []
    all_euclidean_dists = []
    
    for i in range(min(len(pytorch_repr_vectors), len(trt_repr_vectors))):
        pytorch_vec = pytorch_repr_vectors[i]
        trt_vec = trt_repr_vectors[i]
        
        cosine_sim = np.dot(pytorch_vec, trt_vec) / (np.linalg.norm(pytorch_vec) * np.linalg.norm(trt_vec))
        euclidean_dist = np.linalg.norm(pytorch_vec - trt_vec)
        
        all_cosine_sims.append(cosine_sim)
        all_euclidean_dists.append(euclidean_dist)
    
    print(f"\n--- Overall Statistics ---")
    print(f"Average cosine similarity: {np.mean(all_cosine_sims):.4f} ± {np.std(all_cosine_sims):.4f}")
    print(f"Average euclidean distance: {np.mean(all_euclidean_dists):.4f} ± {np.std(all_euclidean_dists):.4f}")
    print(f"Min cosine similarity: {np.min(all_cosine_sims):.4f}")
    print(f"Max cosine similarity: {np.max(all_cosine_sims):.4f}")

# Speed benchmark
def benchmark_speed(num_iterations=100):
    """
    Benchmark inference speed for PyTorch and TensorRT models.
    
    Args:
        num_iterations (int): Number of iterations for timing measurement
    """
    print("\n" + "="*50)
    print("SPEED BENCHMARK")
    print("="*50)
    
    # PyTorch timing
    pytorch_times = []
    for _ in range(num_iterations):
        start_time = time.time()
        _ = fe_pytorch(None, time_infer=True)
        pytorch_times.append(time.time() - start_time)
    
    # TensorRT timing
    trt_times = []
    for _ in range(num_iterations):
        start_time = time.time()
        _ = fe_trt(None, time_infer=True)
        trt_times.append(time.time() - start_time)
    
    pytorch_avg = np.mean(pytorch_times) * 1000  # Convert to ms
    trt_avg = np.mean(trt_times) * 1000
    
    print(f"PyTorch average inference time: {pytorch_avg:.2f} ms")
    print(f"TensorRT average inference time: {trt_avg:.2f} ms")
    print(f"Speed improvement: {pytorch_avg/trt_avg:.2f}x")

# Import pandas for table creation
import pandas as pd

def create_benchmark_table():
    """
    Create comprehensive benchmark table comparing PyTorch and TensorRT performance.
    
    This function generates detailed performance metrics including:
    - Recognition accuracy comparison
    - Inference time analysis
    - Per-class performance breakdown
    - Executive summary with recommendations
    
    Returns:
        tuple: (overall_df, class_df) - Overall and per-class performance dataframes
    """
    
    # Benchmark speed with more iterations for accurate measurement
    print("Running comprehensive speed benchmark...")
    pytorch_times = []
    trt_times = []
    num_iterations = 100
    
    # PyTorch timing
    for i in range(num_iterations):
        if i % 20 == 0:
            print(f"   PyTorch: {i}/{num_iterations}")
        start_time = time.time()
        _ = fe_pytorch(None, time_infer=True)
        pytorch_times.append(time.time() - start_time)
    
    # TensorRT timing  
    for i in range(num_iterations):
        if i % 20 == 0:
            print(f"   TensorRT: {i}/{num_iterations}")
        start_time = time.time()
        _ = fe_trt(None, time_infer=True)
        trt_times.append(time.time() - start_time)
    
    # Calculate metrics
    pytorch_avg_time = np.mean(pytorch_times) * 1000  # ms
    trt_avg_time = np.mean(trt_times) * 1000
    pytorch_std_time = np.std(pytorch_times) * 1000
    trt_std_time = np.std(trt_times) * 1000
    speedup = pytorch_avg_time / trt_avg_time
    
    # Overall performance table
    print("\n" + "="*80)
    print("PYTORCH vs TENSORRT BENCHMARK RESULTS")
    print("="*80)
    
    overall_data = {
        'Metric': [
            'Overall Accuracy (%)',
            'Mean Class Accuracy (%)', 
            'Avg Inference Time (ms)',
            'Std Inference Time (ms)',
            'Min Inference Time (ms)',
            'Max Inference Time (ms)',
            'FPS (Frames/sec)',
            'Speedup Factor'
        ],
        'PyTorch': [
            f"{pytorch_accuracy*100:.2f}",
            f"{np.mean([r['accuracy'] for r in pytorch_class_results.values()])*100:.2f}",
            f"{pytorch_avg_time:.3f}",
            f"{pytorch_std_time:.3f}",
            f"{np.min(pytorch_times)*1000:.3f}",
            f"{np.max(pytorch_times)*1000:.3f}",
            f"{1000/pytorch_avg_time:.1f}",
            "1.00x (baseline)"
        ],
        'TensorRT': [
            f"{trt_accuracy*100:.2f}",
            f"{np.mean([r['accuracy'] for r in trt_class_results.values()])*100:.2f}",
            f"{trt_avg_time:.3f}",
            f"{trt_std_time:.3f}",
            f"{np.min(trt_times)*1000:.3f}",
            f"{np.max(trt_times)*1000:.3f}",
            f"{1000/trt_avg_time:.1f}",
            f"{speedup:.2f}x"
        ],
        'Difference': [
            f"{(trt_accuracy - pytorch_accuracy)*100:+.2f}%",
            f"{(np.mean([r['accuracy'] for r in trt_class_results.values()]) - np.mean([r['accuracy'] for r in pytorch_class_results.values()]))*100:+.2f}%",
            f"{trt_avg_time - pytorch_avg_time:+.3f}",
            f"{trt_std_time - pytorch_std_time:+.3f}",
            f"{(np.min(trt_times) - np.min(pytorch_times))*1000:+.3f}",
            f"{(np.max(trt_times) - np.max(pytorch_times))*1000:+.3f}",
            f"{1000/trt_avg_time - 1000/pytorch_avg_time:+.1f}",
            f"{'Faster' if speedup > 1 else 'Slower'}"
        ]
    }
    
    overall_df = pd.DataFrame(overall_data)
    print("\nOVERALL PERFORMANCE COMPARISON")
    print("-" * 80)
    print(overall_df.to_string(index=False))
    
    # Per-class detailed table
    class_data = {
        'Class': [],
        'PyTorch Acc (%)': [],
        'TensorRT Acc (%)': [],
        'PyTorch Samples': [],
        'TensorRT Samples': [],
        'Acc Diff (%)': [],
        'Status': []
    }
    
    for class_name in CLASSES:
        pytorch_acc = pytorch_class_results[class_name]['accuracy'] * 100
        trt_acc = trt_class_results[class_name]['accuracy'] * 100
        pytorch_samples = pytorch_class_results[class_name]['total']
        trt_samples = trt_class_results[class_name]['total']
        acc_diff = trt_acc - pytorch_acc
        
        class_data['Class'].append(class_name)
        class_data['PyTorch Acc (%)'].append(f"{pytorch_acc:.2f}")
        class_data['TensorRT Acc (%)'].append(f"{trt_acc:.2f}")
        class_data['PyTorch Samples'].append(pytorch_samples)
        class_data['TensorRT Samples'].append(trt_samples) 
        class_data['Acc Diff (%)'].append(f"{acc_diff:+.2f}")
        
        if abs(acc_diff) <= 1:
            status = "Equivalent"
        elif acc_diff > 1:
            status = "TRT Better"
        else:
            status = "PyTorch Better"
        class_data['Status'].append(status)
    
    class_df = pd.DataFrame(class_data)
    print("\nPER-CLASS ACCURACY COMPARISON")
    print("-" * 80)
    print(class_df.to_string(index=False))
    
    # Executive Summary
    acc_diff = abs(trt_accuracy - pytorch_accuracy) * 100
    
    print("\n" + "="*60)
    print("EXECUTIVE SUMMARY")
    print("="*60)
    
    print(f"Accuracy Impact:")
    print(f"   • PyTorch:     {pytorch_accuracy*100:.2f}%")
    print(f"   • TensorRT:    {trt_accuracy*100:.2f}%")
    print(f"   • Difference:  {trt_accuracy*100 - pytorch_accuracy*100:+.2f}%")
    
    print(f"\nPerformance Gain:")
    print(f"   • PyTorch:     {pytorch_avg_time:.3f} ms/frame")
    print(f"   • TensorRT:    {trt_avg_time:.3f} ms/frame") 
    print(f"   • Speedup:     {speedup:.2f}x faster")
    print(f"   • FPS Gain:    {1000/trt_avg_time - 1000/pytorch_avg_time:+.1f} FPS")
    
    print(f"\nRecommendation:")
    if speedup > 2 and acc_diff < 2:
        print("   TensorRT is HIGHLY RECOMMENDED")
        print("   Significant speed improvement with minimal accuracy loss")
    elif speedup > 1.5 and acc_diff < 5:
        print("   TensorRT is RECOMMENDED") 
        print("   Good speed improvement with acceptable accuracy trade-off")
    elif speedup > 1.2:
        print("   TensorRT shows moderate improvement")
        print("   Consider based on your specific requirements")
    else:
        print("   TensorRT optimization may not be worth it")
        print("   Limited speed improvement")
    
    return overall_df, class_df

# Run comparison
compare_results()
create_benchmark_table()

def visualize_embeddings():
    """
    Create t-SNE visualization of PyTorch and TensorRT embeddings.
    
    This function generates 2D t-SNE plots to visualize the distribution
    of feature embeddings from both models, showing how well they separate
    different classes in the embedding space.
    """
    try:
        print("\n=== Creating Visualization ===")
        
        # Combine all vectors for t-SNE
        pytorch_vectors = np.vstack(VECTOR_LIST_PYTORCH)
        trt_vectors = np.vstack(VECTOR_LIST_TRT)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        pytorch_2d = tsne.fit_transform(pytorch_vectors)
        trt_2d = tsne.fit_transform(trt_vectors)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(CLASSES)))

        # PyTorch plot
        for i, class_name in enumerate(CLASSES):
            class_mask = np.array(CLASS_LABELS_PYTORCH) == i
            ax1.scatter(pytorch_2d[class_mask, 0], pytorch_2d[class_mask, 1], 
                        c=[colors[i]], label=class_name, alpha=0.6)
        
        for idx in pytorch_repr_indices:
            ax1.scatter(pytorch_2d[idx, 0], pytorch_2d[idx, 1], 
                        c='red', s=100, marker='x', linewidth=3)
        
        ax1.set_title('PyTorch Embeddings')
        ax1.set_xlabel('t-SNE 1')
        ax1.set_ylabel('t-SNE 2')
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Classes")

        # TensorRT plot
        for i, class_name in enumerate(CLASSES):
            class_mask = np.array(CLASS_LABELS_TRT) == i
            ax2.scatter(trt_2d[class_mask, 0], trt_2d[class_mask, 1], 
                        c=[colors[i]], label=class_name, alpha=0.6)
        
        for idx in trt_repr_indices:
            ax2.scatter(trt_2d[idx, 0], trt_2d[idx, 1], 
                        c='red', s=100, marker='x', linewidth=3)
        
        ax2.set_title('TensorRT Embeddings')
        ax2.set_xlabel('t-SNE 1')
        ax2.set_ylabel('t-SNE 2')
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Classes")

        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Shrink plot area to make room for legend
        plt.savefig('benchmarks/results/pytorch_vs_tensorrt_centroid.png', dpi=300, bbox_inches='tight')
        plt.show()

    except Exception as e:
        print(f"Visualization failed: {e}")

def calculate_separation_scores(vector_dict, repr_indices, repr_vectors, vector_to_class, class_names, model_name):
    """
    Calculate separation scores using anchor method:
    - For each class anchor, compute similarity with all class representatives
    - In-class score = similarity with its own repr
    - Out-class score = average similarity with other class reprs
    """
    # Map from class to repr vector
    class_to_repr = {
        vector_to_class[repr_idx]: repr_vectors[i]
        for i, repr_idx in enumerate(repr_indices)
    }

    # Build full_list_output: similarity matrix [num_classes x num_classes]
    full_list_output = []
    for anchor_class in class_names:
        anchor_vectors = vector_dict[anchor_class]
        anchor_row = []

        # Representative vectors for each class
        for target_class in class_names:
            target_repr = class_to_repr[target_class]
            cosines = [
                np.dot(vec, target_repr) / (np.linalg.norm(vec) * np.linalg.norm(target_repr))
                for vec in anchor_vectors
            ]
            anchor_row.append(np.mean(cosines) if cosines else 0.0)

        full_list_output.append(anchor_row)

    # Compute in_class, out_class, separation (chuẩn anchor method)
    in_class_scores = []
    out_class_scores = []
    separation_scores = []

    num_classes = len(class_names)

    for idx, list_output in enumerate(full_list_output):
        in_class_score = list_output[idx]
        out_class_score = (sum(list_output) - in_class_score) / (num_classes - 1)

        in_class_scores.append(in_class_score)
        out_class_scores.append(out_class_score)
        separation_scores.append(in_class_score - out_class_score)
        
    return in_class_scores, out_class_scores, separation_scores


def plot_separation_analysis():
    """
    Create 3 separation analysis plots: PyTorch, TensorRT, and Combined comparison.
    
    This function generates comprehensive visualization of separation scores:
    1. Individual separation performance for PyTorch model
    2. Individual separation performance for TensorRT model  
    3. Combined comparison showing relative performance
    
    Returns:
        tuple: (pytorch_separation, trt_separation) - Separation scores for both models
    """
    print("\n=== Creating Separation Analysis Plots ===")
    
    # Calculate separation scores for PyTorch
    pytorch_in_class, pytorch_out_class, pytorch_separation = calculate_separation_scores(
        VECTOR_DICT_PYTORCH, pytorch_repr_indices, pytorch_repr_vectors, 
        pytorch_vector_to_class, CLASSES, "PyTorch"
    )
    
    # Calculate separation scores for TensorRT
    trt_in_class, trt_out_class, trt_separation = calculate_separation_scores(
        VECTOR_DICT_TRT, trt_repr_indices, trt_repr_vectors, 
        trt_vector_to_class, CLASSES, "TensorRT"
    )
    
    # Create 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    x = np.arange(len(CLASSES))
    bar_width = 0.6
    
    # Colors
    pytorch_color = '#1f77b4'  # Blue
    trt_color = '#ff7f0e'      # Orange
    
    # Plot 1: PyTorch Separation
    axes[0].bar(x, pytorch_separation, width=bar_width, color=pytorch_color, alpha=0.8, label='PyTorch Separation')
    axes[0].plot(x, pytorch_separation, color='red', linewidth=2, marker='o', label='Trend')
    axes[0].axhline(y=np.mean(pytorch_separation), color='green', linestyle='--', alpha=0.7, 
                   label=f'Mean: {np.mean(pytorch_separation):.3f}')
    
    axes[0].set_xlabel("Class Index")
    axes[0].set_ylabel("Separation Score")
    axes[0].set_title("PyTorch Model Separation Performance")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"C{i}" for i in range(len(CLASSES))], rotation=45)
    
    # Plot 2: TensorRT Separation
    axes[1].bar(x, trt_separation, width=bar_width, color=trt_color, alpha=0.8, label='TensorRT Separation')
    axes[1].plot(x, trt_separation, color='red', linewidth=2, marker='s', label='Trend')
    axes[1].axhline(y=np.mean(trt_separation), color='green', linestyle='--', alpha=0.7,
                   label=f'Mean: {np.mean(trt_separation):.3f}')
    
    axes[1].set_xlabel("Class Index")
    axes[1].set_ylabel("Separation Score")
    axes[1].set_title("TensorRT Model Separation Performance")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"C{i}" for i in range(len(CLASSES))], rotation=45)
    
    # Plot 3: Combined Comparison
    bar_width_combined = 0.35
    x_pytorch = x - bar_width_combined/2
    x_trt = x + bar_width_combined/2
    
    bars1 = axes[2].bar(x_pytorch, pytorch_separation, width=bar_width_combined, 
                       color=pytorch_color, alpha=0.8, label='PyTorch')
    bars2 = axes[2].bar(x_trt, trt_separation, width=bar_width_combined, 
                       color=trt_color, alpha=0.8, label='TensorRT')
    
    # Trend lines
    axes[2].plot(x_pytorch, pytorch_separation, color='blue', linewidth=2, marker='o', alpha=0.7)
    axes[2].plot(x_trt, trt_separation, color='orange', linewidth=2, marker='s', alpha=0.7)
    
    # Mean lines
    axes[2].axhline(y=np.mean(pytorch_separation), color=pytorch_color, linestyle='--', alpha=0.5,
                   label=f'PyTorch Mean: {np.mean(pytorch_separation):.3f}')
    axes[2].axhline(y=np.mean(trt_separation), color=trt_color, linestyle='--', alpha=0.5,
                   label=f'TensorRT Mean: {np.mean(trt_separation):.3f}')
    
    axes[2].set_xlabel("Class Index")
    axes[2].set_ylabel("Separation Score")
    axes[2].set_title("PyTorch vs TensorRT Separation Comparison")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"C{i}" for i in range(len(CLASSES))], rotation=45)
    
    plt.tight_layout()
    
    # Save individual plots
    os.makedirs('benchmarks/results', exist_ok=True)
    
    # Save combined plot
    plt.savefig('benchmarks/results/separation_analysis_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save individual plots separately
    print("\n=== Saving Individual Plots ===")
    
    # Plot 1: PyTorch only
    plt.figure(figsize=(10, 6))
    plt.bar(x, pytorch_separation, width=bar_width, color=pytorch_color, alpha=0.8, label='PyTorch Separation')
    plt.plot(x, pytorch_separation, color='red', linewidth=2, marker='o', label='Trend')
    plt.axhline(y=np.mean(pytorch_separation), color='green', linestyle='--', alpha=0.7, 
               label=f'Mean: {np.mean(pytorch_separation):.3f}')
    plt.xlabel("Class Index")
    plt.ylabel("Separation Score")
    plt.title("PyTorch Model Separation Performance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(x, [f"C{i}" for i in range(len(CLASSES))], rotation=45)
    plt.tight_layout()
    plt.savefig('benchmarks/results/pytorch_separation_only.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   Saved: pytorch_separation_only.png")
    
    # Plot 2: TensorRT only
    plt.figure(figsize=(10, 6))
    plt.bar(x, trt_separation, width=bar_width, color=trt_color, alpha=0.8, label='TensorRT Separation')
    plt.plot(x, trt_separation, color='red', linewidth=2, marker='s', label='Trend')
    plt.axhline(y=np.mean(trt_separation), color='green', linestyle='--', alpha=0.7,
               label=f'Mean: {np.mean(trt_separation):.3f}')
    plt.xlabel("Class Index")
    plt.ylabel("Separation Score")
    plt.title("TensorRT Model Separation Performance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(x, [f"C{i}" for i in range(len(CLASSES))], rotation=45)
    plt.tight_layout()
    plt.savefig('benchmarks/results/tensorrt_separation_only.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   Saved: tensorrt_separation_only.png")
    
    # Plot 3: Comparison only
    plt.figure(figsize=(12, 6))
    bar_width_combined = 0.35
    x_pytorch = x - bar_width_combined/2
    x_trt = x + bar_width_combined/2
    
    bars1 = plt.bar(x_pytorch, pytorch_separation, width=bar_width_combined, 
                   color=pytorch_color, alpha=0.8, label='PyTorch')
    bars2 = plt.bar(x_trt, trt_separation, width=bar_width_combined, 
                   color=trt_color, alpha=0.8, label='TensorRT')
    
    # Trend lines
    plt.plot(x_pytorch, pytorch_separation, color='blue', linewidth=2, marker='o', alpha=0.7)
    plt.plot(x_trt, trt_separation, color='orange', linewidth=2, marker='s', alpha=0.7)
    
    # Mean lines
    plt.axhline(y=np.mean(pytorch_separation), color=pytorch_color, linestyle='--', alpha=0.5,
               label=f'PyTorch Mean: {np.mean(pytorch_separation):.3f}')
    plt.axhline(y=np.mean(trt_separation), color=trt_color, linestyle='--', alpha=0.5,
               label=f'TensorRT Mean: {np.mean(trt_separation):.3f}')
    
    plt.xlabel("Class Index")
    plt.ylabel("Separation Score")
    plt.title("PyTorch vs TensorRT Separation Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(x, [f"C{i}" for i in range(len(CLASSES))], rotation=45)
    plt.tight_layout()
    plt.savefig('benchmarks/results/separation_comparison_only.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   Saved: separation_comparison_only.png")
    
    print("   All individual plots saved successfully!")
    
    # Print separation statistics
    print("\n" + "="*60)
    print("SEPARATION ANALYSIS RESULTS")
    print("="*60)
    
    pytorch_mean_sep = np.mean(pytorch_separation)
    trt_mean_sep = np.mean(trt_separation)
    pytorch_std_sep = np.std(pytorch_separation)
    trt_std_sep = np.std(trt_separation)
    
    print(f"\nPyTorch Separation Statistics:")
    print(f"   Mean Separation: {pytorch_mean_sep:.4f} ± {pytorch_std_sep:.4f}")
    print(f"   Min Separation:  {np.min(pytorch_separation):.4f}")
    print(f"   Max Separation:  {np.max(pytorch_separation):.4f}")
    print(f"   Range:          {np.max(pytorch_separation) - np.min(pytorch_separation):.4f}")
    
    print(f"\nTensorRT Separation Statistics:")
    print(f"   Mean Separation: {trt_mean_sep:.4f} ± {trt_std_sep:.4f}")
    print(f"   Min Separation:  {np.min(trt_separation):.4f}")
    print(f"   Max Separation:  {np.max(trt_separation):.4f}")
    print(f"   Range:          {np.max(trt_separation) - np.min(trt_separation):.4f}")
    
    print(f"\nComparison:")
    print(f"   Separation Difference: {trt_mean_sep - pytorch_mean_sep:+.4f}")
    print(f"   Consistency Difference: {trt_std_sep - pytorch_std_sep:+.4f}")
    
    if trt_mean_sep > pytorch_mean_sep:
        print(f"   TensorRT shows better separation (+{((trt_mean_sep/pytorch_mean_sep - 1)*100):+.2f}%)")
    else:
        print(f"   PyTorch shows better separation (+{((pytorch_mean_sep/trt_mean_sep - 1)*100):+.2f}%)")
    
    # Detailed per-class comparison
    print(f"\nPer-Class Separation Comparison:")
    print("-" * 60)
    print(f"{'Class':<10} {'PyTorch':<10} {'TensorRT':<10} {'Difference':<12} {'Better':<10}")
    print("-" * 60)
    
    for i, class_name in enumerate(CLASSES):
        diff = trt_separation[i] - pytorch_separation[i]
        better = "TensorRT" if diff > 0 else "PyTorch"
        print(f"{class_name:<10} {pytorch_separation[i]:<10.4f} {trt_separation[i]:<10.4f} {diff:<12.4f} {better:<10}")
    
    return pytorch_separation, trt_separation

#create visualization
visualize_embeddings()

# Create separation analysis plots
pytorch_separation, trt_separation = plot_separation_analysis()

print("\n" + "="*50)
print("FEATURE EXTRACTION BENCHMARK ANALYSIS COMPLETE")
print("="*50)