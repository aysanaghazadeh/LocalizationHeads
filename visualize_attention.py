#!/usr/bin/env python3
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from scipy.ndimage import label
import torch
import torch.nn.functional as F
import json
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm

def load_attention_file(file_path: str):
    """Load attention data from a pickle file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Attention file not found: {file_path}")
        
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load attention data: {e}")

def load_metadata_file(file_path: str):
    """Load metadata from a pickle file"""
    metadata_file = file_path.replace(".pkl", "_metadata.pkl")
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load metadata: {e}")
    return None

def load_analysis_file(file_path: str):
    """Load analysis results from a pickle file"""
    analysis_file = file_path.replace(".pkl", "_analysis.pkl")
    if os.path.exists(analysis_file):
        try:
            with open(analysis_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load analysis: {e}")
    return None

def spatial_entropy(attention_map: torch.Tensor, threshold: float = 0.001) -> Dict:
    """
    Calculate spatial entropy of an attention map
    
    Args:
        attention_map: 2D attention map tensor
        threshold: Threshold for binarizing the attention map
        
    Returns:
        Dictionary with spatial entropy and labeled array
    """
    # Step 1: Compute the similarity map (S_h)
    S_h = attention_map

    # Step 2: Compute the mean value of S_h
    mean_value = torch.mean(S_h)

    # Step 3: Thresholding using mean value to create B_h
    B_h = torch.relu(S_h - mean_value)

    # Step 4: Extract connected components using an 8-connectivity relation
    B_h_np = B_h.detach().cpu().to(torch.float32).numpy()
    B_h_binary = (B_h_np > threshold).astype(np.int32)
    labeled_array, num_features = label(B_h_binary, structure=np.ones((3, 3)))
            
    # Step 5: Calculate spatial entropy
    component_sums = []
    total_sum_B_h = torch.sum(B_h).item()
    
    if total_sum_B_h <= 0:
        return {
            "spatial_entropy": float('inf'),
            "labeled_array": labeled_array,
            "num_components": 0
        }

    for i in range(1, num_features + 1):
        component_mask = (labeled_array == i)
        component_sum = B_h_np[component_mask].sum()
        component_sums.append(component_sum)

    # Convert component sums to probabilities
    probabilities = [component_sum / total_sum_B_h for component_sum in component_sums if component_sum > 0]

    # Compute the spatial entropy
    spatial_entropy = -sum(p * np.log(p) for p in probabilities if p > 0) if probabilities else 0

    res = {
        "spatial_entropy": spatial_entropy,
        "labeled_array": labeled_array,
        "num_components": num_features
    }
    return res

def find_largest_component(labeled_array: np.ndarray) -> Tuple[int, int]:
    """
    Find the largest connected component in a labeled array
    
    Args:
        labeled_array: Array with labeled components
        
    Returns:
        Tuple of (max_label, max_size)
    """
    # Calculate frequency of each label (excluding background)
    unique_labels, counts = np.unique(labeled_array[labeled_array > 0], return_counts=True)
    
    if len(counts) == 0:  # No components found
        return None, 0
        
    # Find the largest component
    max_idx = np.argmax(counts)
    max_label = unique_labels[max_idx]
    max_size = counts[max_idx]
    
    return max_label, max_size

def analyze_attention(attention_data, layer: int, head: int, token_idx: int = -1, patch_size: int = None):
    """
    Analyze attention for a specific layer and head
    
    Args:
        attention_data: Attention data loaded from file
        layer: Layer index
        head: Head index
        token_idx: Token index to analyze (default: -1 for last token)
        patch_size: Optional patch size for reshaping (will be estimated if None)
        
    Returns:
        Dictionary with analysis results
    """
    # Get attention dimensions
    attn_gen_wise = attention_data[0]  # First generation step
    attn_map = attn_gen_wise[layer, 0, head]  # [seq_len, seq_len]
    
    # Estimate patch size if not provided
    if patch_size is None:
        patch_size = int(np.sqrt(attn_map.shape[1] // 2))  # Rough estimate
    
    # Focus on the specified token's attention to image tokens
    image_start_idx = 1  # Approximate position
    image_end_idx = image_start_idx + patch_size*patch_size  # Approximate position
    
    try:
        # Extract attention to image tokens and reshape to 2D
        vis_attn = attn_map[token_idx, image_start_idx:image_end_idx].reshape(patch_size, patch_size)
        
        # Calculate spatial entropy
        se_result = spatial_entropy(vis_attn)
        se = se_result["spatial_entropy"]
        labeled_array = se_result["labeled_array"]
        
        # Find largest component
        max_label, max_size = find_largest_component(labeled_array)
        
        # Check if the attention focuses on the bottom row (often a sign of non-localization)
        bottom_row_focus = 1 if (labeled_array[-1, :] > 0).all() else 0
        
        return {
            "layer": layer,
            "head": head,
            "spatial_entropy": se,
            "max_component_size": max_size,
            "bottom_row_focus": bottom_row_focus,
            "num_components": se_result["num_components"],
            "attention_map": vis_attn.detach().cpu().numpy(),
            "labeled_array": labeled_array
        }
    except Exception as e:
        print(f"Error analyzing attention for layer {layer}, head {head}: {e}")
        return {
            "layer": layer,
            "head": head,
            "spatial_entropy": float('inf'),
            "max_component_size": 0,
            "bottom_row_focus": 0,
            "num_components": 0,
            "attention_map": np.zeros((patch_size, patch_size)),
            "labeled_array": np.zeros((patch_size, patch_size))
        }

def visualize_attention_map(cfg: DictConfig, attention_file: str, layer: int = None, head: int = None):
    """
    Visualize attention maps for a specific layer and head or top-k heads
    
    Args:
        cfg: Hydra configuration
        attention_file: Path to attention file
        layer: Layer index (if None, will use top heads from analysis)
        head: Head index (if None, will use top heads from analysis)
    """
    # Load attention data
    attention_data = load_attention_file(attention_file)
    
    # Load metadata if available
    metadata = load_metadata_file(attention_file)
    
    # Load or run analysis
    analysis = load_analysis_file(attention_file)
    if analysis is None:
        print("No analysis file found. Running analysis...")
        # Run analysis for all layers and heads
        attn_gen_wise = attention_data[0]  # First generation step
        L, bsz, H, seq_len, _ = attn_gen_wise.shape
        
        # Estimate patch size
        P = int(np.sqrt(seq_len // 2))
        
        # Analyze each layer and head
        analysis = []
        for l in range(L):
            for h in range(H):
                try:
                    result = analyze_attention(attention_data, l, h, patch_size=P)
                    analysis.append(result)
                except Exception as e:
                    print(f"Error analyzing layer {l}, head {h}: {e}")
                    # Add a placeholder entry
                    analysis.append({
                        "layer": l,
                        "head": h,
                        "spatial_entropy": float('inf'),
                        "max_component_size": 0,
                        "bottom_row_focus": 0,
                        "num_components": 0,
                        "attention_map": np.zeros((P, P)),
                        "labeled_array": np.zeros((P, P))
                    })
        
        # Sort by spatial entropy
        analysis.sort(key=lambda x: x["spatial_entropy"])
        
        # Save analysis
        analysis_file = attention_file.replace(".pkl", "_analysis.pkl")
        os.makedirs(os.path.dirname(analysis_file), exist_ok=True)
        with open(analysis_file, "wb") as f:
            pickle.dump(analysis, f)
        
        print(f"Analysis results saved to: {analysis_file}")
    
    # Determine which heads to visualize
    if layer is not None and head is not None:
        # Visualize specific layer and head
        heads_to_visualize = [next((h for h in analysis if h["layer"] == layer and h["head"] == head), None)]
        if heads_to_visualize[0] is None:
            raise ValueError(f"Layer {layer}, head {head} not found in analysis")
    else:
        # Visualize top-k heads
        heads_to_visualize = analysis[:cfg.top_k]
    
    # Load original image if available
    original_image = None
    if metadata and 'image_file' in metadata:
        try:
            image_file = metadata['image_file']
            if os.path.exists(image_file):
                original_image = Image.open(image_file).convert("RGB")
            elif image_file.startswith(("http://", "https://")):
                import requests
                from io import BytesIO
                response = requests.get(image_file)
                response.raise_for_status()
                original_image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            print(f"Could not load original image: {e}")
    
    # Create visualization
    fig, axes = plt.subplots(1, len(heads_to_visualize) + (1 if original_image else 0), 
                            figsize=(4*len(heads_to_visualize) + (4 if original_image else 0), 4))
    
    # If there's only one subplot, wrap it in a list
    if len(heads_to_visualize) + (1 if original_image else 0) == 1:
        axes = [axes]
    
    # Plot original image if available
    if original_image:
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
    
    # Plot attention maps
    offset = 1 if original_image else 0
    for i, head_info in enumerate(heads_to_visualize):
        l, h = head_info["layer"], head_info["head"]
        
        try:
            # Use pre-computed attention map if available
            if "attention_map" in head_info:
                vis_attn = head_info["attention_map"]
            else:
                # Extract attention map from data
                attn_gen_wise = attention_data[0]  # First generation step
                attn_map = attn_gen_wise[l, 0, h]
                
                # Estimate patch size
                P = int(np.sqrt(attn_map.shape[1] // 2))
                
                # Focus on the last token's attention to image tokens
                last_token_idx = -1
                image_start_idx = 1  # Approximate position
                image_end_idx = image_start_idx + P*P  # Approximate position
                
                vis_attn = attn_map[last_token_idx, image_start_idx:image_end_idx].reshape(P, P).detach().cpu().numpy()
            
            # Plot attention map
            im = axes[i + offset].imshow(vis_attn, cmap='viridis')
            axes[i + offset].set_title(f"L{l}-H{h}\nSE: {head_info['spatial_entropy']:.2f}")
            axes[i + offset].axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i + offset], fraction=0.046, pad=0.04)
        except Exception as e:
            axes[i + offset].text(0.5, 0.5, f"Error visualizing\nattention map\n{str(e)}", 
                                ha='center', va='center', transform=axes[i + offset].transAxes)
            axes[i + offset].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    if cfg.save_fig:
        fig_file = attention_file.replace(".pkl", "_visualization.png")
        os.makedirs(os.path.dirname(fig_file), exist_ok=True)
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {fig_file}")
    
    if cfg.show_plot:
        plt.show()
    else:
        plt.close(fig)

def visualize_components(cfg: DictConfig, attention_file: str, layer: int, head: int):
    """
    Visualize connected components in attention map
    
    Args:
        cfg: Hydra configuration
        attention_file: Path to attention file
        layer: Layer index
        head: Head index
    """
    # Load attention data
    attention_data = load_attention_file(attention_file)
    
    # Load metadata if available
    metadata = load_metadata_file(attention_file)
    
    # Extract attention map
    attn_gen_wise = attention_data[0]  # First generation step
    attn_map = attn_gen_wise[layer, 0, head]  # [seq_len, seq_len]
    
    # Estimate patch size
    P = int(np.sqrt(attn_map.shape[1] // 2))
    
    # Focus on the last token's attention to image tokens
    last_token_idx = -1
    image_start_idx = 1  # Approximate position
    image_end_idx = image_start_idx + P*P  # Approximate position
    
    try:
        # Extract attention to image tokens and reshape to 2D
        vis_attn = attn_map[last_token_idx, image_start_idx:image_end_idx].reshape(P, P)
        
        # Calculate spatial entropy and get labeled array
        se_result = spatial_entropy(vis_attn)
        labeled_array = se_result["labeled_array"]
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Plot original attention map
        im1 = axes[0].imshow(vis_attn.detach().cpu().numpy(), cmap='viridis')
        axes[0].set_title(f"L{layer}-H{head} Attention Map")
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Plot thresholded attention map
        B_h = torch.relu(vis_attn - torch.mean(vis_attn))
        im2 = axes[1].imshow(B_h.detach().cpu().numpy(), cmap='viridis')
        axes[1].set_title("Thresholded Map")
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Plot labeled components with different colors
        from matplotlib.colors import ListedColormap
        num_components = se_result["num_components"]
        cmap = plt.cm.get_cmap('tab20', num_components + 1)
        im3 = axes[2].imshow(labeled_array, cmap=cmap)
        axes[2].set_title(f"Components: {num_components}\nSE: {se_result['spatial_entropy']:.2f}")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        if cfg.save_fig:
            fig_file = attention_file.replace(".pkl", f"_components_L{layer}H{head}.png")
            os.makedirs(os.path.dirname(fig_file), exist_ok=True)
            plt.savefig(fig_file, dpi=300, bbox_inches='tight')
            print(f"Component visualization saved to: {fig_file}")
        
        if cfg.show_plot:
            plt.show()
        else:
            plt.close(fig)
            
    except Exception as e:
        print(f"Error visualizing components: {e}")

def process_jsonl_batch(cfg: DictConfig):
    """
    Process a batch of attention files from a JSONL file
    
    Args:
        cfg: Hydra configuration
    """
    if not cfg.data_file or not os.path.exists(cfg.data_file):
        raise FileNotFoundError(f"Data file not found: {cfg.data_file}")
    
    # Load data from JSONL file
    data = []
    try:
        with open(cfg.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse JSON line: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to read data file: {e}")
    
    print(f"Loaded {len(data)} entries from {cfg.data_file}")
    
    # Determine which entries to process
    start_idx = max(0, cfg.start_index)
    end_idx = min(len(data), cfg.end_index) if cfg.end_index >= 0 else len(data)
    
    if not cfg.process_all:
        data = data[start_idx:end_idx]
        print(f"Processing entries from index {start_idx} to {end_idx}")
    
    # Process each entry
    for i, entry in enumerate(tqdm(data, desc="Processing entries")):
        try:
            # Extract data from entry
            entry_id = entry.get('id', f"entry_{i}")
            attention_file = os.path.join(cfg.output_dir, "attention", f"{entry_id}.pkl")
            
            if not os.path.exists(attention_file):
                print(f"Warning: Attention file not found for entry {entry_id}: {attention_file}")
                continue
            
            print(f"\nProcessing entry {entry_id}")
            
            # Visualize attention
            visualize_attention_map(cfg, attention_file)
            
        except Exception as e:
            print(f"Error processing entry {entry.get('id', i)}: {e}")
    
    print("Batch processing complete")


@hydra.main(config_path="config", config_name="visualization_config", version_base=None)
def main(cfg: DictConfig):
    """Main entry point"""
    print(OmegaConf.to_yaml(cfg))
    
    try:
        if cfg.mode == "single":
            if not cfg.attention_file:
                raise ValueError("For single mode, attention_file must be provided")
            
            if cfg.show_components:
                if cfg.layer is None or cfg.head is None:
                    raise ValueError("For component visualization, both layer and head must be specified")
                visualize_components(cfg, cfg.attention_file, cfg.layer, cfg.head)
            else:
                visualize_attention_map(cfg, cfg.attention_file, cfg.layer, cfg.head)
                
        elif cfg.mode == "batch":
            if not cfg.data_file:
                raise ValueError("For batch mode, data_file must be provided")
            process_jsonl_batch(cfg)
            
        else:
            raise ValueError(f"Invalid mode: {cfg.mode}. Choose from: single, batch")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 