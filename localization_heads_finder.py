#!/usr/bin/env python3
import os
import sys
import torch
import hydra
import pickle
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Tuple, Optional, Union
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
import json
import re
from scipy.ndimage import label
from lab import MetadataStation

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from lab import AttentionStation, StationEngine


class LocalizationHeadsFinder:
    """Tool for finding localization heads in multimodal LLMs"""

    def __init__(self, cfg: DictConfig):
        """Initialize with configuration"""
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None
        self.conv_mode = None
        self.output_dir = os.path.join(
            self.cfg.output_dir, 
            self.cfg.model_name.replace("-", "_").replace("/", "-")
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up AttentionStation for saving attention weights
        AttentionStation.set_flag()
        AttentionStation.set_save_to_path(os.path.join(self.output_dir, "attention"))
        os.makedirs(AttentionStation.get_save_to_path(), exist_ok=True)

        # Initialize model
        self._init_model()

    def _init_model(self):
        """Initialize the model, tokenizer and processor"""
        disable_torch_init()

        model_name = get_model_name_from_path(self.cfg.cache_dir)
        print(f"Loading model: {model_name}")

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.cfg.cache_dir,
            self.cfg.cache_dir_vision_tower,
            self.cfg.model_base,
            model_name
        )

        self.conv_mode = self.cfg.conv_mode

    def load_image(self, image_file):
        """Load image from file or URL"""
        if not os.path.exists(image_file) and not image_file.startswith(("http://", "https://")):
            raise FileNotFoundError(f"Image file not found: {image_file}")

        if image_file.startswith(("http://", "https://")):
            import requests
            from io import BytesIO
            try:
                response = requests.get(image_file)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
            except Exception as e:
                raise RuntimeError(f"Failed to load image from URL: {e}")
        else:
            image = Image.open(image_file).convert("RGB")
        return image

    def collect_attention(self, image_file: str, query: str, save_id: str = None) -> str:
        """
        Stage 1: Collect attention weights for an image and query
        
        Args:
            image_file: Path to image file
            query: Text query to process with the image
            save_id: Optional identifier for saving the attention file
            
        Returns:
            Path to saved attention file
        """
        # Process image
        try:
            image = self.load_image(image_file)
            image_tensor = process_images([image], self.image_processor, self.model.config)
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
            image_sizes = [image.size]
        except Exception as e:
            raise RuntimeError(f"Failed to process image {image_file}: {e}")

        # Format query with image token
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in query:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, query)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, query)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + query
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + query

        # Create conversation and format prompt
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize input
        input_ids = tokenizer_image_token(
            prompt, 
            self.tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors="pt", 
            conv=conv,
        ).unsqueeze(0).to(self.model.device)

        img_idx = input_ids[0].tolist().index(IMAGE_TOKEN_INDEX)
        print(self.tokenizer.batch_decode(input_ids[:, :img_idx]))
        print(self.tokenizer.batch_decode(input_ids[:, img_idx+1:]))
        
        # Generate response with attention
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0),
                image_sizes=image_sizes,
                do_sample=False,
                max_new_tokens=self.cfg.max_new_tokens
            )

        # Process attention weights
        begin_pos_vis = MetadataStation.get_begin_pos('vis')
        vis_len = MetadataStation.get_vis_len()
        attentions_dict = AttentionStation.get_attn_weights()
        if isinstance(attentions_dict, dict):
            layers = sorted(attentions_dict.keys())
            if layers:
                attentions = torch.stack([attentions_dict[l] for l in layers], dim=0)
            else:
                attentions = torch.empty(0)
        else:
            attentions = attentions_dict
        attentions = attentions[:, :, -1:, begin_pos_vis:begin_pos_vis+vis_len] # [L, H, 1, vis_len]

        # Save attention weights
        save_id = save_id or f"query_{len(os.listdir(AttentionStation.get_save_to_path()))}"
        save_path = os.path.join(AttentionStation.get_save_to_path(), f"{save_id}.pkl")

        # Save using AttentionStation
        AttentionStation.save_attn(save_path, attentions)

        # Also save the image and query for reference
        metadata = {
            "image_file": image_file,
            "query": query,
            "image_size": image_sizes[0]
        }
        with open(os.path.join(AttentionStation.get_save_to_path(), f"{save_id}_metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

        print(f"Attention weights saved to: {save_path}")
        return save_path

    def spatial_entropy(self, attention_map: torch.Tensor, threshold: float = 0.001) -> Dict:
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
    
    def find_elbow_point(self, x: List, y: List) -> int:
        """
        Find the elbow point in a list of values using the chord method
        
        Args:
            x: List of head indices (e.g., [0, 1, 2, ...])
            y: List of attention summation values for each head
            
        Returns:
            Threshold index - heads with indices >= this threshold should be considered
        """
        # Convert inputs to numpy arrays
        x_arr = np.array(x)
        y_arr = np.array(y)
        
        # Sort the values based on summation (y values)
        sorted_indices = np.argsort(y_arr)
        x_sorted = x_arr[sorted_indices]
        y_sorted = y_arr[sorted_indices]
        
        # Create points for the chord method
        points = np.array([(i, val) for i, val in enumerate(y_sorted)])
        
        # Start and end points of the chord
        start, end = points[0], points[-1]
        
        # Calculate distances from points to the chord
        line_vec = end - start
        line_mag = np.linalg.norm(line_vec)
        unit_line_vec = line_vec / line_mag
        vec_from_start = points - start
        
        # Calculate the scalar projection and then the vector projection
        scalar_proj = np.dot(vec_from_start, unit_line_vec)
        proj = np.outer(scalar_proj, unit_line_vec)
        
        # Calculate perpendicular distances
        distances = np.linalg.norm(vec_from_start - proj, axis=1)
        
        # Find the point with maximum distance
        elbow_idx = np.argmax(distances)
        
        # Return the actual head index at the elbow point
        # This will be used as the threshold
        return x_sorted[elbow_idx]
        

    def vis_attn_summation(self, attention_map: torch.Tensor) -> List:
        """
        Calculate the summation of attention map across heads
        
        Args:
            attention_map: 2D attention map tensor
            
        Returns:
            List of attention summation values for each head
        """ 
        # Calculate the sum of attention weights
        attn_sum = torch.sum(attention_map).item()
        
        return attn_sum

    def analyze_attention_file(self, attention_file: str) -> Dict:
        """
        Stage 2: Analyze attention file to find localization heads
        
        Args:
            attention_file: Path to attention file saved in Stage 1
            
        Returns:
            Dictionary with analysis results
        """
        if not os.path.exists(attention_file):
            raise FileNotFoundError(f"Attention file not found: {attention_file}")

        # Load attention data
        try:
            with open(attention_file, "rb") as f:
                attention_data = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load attention data: {e}")

        # Load metadata if available
        metadata_file = attention_file.replace(".pkl", "_metadata.pkl")
        if os.path.exists(metadata_file):
            with open(metadata_file, "rb") as f:
                metadata = pickle.load(f)
            print(f"Analyzing attention for query: {metadata['query']}")

        # Convert attention data from dictionary to tensor
        if isinstance(attention_data, dict):
            layers = sorted(attention_data.keys())
            if layers:
                attention = torch.stack([attention_data[l] for l in layers], dim=0)
            else:
                attention = torch.empty(0)
        else:
            attention = attention_data

        L, H, _, vis_len = attention.shape

        # Estimate the patch size (assuming square patches)
        P = int(np.sqrt(vis_len))  # Rough estimate, might need adjustment

        # First pass: collect summation values for all heads
        head_indices = []
        summation_values = []

        for l in range(L):
            for h in range(H):
                # Calculate head index
                head_idx = l * H + h
                head_indices.append(head_idx)
                
                # Extract attention map for this layer and head
                vis_attn = attention[l, h, 0]
                
                # Calculate vis_attn summation
                sum_result = self.vis_attn_summation(vis_attn)
                summation_values.append(sum_result)
        
        # Find threshold using elbow point detection
        threshold_idx = self.find_elbow_point(head_indices, summation_values)
        
        # Results storage
        results = []

        # Second pass: analyze heads based on threshold
        for l in range(L):
            for h in range(H):
                # Calculate head index
                head_idx = l * H + h
                
                # Extract attention map for this layer and head
                vis_attn = attention[l, h, 0]
                
                # Calculate vis_attn summation
                sum_result = self.vis_attn_summation(vis_attn)
                
                # Extract attention to image tokens and reshape to 2D
                vis_attn_map = vis_attn.reshape(P, P)
                
                # Only calculate spatial entropy if head index is >= threshold
                if head_idx >= threshold_idx:
                    # Calculate spatial entropy
                    se_result = self.spatial_entropy(vis_attn_map)
                    se = se_result["spatial_entropy"]
                    labeled_array = se_result["labeled_array"]
                    
                    # Check if the attention focuses on the bottom row (often a sign of non-localization)
                    bottom_row_focus = 1 if (labeled_array[-1, :] > 0).all() else 0
                    
                    num_components = se_result["num_components"]
                else:
                    # For heads below threshold, set default values
                    se = float('inf')  # High entropy (unfocused)
                    bottom_row_focus = 0
                    num_components = 0
                    labeled_array = None

                # Store results
                results.append({
                    "layer": l,
                    "head": h,
                    "head_idx": head_idx,
                    "attn_sum": sum_result,
                    "spatial_entropy": se,
                    "bottom_row_focus": bottom_row_focus,
                    "num_components": num_components,
                    "above_threshold": head_idx >= threshold_idx
                })

        # Sort results by spatial entropy (lower is more focused)
        results.sort(key=lambda x: x["spatial_entropy"])

        # Save analysis results
        analysis_file = attention_file.replace(".pkl", "_analysis.pkl")
        with open(analysis_file, "wb") as f:
            pickle.dump(results, f)

        print(f"Analysis results saved to: {analysis_file}")
        return results

    def visualize_attention(self, attention_file: str, top_k: int = 5, save_fig: bool = True) -> None:
        """
        Visualize attention maps for top localization heads
        
        Args:
            attention_file: Path to attention file
            top_k: Number of top heads to visualize
            save_fig: Whether to save the figure
        """
        if not os.path.exists(attention_file):
            raise FileNotFoundError(f"Attention file not found: {attention_file}")

        # Load attention data
        with open(attention_file, "rb") as f:
            attention_data = pickle.load(f)

        # Load metadata if available
        metadata_file = attention_file.replace(".pkl", "_metadata.pkl")
        metadata = None
        if os.path.exists(metadata_file):
            with open(metadata_file, "rb") as f:
                metadata = pickle.load(f)

        # Load analysis results if available
        analysis_file = attention_file.replace(".pkl", "_analysis.pkl")
        if not os.path.exists(analysis_file):
            print(f"Analysis file not found: {analysis_file}")
            print("Running analysis first...")
            self.analyze_attention_file(attention_file)

        with open(analysis_file, "rb") as f:
            analysis = pickle.load(f)

        # Get top-k heads by spatial entropy
        top_heads = analysis[:top_k]

        # Load original image if available
        original_image = None
        if metadata and 'image_file' in metadata:
            try:
                original_image = self.load_image(metadata['image_file'])
            except Exception as e:
                print(f"Could not load original image: {e}")

        # Create visualization
        fig, axes = plt.subplots(1, top_k + (1 if original_image else 0), figsize=(4*top_k + (4 if original_image else 0), 4))

        # If there's only one subplot, wrap it in a list
        if top_k + (1 if original_image else 0) == 1:
            axes = [axes]

        # Plot original image if available
        if original_image:
            axes[0].imshow(original_image)
            axes[0].set_title("Original Image")
            axes[0].axis('off')

        # Plot attention maps
        offset = 1 if original_image else 0
        attn_gen_wise = attention_data[0]  # First generation step

        for i, head_info in enumerate(top_heads):
            l, h = head_info["layer"], head_info["head"]

            # Extract attention map
            attn_map = attn_gen_wise[l, 0, h]

            # Estimate patch size
            P = int(np.sqrt(attn_map.shape[1] // 2))

            # Focus on the last token's attention to image tokens
            last_token_idx = -1
            image_start_idx = 1  # Approximate position
            image_end_idx = image_start_idx + P*P  # Approximate position

            try:
                vis_attn = attn_map[last_token_idx, image_start_idx:image_end_idx].reshape(P, P)

                # Plot attention map
                im = axes[i + offset].imshow(vis_attn.detach().cpu().numpy(), cmap='viridis')
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
        if save_fig:
            fig_file = attention_file.replace(".pkl", "_visualization.png")
            plt.savefig(fig_file, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {fig_file}")

        if self.cfg.show_plot:
            plt.show()
        else:
            plt.close(fig)

    def run_pipeline(self, image_file: str, query: str, save_id: str = None, visualize: bool = True) -> Dict:
        """
        Run the full pipeline: collect attention and analyze
        
        Args:
            image_file: Path to image file
            query: Text query
            save_id: Optional identifier for saving files
            visualize: Whether to visualize results
            
        Returns:
            Analysis results
        """
        # Stage 1: Collect attention
        attention_file = self.collect_attention(image_file, query, save_id)

        # Stage 2: Analyze attention
        results = self.analyze_attention_file(attention_file)

        # Visualize if requested
        if visualize:
            self.visualize_attention(attention_file, top_k=self.cfg.top_k)

        return results

    def process_jsonl_batch(self) -> List[Dict]:
        """
        Process a batch of data from a JSONL file
        
        Returns:
            List of results for each processed entry
        """
        if not self.cfg.data_file or not os.path.exists(self.cfg.data_file):
            raise FileNotFoundError(f"Data file not found: {self.cfg.data_file}")

        # Load data from JSONL file
        data = []
        try:
            with open(self.cfg.data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Warning: Could not parse JSON line: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to read data file: {e}")

        print(f"Loaded {len(data)} entries from {self.cfg.data_file}")

        # Determine which entries to process
        start_idx = max(0, self.cfg.start_index)
        end_idx = min(len(data), self.cfg.end_index) if self.cfg.end_index >= 0 else len(data)

        if not self.cfg.process_all:
            data = data[start_idx:end_idx]
            print(f"Processing entries from index {start_idx} to {end_idx}")

        results = []

        # Process each entry
        for i, entry in enumerate(tqdm(data, desc="Processing entries")):
            entry_id = entry.get('id', f"entry_{i}")
            prompt = entry.get('prompt', '')
            image_path = entry.get('image_path', '')

            if not image_path:
                print(f"Warning: No image path for entry {entry_id}, skipping")
                continue

            print(f"\nProcessing entry {entry_id}: {prompt[:50]}...")

            # Run pipeline for this entry
            result = self.run_pipeline(
                    image_file=image_path,
                    query=prompt,
                    save_id=entry_id,
                    visualize=self.cfg.visualize_batch  # Control visualization during batch processing
                )

            # Store result with entry ID
            results.append({
                    "entry_id": entry_id,
                    "analysis": result
                })

            # Save intermediate results
            if (i + 1) % self.cfg.batch_size == 0:
                batch_results_file = os.path.join(self.output_dir, f"batch_results_{i//self.cfg.batch_size}.pkl")
                with open(batch_results_file, 'wb') as f:
                    pickle.dump(results[-self.cfg.batch_size:], f)
                print(f"Saved batch results to {batch_results_file}")

        # Save final results
        final_results_file = os.path.join(self.output_dir, "final_results.pkl")
        with open(final_results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Saved all results to {final_results_file}")

        return results


@hydra.main(config_path="config", config_name="find_localization_heads_config", version_base=None)
def main(cfg: DictConfig):
    """Main entry point"""
    print(OmegaConf.to_yaml(cfg))
    
    finder = LocalizationHeadsFinder(cfg)
    
    if cfg.stage == "collect":
        # Stage 1: Collect attention
        if not cfg.image_file or not cfg.query:
            raise ValueError("For collection stage, both image_file and query must be provided")
        
        finder.collect_attention(cfg.image_file, cfg.query, cfg.save_id)
    
    elif cfg.stage == "analyze":
        # Stage 2: Analyze attention
        if not cfg.attention_file:
            raise ValueError("For analysis stage, attention_file must be provided")
        
        results = finder.analyze_attention_file(cfg.attention_file)
        
        # Print top 5 localization heads
        print("\nTop 5 potential localization heads:")
        for i, head in enumerate(results[:5]):
            print(f"{i+1}. Layer {head['layer']}, Head {head['head']} - Spatial Entropy: {head['spatial_entropy']:.4f}")
    
    elif cfg.stage == "visualize":
        # Visualize attention
        if not cfg.attention_file:
            raise ValueError("For visualization stage, attention_file must be provided")
        
        finder.visualize_attention(cfg.attention_file, top_k=cfg.top_k)
    
    elif cfg.stage == "pipeline":
        # Run full pipeline
        if not cfg.image_file or not cfg.query:
            raise ValueError("For pipeline stage, both image_file and query must be provided")
        
        finder.run_pipeline(cfg.image_file, cfg.query, cfg.save_id)
    
    elif cfg.stage == "batch":
        # Process batch from JSONL file
        if not cfg.data_file:
            raise ValueError("For batch stage, data_file must be provided")
        
        finder.process_jsonl_batch()
    
    else:
        raise ValueError(f"Invalid stage: {cfg.stage}. Choose from: collect, analyze, visualize, pipeline, batch")


if __name__ == "__main__":
    main() 
