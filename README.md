# Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding [CVPR 2025 Highlight]

## TL;DR
This repository provides a one-shot evaluation protocol designed to support the discovery and validation of our primary contribution—Localization Heads—in Large Vision-Language Models.

This repository contains tools for finding and analyzing localization heads in multimodal LLMs. The tools help identify which attention heads in a model are most responsible for localizing objects in images.

**Paper**: [https://arxiv.org/abs/2503.06287](https://arxiv.org/abs/2503.06287)

## Features

- Two-stage process for finding localization heads:
  1. Collect attention weights from multimodal LLMs
  2. Criterion 1: Calculate the sum of image attention maps across all attention heads and find the elbow point to filter attention heads
  3. Criterion 2: Analyze attention using spatial entropy to identify localization heads
- Batch processing of multiple images/queries using JSONL files
- Visualization of attention maps and connected components
- Fully configurable using Hydra

## Experiment Dataset

For our experiments, we prepared 1,000 data samples from the RefCOCO training set. The RefCOCO dataset contains images with referring expressions that uniquely identify specific objects in the images. This selected subset allowed us to comprehensively evaluate the localization capabilities of various attention heads in the model. For more detailed information about the dataset preparation and experimental setup, please refer to our paper.

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Localization Heads Finder

The main tool for finding localization heads is `localization_heads_finder.py`. It uses spatial entropy to identify attention heads that focus on coherent regions in images. Lower spatial entropy typically indicates better localization ability.

### How It Works

1. **Collection Stage**: Runs the model on an image-text pair and saves attention weights
2. **Analysis Stage**: Calculates spatial entropy for each attention head
3. **Visualization Stage**: Generates visualizations of the most promising localization heads

### Running the Localization Heads Finder

The script supports several modes of operation through Hydra configuration:

#### Single Image Processing

```bash
# Run the full pipeline (collect + analyze + visualize) for a single image
python localization_heads_finder.py stage=pipeline image_file=path/to/image.jpg query="What is in this image?"

# Collect attention weights only
python localization_heads_finder.py stage=collect image_file=path/to/image.jpg query="What is in this image?"

# Analyze existing attention file
python localization_heads_finder.py stage=analyze attention_file=outputs/localization_heads/llava/attention/example_1.pkl

# Visualize existing attention file
python localization_heads_finder.py stage=visualize attention_file=outputs/localization_heads/llava/attention/example_1.pkl top_k=3
```

#### Batch Processing with JSONL

Process multiple images/queries by providing a JSONL file:

```bash
python localization_heads_finder.py stage=batch data_file=examples/localization_data.jsonl
```

The JSONL file should contain entries with the following format:
```json
{"id": "example_1", "prompt": "What is in this image?", "image_path": "examples/images/cat.jpg"}
```

### Configuration Options

You can override any configuration parameter on the command line:

```bash
python localization_heads_finder.py stage=pipeline image_file=path/to/image.jpg query="What is in this image?" top_k=10 output_dir=my_outputs
```

Common configuration options:
- `stage`: Pipeline stage to run (pipeline, collect, analyze, visualize, batch)
- `image_file`: Path to the image for processing
- `query`: Text query to ask about the image
- `attention_file`: Path to a saved attention file for analysis/visualization
- `top_k`: Number of top attention heads to visualize
- `output_dir`: Directory to save outputs
- `data_file`: Path to JSONL file for batch processing
- `batch_size`: Number of examples to process in each batch
- `visualize_batch`: Whether to visualize attention during batch processing
- `show_plot`: Whether to display plots (set to false for headless environments)

### Output

The tool generates the following outputs:

- Attention weights (.pkl files)
- Analysis results (.pkl files)
- Visualizations (.png files)

All outputs are saved in the `outputs/localization_heads` directory by default.

## Attention Visualization

For more detailed visualization of attention maps, use `visualize_attention.py`:

```bash
# Visualize a single attention file
python visualize_attention.py mode=single attention_file=outputs/localization_heads/llava/attention/example_1.pkl

# Visualize specific layer and head
python visualize_attention.py mode=single attention_file=outputs/localization_heads/llava/attention/example_1.pkl layer=20 head=5 show_components=true

# Batch visualization from JSONL file
python visualize_attention.py mode=batch data_file=examples/visualization_data.jsonl
```

## Project Structure

```
.
├── config/                  # Configuration files
│   ├── localization_config.yaml  # Config for localization heads finder
│   └── visualization_config.yaml # Config for visualization tool
├── examples/                # Example data files
│   └── sample_data.jsonl    # Example JSONL for batch processing
├── lab/                     # Core utilities
│   ├── __init__.py          # Module exports
│   └── stations.py          # AttentionStation and other utilities
├── llava/
│   ├── model/               # A model implementation based on LLaVA
│   │   ├── lanuguage_model/
│   │   ├── llm_modeling/
│   │   ├── multimodal_encoder/
│   │   ├── multimodal_projector/
│   │   ├── ...
│   ├── etc..
├── utils/                   # Helper utilities
│   ├── __init__.py
│   └── hydra_config.py      # Hydra configuration utilities
├── localization_heads_finder.py  # Main script for finding localization heads
├── visualize_attention.py        # Script for visualizing attention maps
└── README.md                # This file
```

## Additional Utilities

### Command Line Interface

```bash
# Start interactive CLI mode
python cli_llava.py cli_mode=true

# Run a single query
python cli_llava.py query="Describe this image" image_file=/path/to/image.jpg
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 