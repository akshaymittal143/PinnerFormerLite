# A Transformer Based Architecture for Focused Recommendations via Sequences

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation of **PinnerFormerLite**, a novel training methodology for transformer-based sequence models that creates specialized user representations within a single, universal architecture.

## 📖 Paper

**Title:** PinnerFormerLite: A Transformer Based Architecture for Focused Recommendations via Sequences  
**Authors:** Vinay Venkatesh, Akshay Mittal, Varun Joshi  
**Conference:** IEEE ICCA 2025

## 🎯 Overview

PinnerFormerLite addresses the challenge of providing precise recommendations for "power users" with highly specific interests (e.g., horror movie enthusiasts) while maintaining a scalable, single-model architecture. Our key innovation is a **weighted loss training process** that dynamically assigns higher weights to interactions from designated domains during training.

### Key Contributions

- **Weighted Loss Training**: Novel training methodology that prioritizes specific domains without requiring separate models
- **Single Universal Architecture**: Scalable transformer-based model that handles multiple domains
- **Empirical Validation**: 17.9% improvement in Recall@10 for power users on MovieLens 25M dataset
- **Production-Ready**: Efficient implementation suitable for large-scale deployment

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA (optional, for GPU acceleration)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/pinnerformer-lite.git
cd pinnerformer-lite
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download the MovieLens 25M dataset:**
```bash
python main.py --download_data
```

### Running Experiments

#### Single Experiment
```bash
# Generic model (baseline)
python main.py --experiment_name generic_baseline

# Weighted model (our approach)
python main.py --experiment_name weighted_model --use_weighted_loss --domain_weight 2.0
```

#### Comparison Experiments
```bash
# Run both generic and weighted models for comparison
python main.py comparison
```

## 📊 Results

Our experiments on the MovieLens 25M dataset demonstrate significant improvements for power users:

| Metric | Generic Model | Weighted Model | Improvement |
|--------|---------------|----------------|-------------|
| Recall@10 | 0.229 | 0.270 | **+17.9%** |
| Interest Entropy@50 | 1.97 | 1.88 | -4.6% |
| P90 Coverage@10 | 0.042 | 0.039 | -7.1% |

*Results are for power users (users with >50% horror movie interactions)*

## 🏗️ Architecture

### Model Components

1. **Transformer Encoder**: Multi-head self-attention with positional encoding
2. **User & Item Embeddings**: Learnable representations for users and items
3. **Weighted Dense All-Action Loss**: Domain-specific weighting during training
4. **Output Projection**: MLP layers for final user representations

### Key Features

- **Domain Weight Hyperparameter**: Configurable weights for different domains (e.g., Horror=2.0)
- **Causal Masking**: Ensures chronological order preservation
- **L2 Normalization**: Normalized embeddings for similarity computation
- **Efficient Training**: Optimized for large-scale datasets

## 📁 Repository Structure

```
pinnerformer-lite/
├── code/
│   ├── main.py                 # Main experiment script
│   ├── pinnerformer_lite.py    # Core model implementation
│   ├── data_processing.py      # Data loading and preprocessing
│   ├── trainer.py              # Training and evaluation utilities
│   ├── requirements.txt        # Python dependencies
│   └── README.md              # This file
├── data/                      # Dataset directory (created automatically)
├── models/                    # Saved model checkpoints
├── results/                   # Experiment results and logs
└── README.md                  # Main repository README
```

## 🔧 Configuration

### Model Configuration
```python
model_config = {
    'd_model': 256,              # Embedding dimension
    'nhead': 8,                  # Number of attention heads
    'num_layers': 4,             # Number of transformer layers
    'dim_feedforward': 1024,     # Feedforward network dimension
    'dropout': 0.1,              # Dropout rate
    'max_seq_length': 100        # Maximum sequence length
}
```

### Training Configuration
```python
training_config = {
    'batch_size': 256,           # Training batch size
    'num_epochs': 10,            # Number of training epochs
    'learning_rate': 0.001,      # Learning rate
    'weight_decay': 0.01,        # Weight decay
    'use_weighted_loss': True,   # Enable weighted loss
    'domain_weight': 2.0         # Weight for domain-specific interactions
}
```

## 📈 Evaluation Metrics

The implementation includes comprehensive evaluation metrics:

- **Recall@K**: Measures recommendation accuracy
- **Interest Entropy@K**: Measures recommendation diversity
- **P90 Coverage@K**: Measures global diversity across users

## 🎯 Use Cases

PinnerFormerLite is particularly effective for:

- **E-commerce Platforms**: Catering to users with niche product interests
- **Content Recommendation**: Serving users with specific content preferences
- **Social Media**: Personalizing feeds for users with focused interests
- **Gaming Platforms**: Recommending games to genre-specific players

## 🔬 Research Applications

This implementation supports various research directions:

- **Multi-Domain Weighting**: Extend to multiple domains simultaneously
- **Dynamic Weighting**: Adaptive weights based on user behavior
- **Cross-Domain Transfer**: Knowledge transfer between domains
- **Fairness Studies**: Investigating bias and fairness implications

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{venkatesh2025pinnerformerlite,
  title={PinnerFormerLite: A Transformer Based Architecture for Focused Recommendations via Sequences},
  author={Venkatesh, Vinay and Mittal, Akshay and Joshi, Varun},
  year={2025},
  booktitle={Proceedings of IEEE ICCA},
  publisher={IEEE}
}
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup
```bash
# Clone and setup development environment
git clone https://github.com/your-username/pinnerformer-lite.git
cd pinnerformer-lite
pip install -r requirements.txt
pip install -e .  # Install in development mode
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MovieLens Dataset**: Provided by GroupLens Research at the University of Minnesota
- **PyTorch**: Deep learning framework
- **PinnerFormer**: Original architecture that inspired this work
- **Research Community**: For valuable feedback and discussions

## 📞 Contact

- **Vinay Venkatesh**: vinay.venkatesh@ieee.org
- **Akshay Mittal**: akshay.mittal@ieee.org
- **Varun Joshi**: engineervarunjoshi@ieee.org

## 🔗 Links

- [Paper (PDF)](link-to-paper)
- [Dataset](https://grouplens.org/datasets/movielens/25m/)
- [PinnerFormer Original Paper](https://dl.acm.org/doi/10.1145/3534678.3539403)

---

**Note**: This implementation is designed for research reproducibility. For production deployment, additional optimizations may be required.
