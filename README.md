```
 ████████╗██████╗  █████╗ ███╗   ██╗███████╗██╗      █████╗ ████████╗ ██████╗ ██████╗                      
 ╚══██╔══╝██╔══██╗██╔══██╗████╗  ██║██╔════╝██║     ██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗                     
    ██║   ██████╔╝███████║██╔██╗ ██║███████╗██║     ███████║   ██║   ██║   ██║██████╔╝                     
    ██║   ██╔══██╗██╔══██║██║╚██╗██║╚════██║██║     ██╔══██║   ██║   ██║   ██║██╔══██╗                     
    ██║   ██║  ██║██║  ██║██║ ╚████║███████║███████╗██║  ██║   ██║   ╚██████╔╝██║  ██║                     
    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝                     
```

# Transformer from Scratch

An implementation of the Transformer architecture from the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) for English-to-French translation.
This is a project of mine, that I am using to learn about AI and AI development.
My goal for this project is to get it efficient enough to rent google colab computers and train the model through google-colab.

## 🚀 Features

- **Full Transformer Implementation**: Encoder-decoder architecture with multi-head attention
- **Tokenizers**: Custom BPE tokenizers for English and French
- **Training Pipeline**: Complete training loop with checkpointing
- **Translation**: Interactive translation script
- **Visualization**: Attention weights and training metrics

## 📁 Project Structure

```
.
├── Transformer_from_scratch.py   # Core transformer implementation
├── train.py                      # Training script
├── translate.py                  # Translation inference
├── dataset.py                    # Data loading and preprocessing
├── config.py                     # Configuration settings
├── Training_performance_plotter  # A performance plotter for training and inference data
├── tokenizer_en.json             # English tokenizer
├── tokenizer_fr.json             # French tokenizer
├── weights/                      # Model checkpoints
└── logs/                         # Training logs
```

## 🛠️ Setup

```bash
# Install dependencies
pip install torch numpy matplotlib tensorboard

# Train the model
python train.py

# Translate text
python translate.py

# To load an interactive app with training/inference data charts
python Transformer_from_scratch.py
```

## 📝 Usage

### Training

```bash
python train.py
```

The training script will:
- Load and preprocess the dataset
- Train the transformer model
- Save checkpoints to `weights/`
- Log metrics to TensorBoard

### Translation

```bash
python translate.py
```

Enter English text and get French translations!

## 📊 Results

The model achieves competitive BLEU scores on English-to-French translation tasks after training.

## 📖 References

- Vaswani et al., "Attention Is All You Need", NeurIPS 2017

- Base transformer model before modifications are based on Umar Jamil's YouTube Video on creating a transformer from scratch: https://youtu.be/ISNdQcPhsts
- Umar Jamil's GitHub:  https://github.com/hkproj
