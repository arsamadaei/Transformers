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

I will be documenting my learning by creating and postin high-quality videos on a [YouTube channel](https://www.youtube.com/@OpenML-x2d)

## AI Use
### AI use for writing code
Since the scope of my project is to learn about AI and the transformer model, I will develop code related to the model myself.
However, work out of scope of this project, such as app development is done through AI vibe-coding since I have no time or resources to spend myself on app development, and I have no team of developers (but am looking forward to having one).
I will be 100% transparent with my use of AI in my commit messages.

### AI use for writing documentation
I will sometimes use AI for creating documentation files. This is because as a student single handedly working on this project, I do not have enough time to dedicate to creating high quality and well structured documentation files. However, I will make sure to read through and make changes if needed. All these files are reviewed by me as I will have to also use them myself to catch up to the most recent developments and implementation I have made in my code. I will put warning in the file names or inside the documentation letting you know if a file was made using AI.


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
├── processes.py                  # Process time tracking
├── Training_performance_plotter  # A performance plotter for training and inference data
├── tokenizer_en.json             # English tokenizer
├── tokenizer_fr.json             # French tokenizer
├── weights/                      # Model checkpoints
├── Documentation/                # Is a folder containing Document pdfs
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
