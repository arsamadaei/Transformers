import torch
import torch.nn as nn
import math
import time

from processes import Process, TimeStamp


# Paren Process creation
#-----------------------
importTime = time.time()
parentProcess = Process("Transformer_from_scratch.py", 
                        importTime, epoch=0)
#-----------------------

# Global tracking for forward pass monitoring
_current_training_epoch = 0
_current_epoch_process = None  # Reference to the train_epoch process
_current_batch_id = None       # Current batch being processed

def set_training_epoch(epoch: int, epoch_process=None, batch_id: str = None):
    """Set current epoch and batch for forward pass tracking. Called from train.py."""
    global _current_training_epoch, _current_epoch_process, _current_batch_id
    _current_training_epoch = epoch
    _current_epoch_process = epoch_process
    _current_batch_id = batch_id

def get_training_epoch():
    """Get current training epoch."""
    return _current_training_epoch

def get_current_batch_id():
    """Get current batch ID for data flow tracking."""
    global _current_batch_id
    if _current_batch_id is None:
        import time
        return f"batch_{int(time.time()*1000) % 10000}"
    return _current_batch_id


class ForwardTracker:
    """
    Context manager for tracking forward pass execution.
    Only tracks the FIRST batch of each epoch to avoid overwhelming the Gantt chart.
    """
    _tracked_batches = set()  # Class-level set to track which (epoch, batch) we've logged
    
    def __init__(self, module, name: str, layer: int = None):
        self.module = module
        self.name = name
        self.layer = layer if layer is not None else getattr(module, 'layer', 0)
        self.proc = None
        self.should_track = False
        
    def __enter__(self):
        epoch = get_training_epoch()
        batch_id = get_current_batch_id()
        
        # Only track first batch of each epoch to avoid too many processes
        track_key = (epoch, batch_id)
        if track_key not in ForwardTracker._tracked_batches:
            ForwardTracker._tracked_batches.add(track_key)
            self.should_track = True
        
        if not self.should_track:
            return None
        
        # Create process that includes batch_id to track data flow
        proc_name = f"E{epoch}_{batch_id}_{self.name}"
        
        self.proc = Process(
            name=proc_name,
            _init=time.time(),
            epoch=epoch,
            layer=self.layer
        )
        
        # Link to epoch parent if available
        global _current_epoch_process
        if _current_epoch_process:
            _current_epoch_process.add_subtask(self.proc)
        
        # Store reference on module
        self.module._forward_proc = self.proc
        
        return self.proc
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.proc and self.should_track:
            self.proc._term = time.time()
            self.module._forward_proc = None
        return False  # Don't suppress exceptions
    
    @classmethod
    def reset_tracking(cls):
        """Reset tracking at the start of a new training run."""
        cls._tracked_batches.clear()


class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps:float=10**-6, layer: int = 0, parent_process=None) -> None:
        # Store layer number for tracking
        self.layer = layer
        
        # Process creation
        # -------------------
        exeTime = time.time()
        # I would prefer to manually track parent child relations for now
        self.__p0 = Process("LayerNormalization", exeTime, layer=layer, epoch=0)
        if parent_process:
            parent_process.add_subtask(self.__p0)
        else:
            parentProcess.add_subtask(self.__p0)
        #--------------------

        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter
        
        # Process modification
        #--------------------
        self.__p0._term = time.time()
        #--------------------
        
    def forward(self, x):
        with ForwardTracker(self, "LayerNormalization"):
            # x: (batch, seq_len, hidden_size)
             # Keep the dimension for broadcasting
            mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
            # Keep the dimension for broadcasting
            std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
            # eps is to prevent dividing by zero or when std is very small
            
            output = self.alpha * (x - mean) / (std + self.eps) + self.bias
            return output

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float, layer: int = 0, parent_process=None) -> None:
        # Store layer number for tracking
        self.layer = layer
        
        # Process creation
        # -------------------
        exeTime = time.time()
        self.__p0 = Process(f"FeedForwardBlock_L{layer}", exeTime, layer=layer, epoch=0)
        if parent_process:
            parent_process.add_subtask(self.__p0)
        else:
            parentProcess.add_subtask(self.__p0)
        #--------------------

        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

        # Process modification
        #--------------------
        self.__p0._term = time.time()
        #--------------------


    def forward(self, x):
        with ForwardTracker(self, f"FeedForwardBlock_L{self.layer}"):
            # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)    
            output = self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
            return output

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, layer: int = 0, parent_process=None) -> None:
        # Store layer number for tracking
        self.layer = layer
        
        # Process creation
        # -------------------
        exeTime = time.time()
        self.__p0 = Process("InputEmbeddings", exeTime, layer=layer, epoch=0)
        if parent_process:
            parent_process.add_subtask(self.__p0)
        else:
            parentProcess.add_subtask(self.__p0)
        #--------------------

        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Process modification
        #--------------------
        self.__p0._term = time.time()
        #--------------------

    def forward(self, x):
        with ForwardTracker(self, "InputEmbeddings"):
            # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
            return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float, layer: int = 0, parent_process=None) -> None:
        # Store layer number for tracking
        self.layer = layer
        
        # Process creation
        # -------------------
        exeTime = time.time()
        self.__p0 = Process("PositionalEncoding", exeTime, layer=layer, epoch=0)
        if parent_process:
            parent_process.add_subtask(self.__p0)
        else:
            parentProcess.add_subtask(self.__p0)
        #--------------------

        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

        # Process modification
        #--------------------
        self.__p0._term = time.time()
        #--------------------

    def forward(self, x):
        with ForwardTracker(self, "PositionalEncoding"):
            output = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
            output = self.dropout(output)
            return output

class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float, layer: int = 0, parent_process=None) -> None:
            # Store layer number for tracking
            self.layer = layer
            
            # Process creation
            # -------------------
            exeTime = time.time()
            self.__p0 = Process("ResidualConnection", exeTime, layer=layer, epoch=0)
            if parent_process:
                parent_process.add_subtask(self.__p0)
            else:
                parentProcess.add_subtask(self.__p0)
            #--------------------

            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features, parent_process=self.__p0)

            # Process modification
            #--------------------
            self.__p0._term = time.time()
            #--------------------
    
        def forward(self, x, sublayer):
            with ForwardTracker(self, "ResidualConnection"):
                output = x + self.dropout(sublayer(self.norm(x)))
                return output


class MultiHeadLatentAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float, layer: int = 0, parent_process=None):
        # Store layer number for tracking
        self.layer = layer
        
        # Process creation
        # -------------------
        exeTime = time.time()
        self.__p0 = Process(f"MultiHeadLatentAttentionBlock_L{layer}", exeTime, layer=layer, epoch=0)
        if parent_process:
            parent_process.add_subtask(self.__p0)
        else:
            parentProcess.add_subtask(self.__p0)
        #--------------------

        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h
        
        assert d_model % 4 == 0, "d_model is not divisible by h"
        self.d_qk = d_model // 4
        
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_kv_compress = nn.Linear(d_model, self.d_qk, bias=False) # W_kv compression
        self.w_k_decompress = nn.Linear(self.d_qk, d_model, bias=False) # W_k decompression
        self.w_v_decompress = nn.Linear(self.d_qk, d_model, bias=False) # W_v decompression
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)
        
        # Process modification
        #--------------------
        self.__p0._term = time.time()
        #--------------------


    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k, v, mask=None):
        with ForwardTracker(self, f"MultiHeadLatentAttention_L{self.layer}"):
            batch_size, q_seq_len, _ = q.shape
            
            kv_latent = self.w_kv_compress(k)
            k_seq_len = kv_latent.shape[1]
            
            key = self.w_k_decompress(kv_latent)
            value = self.w_v_decompress(kv_latent)
            query = self.w_q(q)
            
            # reshape for multi head
            query = query.view(batch_size, q_seq_len, self.h, self.d_k).transpose(1, 2)
            key = key.view(batch_size, k_seq_len, self.h, self.d_k).transpose(1, 2)
            value = value.view(batch_size, k_seq_len, self.h, self.d_k).transpose(1, 2)
      
            
            x, self.attention_scores = MultiHeadLatentAttentionBlock.attention(query, key, value, mask, self.dropout)
            
            # Combine all the heads together
            # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
            x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

            # Multiply by Wo
            # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
            output = self.w_o(x)
            return output


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float, layer: int = 0, parent_process=None) -> None:
        # Store layer number for tracking
        self.layer = layer
        
        # Process creation
        # -------------------
        exeTime = time.time()
        self.__p0 = Process(f"MultiHeadAttentionBlock_L{layer}", exeTime, layer=layer, epoch=0)
        if parent_process:
            parent_process.add_subtask(self.__p0)
        else:
            parentProcess.add_subtask(self.__p0)
        #--------------------

        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)
        
        # Process modification
        #--------------------
        self.__p0._term = time.time()
        #--------------------


    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        with ForwardTracker(self, f"MultiHeadAttention_L{self.layer}"):
            query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
            key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
            value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

            # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
            query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
            key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
            value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

            # Calculate attention
            x, self.attention_scores = MultiHeadLatentAttentionBlock.attention(query, key, value, mask, self.dropout)
            
            # Combine all the heads together
            # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
            x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

            # Multiply by Wo
            # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
            output = self.w_o(x)
            return output
        
class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadLatentAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float, layer: int = 0, parent_process=None) -> None:
        # Store layer number for tracking
        self.layer = layer
        
        # Process creation
        # -------------------
        exeTime = time.time()
        self.__p0 = Process(f"EncoderBlock_L{layer}", exeTime, layer=layer, epoch=0)
        if parent_process:
            parent_process.add_subtask(self.__p0)
        else:
            parentProcess.add_subtask(self.__p0)
        #--------------------

        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout, layer=layer, parent_process=self.__p0) for _ in range(2)])
        self.__p0.add_subtask(self_attention_block._MultiHeadLatentAttentionBlock__p0)
        self.__p0.add_subtask(feed_forward_block._FeedForwardBlock__p0)

        # Process modification
        #--------------------
        self.__p0._term = time.time()
        #--------------------

    def forward(self, x, src_mask):
        with ForwardTracker(self, f"EncoderBlock_L{self.layer}"):
            x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
            x = self.residual_connections[1](x, self.feed_forward_block)
            return x
    
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList, layer: int = 0, parent_process=None) -> None:
        # Store layer number for tracking
        self.layer = layer
        
        # Process creation
        # -------------------
        exeTime = time.time()
        self.__p0 = Process("Encoder", exeTime, layer=layer, epoch=0)
        if parent_process:
            parent_process.add_subtask(self.__p0)
        else:
            parentProcess.add_subtask(self.__p0)
        #--------------------

        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features, parent_process=self.__p0)
        for layer_block in layers:
            self.__p0.add_subtask(layer_block._EncoderBlock__p0)

        # Process modification
        #--------------------
        self.__p0._term = time.time()
        #--------------------

    def forward(self, x, mask):
        with ForwardTracker(self, "Encoder"):
            for layer in self.layers:
                x = layer(x, mask)
            output = self.norm(x)
            return output

class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadLatentAttentionBlock, cross_attention_block: MultiHeadLatentAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float, layer: int = 0, parent_process=None) -> None:
        # Store layer number for tracking
        self.layer = layer
        
        # Process creation
        # -------------------
        exeTime = time.time()
        self.__p0 = Process(f"DecoderBlock_L{layer}", exeTime, layer=layer, epoch=0)
        if parent_process:
            parent_process.add_subtask(self.__p0)
        else:
            parentProcess.add_subtask(self.__p0)
        #--------------------

        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout, layer=layer, parent_process=self.__p0) for _ in range(3)])
        self.__p0.add_subtask(self_attention_block._MultiHeadLatentAttentionBlock__p0)
        self.__p0.add_subtask(cross_attention_block._MultiHeadLatentAttentionBlock__p0)
        self.__p0.add_subtask(feed_forward_block._FeedForwardBlock__p0)

        # Process modification
        #--------------------
        self.__p0._term = time.time()
        #--------------------

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        with ForwardTracker(self, f"DecoderBlock_L{self.layer}"):
            x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
            x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
            x = self.residual_connections[2](x, self.feed_forward_block)
            return x
    
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList, layer: int = 0, parent_process=None) -> None:
        # Store layer number for tracking
        self.layer = layer
        
        # Process creation
        # -------------------
        exeTime = time.time()
        self.__p0 = Process("Decoder", exeTime, layer=layer, epoch=0)
        if parent_process:
            parent_process.add_subtask(self.__p0)
        else:
            parentProcess.add_subtask(self.__p0)
        #--------------------

        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features, parent_process=self.__p0)
        for layer_block in layers:
            self.__p0.add_subtask(layer_block._DecoderBlock__p0)

        # Process modification
        #--------------------
        self.__p0._term = time.time()
        #--------------------

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        with ForwardTracker(self, "Decoder"):
            for layer in self.layers:
                x = layer(x, encoder_output, src_mask, tgt_mask)
            output = self.norm(x)
            return output

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size, layer: int = 0, parent_process=None) -> None:
        # Store layer number for tracking
        self.layer = layer
        
        # Process creation
        # -------------------
        exeTime = time.time()
        self.__p0 = Process("ProjectionLayer", exeTime, layer=layer, epoch=0)
        if parent_process:
            parent_process.add_subtask(self.__p0)
        else:
            parentProcess.add_subtask(self.__p0)
        #--------------------

        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

        # Process modification
        #--------------------
        self.__p0._term = time.time()
        #--------------------

    def forward(self, x) -> None:
        with ForwardTracker(self, "ProjectionLayer"):
            # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
            output = self.proj(x)
            return output

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer, layer: int = 0, parent_process=None) -> None:
        # Store layer number for tracking
        self.layer = layer
        
        # Process creation
        # -------------------
        exeTime = time.time()
        self.__p0 = Process("Transformer", exeTime, layer=layer, epoch=0)
        if parent_process:
            parent_process.add_subtask(self.__p0)
        else:
            parentProcess.add_subtask(self.__p0)
        #--------------------

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        self.__p0.add_subtask(encoder._Encoder__p0)
        self.__p0.add_subtask(decoder._Decoder__p0)
        self.__p0.add_subtask(src_embed._InputEmbeddings__p0)
        self.__p0.add_subtask(tgt_embed._InputEmbeddings__p0)
        self.__p0.add_subtask(src_pos._PositionalEncoding__p0)
        self.__p0.add_subtask(tgt_pos._PositionalEncoding__p0)
        self.__p0.add_subtask(projection_layer._ProjectionLayer__p0)

        # Process modification
        #--------------------
        self.__p0._term = time.time()
        #--------------------

    def encode(self, src, src_mask):
        with ForwardTracker(self, "Transformer_encode"):
            # (batch, seq_len, d_model)
            src = self.src_embed(src)
            src = self.src_pos(src)
            output = self.encoder(src, src_mask)
            return output
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        with ForwardTracker(self, "Transformer_decode"):
            # (batch, seq_len, d_model)
            tgt = self.tgt_embed(tgt)
            tgt = self.tgt_pos(tgt)
            output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
            return output
    
    def project(self, x):
        with ForwardTracker(self, "Transformer_project"):
            # (batch, seq_len, vocab_size)
            output = self.projection_layer(x)
            return output

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Process creation
    #--------------------
    exeTime = time.time()
    p0 = Process("build_transformer", exeTime, epoch=0)
    parentProcess.add_subtask(p0)
    #--------------------

    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size, parent_process=p0)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size, parent_process=p0)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout, parent_process=p0)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout, parent_process=p0)
    
    # Create the encoder blocks
    encoder_blocks = []
    for layer_idx in range(N):
        encoder_self_attention_block = MultiHeadLatentAttentionBlock(d_model, h, dropout, layer=layer_idx)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout, layer=layer_idx)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout, layer=layer_idx)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for layer_idx in range(N):
        decoder_self_attention_block = MultiHeadLatentAttentionBlock(d_model, h, dropout, layer=layer_idx)
        decoder_cross_attention_block = MultiHeadLatentAttentionBlock(d_model, h, dropout, layer=layer_idx)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout, layer=layer_idx)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout, layer=layer_idx)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks), parent_process=p0)
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks), parent_process=p0)

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size, parent_process=p0)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer, parent_process=p0)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Process modification
    #--------------------
    p0._term = time.time()
    #--------------------

    # Save all processes
    Process.storeAll()

    return transformer