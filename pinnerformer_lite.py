"""
PinnerFormerLite: A Transformer Based Architecture for Focused Recommendations via Sequences

This module implements the core PinnerFormerLite model with weighted loss training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer sequences."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class PinnerFormerLite(nn.Module):
    """
    PinnerFormerLite: Transformer-based sequence model for focused recommendations.
    
    This model implements the core architecture described in the paper, with support
    for weighted loss training to prioritize specific domains.
    """
    
    def __init__(
        self,
        num_items: int,
        num_users: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_length: int = 100
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Embeddings
        self.item_embedding = nn.Embedding(num_items, d_model)
        self.user_embedding = nn.Embedding(num_users, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # Item projection for final embeddings
        self.item_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_sequences: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            user_ids: User IDs [batch_size]
            item_sequences: Item sequences [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            user_embeddings: User embeddings [batch_size, d_model]
            item_embeddings: Item embeddings [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = item_sequences.shape
        
        # Get embeddings
        item_emb = self.item_embedding(item_sequences)  # [batch_size, seq_len, d_model]
        user_emb = self.user_embedding(user_ids)  # [batch_size, d_model]
        
        # Add positional encoding
        item_emb = self.pos_encoding(item_emb.transpose(0, 1)).transpose(0, 1)
        
        # Create causal mask for transformer
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        # Apply transformer
        transformer_output = self.transformer(
            item_emb,
            src_key_padding_mask=~attention_mask
        )
        
        # Get user representation (mean pooling over sequence)
        user_embeddings = torch.mean(transformer_output, dim=1)  # [batch_size, d_model]
        user_embeddings = self.output_projection(user_embeddings)
        
        # L2 normalize user embeddings
        user_embeddings = F.normalize(user_embeddings, p=2, dim=1)
        
        # Project item embeddings
        item_embeddings = self.item_projection(transformer_output)
        
        return user_embeddings, item_embeddings


class WeightedDenseAllActionLoss(nn.Module):
    """
    Weighted Dense All-Action Loss for domain-specific training.
    
    This loss function implements the weighted training objective described in the paper,
    allowing the model to prioritize specific domains during training.
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        positive_items: torch.Tensor,
        domain_weights: torch.Tensor,
        user_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute weighted dense all-action loss.
        
        Args:
            user_embeddings: User embeddings [batch_size, d_model]
            item_embeddings: Item embeddings [batch_size, seq_len, d_model]
            positive_items: Positive item indices [batch_size, num_positives]
            domain_weights: Domain-specific weights [batch_size, num_positives]
            user_weights: User-level weights [batch_size] (optional)
            
        Returns:
            loss: Weighted loss value
        """
        batch_size, d_model = user_embeddings.shape
        
        # Get positive item embeddings
        pos_item_emb = self.item_embedding(positive_items)  # [batch_size, num_pos, d_model]
        
        # Compute similarities
        similarities = torch.bmm(
            user_embeddings.unsqueeze(1),  # [batch_size, 1, d_model]
            pos_item_emb.transpose(1, 2)   # [batch_size, d_model, num_pos]
        ).squeeze(1)  # [batch_size, num_pos]
        
        # Apply temperature scaling
        similarities = similarities / self.temperature
        
        # Compute log probabilities
        log_probs = F.log_softmax(similarities, dim=1)
        
        # Apply domain weights
        weighted_log_probs = log_probs * domain_weights
        
        # Apply user weights if provided
        if user_weights is not None:
            weighted_log_probs = weighted_log_probs * user_weights.unsqueeze(1)
        
        # Compute loss (negative log likelihood)
        loss = -torch.mean(weighted_log_probs)
        
        return loss


class MovieLensDataset(torch.utils.data.Dataset):
    """Dataset for MovieLens sequence modeling."""
    
    def __init__(
        self,
        sequences: Dict[int, List[int]],
        user_map: Dict[int, int],
        item_map: Dict[int, int],
        max_seq_length: int = 100,
        min_seq_length: int = 5
    ):
        self.sequences = sequences
        self.user_map = user_map
        self.item_map = item_map
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        
        # Filter sequences by length
        self.valid_sequences = [
            (user_id, seq) for user_id, seq in sequences.items()
            if len(seq) >= min_seq_length
        ]
    
    def __len__(self):
        return len(self.valid_sequences)
    
    def __getitem__(self, idx):
        user_id, sequence = self.valid_sequences[idx]
        
        # Truncate or pad sequence
        if len(sequence) > self.max_seq_length:
            sequence = sequence[-self.max_seq_length:]
        else:
            sequence = sequence + [0] * (self.max_seq_length - len(sequence))
        
        return {
            'user_id': torch.tensor(self.user_map[user_id], dtype=torch.long),
            'sequence': torch.tensor(sequence, dtype=torch.long),
            'seq_length': torch.tensor(len(sequence), dtype=torch.long)
        }


def create_horror_domain_weights(
    sequences: Dict[int, List[int]],
    horror_items: set,
    weight_value: float = 2.0
) -> Dict[int, List[float]]:
    """
    Create domain-specific weights for Horror movie interactions.
    
    Args:
        sequences: User sequences
        horror_items: Set of horror movie item IDs
        weight_value: Weight to apply to horror interactions
        
    Returns:
        Dictionary mapping user_id to list of weights
    """
    domain_weights = {}
    
    for user_id, sequence in sequences.items():
        weights = []
        for item_id in sequence:
            if item_id in horror_items:
                weights.append(weight_value)
            else:
                weights.append(1.0)
        domain_weights[user_id] = weights
    
    return domain_weights
