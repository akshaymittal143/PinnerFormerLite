"""
Training and evaluation utilities for PinnerFormerLite.

This module provides the training loop, evaluation metrics, and experiment management
for the PinnerFormerLite model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import time
from collections import defaultdict
import json
import os

from pinnerformer_lite import PinnerFormerLite, WeightedDenseAllActionLoss, MovieLensDataset

logger = logging.getLogger(__name__)


class PinnerFormerLiteTrainer:
    """Trainer for PinnerFormerLite model."""
    
    def __init__(
        self,
        model: PinnerFormerLite,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 0.01
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize optimizer and loss
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = WeightedDenseAllActionLoss(temperature=0.1)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        domain_weights: Optional[Dict[int, List[float]]] = None,
        epoch: int = 0
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            domain_weights: Domain-specific weights for each user
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            user_ids = batch['user_id'].to(self.device)
            sequences = batch['sequence'].to(self.device)
            seq_lengths = batch['seq_length'].to(self.device)
            
            # Create attention mask
            attention_mask = torch.arange(sequences.size(1)).unsqueeze(0) < seq_lengths.unsqueeze(1)
            attention_mask = attention_mask.to(self.device)
            
            # Forward pass
            user_embeddings, item_embeddings = self.model(user_ids, sequences, attention_mask)
            
            # Prepare positive items (last item in each sequence)
            positive_items = sequences[:, -1].unsqueeze(1)  # [batch_size, 1]
            
            # Create domain weights for this batch
            if domain_weights is not None:
                batch_weights = []
                for user_id in user_ids.cpu().numpy():
                    user_weights = domain_weights.get(user_id, [1.0])
                    batch_weights.append(user_weights[-1] if user_weights else 1.0)
                domain_weights_tensor = torch.tensor(batch_weights, dtype=torch.float).to(self.device)
            else:
                domain_weights_tensor = torch.ones(sequences.size(0), 1).to(self.device)
            
            # Compute loss
            loss = self.criterion(
                user_embeddings,
                item_embeddings,
                positive_items,
                domain_weights_tensor
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def evaluate(
        self,
        val_loader: DataLoader,
        domain_weights: Optional[Dict[int, List[float]]] = None
    ) -> float:
        """
        Evaluate the model on validation data.
        
        Args:
            val_loader: Validation data loader
            domain_weights: Domain-specific weights for each user
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                user_ids = batch['user_id'].to(self.device)
                sequences = batch['sequence'].to(self.device)
                seq_lengths = batch['seq_length'].to(self.device)
                
                # Create attention mask
                attention_mask = torch.arange(sequences.size(1)).unsqueeze(0) < seq_lengths.unsqueeze(1)
                attention_mask = attention_mask.to(self.device)
                
                # Forward pass
                user_embeddings, item_embeddings = self.model(user_ids, sequences, attention_mask)
                
                # Prepare positive items
                positive_items = sequences[:, -1].unsqueeze(1)
                
                # Create domain weights for this batch
                if domain_weights is not None:
                    batch_weights = []
                    for user_id in user_ids.cpu().numpy():
                        user_weights = domain_weights.get(user_id, [1.0])
                        batch_weights.append(user_weights[-1] if user_weights else 1.0)
                    domain_weights_tensor = torch.tensor(batch_weights, dtype=torch.float).to(self.device)
                else:
                    domain_weights_tensor = torch.ones(sequences.size(0), 1).to(self.device)
                
                # Compute loss
                loss = self.criterion(
                    user_embeddings,
                    item_embeddings,
                    positive_items,
                    domain_weights_tensor
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 10,
        domain_weights: Optional[Dict[int, List[float]]] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            domain_weights: Domain-specific weights for each user
            save_path: Path to save the best model
            
        Returns:
            Dictionary containing training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader, domain_weights, epoch)
            
            # Evaluate
            val_loss = self.evaluate(val_loader, domain_weights)
            
            epoch_time = time.time() - start_time
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Save best model
            if val_loss < best_val_loss and save_path:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"Saved best model with validation loss: {val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def save_model(self, path: str):
        """Save the model."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load the model."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Model loaded from {path}")


class Evaluator:
    """Evaluator for computing recommendation metrics."""
    
    def __init__(self, model: PinnerFormerLite, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device
    
    def compute_recall_at_k(
        self,
        user_embeddings: torch.Tensor,
        positive_items: torch.Tensor,
        all_items: torch.Tensor,
        k: int = 10
    ) -> float:
        """
        Compute Recall@K.
        
        Args:
            user_embeddings: User embeddings [batch_size, d_model]
            positive_items: Positive item indices [batch_size]
            all_items: All item embeddings [num_items, d_model]
            k: Number of top items to consider
            
        Returns:
            Recall@K score
        """
        # Compute similarities
        similarities = torch.mm(user_embeddings, all_items.t())  # [batch_size, num_items]
        
        # Get top-k items
        _, top_k_indices = torch.topk(similarities, k, dim=1)  # [batch_size, k]
        
        # Check if positive items are in top-k
        positive_items_expanded = positive_items.unsqueeze(1)  # [batch_size, 1]
        hits = (top_k_indices == positive_items_expanded).any(dim=1)  # [batch_size]
        
        recall = hits.float().mean().item()
        return recall
    
    def compute_interest_entropy(
        self,
        user_embeddings: torch.Tensor,
        all_items: torch.Tensor,
        k: int = 50
    ) -> float:
        """
        Compute Interest Entropy@K.
        
        Args:
            user_embeddings: User embeddings [batch_size, d_model]
            all_items: All item embeddings [num_items, d_model]
            k: Number of top items to consider
            
        Returns:
            Interest Entropy@K score
        """
        # Compute similarities
        similarities = torch.mm(user_embeddings, all_items.t())  # [batch_size, num_items]
        
        # Get top-k items
        _, top_k_indices = torch.topk(similarities, k, dim=1)  # [batch_size, k]
        
        # Compute entropy for each user
        entropies = []
        for user_top_k in top_k_indices:
            # Count occurrences of each item
            item_counts = torch.bincount(user_top_k, minlength=all_items.size(0))
            item_probs = item_counts.float() / k
            
            # Compute entropy
            entropy = -torch.sum(item_probs * torch.log(item_probs + 1e-8))
            entropies.append(entropy.item())
        
        return np.mean(entropies)
    
    def compute_coverage(
        self,
        user_embeddings: torch.Tensor,
        all_items: torch.Tensor,
        k: int = 10,
        percentile: float = 90
    ) -> float:
        """
        Compute P90 Coverage@K.
        
        Args:
            user_embeddings: User embeddings [batch_size, d_model]
            all_items: All item embeddings [num_items, d_model]
            k: Number of top items to consider
            percentile: Percentile for coverage calculation
            
        Returns:
            Coverage score
        """
        # Compute similarities
        similarities = torch.mm(user_embeddings, all_items.t())  # [batch_size, num_items]
        
        # Get top-k items for each user
        _, top_k_indices = torch.topk(similarities, k, dim=1)  # [batch_size, k]
        
        # Flatten all top-k items
        all_top_items = top_k_indices.flatten()
        
        # Count occurrences of each item
        item_counts = torch.bincount(all_top_items, minlength=all_items.size(0))
        
        # Sort counts
        sorted_counts, _ = torch.sort(item_counts, descending=True)
        
        # Find the number of items that account for the specified percentile
        total_recommendations = user_embeddings.size(0) * k
        target_count = int(total_recommendations * percentile / 100)
        
        cumulative_count = 0
        num_items = 0
        
        for count in sorted_counts:
            cumulative_count += count.item()
            num_items += 1
            if cumulative_count >= target_count:
                break
        
        coverage = num_items / all_items.size(0)
        return coverage
    
    def evaluate_model(
        self,
        test_loader: DataLoader,
        all_items_embedding: torch.Tensor,
        power_users: Optional[set] = None
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader
            all_items_embedding: All item embeddings
            power_users: Set of power user IDs for focused evaluation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        
        all_user_embeddings = []
        all_positive_items = []
        all_user_ids = []
        
        with torch.no_grad():
            for batch in test_loader:
                user_ids = batch['user_id'].to(self.device)
                sequences = batch['sequence'].to(self.device)
                seq_lengths = batch['seq_length'].to(self.device)
                
                # Create attention mask
                attention_mask = torch.arange(sequences.size(1)).unsqueeze(0) < seq_lengths.unsqueeze(1)
                attention_mask = attention_mask.to(self.device)
                
                # Forward pass
                user_embeddings, _ = self.model(user_ids, sequences, attention_mask)
                
                # Store results
                all_user_embeddings.append(user_embeddings.cpu())
                all_positive_items.append(sequences[:, -1].cpu())
                all_user_ids.extend(user_ids.cpu().numpy())
        
        # Concatenate results
        user_embeddings = torch.cat(all_user_embeddings, dim=0)
        positive_items = torch.cat(all_positive_items, dim=0)
        
        # Filter for power users if specified
        if power_users is not None:
            power_user_mask = torch.tensor([uid in power_users for uid in all_user_ids])
            user_embeddings = user_embeddings[power_user_mask]
            positive_items = positive_items[power_user_mask]
        
        # Compute metrics
        recall_at_10 = self.compute_recall_at_k(user_embeddings, positive_items, all_items_embedding, k=10)
        interest_entropy_50 = self.compute_interest_entropy(user_embeddings, all_items_embedding, k=50)
        coverage_10 = self.compute_coverage(user_embeddings, all_items_embedding, k=10, percentile=90)
        
        return {
            'recall_at_10': recall_at_10,
            'interest_entropy_50': interest_entropy_50,
            'p90_coverage_10': coverage_10
        }


def run_experiment(
    data_dict: Dict,
    model_config: Dict,
    training_config: Dict,
    experiment_name: str = "pinnerformer_lite_experiment"
) -> Dict:
    """
    Run a complete PinnerFormerLite experiment.
    
    Args:
        data_dict: Dictionary containing processed data
        model_config: Model configuration
        training_config: Training configuration
        experiment_name: Name of the experiment
        
    Returns:
        Dictionary containing experiment results
    """
    logger.info(f"Starting experiment: {experiment_name}")
    
    # Extract data
    user_map = data_dict['user_map']
    item_map = data_dict['item_map']
    train_sequences = data_dict['train_sequences']
    val_sequences = data_dict['val_sequences']
    test_sequences = data_dict['test_sequences']
    horror_items = data_dict['horror_items']
    power_users = data_dict['power_users']
    
    # Create datasets
    train_dataset = MovieLensDataset(train_sequences, user_map, item_map)
    val_dataset = MovieLensDataset(val_sequences, user_map, item_map)
    test_dataset = MovieLensDataset(test_sequences, user_map, item_map)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=training_config.get('num_workers', 0)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=training_config.get('num_workers', 0)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=training_config.get('num_workers', 0)
    )
    
    # Create model
    model = PinnerFormerLite(
        num_items=len(item_map),
        num_users=len(user_map),
        **model_config
    )
    
    # Create trainer
    trainer = PinnerFormerLiteTrainer(
        model,
        device=training_config['device'],
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config.get('weight_decay', 0.01)
    )
    
    # Create domain weights for weighted training
    domain_weights = None
    if training_config.get('use_weighted_loss', False):
        from pinnerformer_lite import create_horror_domain_weights
        domain_weights = create_horror_domain_weights(
            train_sequences,
            horror_items,
            weight_value=training_config.get('domain_weight', 2.0)
        )
    
    # Train model
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=training_config['num_epochs'],
        domain_weights=domain_weights,
        save_path=f"models/{experiment_name}_best.pth"
    )
    
    # Evaluate model
    evaluator = Evaluator(model, device=training_config['device'])
    
    # Get all item embeddings for evaluation
    with torch.no_grad():
        all_items = torch.arange(len(item_map)).to(training_config['device'])
        all_item_embeddings = model.item_embedding(all_items)
    
    # Evaluate on all users
    all_user_metrics = evaluator.evaluate_model(test_loader, all_item_embeddings)
    
    # Evaluate on power users only
    power_user_metrics = evaluator.evaluate_model(test_loader, all_item_embeddings, power_users)
    
    # Save results
    results = {
        'experiment_name': experiment_name,
        'model_config': model_config,
        'training_config': training_config,
        'training_history': history,
        'all_user_metrics': all_user_metrics,
        'power_user_metrics': power_user_metrics,
        'dataset_stats': data_dict['stats']
    }
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    with open(f'results/{experiment_name}_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Experiment completed. Results saved to results/{experiment_name}_results.json")
    
    return results
