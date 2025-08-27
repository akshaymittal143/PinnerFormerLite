"""
Data processing utilities for MovieLens dataset preparation.

This module handles the loading, preprocessing, and filtering of the MovieLens 25M dataset
for the PinnerFormerLite experiments.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import os
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieLensDataProcessor:
    """Data processor for MovieLens dataset."""
    
    def __init__(self, data_dir: str = "./data/ml-25m"):
        self.data_dir = data_dir
        self.ratings_path = os.path.join(data_dir, "ratings.csv")
        self.movies_path = os.path.join(data_dir, "movies.csv")
        
        # Data containers
        self.ratings_df = None
        self.movies_df = None
        self.user_map = {}
        self.item_map = {}
        self.horror_items = set()
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load MovieLens ratings and movies data."""
        logger.info("Loading MovieLens dataset...")
        
        if not os.path.exists(self.ratings_path):
            raise FileNotFoundError(
                f"Ratings file not found at {self.ratings_path}. "
                "Please download the MovieLens 25M dataset and extract it to the data directory."
            )
        
        if not os.path.exists(self.movies_path):
            raise FileNotFoundError(
                f"Movies file not found at {self.movies_path}. "
                "Please download the MovieLens 25M dataset and extract it to the data directory."
            )
        
        # Load data
        self.ratings_df = pd.read_csv(self.ratings_path)
        self.movies_df = pd.read_csv(self.movies_path)
        
        logger.info(f"Loaded {len(self.ratings_df):,} ratings and {len(self.movies_df):,} movies")
        
        return self.ratings_df, self.movies_df
    
    def preprocess_data(self) -> Tuple[Dict[int, int], Dict[int, int], Set[int]]:
        """Preprocess the data and create mappings."""
        logger.info("Preprocessing data...")
        
        # Convert timestamps to datetime
        self.ratings_df['timestamp'] = pd.to_datetime(self.ratings_df['timestamp'], unit='s')
        
        # Sort by timestamp
        self.ratings_df = self.ratings_df.sort_values('timestamp')
        
        # Create user and item mappings
        unique_users = self.ratings_df['userId'].unique()
        unique_items = self.ratings_df['movieId'].unique()
        
        self.user_map = {user_id: i for i, user_id in enumerate(unique_users)}
        self.item_map = {item_id: i for i, item_id in enumerate(unique_items)}
        
        # Add mapped columns
        self.ratings_df['userId_mapped'] = self.ratings_df['userId'].map(self.user_map)
        self.ratings_df['movieId_mapped'] = self.ratings_df['movieId'].map(self.item_map)
        
        # Extract horror movies
        horror_movies = self.movies_df[self.movies_df['genres'].str.contains('Horror', na=False)]
        horror_movie_ids = horror_movies['movieId'].unique()
        
        self.horror_items = {
            self.item_map[mid] for mid in horror_movie_ids 
            if mid in self.item_map
        }
        
        logger.info(f"Created mappings for {len(self.user_map):,} users and {len(self.item_map):,} items")
        logger.info(f"Found {len(self.horror_items):,} horror movies")
        
        return self.user_map, self.item_map, self.horror_items
    
    def create_sequences(self) -> Dict[int, List[int]]:
        """Create user sequences from ratings data."""
        logger.info("Creating user sequences...")
        
        sequences = defaultdict(list)
        
        for _, row in self.ratings_df.iterrows():
            user_id = row['userId_mapped']
            movie_id = row['movieId_mapped']
            sequences[user_id].append(movie_id)
        
        # Filter sequences by minimum length
        min_seq_length = 5
        filtered_sequences = {
            user_id: seq for user_id, seq in sequences.items()
            if len(seq) >= min_seq_length
        }
        
        logger.info(f"Created {len(filtered_sequences):,} user sequences")
        
        return filtered_sequences
    
    def identify_power_users(
        self, 
        sequences: Dict[int, List[int]], 
        horror_threshold: float = 0.5
    ) -> Set[int]:
        """
        Identify power users with high horror movie affinity.
        
        Args:
            sequences: User sequences
            horror_threshold: Minimum fraction of horror movies to be considered a power user
            
        Returns:
            Set of power user IDs
        """
        logger.info("Identifying power users...")
        
        power_users = set()
        
        for user_id, sequence in sequences.items():
            horror_count = sum(1 for item_id in sequence if item_id in self.horror_items)
            horror_fraction = horror_count / len(sequence)
            
            if horror_fraction >= horror_threshold:
                power_users.add(user_id)
        
        logger.info(f"Identified {len(power_users):,} power users")
        
        return power_users
    
    def identify_general_users(
        self,
        sequences: Dict[int, List[int]],
        horror_threshold: float = 0.1
    ) -> Set[int]:
        """
        Identify general users with low horror movie affinity for fairness analysis.
        
        Args:
            sequences: User sequences
            horror_threshold: Maximum fraction of horror movies to be considered a general user
            
        Returns:
            Set of general user IDs
        """
        logger.info("Identifying general users for fairness analysis...")
        
        general_users = set()
        
        for user_id, sequence in sequences.items():
            horror_count = sum(1 for item_id in sequence if item_id in self.horror_items)
            horror_fraction = horror_count / len(sequence)
            
            if horror_fraction <= horror_threshold:
                general_users.add(user_id)
        
        logger.info(f"Identified {len(general_users):,} general users")
        
        return general_users
    
    def sensitivity_analysis_power_users(
        self,
        sequences: Dict[int, List[int]],
        thresholds: List[float] = [0.4, 0.5, 0.6]
    ) -> Dict[float, Dict[str, int]]:
        """
        Perform sensitivity analysis for different power user thresholds.
        
        Args:
            sequences: User sequences
            thresholds: List of horror affinity thresholds to test
            
        Returns:
            Dictionary with threshold as key and statistics as value
        """
        logger.info("Performing power user threshold sensitivity analysis...")
        
        results = {}
        
        for threshold in thresholds:
            power_users = self.identify_power_users(sequences, horror_threshold=threshold)
            
            # Calculate statistics
            total_users = len(sequences)
            power_user_count = len(power_users)
            power_user_percentage = (power_user_count / total_users) * 100 if total_users > 0 else 0
            
            # Calculate average horror affinity for power users
            total_horror_affinity = 0
            for user_id in power_users:
                if user_id in sequences:
                    sequence = sequences[user_id]
                    horror_count = sum(1 for item_id in sequence if item_id in self.horror_items)
                    horror_fraction = horror_count / len(sequence)
                    total_horror_affinity += horror_fraction
            
            avg_horror_affinity = total_horror_affinity / power_user_count if power_user_count > 0 else 0
            
            results[threshold] = {
                'power_user_count': power_user_count,
                'power_user_percentage': power_user_percentage,
                'avg_horror_affinity': avg_horror_affinity,
                'total_users': total_users
            }
            
            logger.info(f"Threshold {threshold:.1%}: {power_user_count:,} power users "
                       f"({power_user_percentage:.1f}%), avg affinity: {avg_horror_affinity:.3f}")
        
        return results
    
    def create_distilled_dataset(
        self, 
        sequences: Dict[int, List[int]], 
        power_users: Set[int]
    ) -> Dict[int, List[int]]:
        """
        Create distilled dataset for power users.
        
        Args:
            sequences: All user sequences
            power_users: Set of power user IDs
            
        Returns:
            Distilled sequences containing only horror interactions for power users
        """
        logger.info("Creating distilled dataset...")
        
        distilled_sequences = {}
        
        for user_id in power_users:
            if user_id in sequences:
                # Filter to only horror movies
                horror_sequence = [
                    item_id for item_id in sequences[user_id]
                    if item_id in self.horror_items
                ]
                
                if len(horror_sequence) >= 5:  # Minimum sequence length
                    distilled_sequences[user_id] = horror_sequence
        
        logger.info(f"Created distilled dataset with {len(distilled_sequences):,} sequences")
        
        return distilled_sequences
    
    def split_sequences(
        self, 
        sequences: Dict[int, List[int]], 
        test_ratio: float = 0.2,
        val_ratio: float = 0.1
    ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]]]:
        """
        Split sequences into train/validation/test sets.
        
        Args:
            sequences: User sequences
            test_ratio: Fraction of data for testing
            val_ratio: Fraction of data for validation
            
        Returns:
            Tuple of (train_sequences, val_sequences, test_sequences)
        """
        logger.info("Splitting sequences into train/val/test sets...")
        
        train_sequences = {}
        val_sequences = {}
        test_sequences = {}
        
        for user_id, sequence in sequences.items():
            if len(sequence) < 10:  # Skip very short sequences
                continue
                
            # Split sequence chronologically
            split_idx = int(len(sequence) * (1 - test_ratio - val_ratio))
            val_split_idx = int(len(sequence) * (1 - test_ratio))
            
            train_seq = sequence[:split_idx]
            val_seq = sequence[split_idx:val_split_idx]
            test_seq = sequence[val_split_idx:]
            
            if len(train_seq) >= 5:
                train_sequences[user_id] = train_seq
            if len(val_seq) >= 2:
                val_sequences[user_id] = val_seq
            if len(test_seq) >= 2:
                test_sequences[user_id] = test_seq
        
        logger.info(f"Split into {len(train_sequences):,} train, {len(val_sequences):,} val, {len(test_sequences):,} test sequences")
        
        return train_sequences, val_sequences, test_sequences
    
    def get_dataset_statistics(self, sequences: Dict[int, List[int]]) -> Dict:
        """Get statistics about the dataset."""
        stats = {
            'num_users': len(sequences),
            'total_interactions': sum(len(seq) for seq in sequences.values()),
            'avg_seq_length': np.mean([len(seq) for seq in sequences.values()]),
            'median_seq_length': np.median([len(seq) for seq in sequences.values()]),
            'min_seq_length': min(len(seq) for seq in sequences.values()),
            'max_seq_length': max(len(seq) for seq in sequences.values()),
            'horror_interactions': sum(
                sum(1 for item_id in seq if item_id in self.horror_items)
                for seq in sequences.values()
            )
        }
        
        return stats
    
    def process_full_pipeline(self) -> Dict:
        """
        Run the complete data processing pipeline.
        
        Returns:
            Dictionary containing all processed data and mappings
        """
        # Load data
        self.load_data()
        
        # Preprocess
        user_map, item_map, horror_items = self.preprocess_data()
        
        # Create sequences
        sequences = self.create_sequences()
        
        # Perform sensitivity analysis for power user thresholds
        sensitivity_results = self.sensitivity_analysis_power_users(sequences)
        
        # Identify power users (using 50% threshold as per paper)
        power_users = self.identify_power_users(sequences, horror_threshold=0.5)
        
        # Identify general users for fairness analysis
        general_users = self.identify_general_users(sequences, horror_threshold=0.1)
        
        # Create distilled dataset
        distilled_sequences = self.create_distilled_dataset(sequences, power_users)
        
        # Split sequences
        train_sequences, val_sequences, test_sequences = self.split_sequences(sequences)
        train_distilled, val_distilled, test_distilled = self.split_sequences(distilled_sequences)
        
        # Get statistics
        stats = self.get_dataset_statistics(sequences)
        distilled_stats = self.get_dataset_statistics(distilled_sequences)
        
        logger.info("Data processing pipeline completed successfully")
        
        return {
            'user_map': user_map,
            'item_map': item_map,
            'horror_items': horror_items,
            'sequences': sequences,
            'distilled_sequences': distilled_sequences,
            'power_users': power_users,
            'general_users': general_users,
            'sensitivity_results': sensitivity_results,
            'train_sequences': train_sequences,
            'val_sequences': val_sequences,
            'test_sequences': test_sequences,
            'train_distilled': train_distilled,
            'val_distilled': val_distilled,
            'test_distilled': test_distilled,
            'stats': stats,
            'distilled_stats': distilled_stats
        }


def download_movielens_data(data_dir: str = "./data") -> str:
    """
    Download MovieLens 25M dataset if not already present.
    
    Args:
        data_dir: Directory to store the data
        
    Returns:
        Path to the extracted data directory
    """
    import urllib.request
    import zipfile
    
    os.makedirs(data_dir, exist_ok=True)
    ml25m_dir = os.path.join(data_dir, "ml-25m")
    
    if os.path.exists(ml25m_dir):
        logger.info("MovieLens 25M dataset already exists")
        return ml25m_dir
    
    # Download URL
    url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    zip_path = os.path.join(data_dir, "ml-25m.zip")
    
    logger.info("Downloading MovieLens 25M dataset...")
    urllib.request.urlretrieve(url, zip_path)
    
    logger.info("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Clean up zip file
    os.remove(zip_path)
    
    logger.info("Dataset downloaded and extracted successfully")
    return ml25m_dir
