#!/usr/bin/env python3
"""
Main script for running PinnerFormerLite experiments.

This script demonstrates the complete pipeline for training and evaluating
the PinnerFormerLite model on the MovieLens 25M dataset.
"""

import os
import sys
import logging
import json
import argparse
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing import MovieLensDataProcessor, download_movielens_data
from trainer import run_experiment
from pinnerformer_lite import PinnerFormerLite

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the PinnerFormerLite experiment."""
    
    parser = argparse.ArgumentParser(description='PinnerFormerLite Experiment')
    parser.add_argument('--download_data', action='store_true', 
                       help='Download MovieLens 25M dataset')
    parser.add_argument('--data_dir', type=str, default='./data/ml-25m',
                       help='Directory containing MovieLens data')
    parser.add_argument('--experiment_name', type=str, default='pinnerformer_lite',
                       help='Name of the experiment')
    parser.add_argument('--use_weighted_loss', action='store_true',
                       help='Use weighted loss for domain-specific training')
    parser.add_argument('--domain_weight', type=float, default=2.0,
                       help='Weight for horror domain interactions')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Download data if requested
    if args.download_data:
        logger.info("Downloading MovieLens 25M dataset...")
        data_dir = download_movielens_data()
    else:
        data_dir = args.data_dir
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Initialize data processor
    processor = MovieLensDataProcessor(data_dir)
    
    # Process data
    logger.info("Processing MovieLens dataset...")
    data_dict = processor.process_full_pipeline()
    
    # Print dataset statistics
    logger.info("Dataset Statistics:")
    logger.info(f"  Total users: {data_dict['stats']['num_users']:,}")
    logger.info(f"  Total interactions: {data_dict['stats']['total_interactions']:,}")
    logger.info(f"  Average sequence length: {data_dict['stats']['avg_seq_length']:.2f}")
    logger.info(f"  Horror interactions: {data_dict['stats']['horror_interactions']:,}")
    logger.info(f"  Power users: {len(data_dict['power_users']):,}")
    
    logger.info("Distilled Dataset Statistics:")
    logger.info(f"  Users: {data_dict['distilled_stats']['num_users']:,}")
    logger.info(f"  Interactions: {data_dict['distilled_stats']['total_interactions']:,}")
    logger.info(f"  Average sequence length: {data_dict['distilled_stats']['avg_seq_length']:.2f}")
    
    # Model configuration
    model_config = {
        'd_model': 256,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'max_seq_length': 100
    }
    
    # Training configuration
    training_config = {
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': 0.01,
        'device': device,
        'use_weighted_loss': args.use_weighted_loss,
        'domain_weight': args.domain_weight,
        'num_workers': 0  # Set to higher value if you have multiple CPU cores
    }
    
    # Run experiment
    logger.info(f"Starting experiment: {args.experiment_name}")
    results = run_experiment(
        data_dict=data_dict,
        model_config=model_config,
        training_config=training_config,
        experiment_name=args.experiment_name
    )
    
    # Print results
    logger.info("Experiment Results:")
    logger.info("All Users Metrics:")
    for metric, value in results['all_user_metrics'].items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("Power Users Metrics:")
    for metric, value in results['power_user_metrics'].items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save final results summary
    summary = {
        'experiment_name': args.experiment_name,
        'model_config': model_config,
        'training_config': training_config,
        'results': {
            'all_users': results['all_user_metrics'],
            'power_users': results['power_user_metrics']
        },
        'dataset_stats': data_dict['stats']
    }
    
    with open(f'results/{args.experiment_name}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Experiment completed successfully!")
    logger.info(f"Results saved to: results/{args.experiment_name}_results.json")
    logger.info(f"Summary saved to: results/{args.experiment_name}_summary.json")


def run_comparison_experiments():
    """Run comparison experiments between generic and weighted models."""
    
    logger.info("Running comparison experiments...")
    
    # Download and process data
    data_dir = download_movielens_data()
    processor = MovieLensDataProcessor(data_dir)
    data_dict = processor.process_full_pipeline()
    
    # Model configuration
    model_config = {
        'd_model': 256,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'max_seq_length': 100
    }
    
    # Experiment 1: Generic Model (no weighted loss)
    logger.info("Running Generic Model Experiment...")
    generic_config = {
        'batch_size': 256,
        'num_epochs': 5,  # Reduced for demonstration
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'use_weighted_loss': False,
        'num_workers': 0
    }
    
    generic_results = run_experiment(
        data_dict=data_dict,
        model_config=model_config,
        training_config=generic_config,
        experiment_name="generic_model"
    )
    
    # Experiment 2: Weighted Model (with weighted loss)
    logger.info("Running Weighted Model Experiment...")
    weighted_config = {
        'batch_size': 256,
        'num_epochs': 5,  # Reduced for demonstration
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'use_weighted_loss': True,
        'domain_weight': 2.0,
        'num_workers': 0
    }
    
    weighted_results = run_experiment(
        data_dict=data_dict,
        model_config=model_config,
        training_config=weighted_config,
        experiment_name="weighted_model"
    )
    
    # Print comparison
    logger.info("\n" + "="*50)
    logger.info("EXPERIMENT COMPARISON RESULTS")
    logger.info("="*50)
    
    logger.info("\nGeneric Model (All Users):")
    for metric, value in generic_results['all_user_metrics'].items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("\nWeighted Model (All Users):")
    for metric, value in weighted_results['all_user_metrics'].items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("\nGeneric Model (Power Users):")
    for metric, value in generic_results['power_user_metrics'].items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("\nWeighted Model (Power Users):")
    for metric, value in weighted_results['power_user_metrics'].items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Calculate improvements
    logger.info("\nIMPROVEMENTS (Power Users):")
    for metric in generic_results['power_user_metrics'].keys():
        generic_val = generic_results['power_user_metrics'][metric]
        weighted_val = weighted_results['power_user_metrics'][metric]
        improvement = ((weighted_val - generic_val) / generic_val) * 100
        logger.info(f"  {metric}: {improvement:+.1f}%")
    
    logger.info("\nExperiments completed successfully!")


if __name__ == "__main__":
    import torch
    
    # Check if running comparison or single experiment
    if len(sys.argv) > 1 and sys.argv[1] == "comparison":
        run_comparison_experiments()
    else:
        main()
