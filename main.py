"""
Main Script for NAD+ Supplement Study Analysis Pipeline

This script combines the data processing and clustering optimization stages
to perform a complete analysis of NAD+ supplement study data.

Author: [Your Name]
Date: May 21, 2025
"""

import os
import argparse
import time
from data_processing import DataProcessor
from clustering import ClusteringOptimizer


def main():
    """Main function to run the complete NAD+ analysis pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='NAD+ Supplement Study Analysis Pipeline')
    
    parser.add_argument('--data_dir', type=str, default='statistics',#'processed_data_unimodal_analysed/filtered_data',
                      help='Directory containing raw data files')
    parser.add_argument('--processed_dir', type=str, default='processed_data',
                      help='Directory to save processed data')
    parser.add_argument('--results_dir', type=str, default='results',
                      help='Directory to save analysis results')
    parser.add_argument('--balance_tolerance', type=float, default=0.15,
                      help='Balance tolerance for clustering (default: 0.15)')
    parser.add_argument('--alpha_threshold', type=float, default=0.05,
                      help='Alpha threshold for significance testing (default: 0.05)')
    parser.add_argument('--skip_processing', action='store_true',
                      help='Skip data processing and use existing processed data')
    parser.add_argument('--min_subjects', type=int, default=20,
                      help='Minimum number of subjects required (default: 20)')
    parser.add_argument('--memory_efficient', action='store_true',
                      help='Use memory-efficient processing')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("NAD+ SUPPLEMENT STUDY ANALYSIS PIPELINE")
    print("=" * 80)
    
    start_time = time.time()
    
    # Stage 1: Data Processing
    if not args.skip_processing:
        print("\nSTAGE 1: DATA PROCESSING")
        print("-" * 24)
        
        # Initialize processor
        processor = DataProcessor(
            alpha_threshold=args.alpha_threshold,
            memory_efficient=args.memory_efficient
        )
        
        # Identify raw data files
        file_paths = []
        precomputed_contrasts = []
        
        # Walk through data directory and identify files
        for file in os.listdir(args.data_dir):
            if file.endswith('_cntr.csv'):
                precomputed_contrasts.append(os.path.join(args.data_dir, file))
            elif file.endswith('.csv'):
                file_paths.append(os.path.join(args.data_dir, file))
        
        print(f"Found {len(file_paths)} raw data files and {len(precomputed_contrasts)} pre-contrasted files")
        
        # Process data
        processing_results = processor.process_all_data(
            file_paths=file_paths,
            precomputed_contrasts=precomputed_contrasts,
            output_dir=args.processed_dir
        )
        
        processing_time = time.time() - start_time
        print(f"\nData processing completed in {processing_time:.2f} seconds")
    else:
        print("\nSkipping data processing, using existing processed data")
    
    # Stage 2: Clustering Optimization
    print("\nSTAGE 2: CLUSTERING OPTIMIZATION")
    print("-" * 30)
    
    clustering_start = time.time()
    
    # Initialize optimizer
    optimizer = ClusteringOptimizer(
        balance_tolerance=args.balance_tolerance,
        random_state=42,
        min_subjects=70
    )
    
    # Run clustering analysis
    clustering_results = optimizer.run_complete_analysis(
        data_dir=args.processed_dir,
        output_dir=args.results_dir
    )
    
    clustering_time = time.time() - clustering_start
    print(f"\nClustering optimization completed in {clustering_time:.2f} seconds")
    
    # Print final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("ANALYSIS PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Total execution time: {total_time:.2f} seconds")
    
    # Print best model summary if available
    if clustering_results['best_model']:
        best = clustering_results['best_model']
        print("\nBEST CLUSTERING MODEL:")
        print(f"Silhouette Score: {best['silhouette_score']:.4f}")
        print(f"Balance Ratio: {best['balance_ratio']:.4f}")
        print(f"Algorithm: {best['algorithm']}")
        print(f"Dataset: {best['dataset']}")
        
        # Get cluster assignments
        import numpy as np
        clusters = np.bincount(best['labels'])
        print(f"Cluster Distribution: {clusters[0]} vs {clusters[1]} participants")
    else:
        print("\nNo suitable clustering model found")
    
    print("\nResults saved to:")
    print(f"- Processed data: {args.processed_dir}/")
    print(f"- Analysis results: {args.results_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()