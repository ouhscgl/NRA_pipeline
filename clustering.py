"""
Clustering Optimization Module for NAD+ Supplement Study

This module handles the clustering optimization to maximize silhouette score for the
NAD+ supplement study data. It implements various clustering algorithms, feature
selection methods, and optimization techniques.

Author: [Your Name]
Date: May 21, 2025
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from sklearn.impute import KNNImputer
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class ClusteringOptimizer:
    """
    Optimizer for clustering NAD+ supplement study data.
    
    This class handles:
    1. Loading processed contrast data
    2. Feature selection and engineering
    3. Clustering with multiple algorithms
    4. Optimizing for silhouette score with balanced groups
    5. Iterative optimization to find theoretical maximum silhouette score
    """
    
    def __init__(self, balance_tolerance: float = 0.15, random_state: int = 42, 
                 min_subjects: int = 20, primary_metric: str = 'silhouette'):
        """
        Initialize the ClusteringOptimizer.
        
        Args:
            balance_tolerance: Maximum allowed deviation from balanced clusters (default: 0.15)
            random_state: Random seed for reproducibility (default: 42)
            min_subjects: Minimum number of subjects required (default: 20)
            primary_metric: Primary metric to optimize ('silhouette' or 'nad_specific')
        """
        self.balance_tolerance = balance_tolerance
        self.random_state = random_state
        self.min_subjects = min_subjects
        self.primary_metric = primary_metric
        self.clustering_results = []
        self.best_model = None
        self.feature_importance = None
        self.optimization_history = []
        self.best_overall = None
        
        # Available evaluation metrics
        self.available_metrics = {
            'silhouette': self._calculate_silhouette_score,
            'calinski_harabasz': self._calculate_calinski_harabasz_score,
            'davies_bouldin': self._calculate_davies_bouldin_score,
            'nad_specific': self._calculate_nad_specific_metric,
            'dunn_index': self._calculate_dunn_index
        }
    
    def run_clustering_analysis(self, preprocessed_datasets):
        """
        Run comprehensive clustering analysis with multiple algorithms.
        
        Args:
            preprocessed_datasets: Dictionary of preprocessed datasets
            
        Returns:
            List of clustering results
        """
        print("\nRunning standard clustering analysis...")
        
        # Define the clustering methods to use
        clustering_methods = {
            'kmeans': self._try_kmeans,
            'hierarchical': self._try_hierarchical,
            'gmm': self._try_gmm,
            'spectral': self._try_spectral,
            'ensemble': self._try_ensemble_clustering
        }
        
        all_results = []
        
        # Try each dataset with each clustering algorithm
        for dataset_name, dataset_info in preprocessed_datasets.items():
            X = dataset_info['X']
            
            # Skip if too few samples or features
            if X.shape[0] < self.min_subjects or X.shape[1] < 2:
                continue
            
            print(f"  Analyzing {dataset_name} ({X.shape[0]} samples, {X.shape[1]} features)")
            
            # Try each clustering algorithm
            for method_name, method_func in clustering_methods.items():
                try:
                    # Call the method with the dataset
                    results = method_func(X)
                    
                    # Add dataset info to each result
                    for result in results:
                        result['dataset'] = dataset_name
                        result['dataset_info'] = dataset_info
                        all_results.append(result)
                except Exception as e:
                    print(f"    Error with {method_name}: {str(e)}")
        
        # Sort results by silhouette score
        all_results.sort(key=lambda x: (x['is_balanced'], x['silhouette_score']), reverse=True)
        
        # Store all results
        self.clustering_results = all_results
        
        # Get best model
        if all_results:
            self.best_model = all_results[0]
            print(f"\nBest standard clustering model:")
            print(f"  Dataset: {self.best_model['dataset']}")
            print(f"  Algorithm: {self.best_model['algorithm']}")
            print(f"  Silhouette Score: {self.best_model['silhouette_score']:.4f}")
            print(f"  Balanced: {self.best_model['is_balanced']}")
            print(f"  Balance Ratio: {self.best_model['balance_ratio']:.4f}")
        else:
            print("\nNo valid clustering results found.")
        
        return all_results
    
    def preprocess_datasets(self, datasets: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Preprocess datasets with different scalers and dimensionality reduction.
        
        Args:
            datasets: Dictionary mapping dataset names to their data
            
        Returns:
            Dictionary of preprocessed datasets
        """
        print("\nPreprocessing datasets...")
        
        preprocessed_datasets = {}
        
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler(),
            'quantile': QuantileTransformer(output_distribution='normal')
        }
        
        dim_reduction = {
            'none': None,
            'pca_90': ('pca', 0.90),
            'pca_95': ('pca', 0.95),
            'kernel_pca': ('kernel_pca', 10)
        }
        
        for dataset_name, dataset_info in datasets.items():
            X = dataset_info['X']
            feature_names = dataset_info['feature_names']
            base_names = dataset_info['base_names']
            
            # Skip if too few samples or features
            if X.shape[0] < self.min_subjects or X.shape[1] < 2:
                continue
                
            # Handle missing values
            if np.isnan(X).any():
                imputer = KNNImputer(n_neighbors=min(5, X.shape[0]-1))
                X = imputer.fit_transform(X)
            
            # Create enhanced features variant
            enhanced_X, enhanced_feature_names = self.enhance_features(X, feature_names, max_features=None)
            
            # Process both original and enhanced datasets
            for data_X, data_feature_names, suffix in [
                (X, feature_names, ""),
                (enhanced_X, enhanced_feature_names, "_enhanced")
            ]:
                # Process with different scalers
                for scaler_name, scaler in scalers.items():
                    # Apply scaling
                    X_scaled = scaler.fit_transform(data_X)
                    
                    # Process with different dimensionality reduction
                    for dim_name, dim_params in dim_reduction.items():
                        X_final = X_scaled.copy()
                        final_feature_names = data_feature_names.copy()
                        
                        if dim_params is not None:
                            method, param = dim_params
                            
                            if method == 'pca' and X_final.shape[1] > 2:
                                try:
                                    pca = PCA(n_components=param, random_state=self.random_state)
                                    X_final = pca.fit_transform(X_scaled)
                                    final_feature_names = [f'PC{i+1}' for i in range(X_final.shape[1])]
                                except:
                                    continue
                            
                            elif method == 'kernel_pca' and X_final.shape[1] > 2:
                                try:
                                    n_components = min(param, X_scaled.shape[1], X_scaled.shape[0]-1)
                                    kpca = KernelPCA(n_components=n_components, kernel='rbf', 
                                                    random_state=self.random_state)
                                    X_final = kpca.fit_transform(X_scaled)
                                    final_feature_names = [f'KPCA{i+1}' for i in range(X_final.shape[1])]
                                except:
                                    continue
                        
                        # Store preprocessed dataset
                        key = f"{dataset_name}{suffix}_{scaler_name}_{dim_name}"
                        preprocessed_datasets[key] = {
                            'X': X_final,
                            'feature_names': final_feature_names,
                            'base_names': base_names,
                            'original_dataset': dataset_name,
                            'preprocessing': f"{scaler_name}_{dim_name}"
                        }
        
        print(f"Created {len(preprocessed_datasets)} preprocessed variants")
        return preprocessed_datasets
    
    def load_processed_data(self, data_dir: str = 'processed_data') -> Dict[str, Dict]:
        """
        Load all processed datasets.
        
        Args:
            data_dir: Directory containing processed data files
            
        Returns:
            Dictionary mapping dataset names to their data
        """
        print(f"\nLoading processed datasets from {data_dir}/...")
        
        datasets = {}
        
        # Get all CSV files in the directory
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('_processed.csv')]
        
        for file in csv_files:
            # Extract method name from filename
            method_name = file.replace('_processed.csv', '')
            
            # Load dataset
            df = pd.read_csv(os.path.join(data_dir, file))
            
            # Verify 'Name' column exists
            if 'Name' not in df.columns:
                print(f"  Warning: 'Name' column missing in {file}, skipping")
                continue
            
            # Extract feature columns and data
            feature_cols = [col for col in df.columns if col != 'Name']
            base_names = df['Name'].values
            X = df[feature_cols].values
            
            # Store dataset
            datasets[method_name] = {
                'X': X,
                'feature_names': feature_cols,
                'base_names': base_names,
                'df': df
            }
            
            print(f"  Loaded {method_name}: {X.shape[0]} samples, {X.shape[1]} features")
        
        return datasets
    
    def enhance_features(self, X: np.ndarray, feature_names: List[str], 
                        max_features: Optional[int] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Create more discriminative features from existing ones.
        
        Args:
            X: Feature matrix
            feature_names: Names of features
            max_features: Maximum number of output features (default: X.shape[1] + 50)
            
        Returns:
            Tuple containing:
                - Enhanced feature matrix
                - List of enhanced feature names
        """
        print("  Creating enhanced features...")
        
        # Set default max_features to be original features + 50 additional ones
        if max_features is None:
            max_features = X.shape[1] + 50
        
        # Ensure max_features is at least as large as the input feature count
        max_features = max(max_features, X.shape[1])
        
        # Calculate enhanced features to add (not including original features)
        max_additional_features = max_features - X.shape[1]
        
        # Select features with highest variance
        variances = np.nanvar(X, axis=0)
        top_variance_indices = np.argsort(variances)[::-1][:min(5, X.shape[1])]
        
        # Pre-allocate the final array
        enhanced_X = np.zeros((X.shape[0], X.shape[1] + max_additional_features))
        enhanced_X[:, :X.shape[1]] = X  # Copy original features
        enhanced_features = feature_names.copy()
        
        # Track current column position (start after original features)
        current_col = X.shape[1]
        
        # 1. Add non-linear transformations for top variance features
        nonlinear_count = 0
        max_nonlinear = min(10, X.shape[1])
        
        for i, feature_idx in enumerate(top_variance_indices):
            if nonlinear_count >= max_nonlinear:
                break
                
            # Log transformation (with sign preservation)
            feature_data = X[:, feature_idx]
            if np.sum(~np.isnan(feature_data)) >= 5:  # Require at least 5 non-nan values
                log_transformed = np.log1p(np.abs(feature_data)) * np.sign(feature_data)
                enhanced_X[:, current_col] = log_transformed
                enhanced_features.append(f"log_{feature_names[feature_idx]}")
                current_col += 1
                nonlinear_count += 1
                
                # Square transformation
                squared = feature_data ** 2 * np.sign(feature_data)
                enhanced_X[:, current_col] = squared
                enhanced_features.append(f"squared_{feature_names[feature_idx]}")
                current_col += 1
                nonlinear_count += 1
        
        # 2. Add pairwise feature interactions
        interaction_count = 0
        max_interactions = 10
        
        from itertools import combinations
        for i, j in combinations(top_variance_indices, 2):
            if interaction_count >= max_interactions:
                break
                
            # Skip if too many NaN values
            if np.sum(~np.isnan(X[:, i]) & ~np.isnan(X[:, j])) < 5:
                continue
                
            # Create interaction feature
            interaction = X[:, i] * X[:, j]
            enhanced_X[:, current_col] = interaction
            enhanced_features.append(f"interaction_{feature_names[i]}_{feature_names[j]}")
            current_col += 1
            interaction_count += 1
        
        # 3. Add statistical moments if enough features
        if X.shape[1] >= 5:
            # Row means (across features)
            row_means = np.nanmean(X, axis=1)
            enhanced_X[:, current_col] = row_means
            enhanced_features.append("global_mean")
            current_col += 1
            
            # Row medians (more robust)
            row_medians = np.nanmedian(X, axis=1)
            enhanced_X[:, current_col] = row_medians
            enhanced_features.append("global_median")
            current_col += 1
        
        # Truncate if we didn't use all allocated space
        if current_col < enhanced_X.shape[1]:
            enhanced_X = enhanced_X[:, :current_col]
            
        # Handle NaN values
        if np.isnan(enhanced_X).any():
            enhanced_X = np.nan_to_num(enhanced_X, nan=0)
        
        print(f"  Created {current_col - X.shape[1]} enhanced features")
        
        return enhanced_X, enhanced_features
    
    def prepare_optimization_data(self, preprocessed_datasets):
        """
        Prepare data for optimization by combining all viable datasets.
        
        Args:
            preprocessed_datasets: Output from preprocessing
        
        Returns:
            Dictionary: Optimization-ready datasets
        """
        print("Preparing data for optimization...")
        
        # Filter datasets with sufficient samples and features
        viable_datasets = {}
        for name, dataset_info in preprocessed_datasets.items():
            X = dataset_info['X']
            if X.shape[0] >= self.min_subjects and X.shape[1] > 0:
                viable_datasets[name] = dataset_info
        
        print(f"Found {len(viable_datasets)} viable datasets for optimization")
        return viable_datasets
        
    def genetic_algorithm_optimizer(self, dataset_info, n_generations=30, population_size=50):
        """
        Use genetic algorithm to optimize subject/feature selection.
        
        Args:
            dataset_info: Dataset information
            n_generations: Number of generations
            population_size: Population size
        
        Returns:
            Dictionary: Best configuration found
        """
        print(f"Running genetic algorithm optimization ({n_generations} generations, {population_size} population)...")
        
        import random
        from deap import base, creator, tools, algorithms
        
        X = dataset_info['X']
        n_subjects, n_features = X.shape
        
        # Define fitness function
        def fitness_function(individual):
            # Decode individual
            subject_mask = np.array(individual[:n_subjects], dtype=bool)
            feature_mask = np.array(individual[n_subjects:n_subjects+n_features], dtype=bool)
            
            # Clustering parameters (encoded in remaining genes)
            clustering_idx = individual[n_subjects+n_features] if len(individual) > n_subjects+n_features else 0
            preprocessing_idx = individual[n_subjects+n_features+1] if len(individual) > n_subjects+n_features+1 else 0
            
            clustering_options = [
                {'algorithm': 'kmeans'},
                {'algorithm': 'hierarchical', 'linkage': 'ward'},
                {'algorithm': 'hierarchical', 'linkage': 'complete'},
                {'algorithm': 'gaussian_mixture', 'covariance_type': 'full'},
                {'algorithm': 'spectral', 'affinity': 'rbf'}
            ]
            
            preprocessing_options = [
                {'scaler': None},
                {'scaler': 'standard'},
                {'scaler': 'robust'},
                {'scaler': 'standard', 'pca_variance': 0.95},
                {'scaler': 'robust', 'pca_variance': 0.90}
            ]
            
            clustering_params = clustering_options[clustering_idx % len(clustering_options)]
            preprocessing_params = preprocessing_options[preprocessing_idx % len(preprocessing_options)]
            
            # Ensure minimum subjects and features
            if np.sum(subject_mask) < self.min_subjects or np.sum(feature_mask) == 0:
                return (-1,)
            
            # Evaluate configuration
            eval_result = self.evaluate_configuration(
                X, subject_mask, feature_mask, clustering_params, preprocessing_params
            )
            
            silhouette, balance_ratio, n_used, metrics = eval_result
            
            return (silhouette,)
        
        # Set up DEAP
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # Gene definition: subject mask + feature mask + algorithm choices
        gene_length = n_subjects + n_features + 2
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, 
                        toolbox.attr_bool, gene_length)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Genetic operators
        toolbox.register("evaluate", fitness_function)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Initialize population
        population = toolbox.population(n=population_size)
        
        # Evolution parameters
        NGEN = n_generations
        CXPB = 0.7  # Crossover probability
        MUTPB = 0.2  # Mutation probability
        
        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Evolution loop
        from tqdm import tqdm
        for generation in tqdm(range(NGEN), desc="GA generations"):
            # Select and clone the next generation individuals
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))
            
            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate individuals with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population
            population[:] = offspring
            
            # Track best fitness
            fits = [ind.fitness.values[0] for ind in population]
            best_fitness = max(fits)
            self.optimization_history.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'method': 'genetic_algorithm'
            })
        
        # Get best individual
        best_individual = tools.selBest(population, 1)[0]
        
        # Decode best configuration
        subject_mask = np.array(best_individual[:n_subjects], dtype=bool)
        feature_mask = np.array(best_individual[n_subjects:n_subjects+n_features], dtype=bool)
        
        clustering_idx = best_individual[n_subjects+n_features] if len(best_individual) > n_subjects+n_features else 0
        preprocessing_idx = best_individual[n_subjects+n_features+1] if len(best_individual) > n_subjects+n_features+1 else 0
        
        clustering_options = [
            {'algorithm': 'kmeans'},
            {'algorithm': 'hierarchical', 'linkage': 'ward'},
            {'algorithm': 'hierarchical', 'linkage': 'complete'},
            {'algorithm': 'gaussian_mixture', 'covariance_type': 'full'},
            {'algorithm': 'spectral', 'affinity': 'rbf'}
        ]
        
        preprocessing_options = [
            {'scaler': None},
            {'scaler': 'standard'},
            {'scaler': 'robust'},
            {'scaler': 'standard', 'pca_variance': 0.95},
            {'scaler': 'robust', 'pca_variance': 0.90}
        ]
        
        best_clustering = clustering_options[clustering_idx % len(clustering_options)]
        best_preprocessing = preprocessing_options[preprocessing_idx % len(preprocessing_options)]
        
        # Final evaluation
        final_silhouette, final_balance, final_n_subjects, _ = self.evaluate_configuration(
            X, subject_mask, feature_mask, best_clustering, best_preprocessing
        )
        
        return {
            'method': 'genetic_algorithm',
            'silhouette_score': final_silhouette,
            'balance_ratio': final_balance,
            'n_subjects_used': final_n_subjects,
            'n_features_used': np.sum(feature_mask),
            'subject_mask': subject_mask,
            'feature_mask': feature_mask,
            'clustering_params': best_clustering,
            'preprocessing_params': best_preprocessing,
            'dataset_info': dataset_info
        }
    
    def bayesian_optimization(self, dataset_info, n_trials=100):
        """
        Use Optuna for Bayesian optimization.
        
        Args:
            dataset_info: Dataset information
            n_trials: Number of optimization trials
        
        Returns:
            Dictionary: Best configuration found
        """
        print(f"Running Bayesian optimization ({n_trials} trials)...")
        
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            print("Optuna not installed. Please install with 'pip install optuna'")
            return None
        
        X = dataset_info['X']
        n_subjects, n_features = X.shape
        
        def objective(trial):
            # Sample subject inclusion probability
            subject_inclusion_prob = trial.suggest_float('subject_prob', 0.5, 1.0)
            subject_mask = np.random.choice([True, False], size=n_subjects, 
                                          p=[subject_inclusion_prob, 1-subject_inclusion_prob])
            
            # Ensure minimum subjects
            if np.sum(subject_mask) < self.min_subjects:
                # Randomly select minimum subjects
                false_indices = np.where(~subject_mask)[0]
                needed = self.min_subjects - np.sum(subject_mask)
                if needed <= len(false_indices):
                    selected_indices = np.random.choice(false_indices, needed, replace=False)
                    subject_mask[selected_indices] = True
                else:
                    subject_mask = np.random.choice([True, False], size=n_subjects,
                                                   p=[max(0.5, self.min_subjects/n_subjects), 1-max(0.5, self.min_subjects/n_subjects)])
            
            # Sample feature inclusion probability
            feature_inclusion_prob = trial.suggest_float('feature_prob', 0.1, 1.0)
            feature_mask = np.random.choice([True, False], size=n_features,
                                          p=[feature_inclusion_prob, 1-feature_inclusion_prob])
            
            # Ensure at least some features
            if np.sum(feature_mask) == 0:
                feature_mask[np.random.choice(n_features, 1)] = True
            
            # Sample algorithm and parameters
            algorithm = trial.suggest_categorical('algorithm', 
                ['kmeans', 'hierarchical', 'gaussian_mixture', 'spectral'])
            
            clustering_params = {'algorithm': algorithm}
            
            if algorithm == 'hierarchical':
                clustering_params['linkage'] = trial.suggest_categorical('linkage', 
                    ['ward', 'complete', 'average'])
            elif algorithm == 'gaussian_mixture':
                clustering_params['covariance_type'] = trial.suggest_categorical('covariance_type',
                    ['full', 'tied', 'diag', 'spherical'])
            elif algorithm == 'spectral':
                clustering_params['affinity'] = trial.suggest_categorical('affinity',
                    ['rbf', 'nearest_neighbors', 'polynomial'])
            
            # Sample preprocessing parameters
            preprocessing_params = {}
            preprocessing_params['scaler'] = trial.suggest_categorical('scaler', 
                [None, 'standard', 'robust'])
            
            use_pca = trial.suggest_categorical('use_pca', [True, False])
            if use_pca:
                preprocessing_params['pca_variance'] = trial.suggest_float('pca_variance', 0.8, 0.99)
            
            # Evaluate configuration
            eval_result = self.evaluate_configuration(
                X, subject_mask, feature_mask, clustering_params, preprocessing_params
            )
            
            silhouette, balance_ratio, n_used, metrics = eval_result
            
            # Store trial information
            trial.set_user_attr('balance_ratio', balance_ratio)
            trial.set_user_attr('n_subjects_used', n_used)
            trial.set_user_attr('n_features_used', np.sum(feature_mask))
            
            return silhouette
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get best configuration
        best_trial = study.best_trial
        
        # Reconstruct best configuration
        subject_inclusion_prob = best_trial.params['subject_prob']
        feature_inclusion_prob = best_trial.params['feature_prob']
        
        # Note: We need to reconstruct the exact masks, which is challenging with probabilistic sampling
        # For now, we'll use the best trial's stored attributes
        
        return {
            'method': 'bayesian_optimization',
            'silhouette_score': best_trial.value,
            'balance_ratio': best_trial.user_attrs['balance_ratio'],
            'n_subjects_used': best_trial.user_attrs['n_subjects_used'],
            'n_features_used': best_trial.user_attrs['n_features_used'],
            'best_params': best_trial.params,
            'dataset_info': dataset_info,
            'study': study
        }
    
    def exhaustive_subset_search(self, dataset_info, max_subjects_to_exclude=5, max_features_to_exclude=5):
        """
        Exhaustive search over small subset exclusions.
        
        Args:
            dataset_info: Dataset information
            max_subjects_to_exclude: Maximum number of subjects to exclude
            max_features_to_exclude: Maximum number of features to exclude
        
        Returns:
            Dictionary: Best configuration found
        """
        print(f"Running exhaustive subset search (max exclude: {max_subjects_to_exclude} subjects, {max_features_to_exclude} features)...")
        
        from itertools import combinations
        
        X = dataset_info['X']
        n_subjects, n_features = X.shape
        
        # Limit search space to manageable size
        max_subjects_to_exclude = min(max_subjects_to_exclude, n_subjects - self.min_subjects)
        max_features_to_exclude = min(max_features_to_exclude, n_features - 1)
        
        best_config = None
        best_score = -1
        
        # Algorithm and preprocessing combinations
        algorithm_options = [
            {'algorithm': 'kmeans'},
            {'algorithm': 'hierarchical', 'linkage': 'ward'},
            {'algorithm': 'gaussian_mixture', 'covariance_type': 'full'},
        ]
        
        preprocessing_options = [
            {'scaler': 'standard'},
            {'scaler': 'robust'},
            {'scaler': 'standard', 'pca_variance': 0.95},
        ]
        
        total_combinations = 0
        for n_subjects_exclude in range(max_subjects_to_exclude + 1):
            for n_features_exclude in range(max_features_to_exclude + 1):
                subject_combinations = list(combinations(range(n_subjects), n_subjects_exclude))
                feature_combinations = list(combinations(range(n_features), n_features_exclude))
                total_combinations += len(subject_combinations) * len(feature_combinations) * len(algorithm_options) * len(preprocessing_options)
        
        print(f"Total combinations to evaluate: {total_combinations}")
        
        from tqdm import tqdm
        with tqdm(total=total_combinations, desc="Exhaustive search") as pbar:
            for n_subjects_exclude in range(max_subjects_to_exclude + 1):
                for n_features_exclude in range(max_features_to_exclude + 1):
                    # Generate all combinations of subjects/features to exclude
                    subject_combinations = list(combinations(range(n_subjects), n_subjects_exclude))
                    feature_combinations = list(combinations(range(n_features), n_features_exclude))
                    
                    for subjects_to_exclude in subject_combinations:
                        for features_to_exclude in feature_combinations:
                            # Create masks
                            subject_mask = np.ones(n_subjects, dtype=bool)
                            feature_mask = np.ones(n_features, dtype=bool)
                            
                            subject_mask[list(subjects_to_exclude)] = False
                            feature_mask[list(features_to_exclude)] = False
                            
                            # Try different algorithms and preprocessing
                            for clustering_params in algorithm_options:
                                for preprocessing_params in preprocessing_options:
                                    eval_result = self.evaluate_configuration(
                                        X, subject_mask, feature_mask, clustering_params, preprocessing_params
                                    )
                                    
                                    silhouette, balance_ratio, n_used, metrics = eval_result
                                    
                                    if silhouette > best_score:
                                        best_score = silhouette
                                        best_config = {
                                            'method': 'exhaustive_search',
                                            'silhouette_score': silhouette,
                                            'balance_ratio': balance_ratio,
                                            'n_subjects_used': n_used,
                                            'n_features_used': np.sum(feature_mask),
                                            'subject_mask': subject_mask.copy(),
                                            'feature_mask': feature_mask.copy(),
                                            'clustering_params': clustering_params.copy(),
                                            'preprocessing_params': preprocessing_params.copy(),
                                            'subjects_excluded': list(subjects_to_exclude),
                                            'features_excluded': list(features_to_exclude),
                                            'dataset_info': dataset_info
                                        }
                                    
                                    pbar.update(1)
        
        return best_config
    
    def random_search_optimizer(self, dataset_info, n_iterations=1000):
        """
        Random search over configurations.
        
        Args:
            dataset_info: Dataset information
            n_iterations: Number of random iterations
        
        Returns:
            Dictionary: Best configuration found
        """
        print(f"Running random search ({n_iterations} iterations)...")
        
        X = dataset_info['X']
        n_subjects, n_features = X.shape
        
        best_config = None
        best_score = -1
        
        algorithm_options = [
            {'algorithm': 'kmeans'},
            {'algorithm': 'hierarchical', 'linkage': 'ward'},
            {'algorithm': 'hierarchical', 'linkage': 'complete'},
            {'algorithm': 'hierarchical', 'linkage': 'average'},
            {'algorithm': 'gaussian_mixture', 'covariance_type': 'full'},
            {'algorithm': 'gaussian_mixture', 'covariance_type': 'tied'},
            {'algorithm': 'spectral', 'affinity': 'rbf'},
            {'algorithm': 'spectral', 'affinity': 'nearest_neighbors'},
        ]
        
        preprocessing_options = [
            {'scaler': None},
            {'scaler': 'standard'},
            {'scaler': 'robust'},
            {'scaler': 'standard', 'pca_variance': 0.95},
            {'scaler': 'standard', 'pca_variance': 0.90},
            {'scaler': 'robust', 'pca_variance': 0.95},
            {'scaler': 'robust', 'pca_variance': 0.90},
        ]
        
        from tqdm import tqdm
        for iteration in tqdm(range(n_iterations), desc="Random search"):
            # Random subject inclusion (ensure minimum)
            n_subjects_to_include = np.random.randint(self.min_subjects, n_subjects + 1)
            subjects_to_include = np.random.choice(n_subjects, n_subjects_to_include, replace=False)
            subject_mask = np.zeros(n_subjects, dtype=bool)
            subject_mask[subjects_to_include] = True
            
            # Random feature inclusion (ensure at least 1)
            n_features_to_include = np.random.randint(1, n_features + 1)
            features_to_include = np.random.choice(n_features, n_features_to_include, replace=False)
            feature_mask = np.zeros(n_features, dtype=bool)
            feature_mask[features_to_include] = True
            
            # Random algorithm and preprocessing
            clustering_params = algorithm_options[np.random.choice(len(algorithm_options))]
            preprocessing_params = preprocessing_options[np.random.choice(len(preprocessing_options))]
            
            # Evaluate
            eval_result = self.evaluate_configuration(
                X, subject_mask, feature_mask, clustering_params, preprocessing_params
            )
            
            silhouette, balance_ratio, n_used, metrics = eval_result
            
            if silhouette > best_score:
                best_score = silhouette
                best_config = {
                    'method': 'random_search',
                    'silhouette_score': silhouette,
                    'balance_ratio': balance_ratio,
                    'n_subjects_used': n_used,
                    'n_features_used': np.sum(feature_mask),
                    'subject_mask': subject_mask.copy(),
                    'feature_mask': feature_mask.copy(),
                    'clustering_params': clustering_params.copy(),
                    'preprocessing_params': preprocessing_params.copy(),
                    'dataset_info': dataset_info
                }
            
            # Track progress
            if iteration % 100 == 0:
                self.optimization_history.append({
                    'iteration': iteration,
                    'best_score': best_score,
                    'method': 'random_search'
                })
        
        return best_config
    
    def _calculate_silhouette_score(self, X, labels):
        """Calculate silhouette score."""
        if len(np.unique(labels)) < 2:
            return -1
        return silhouette_score(X, labels)
    
    def _calculate_calinski_harabasz_score(self, X, labels):
        """Calculate Calinski-Harabasz score (higher is better)."""
        if len(np.unique(labels)) < 2:
            return 0
        return calinski_harabasz_score(X, labels)
    
    def _calculate_davies_bouldin_score(self, X, labels):
        """Calculate Davies-Bouldin score (lower is better, so we return negative)."""
        if len(np.unique(labels)) < 2:
            return -np.inf
        return -davies_bouldin_score(X, labels)  # Negative because lower is better
    
    def _calculate_dunn_index(self, X, labels):
        """Calculate Dunn index (higher is better)."""
        if len(np.unique(labels)) < 2:
            return 0
        
        # Calculate pairwise distances
        distances = pdist(X)
        distance_matrix = squareform(distances)
        
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters < 2:
            return 0
        
        # Calculate minimum inter-cluster distance
        min_inter_cluster = np.inf
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                cluster_i_indices = np.where(labels == unique_labels[i])[0]
                cluster_j_indices = np.where(labels == unique_labels[j])[0]
                
                inter_distances = distance_matrix[np.ix_(cluster_i_indices, cluster_j_indices)]
                min_inter_cluster = min(min_inter_cluster, np.min(inter_distances))
        
        # Calculate maximum intra-cluster distance
        max_intra_cluster = 0
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) > 1:
                intra_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
                max_intra_cluster = max(max_intra_cluster, np.max(intra_distances))
        
        if max_intra_cluster == 0:
            return min_inter_cluster
        
        return min_inter_cluster / max_intra_cluster
    
    def _calculate_nad_specific_metric(self, X, labels):
        """
        Calculate NAD+ supplement study specific metric combining:
        - Silhouette score (cluster separation)
        - Effect size (biological significance)
        - Balance (study design consideration)
        """
        # Base silhouette score
        silhouette = self._calculate_silhouette_score(X, labels)
        
        # Extract the two clusters
        cluster0 = X[labels == 0]
        cluster1 = X[labels == 1]
        
        # Calculate effect sizes between groups for each feature
        effect_sizes = []
        for feature_idx in range(X.shape[1]):
            group0_vals = cluster0[:, feature_idx]
            group1_vals = cluster1[:, feature_idx]
            
            if len(group0_vals) > 0 and len(group1_vals) > 0:
                # Calculate Cohen's d effect size
                pooled_std = np.sqrt((np.var(group0_vals) * (len(group0_vals) - 1) + 
                                    np.var(group1_vals) * (len(group1_vals) - 1)) / 
                                    (len(group0_vals) + len(group1_vals) - 2))
                
                # Avoid division by zero
                if pooled_std > 0:
                    effect = np.abs(np.mean(group1_vals) - np.mean(group0_vals)) / pooled_std
                    effect_sizes.append(effect)
        
        # Calculate mean and max effect sizes
        mean_effect = np.mean(effect_sizes) if effect_sizes else 0
        max_effect = np.max(effect_sizes) if effect_sizes else 0
        
        # Calculate cluster sizes ratio (balance measure)
        unique, counts = np.unique(labels, return_counts=True)
        if len(counts) == 2:
            balance_ratio = min(counts) / max(counts)
        else:
            balance_ratio = 0
        
        # Combine into NAD+ specific metric
        # Weight silhouette more heavily, but also consider effect size and balance
        nad_metric = (0.6 * silhouette + 
                     0.2 * min(1.0, mean_effect / 0.8) +  # Cap effect size contribution
                     0.1 * min(1.0, max_effect / 1.2) +   # Consider maximum effect size too
                     0.1 * balance_ratio)                 # Consider balance, but not as crucial
        
        return nad_metric
        
    def evaluate_configuration(self, dataset_X, subject_mask, feature_mask, 
                              clustering_params, preprocessing_params, min_features = 5):
        """
        Evaluate a specific configuration and return silhouette score and balance.
        
        Args:
            dataset_X: Full dataset
            subject_mask: Boolean mask for subject inclusion
            feature_mask: Boolean mask for feature inclusion
            clustering_params: Clustering algorithm parameters
            preprocessing_params: Preprocessing parameters
            
        Returns:
            tuple: (metric_score, balance_ratio, n_subjects_used, metrics_dict)
        """
        try:
            # Apply subject and feature masks
            X_subset = dataset_X[subject_mask][:, feature_mask]
            
            # Check minimum requirements
            if (X_subset.shape[0] < self.min_subjects or 
                X_subset.shape[1] == 0 or 
                X_subset.shape[1] < min_features):  # NEW CONSTRAINT
                return -1, 0, X_subset.shape[0], {}
            
            # Apply preprocessing
            if preprocessing_params.get('scaler') == 'standard':
                scaler = StandardScaler()
                X_processed = scaler.fit_transform(X_subset)
            elif preprocessing_params.get('scaler') == 'robust':
                scaler = RobustScaler()
                X_processed = scaler.fit_transform(X_subset)
            elif preprocessing_params.get('scaler') == 'minmax':
                scaler = MinMaxScaler()
                X_processed = scaler.fit_transform(X_subset)
            elif preprocessing_params.get('scaler') == 'quantile':
                scaler = QuantileTransformer(output_distribution='normal', 
                                           n_quantiles=min(1000, X_subset.shape[0]))
                X_processed = scaler.fit_transform(X_subset)
            else:
                X_processed = X_subset.copy()
            
            # Apply dimensionality reduction if specified
            if preprocessing_params.get('pca_variance'):
                pca = PCA(n_components=preprocessing_params['pca_variance'], 
                         random_state=self.random_state)
                X_processed = pca.fit_transform(X_processed)
                
                # Check if we have enough components
                if X_processed.shape[1] < 2:
                    return -1, 0, X_subset.shape[0], {}
            
            # Apply kernel PCA if specified
            elif preprocessing_params.get('kernel_pca'):
                kpca = KernelPCA(n_components=preprocessing_params['kernel_pca'],
                                kernel='rbf', random_state=self.random_state)
                X_processed = kpca.fit_transform(X_processed)
                
                # Check if we have enough components
                if X_processed.shape[1] < 2:
                    return -1, 0, X_subset.shape[0], {}
            
            # Apply clustering
            algorithm = clustering_params['algorithm']
            
            if algorithm == 'kmeans':
                clusterer = KMeans(
                    n_clusters=2,
                    random_state=self.random_state,
                    n_init=10,
                    max_iter=300
                )
            elif algorithm == 'hierarchical':
                clusterer = AgglomerativeClustering(
                    n_clusters=2,
                    linkage=clustering_params.get('linkage', 'ward')
                )
            elif algorithm == 'gaussian_mixture':
                clusterer = GaussianMixture(
                    n_components=2,
                    covariance_type=clustering_params.get('covariance_type', 'full'),
                    random_state=self.random_state,
                    n_init=5
                )
            elif algorithm == 'spectral':
                clusterer = SpectralClustering(
                    n_clusters=2,
                    affinity=clustering_params.get('affinity', 'rbf'),
                    random_state=self.random_state,
                    n_init=5
                )
            elif algorithm == 'ensemble':
                return self._run_ensemble_clustering(X_processed, preprocessing_params)
            else:
                return -1, 0, X_subset.shape[0], {}
            
            # Fit and predict
            labels = clusterer.fit_predict(X_processed)
            
            # Check if we got 2 clusters
            if len(np.unique(labels)) != 2:
                return -1, 0, X_subset.shape[0], {}
            
            # Calculate balance
            unique, counts = np.unique(labels, return_counts=True)
            balance_ratio = min(counts) / max(counts)
            
            # Check balance constraint
            if balance_ratio < (1 - self.balance_tolerance):
                return -1, balance_ratio, X_subset.shape[0], {}
            
            # Calculate all available metrics
            metrics = {}
            for metric_name, metric_func in self.available_metrics.items():
                metrics[metric_name] = metric_func(X_processed, labels)
            
            # Return primary metric and all metrics
            primary_score = metrics[self.primary_metric]
            
            return primary_score, balance_ratio, X_subset.shape[0], metrics
            
        except Exception as e:
            return -1, 0, 0, {}
            
    def _run_ensemble_clustering(self, X, preprocessing_params):
        """Run ensemble clustering instead of a single algorithm."""
        if X.shape[0] < 5:
            return -1, 0, X.shape[0], {}
            
        try:
            # Create base clusterers
            base_clusterers = [
                KMeans(n_clusters=2, random_state=self.random_state),
                KMeans(n_clusters=2, random_state=self.random_state+1),
                AgglomerativeClustering(n_clusters=2, linkage='ward'),
                AgglomerativeClustering(n_clusters=2, linkage='complete'),
                GaussianMixture(n_components=2, random_state=self.random_state)
            ]
            
            # Generate base clusterings
            base_labels = np.zeros((len(base_clusterers), X.shape[0]))
            base_scores = np.zeros(len(base_clusterers))
            
            for i, clusterer in enumerate(base_clusterers):
                labels = clusterer.fit_predict(X)
                base_labels[i] = labels
                
                if len(np.unique(labels)) > 1:
                    base_scores[i] = max(0, silhouette_score(X, labels))
                else:
                    base_scores[i] = 0
            
            # Normalize weights if any are positive
            if np.sum(base_scores) > 0:
                weights = base_scores / np.sum(base_scores)
            else:
                weights = np.ones(len(base_clusterers)) / len(base_clusterers)
            
            # Create co-association matrix
            co_assoc = np.zeros((X.shape[0], X.shape[0]))
            
            for i in range(len(base_clusterers)):
                # Skip if only one cluster
                if len(np.unique(base_labels[i])) < 2:
                    continue
                    
                # Create binary co-association matrix for this clustering
                labels = base_labels[i]
                for j in range(X.shape[0]):
                    for k in range(j+1, X.shape[0]):
                        if labels[j] == labels[k]:
                            co_assoc[j, k] += weights[i]
                            co_assoc[k, j] += weights[i]
            
            # Convert to distance matrix
            dist_matrix = 1 - co_assoc
            
            # Final clustering using the consensus matrix
            consensus = AgglomerativeClustering(
                n_clusters=2,
                affinity='precomputed',
                linkage='average'
            )
            
            labels = consensus.fit_predict(dist_matrix)
            
            # Evaluate result
            if len(np.unique(labels)) != 2:
                return -1, 0, X.shape[0], {}
                
            # Calculate balance
            unique, counts = np.unique(labels, return_counts=True)
            balance_ratio = min(counts) / max(counts)
            
            # Check balance constraint
            if balance_ratio < (1 - self.balance_tolerance):
                return -1, balance_ratio, X.shape[0], {}
            
            # Calculate all available metrics
            metrics = {}
            for metric_name, metric_func in self.available_metrics.items():
                metrics[metric_name] = metric_func(X, labels)
            
            # Return primary metric and all metrics
            primary_score = metrics[self.primary_metric]
            
            return primary_score, balance_ratio, X.shape[0], metrics
            
        except Exception as e:
            return -1, 0, 0, {}
    
    def optimize_all_datasets(self, viable_datasets, methods=None, save_results=True):
        """
        Run optimization on all viable datasets using multiple methods.
        
        Args:
            viable_datasets: Viable datasets from prepare_optimization_data
            methods: List of optimization methods to use
            save_results: Whether to save intermediate results
            
        Returns:
            Dictionary: Optimization results for all datasets and methods
        """
        if methods is None:
            methods = ['random_search', 'genetic_algorithm', 'exhaustive_search', 'bayesian_optimization']
        
        print(f"\nRunning comprehensive optimization on {len(viable_datasets)} datasets")
        print(f"Methods: {methods}")
        print("=" * 60)
        
        all_results = {}
        
        for dataset_name, dataset_info in viable_datasets.items():
            print(f"\nOptimizing dataset: {dataset_name}")
            print(f"Shape: {dataset_info['X'].shape}")
            
            dataset_results = {}
            
            # Random search (fast baseline)
            if 'random_search' in methods:
                try:
                    result = self.random_search_optimizer(dataset_info, n_iterations=1000)
                    dataset_results['random_search'] = result
                    print(f"  Random search: {result['silhouette_score']:.4f}")
                except Exception as e:
                    print(f"  Random search failed: {e}")
            
            # Genetic algorithm (good for complex spaces)
            if 'genetic_algorithm' in methods:
                try:
                    result = self.genetic_algorithm_optimizer(dataset_info, n_generations=25, population_size=40)
                    dataset_results['genetic_algorithm'] = result
                    print(f"  Genetic algorithm: {result['silhouette_score']:.4f}")
                except Exception as e:
                    print(f"  Genetic algorithm failed: {e}")
            
            # Exhaustive search (for small datasets)
            if 'exhaustive_search' in methods and dataset_info['X'].shape[0] <= 50:
                try:
                    result = self.exhaustive_subset_search(dataset_info, max_subjects_to_exclude=3, max_features_to_exclude=3)
                    dataset_results['exhaustive_search'] = result
                    print(f"  Exhaustive search: {result['silhouette_score']:.4f}")
                except Exception as e:
                    print(f"  Exhaustive search failed: {e}")
            
            # Bayesian optimization (efficient for continuous spaces)
            if 'bayesian_optimization' in methods:
                try:
                    result = self.bayesian_optimization(dataset_info, n_trials=100)
                    dataset_results['bayesian_optimization'] = result
                    print(f"  Bayesian optimization: {result['silhouette_score']:.4f}")
                except Exception as e:
                    print(f"  Bayesian optimization failed: {e}")
            
            all_results[dataset_name] = dataset_results
            
            # Save intermediate results
            if save_results:
                import pickle
                os.makedirs('optimization_results', exist_ok=True)
                with open(f'optimization_results/{dataset_name}_results.pkl', 'wb') as f:
                    pickle.dump(dataset_results, f)
        
        # Find overall best result
        best_overall = None
        best_score = -1
        
        for dataset_name, dataset_results in all_results.items():
            for method_name, result in dataset_results.items():
                # Get the silhouette score for comparison
                if result is None:
                    continue
                current_score = result.get('silhouette_score', -1)
                if current_score > best_score:
                    best_score = current_score
                    best_overall = result.copy()
                    best_overall['dataset_name'] = dataset_name
                    best_overall['method_name'] = method_name
        
        self.best_results = all_results
        self.best_overall = best_overall
        
        return all_results
    
    def analyze_optimal_solution(self, best_result):
        """
        Analyze the optimal solution in detail.
        
        Args:
            best_result: Best optimization result
            
        Returns:
            Dictionary: Detailed analysis
        """
        print("\nAnalyzing optimal solution...")
        
        X = best_result['dataset_info']['X']
        if 'subject_mask' in best_result:
            subject_mask = best_result['subject_mask']
        elif 'best_params' in best_result:
            # For Bayesian optimization, reconstruct masks from parameters
            # This is approximate and would need to be refined based on your exact implementation
            n_subjects = X.shape[0]
            subject_inclusion_prob = best_result['best_params'].get('subject_prob', 0.5)
            np.random.seed(self.random_state)  # Ensure reproducibility
            subject_mask = np.random.random(n_subjects) < subject_inclusion_prob
            # Ensure minimum subjects
            if np.sum(subject_mask) < self.min_subjects:
                indices = np.random.choice(np.where(~subject_mask)[0], 
                                         self.min_subjects - np.sum(subject_mask),
                                         replace=False)
                subject_mask[indices] = True
        else:
            # Fallback to using all subjects
            print("Warning: No subject mask found in optimization result. Using all subjects.")
            subject_mask = np.ones(X.shape[0], dtype=bool)
        
        # Similar approach for feature_mask
        if 'feature_mask' in best_result:
            feature_mask = best_result['feature_mask']
        elif 'best_params' in best_result:
            # Reconstruct feature mask
            n_features = X.shape[1]
            feature_inclusion_prob = best_result['best_params'].get('feature_prob', 0.5)
            np.random.seed(self.random_state + 1)  # Different seed from subject mask
            feature_mask = np.random.random(n_features) < feature_inclusion_prob
            # Ensure at least one feature
            if not np.any(feature_mask):
                feature_mask[np.random.choice(n_features)] = True
        else:
            # Fallback to using all features
            print("Warning: No feature mask found in optimization result. Using all features.")
            feature_mask = np.ones(X.shape[1], dtype=bool)
        
        # Add masks to result dictionary for future use
        if 'subject_mask' not in best_result:
            best_result['subject_mask'] = subject_mask
        if 'feature_mask' not in best_result:
            best_result['feature_mask'] = feature_mask
        
        # Get the optimal subset
        X_optimal = X[subject_mask][:, feature_mask]
        
        # Re-run clustering to get labels
        clustering_params = best_result.get('clustering_params', 
            {'algorithm': 'kmeans', 'linkage': 'ward'})

        preprocessing_params = best_result.get('preprocessing_params', 
            {'scaler': 'robust', 'pca_variance': None})
        #clustering_params = best_result['clustering_params']
        #preprocessing_params = best_result['preprocessing_params']
        
        # Apply preprocessing
        if preprocessing_params.get('scaler') == 'standard':
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X_optimal)
        elif preprocessing_params.get('scaler') == 'robust':
            scaler = RobustScaler()
            X_processed = scaler.fit_transform(X_optimal)
        else:
            X_processed = X_optimal.copy()
        
        # Apply PCA if specified
        if preprocessing_params.get('pca_variance'):
            pca = PCA(n_components=preprocessing_params['pca_variance'], random_state=self.random_state)
            X_processed = pca.fit_transform(X_processed)
        
        # Apply clustering
        algorithm = clustering_params.get('algorithm', 'kmeans')
        if algorithm == 'kmeans':
            clusterer = KMeans(n_clusters=2, random_state=self.random_state, n_init=10)
        elif algorithm == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=2, linkage=clustering_params.get('linkage', 'ward'))
        elif algorithm == 'gaussian_mixture':
            clusterer = GaussianMixture(n_components=2, covariance_type=clustering_params.get('covariance_type', 'full'), random_state=self.random_state)
        elif algorithm == 'spectral':
            clusterer = SpectralClustering(n_clusters=2, affinity=clustering_params.get('affinity', 'rbf'), random_state=self.random_state)
        
        labels = clusterer.fit_predict(X_processed)
        
        # Analysis
        analysis = {
            'silhouette_score': silhouette_score(X_processed, labels),
            'cluster_sizes': np.bincount(labels),
            'balance_ratio': min(np.bincount(labels)) / max(np.bincount(labels)),
            'n_subjects_total': X.shape[0],
            'n_subjects_used': np.sum(subject_mask),
            'n_features_total': X.shape[1],
            'n_features_used': np.sum(feature_mask),
            'subjects_excluded': np.where(~subject_mask)[0],
            'features_excluded': np.where(~feature_mask)[0],
            'clustering_algorithm': clustering_params,
            'preprocessing': preprocessing_params,
            'optimization_method': best_result['method'],
            'labels': labels
        }
        
        # Calculate feature importance in optimal solution
        feature_names = best_result['dataset_info']['feature_names']
        selected_features = [feature_names[i] for i in np.where(feature_mask)[0]]
        
        # Group differences for selected features
        group_diffs = []
        for i, feature_idx in enumerate(np.where(feature_mask)[0]):
            group0_vals = X_optimal[labels == 0, i]
            group1_vals = X_optimal[labels == 1, i]
            
            if len(group0_vals) > 0 and len(group1_vals) > 0:
                from scipy import stats
                t_stat, p_val = stats.ttest_ind(group0_vals, group1_vals)
                effect_size = np.abs(group1_vals.mean() - group0_vals.mean()) / np.sqrt((group0_vals.var() + group1_vals.var()) / 2)
                group_diffs.append({
                    'feature': feature_names[feature_idx],
                    'feature_index': feature_idx,
                    't_stat': np.abs(t_stat),
                    'p_value': p_val,
                    'effect_size': effect_size,
                    'group0_mean': group0_vals.mean(),
                    'group1_mean': group1_vals.mean()
                })
        
        analysis['feature_analysis'] = sorted(group_diffs, key=lambda x: x['effect_size'], reverse=True)
        
        # Add PCA component analysis
        if 'pca_variance' in best_result.get('preprocessing_params', {}):
            pca = PCA().fit(X_optimal)
            components = pca.components_
            feature_names = best_result['dataset_info']['feature_names']
            selected_features = [feature_names[i] for i in np.where(feature_mask)[0]]
            
            # Store component breakdown
            component_analysis = []
            for i, component in enumerate(components):
                top_indices = np.argsort(np.abs(component))[::-1][:10]
                top_features = [(selected_features[j], component[j]) for j in top_indices]
                component_analysis.append({
                    'component': i+1,
                    'explained_variance': pca.explained_variance_ratio_[i],
                    'top_features': top_features
                })
            
            analysis['pca_components'] = component_analysis
    
        return analysis

    
    def get_cluster_membership(self, best_result):
        """
        Generate lists of participant IDs grouped by their cluster assignments.
        
        Args:
            best_result: Best optimization result
            
        Returns:
            Dictionary: Mapping of cluster labels to lists of participant IDs
        """
        print("\nAnalyzing participant clusters...")
        
        # Get dataset information
        X = best_result['dataset_info']['X']
        subject_mask = best_result['subject_mask']
        feature_mask = best_result['feature_mask']
        
        # Get participant IDs (base_names)
        base_names = best_result['dataset_info'].get('base_names', None)
        # Fix: Properly check for numpy arrays and their length
        if base_names is None or (isinstance(base_names, (list, np.ndarray)) and len(base_names) != X.shape[0]):
            # Create generic IDs if none exist
            base_names = [f"Subject_{i}" for i in range(X.shape[0])]
        
        # Get subset of data and participants
        X_optimal = X[subject_mask][:, feature_mask]
        included_participants = np.array(base_names)[subject_mask]
        
        # Apply the same preprocessing as in analyze_optimal_solution
        clustering_params = best_result.get('clustering_params', 
            {'algorithm': 'kmeans', 'linkage': 'ward'})
        
        preprocessing_params = best_result.get('preprocessing_params', 
            {'scaler': 'robust', 'pca_variance': None})
        
        # Apply preprocessing
        if preprocessing_params.get('scaler') == 'standard':
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X_optimal)
        elif preprocessing_params.get('scaler') == 'robust':
            scaler = RobustScaler()
            X_processed = scaler.fit_transform(X_optimal)
        else:
            X_processed = X_optimal.copy()
        
        # Apply PCA if specified
        if preprocessing_params.get('pca_variance'):
            pca = PCA(n_components=preprocessing_params['pca_variance'], random_state=self.random_state)
            X_processed = pca.fit_transform(X_processed)
        
        # Apply clustering
        algorithm = clustering_params['algorithm']
        if algorithm == 'kmeans':
            clusterer = KMeans(n_clusters=2, random_state=self.random_state, n_init=10)
        elif algorithm == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=2, linkage=clustering_params.get('linkage', 'ward'))
        elif algorithm == 'gaussian_mixture':
            clusterer = GaussianMixture(n_components=2, covariance_type=clustering_params.get('covariance_type', 'full'), random_state=self.random_state)
        elif algorithm == 'spectral':
            clusterer = SpectralClustering(n_clusters=2, affinity=clustering_params.get('affinity', 'rbf'), random_state=self.random_state)
        
        # Get cluster labels
        labels = clusterer.fit_predict(X_processed)
        
        # Group participants by cluster
        cluster_groups = {}
        for label in np.unique(labels):
            # Convert numpy array elements to strings to avoid potential issues
            cluster_groups[f"Cluster_{label}"] = [str(p) for p in included_participants[labels == label].tolist()]
        
        return cluster_groups
        
    def generate_optimization_report(self, all_results, output_file='optimization_report.txt'):
        """
        Generate comprehensive optimization report.
        
        Args:
            all_results: All optimization results
            output_file: Output file path
        """
        with open(output_file, 'w') as f:
            f.write("THEORETICAL MAXIMUM CLUSTERING OPTIMIZATION REPORT\n")
            f.write("=" * 55 + "\n\n")
            
            # Executive summary
            if hasattr(self, 'best_overall') and self.best_overall:
                best = self.best_overall
                f.write("THEORETICAL MAXIMUM ACHIEVED\n")
                f.write("-" * 28 + "\n")
                f.write(f"🏆 Best Silhouette Score: {best['silhouette_score']:.4f}\n")
                f.write(f"🏆 Balance Ratio: {best['balance_ratio']:.4f}\n")
                f.write(f"🏆 Method: {best['method_name']}\n")
                f.write(f"🏆 Dataset: {best['dataset_name']}\n")
                f.write(f"🏆 Subjects Used: {best['n_subjects_used']}/{best['dataset_info']['X'].shape[0]} "
                       f"({best['n_subjects_used']/best['dataset_info']['X'].shape[0]*100:.1f}%)\n")
                f.write(f"🏆 Features Used: {best['n_features_used']}/{best['dataset_info']['X'].shape[1]} "
                       f"({best['n_features_used']/best['dataset_info']['X'].shape[1]*100:.1f}%)\n\n")
                
                # Analyze best result
                analysis = self.analyze_optimal_solution(best)
                f.write("OPTIMAL CONFIGURATION DETAILS\n")
                f.write("-" * 28 + "\n")
                f.write(f"Clustering Algorithm: {analysis['clustering_algorithm']}\n")
                f.write(f"Preprocessing: {analysis['preprocessing']}\n")
                f.write(f"Group 0 Size: {analysis['cluster_sizes'][0]}\n")
                f.write(f"Group 1 Size: {analysis['cluster_sizes'][1]}\n\n")
                
                # Feature analysis
                if 'feature_analysis' in analysis and analysis['feature_analysis']:
                    f.write("TOP 10 DISCRIMINATING FEATURES IN OPTIMAL SOLUTION\n")
                    f.write("-" * 45 + "\n")
                    for i, feature_info in enumerate(analysis['feature_analysis'][:10]):
                        f.write(f"{i+1}. {feature_info['feature']}\n")
                        f.write(f"   Effect Size: {feature_info['effect_size']:.4f}\n")
                        f.write(f"   P-value: {feature_info['p_value']:.6f}\n")
                        f.write(f"   Group 0 Mean: {feature_info['group0_mean']:.4f}\n")
                        f.write(f"   Group 1 Mean: {feature_info['group1_mean']:.4f}\n\n")
            
            # Method comparison
            f.write("OPTIMIZATION METHOD COMPARISON\n")
            f.write("-" * 30 + "\n")
            
            method_summary = {}
            for dataset_name, dataset_results in all_results.items():
                for method_name, result in dataset_results.items():
                    if method_name not in method_summary:
                        method_summary[method_name] = []
                    if result is not None: 
                        method_summary[method_name].append(result['silhouette_score'])
            
            for method_name, scores in method_summary.items():
                f.write(f"{method_name}:\n")
                f.write(f"  Best Score: {max(scores):.4f}\n")
                f.write(f"  Mean Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}\n")
                f.write(f"  Attempts: {len(scores)}\n\n")
            
            # Dataset-specific results
            f.write("RESULTS BY DATASET\n")
            f.write("-" * 18 + "\n")
            for dataset_name, dataset_results in all_results.items():
                f.write(f"\n{dataset_name}:\n")
                for method_name, result in dataset_results.items():
                    if result is not None:
                        f.write(f"  {method_name}: {result['silhouette_score']:.4f} "
                               f"(balance: {result['balance_ratio']:.3f}, "
                               f"subjects: {result['n_subjects_used']}, "
                               f"features: {result['n_features_used']})\n")
            
            if hasattr(self, 'best_overall') and self.best_overall:
                f.write("\nPARTICIPANT CLUSTER MEMBERSHIP\n")
                f.write("-" * 28 + "\n")
                
                # Get cluster memberships
                cluster_groups = self.get_cluster_membership(self.best_overall)
                
                # Write to report
                for cluster_name, participants in cluster_groups.items():
                    f.write(f"{cluster_name} ({len(participants)} participants):\n")
                    f.write(", ".join(participants) + "\n\n")

            f.write("\n" + "=" * 55 + "\n")
            f.write("Optimization completed! Theoretical maximum identified.\n")
        
        print(f"Optimization report saved to {output_file}")
        
    def create_optimization_visualizations(self, all_results, output_dir='optimization_results'):
        """
        Create comprehensive visualizations of optimization results.
        
        Args:
            all_results: All optimization results
            output_dir: Output directory
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nCreating optimization visualizations in {output_dir}/...")
        
        # 1. Performance comparison across methods and datasets
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Collect results for plotting
        results_data = []
        for dataset_name, dataset_results in all_results.items():
            for method_name, result in dataset_results.items():
                if result is not None:
                    results_data.append({
                        'dataset': dataset_name,
                        'method': method_name,
                        'silhouette': result['silhouette_score'],
                        'balance': result['balance_ratio'],
                        'n_subjects': result['n_subjects_used'],
                        'n_features': result['n_features_used']
                    })
        
        results_df = pd.DataFrame(results_data)
        
        # Silhouette scores by method
        method_performance = results_df.groupby('method')['silhouette'].agg(['mean', 'std', 'max']).round(4)
        axes[0, 0].bar(range(len(method_performance)), method_performance['mean'], 
                      yerr=method_performance['std'], capsize=5)
        axes[0, 0].set_xticks(range(len(method_performance)))
        axes[0, 0].set_xticklabels(method_performance.index, rotation=45)
        axes[0, 0].set_ylabel('Mean Silhouette Score')
        axes[0, 0].set_title('Performance by Optimization Method')
        
        # Balance ratios
        axes[0, 1].hist(results_df['balance'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=1-self.balance_tolerance, color='red', linestyle='--', 
                          label=f'Target threshold ({1-self.balance_tolerance:.2f})')
        axes[0, 1].set_xlabel('Balance Ratio')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Balance Ratios')
        axes[0, 1].legend()
        
        # Subject utilization
        axes[1, 0].scatter(results_df['n_subjects'], results_df['silhouette'], alpha=0.6)
        axes[1, 0].set_xlabel('Number of Subjects Used')
        axes[1, 0].set_ylabel('Silhouette Score')
        axes[1, 0].set_title('Silhouette Score vs Subject Utilization')
        
        # Feature utilization
        axes[1, 1].scatter(results_df['n_features'], results_df['silhouette'], alpha=0.6)
        axes[1, 1].set_xlabel('Number of Features Used')
        axes[1, 1].set_ylabel('Silhouette Score')
        axes[1, 1].set_title('Silhouette Score vs Feature Utilization')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/optimization_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Best result analysis
        if hasattr(self, 'best_overall') and self.best_overall:
            self._plot_best_result_analysis(output_dir)
        
        # 3. Optimization history
        if self.optimization_history:
            self._plot_optimization_history(output_dir)
        
        print("Optimization visualizations completed!")
        
    def _plot_best_result_analysis(self, output_dir):
        """Plot detailed analysis of the best result."""
        import matplotlib.pyplot as plt
        
        best_result = self.best_overall
        analysis = self.analyze_optimal_solution(best_result)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Cluster sizes
        cluster_sizes = analysis['cluster_sizes']
        ax1.bar(['Group 0', 'Group 1'], cluster_sizes, color=['red', 'blue'])
        ax1.set_ylabel('Number of Subjects')
        ax1.set_title(f'Optimal Cluster Sizes\n(Balance Ratio: {analysis["balance_ratio"]:.3f})')
        
        # Subject utilization
        utilization_data = {
            'Total Subjects': analysis['n_subjects_total'],
            'Used Subjects': analysis['n_subjects_used'],
            'Excluded Subjects': analysis['n_subjects_total'] - analysis['n_subjects_used']
        }
        ax2.bar(utilization_data.keys(), utilization_data.values(), color=['gray', 'green', 'red'])
        ax2.set_ylabel('Count')
        ax2.set_title('Subject Utilization in Optimal Solution')
        
        # Feature utilization
        feature_utilization = {
            'Total Features': analysis['n_features_total'],
            'Used Features': analysis['n_features_used'],
            'Excluded Features': analysis['n_features_total'] - analysis['n_features_used']
        }
        ax3.bar(feature_utilization.keys(), feature_utilization.values(), color=['gray', 'green', 'red'])
        ax3.set_ylabel('Count')
        ax3.set_title('Feature Utilization in Optimal Solution')
        
        # Top discriminating features
        if 'feature_analysis' in analysis and analysis['feature_analysis']:
            top_features = analysis['feature_analysis'][:10]
            feature_names = [f['feature'][:20] + '...' if len(f['feature']) > 20 else f['feature'] for f in top_features]
            effect_sizes = [f['effect_size'] for f in top_features]
            
            ax4.barh(range(len(top_features)), effect_sizes)
            ax4.set_yticks(range(len(top_features)))
            ax4.set_yticklabels(feature_names, fontsize=8)
            ax4.set_xlabel('Effect Size (Cohen\'s d)')
            ax4.set_title('Top 10 Discriminating Features')
            ax4.invert_yaxis()
        
        plt.suptitle(f'Best Optimization Result Analysis\n'
                    f'Method: {best_result["method"]} | '
                    f'Dataset: {best_result["dataset_name"]} | '
                    f'Silhouette: {analysis["silhouette_score"]:.4f}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/best_result_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_optimization_history(self, output_dir):
        """Plot optimization history."""
        import matplotlib.pyplot as plt
        import pandas as pd
        
        if not self.optimization_history:
            return
        
        history_df = pd.DataFrame(self.optimization_history)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot by method
        for method in history_df['method'].unique():
            method_data = history_df[history_df['method'] == method]
            if 'generation' in method_data.columns:
                ax1.plot(method_data['generation'], method_data['best_fitness'], 
                        label=method, marker='o')
                ax1.set_xlabel('Generation')
            elif 'iteration' in method_data.columns:
                ax1.plot(method_data['iteration'], method_data['best_score'], 
                        label=method, marker='o')
                ax1.set_xlabel('Iteration')
        
        ax1.set_ylabel('Best Score')
        ax1.set_title('Optimization Progress by Method')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Distribution of final scores by method
        if len(history_df['method'].unique()) > 1:
            for method in history_df['method'].unique():
                method_data = history_df[history_df['method'] == method]
                final_scores = method_data.groupby('method').last()
                if 'best_fitness' in method_data.columns:
                    scores = method_data['best_fitness'].values
                else:
                    scores = method_data['best_score'].values
                scores = scores[~np.isnan(scores)]
                if len(scores) > 0:
                    ax2.hist(scores, alpha=0.7, label=method, bins=20)
        
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Score Distribution During Optimization')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/optimization_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _try_kmeans(self, X: np.ndarray, n_attempts: int = 20) -> List[Dict]:
        """
        Try K-means clustering with different random states.
        
        Args:
            X: Feature matrix
            n_attempts: Number of attempts (default: 20)
            
        Returns:
            List of clustering results
        """
        results = []
        
        for attempt in range(n_attempts):
            try:
                kmeans = KMeans(
                    n_clusters=2, 
                    random_state=self.random_state + attempt,
                    n_init=10
                )
                labels = kmeans.fit_predict(X)
                
                # Evaluate clustering
                if len(np.unique(labels)) == 2:
                    silhouette = silhouette_score(X, labels)
                    is_balanced, balance_ratio = self._check_balance(labels)
                    
                    results.append({
                        'algorithm': f'kmeans_{attempt}',
                        'labels': labels,
                        'silhouette_score': silhouette,
                        'is_balanced': is_balanced,
                        'balance_ratio': balance_ratio,
                        'calinski_harabasz': calinski_harabasz_score(X, labels),
                        'davies_bouldin': davies_bouldin_score(X, labels)
                    })
            except:
                continue
        
        return results
    
    def _try_hierarchical(self, X: np.ndarray) -> List[Dict]:
        """
        Try hierarchical clustering with different linkage methods.
        
        Args:
            X: Feature matrix
            
        Returns:
            List of clustering results
        """
        results = []
        
        linkage_methods = ['ward', 'complete', 'average']
        
        for method in linkage_methods:
            try:
                hac = AgglomerativeClustering(
                    n_clusters=2,
                    linkage=method
                )
                labels = hac.fit_predict(X)
                
                # Evaluate clustering
                if len(np.unique(labels)) == 2:
                    silhouette = silhouette_score(X, labels)
                    is_balanced, balance_ratio = self._check_balance(labels)
                    
                    results.append({
                        'algorithm': f'hierarchical_{method}',
                        'labels': labels,
                        'silhouette_score': silhouette,
                        'is_balanced': is_balanced,
                        'balance_ratio': balance_ratio,
                        'calinski_harabasz': calinski_harabasz_score(X, labels),
                        'davies_bouldin': davies_bouldin_score(X, labels)
                    })
            except:
                continue
        
        return results
    
    def _try_gmm(self, X: np.ndarray, n_attempts: int = 10) -> List[Dict]:
        """
        Try Gaussian Mixture Models with different covariance types.
        
        Args:
            X: Feature matrix
            n_attempts: Number of attempts per covariance type (default: 10)
            
        Returns:
            List of clustering results
        """
        results = []
        
        covariance_types = ['full', 'tied', 'diag', 'spherical']
        
        for cov_type in covariance_types:
            for attempt in range(n_attempts):
                try:
                    gmm = GaussianMixture(
                        n_components=2,
                        covariance_type=cov_type,
                        random_state=self.random_state + attempt,
                        n_init=5
                    )
                    labels = gmm.fit_predict(X)
                    
                    # Evaluate clustering
                    if len(np.unique(labels)) == 2:
                        silhouette = silhouette_score(X, labels)
                        is_balanced, balance_ratio = self._check_balance(labels)
                        
                        results.append({
                            'algorithm': f'gmm_{cov_type}_{attempt}',
                            'labels': labels,
                            'silhouette_score': silhouette,
                            'is_balanced': is_balanced,
                            'balance_ratio': balance_ratio,
                            'calinski_harabasz': calinski_harabasz_score(X, labels),
                            'davies_bouldin': davies_bouldin_score(X, labels)
                        })
                except:
                    continue
        
        return results
    
    def _try_spectral(self, X: np.ndarray) -> List[Dict]:
        """
        Try spectral clustering with different affinity types.
        
        Args:
            X: Feature matrix
            
        Returns:
            List of clustering results
        """
        results = []
        
        affinity_types = ['rbf', 'nearest_neighbors']
        
        for affinity in affinity_types:
            try:
                spectral = SpectralClustering(
                    n_clusters=2,
                    affinity=affinity,
                    random_state=self.random_state,
                    n_init=5
                )
                labels = spectral.fit_predict(X)
                
                # Evaluate clustering
                if len(np.unique(labels)) == 2:
                    silhouette = silhouette_score(X, labels)
                    is_balanced, balance_ratio = self._check_balance(labels)
                    
                    results.append({
                        'algorithm': f'spectral_{affinity}',
                        'labels': labels,
                        'silhouette_score': silhouette,
                        'is_balanced': is_balanced,
                        'balance_ratio': balance_ratio,
                        'calinski_harabasz': calinski_harabasz_score(X, labels),
                        'davies_bouldin': davies_bouldin_score(X, labels)
                    })
            except:
                continue
        
        return results
    
    def _try_ensemble_clustering(self, X: np.ndarray) -> List[Dict]:
        """
        Try ensemble clustering (consensus from multiple base clusterers).
        
        Args:
            X: Feature matrix
            
        Returns:
            List of clustering results
        """
        results = []
        
        try:
            # Create base clusterers
            base_clusterers = [
                KMeans(n_clusters=2, random_state=self.random_state),
                KMeans(n_clusters=2, random_state=self.random_state+1),
                AgglomerativeClustering(n_clusters=2, linkage='ward'),
                AgglomerativeClustering(n_clusters=2, linkage='complete'),
                GaussianMixture(n_components=2, random_state=self.random_state)
            ]
            
            # Generate base clusterings
            base_labels = np.zeros((len(base_clusterers), X.shape[0]))
            base_scores = np.zeros(len(base_clusterers))
            
            for i, clusterer in enumerate(base_clusterers):
                labels = clusterer.fit_predict(X)
                base_labels[i] = labels
                
                if len(np.unique(labels)) > 1:
                    base_scores[i] = max(0, silhouette_score(X, labels))
                else:
                    base_scores[i] = 0
            
            # Normalize weights if any are positive
            if np.sum(base_scores) > 0:
                weights = base_scores / np.sum(base_scores)
            else:
                weights = np.ones(len(base_clusterers)) / len(base_clusterers)
            
            # Create co-association matrix
            co_assoc = np.zeros((X.shape[0], X.shape[0]))
            
            for i in range(len(base_clusterers)):
                # Skip if only one cluster
                if len(np.unique(base_labels[i])) < 2:
                    continue
                    
                # Create binary co-association matrix for this clustering
                labels = base_labels[i]
                for j in range(X.shape[0]):
                    for k in range(j+1, X.shape[0]):
                        if labels[j] == labels[k]:
                            co_assoc[j, k] += weights[i]
                            co_assoc[k, j] += weights[i]
            
            # Convert to distance matrix
            dist_matrix = 1 - co_assoc
            
            # Final clustering using the consensus matrix
            consensus = AgglomerativeClustering(
                n_clusters=2,
                affinity='precomputed',
                linkage='average'
            )
            
            labels = consensus.fit_predict(dist_matrix)
            
            # Evaluate clustering
            if len(np.unique(labels)) == 2:
                silhouette = silhouette_score(X, labels)
                is_balanced, balance_ratio = self._check_balance(labels)
                
                results.append({
                    'algorithm': 'ensemble_consensus',
                    'labels': labels,
                    'silhouette_score': silhouette,
                    'is_balanced': is_balanced,
                    'balance_ratio': balance_ratio,
                    'calinski_harabasz': calinski_harabasz_score(X, labels),
                    'davies_bouldin': davies_bouldin_score(X, labels)
                })
        except:
            pass
        
        return results
    
    def analyze_feature_importance(self) -> pd.DataFrame:
        """
        Analyze feature importance in the best clustering model.
        
        Returns:
            DataFrame with feature importance information
        """
        if not self.best_model:
            print("No best model found. Run clustering analysis first.")
            return None
        
        print("\nAnalyzing feature importance...")
        
        X = self.best_model['dataset_info']['X']
        labels = self.best_model['labels']
        feature_names = self.best_model['dataset_info']['feature_names']
        
        # Create DataFrame for feature analysis
        feature_analysis = pd.DataFrame({'feature': feature_names})
        
        # 1. Random Forest importance
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rf.fit(X, labels)
        feature_analysis['rf_importance'] = rf.feature_importances_
        
        # 2. Univariate statistics
        from sklearn.feature_selection import f_classif
        f_scores, f_pvalues = f_classif(X, labels)
        feature_analysis['f_score'] = f_scores
        feature_analysis['f_pvalue'] = f_pvalues
        
        # 3. Mean difference between groups
        group0_mean = X[labels == 0].mean(axis=0)
        group1_mean = X[labels == 1].mean(axis=0)
        feature_analysis['mean_diff'] = np.abs(group1_mean - group0_mean)
        
        # 4. Effect size (Cohen's d)
        cohens_d = []
        for i in range(X.shape[1]):
            group0_vals = X[labels == 0, i]
            group1_vals = X[labels == 1, i]
            
            pooled_std = np.sqrt(((len(group0_vals) - 1) * group0_vals.var() + 
                                 (len(group1_vals) - 1) * group1_vals.var()) / 
                                (len(group0_vals) + len(group1_vals) - 2))
            
            if pooled_std > 0:
                d = np.abs(group1_vals.mean() - group0_vals.mean()) / pooled_std
            else:
                d = 0
            cohens_d.append(d)
        
        feature_analysis['cohens_d'] = cohens_d
        
        # Calculate composite importance score
        # Normalize each metric
        metrics = ['rf_importance', 'f_score', 'mean_diff', 'cohens_d']
        for metric in metrics:
            min_val = feature_analysis[metric].min()
            max_val = feature_analysis[metric].max()
            if max_val > min_val:
                feature_analysis[f'{metric}_norm'] = (feature_analysis[metric] - min_val) / (max_val - min_val)
            else:
                feature_analysis[f'{metric}_norm'] = 0
        
        # Calculate composite score
        norm_cols = [f'{metric}_norm' for metric in metrics]
        feature_analysis['composite_importance'] = feature_analysis[norm_cols].mean(axis=1)
        
        # Sort by composite importance
        feature_analysis = feature_analysis.sort_values('composite_importance', ascending=False)
        
        # Store feature importance
        self.feature_importance = feature_analysis
        
        # Display top features
        print("\nTop 10 most important features:")
        print(feature_analysis.head(10)[['feature', 'composite_importance', 'cohens_d', 'f_pvalue']].to_string(index=False))
        
        return feature_analysis
    
    def generate_visualizations(self, output_dir: str = 'clustering_results') -> None:
        """
        Generate visualizations of clustering results.
        
        Args:
            output_dir: Directory to save visualizations
        """
        if not self.best_model:
            print("No best model found. Run clustering analysis first.")
            return
        
        print(f"\nGenerating visualizations in {output_dir}/...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Silhouette Plot
        self._generate_silhouette_plot(output_dir)
        
        # 2. Feature Importance Plot
        if self.feature_importance is not None:
            self._generate_feature_importance_plot(output_dir)
        
        # 3. Cluster Distribution
        self._generate_cluster_distribution_plot(output_dir)
        
        # 4. PCA Visualization
        self._generate_pca_visualization(output_dir)
        
        print("Visualization generation complete!")
    
    def _generate_silhouette_plot(self, output_dir: str) -> None:
        """
        Generate silhouette plot for the best model.
        
        Args:
            output_dir: Directory to save the plot
        """
        X = self.best_model['dataset_info']['X']
        labels = self.best_model['labels']
        
        plt.figure(figsize=(10, 6))
        
        # Get silhouette values
        silhouette_vals = silhouette_samples(X, labels)
        
        # Plot silhouette for each cluster
        y_lower = 10
        colors = ['red', 'blue']
        
        for i in range(2):
            cluster_silhouette_vals = silhouette_vals[labels == i]
            cluster_silhouette_vals.sort()
            
            size_cluster_i = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i
            
            plt.fill_betweenx(np.arange(y_lower, y_upper),
                             0, cluster_silhouette_vals,
                             alpha=0.7, color=colors[i])
            
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, f'Cluster {i}')
            y_lower = y_upper + 10
        
        plt.axvline(x=silhouette_score(X, labels), color='red', linestyle='--',
                   label=f'Average Score: {silhouette_score(X, labels):.3f}')
        
        plt.title(f'Silhouette Analysis for {self.best_model["algorithm"]}')
        plt.xlabel('Silhouette Coefficient')
        plt.ylabel('Cluster')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'silhouette_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_feature_importance_plot(self, output_dir: str) -> None:
        """
        Generate feature importance plot.
        
        Args:
            output_dir: Directory to save the plot
        """
        # Get top 15 features
        top_features = self.feature_importance.head(15)
        
        plt.figure(figsize=(12, 8))
        
        # Plot composite importance scores
        plt.barh(range(len(top_features)), top_features['composite_importance'], align='center')
        plt.yticks(range(len(top_features)), 
                  [f[:30] + '...' if len(f) > 30 else f for f in top_features['feature']])
        plt.xlabel('Composite Importance Score')
        plt.title('Top 15 Features by Importance')
        plt.gca().invert_yaxis()  # Highest importance at the top
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_cluster_distribution_plot(self, output_dir: str) -> None:
        """
        Generate cluster distribution plot.
        
        Args:
            output_dir: Directory to save the plot
        """
        labels = self.best_model['labels']
        
        plt.figure(figsize=(10, 6))
        
        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        plt.bar(unique, counts, color=['red', 'blue'])
        
        # Add percentages
        total = len(labels)
        for i, count in enumerate(counts):
            percentage = count / total * 100
            plt.text(unique[i], count + 1, f'{count}\n({percentage:.1f}%)', 
                    ha='center', va='bottom')
        
        plt.xlabel('Cluster')
        plt.ylabel('Number of Participants')
        plt.title('Cluster Distribution')
        plt.xticks(unique, [f'Cluster {u}' for u in unique])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_pca_visualization(self, output_dir: str) -> None:
        X = self.best_model['dataset_info']['X']
        labels = self.best_model['labels']
        feature_names = self.best_model['dataset_info']['feature_names']
        
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X)
        
        # Plot clusters
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels, palette='viridis', s=100)
        
        # Add feature vectors
        if hasattr(self, 'feature_importance'):
            top_features = self.feature_importance.head(5)['feature']
            feature_indices = [feature_names.index(f) for f in top_features]
            
            for i in feature_indices:
                plt.arrow(0, 0, pca.components_[0,i]*np.std(X[:,i]),
                          pca.components_[1,i]*np.std(X[:,i]),
                          color='r', width=0.01)
                plt.text(pca.components_[0,i]*1.2, 
                         pca.components_[1,i]*1.2,
                         feature_names[i], color='r')
    
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.title('PCA Cluster Visualization with Feature Loadings')
        plt.savefig(os.path.join(output_dir, 'enhanced_pca_plot.png'))
        plt.close()
    
    # New method for delta significance filtering
    def filter_significant_deltas(self, datasets, alpha=0.05):
        """
        Filter features based on paired t-test of deltas being different from 0
        Returns datasets with only significant features
        """
        filtered_datasets = {}
        for name, data in datasets.items():
            X = data['X']
            pvals = [stats.ttest_1samp(col, 0).pvalue for col in X.T]
            sig_mask = np.array(pvals) < alpha
            filtered_datasets[name] = {
                'X': X[:, sig_mask],
                'feature_names': [f for f, m in zip(data['feature_names'], sig_mask) if m],
                'base_names': data['base_names']
            }
        return filtered_datasets
    
    def save_cluster_assignments(self, output_file: str = 'cluster_assignments.csv') -> None:
        """
        Save cluster assignments to a CSV file.
        
        Args:
            output_file: Output file path
        """
        if not self.best_model:
            print("No best model found. Run clustering analysis first.")
            return
        
        print(f"\nSaving cluster assignments to {output_file}...")
        
        # Get participant names and cluster labels
        base_names = self.best_model['dataset_info']['base_names']
        labels = self.best_model['labels']
        
        # Create DataFrame
        assignments = pd.DataFrame({
            'Participant': base_names,
            'Cluster': labels,
            'Group': [f'Group_{i}' for i in labels]
        })
        
        # Save to CSV
        assignments.to_csv(output_file, index=False)
        print(f"Saved assignments for {len(assignments)} participants")
    
    def generate_comprehensive_report(self, output_file: str = 'clustering_report.txt') -> None:
        """
        Generate comprehensive clustering report.
        
        Args:
            output_file: Output file path
        """
        if not self.best_model:
            print("No best model found. Run clustering analysis first.")
            return
        
        print(f"\nGenerating comprehensive report to {output_file}...")
        
        with open(output_file, 'w') as f:
            f.write("NAD+ SUPPLEMENT STUDY CLUSTERING REPORT\n")
            f.write("=" * 40 + "\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 17 + "\n")
            f.write(f"Best Silhouette Score: {self.best_model['silhouette_score']:.4f}\n")
            f.write(f"Balance Ratio: {self.best_model['balance_ratio']:.4f} (target: ≥{1-self.balance_tolerance:.2f})\n")
            f.write(f"Algorithm: {self.best_model['algorithm']}\n")
            f.write(f"Dataset: {self.best_model['dataset']}\n\n")
            
            # Cluster Statistics
            labels = self.best_model['labels']
            unique, counts = np.unique(labels, return_counts=True)
            
            f.write("CLUSTER STATISTICS\n")
            f.write("-" * 18 + "\n")
            for i, (label, count) in enumerate(zip(unique, counts)):
                percentage = count / len(labels) * 100
                f.write(f"Cluster {label}: {count} participants ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Top Features
            if self.feature_importance is not None:
                f.write("TOP 15 MOST IMPORTANT FEATURES\n")
                f.write("-" * 28 + "\n")
                
                top_features = self.feature_importance.head(15)
                for i, (_, row) in enumerate(top_features.iterrows()):
                    f.write(f"{i+1}. {row['feature']}\n")
                    f.write(f"   Importance Score: {row['composite_importance']:.4f}\n")
                    f.write(f"   Effect Size (Cohen's d): {row['cohens_d']:.4f}\n")
                    f.write(f"   p-value: {row['f_pvalue']:.6f}\n\n")
            
            # Dataset Information
            f.write("DATASET INFORMATION\n")
            f.write("-" * 19 + "\n")
            f.write(f"Original Dataset: {self.best_model['dataset_info']['original_dataset']}\n")
            f.write(f"Preprocessing: {self.best_model['dataset_info']['preprocessing']}\n")
            f.write(f"Number of Features: {len(self.best_model['dataset_info']['feature_names'])}\n")
            f.write(f"Number of Participants: {len(self.best_model['dataset_info']['base_names'])}\n\n")
            
            # Top Models
            f.write("TOP 5 CLUSTERING MODELS\n")
            f.write("-" * 21 + "\n")
            for i, model in enumerate(self.clustering_results[:5]):
                f.write(f"{i+1}. {model['algorithm']}\n")
                f.write(f"   Dataset: {model['dataset']}\n")
                f.write(f"   Silhouette Score: {model['silhouette_score']:.4f}\n")
                f.write(f"   Balance Ratio: {model['balance_ratio']:.4f}\n")
                f.write(f"   Balanced: {'Yes' if model['is_balanced'] else 'No'}\n\n")
            
            f.write("=" * 40 + "\n")
            f.write("Analysis completed successfully!\n")
        
        print(f"Comprehensive report saved to {output_file}")
    
    def run_complete_analysis(self, data_dir: str = 'processed_data', 
                            output_dir: str = 'clustering_results') -> Dict:
        """
        Run complete clustering analysis pipeline.
        
        Args:
            data_dir: Directory containing processed data
            output_dir: Directory to save results
            
        Returns:
            Dictionary with analysis results
        """
        print("STARTING CLUSTERING OPTIMIZATION PIPELINE")
        print("=" * 40)
        
        # Step 1: Load processed data
        datasets = self.load_processed_data(data_dir)
        
        # Step 2: Preprocess datasets
        preprocessed_datasets = self.preprocess_datasets(datasets)
        
        # Step 3: One/paired t-test
        #preprocessed_datasets = self.filter_significant_deltas(datasets, 0.10)
        
        # Step 3: Prepare for optimization
        viable_datasets = self.prepare_optimization_data(preprocessed_datasets)
        
        # Step 4: Run iterative optimization
        # Determine which optimization methods to use based on dataset sizes
        methods = ['random_search']  # Always include random search
        
        # Add other methods based on dataset size
        for name, info in viable_datasets.items():
            # If any dataset is large enough, include genetic algorithm
            if info['X'].shape[0] >= 30:
                if 'genetic_algorithm' not in methods:
                    methods.append('genetic_algorithm')
            
            # For smaller datasets, include exhaustive search
            if info['X'].shape[0] <= 50:
                if 'exhaustive_search' not in methods:
                    methods.append('exhaustive_search')
            
            # Include Bayesian optimization if optuna is available
            try:
                import optuna
                if 'bayesian_optimization' not in methods:
                    methods.append('bayesian_optimization')
            except ImportError:
                pass
        
        print(f"Using optimization methods: {methods}")
        
        # Run optimization
        all_results = self.optimize_all_datasets(viable_datasets, methods=methods)
        
        # Step 5: Analyze results
        if hasattr(self, 'best_overall') and self.best_overall:
            print("\nAnalyzing optimal solution...")
            best_analysis = self.analyze_optimal_solution(self.best_overall)
            cluster_groups = self.get_cluster_membership(self.best_overall)
        
        # Step 6: Create visualizations
        os.makedirs(output_dir, exist_ok=True)
        self.create_optimization_visualizations(all_results, output_dir)
        
        # Step 7: Generate report
        self.generate_optimization_report(all_results, os.path.join(output_dir, 'optimization_report.txt'))
        
        # Step 8: Save final cluster assignments
        if hasattr(self, 'best_overall') and self.best_overall:
            base_names = self.best_overall['dataset_info']['base_names']
            labels = best_analysis['labels']
            
            # Get mapping of excluded subjects
            all_subjects = np.zeros(self.best_overall['dataset_info']['X'].shape[0], dtype=int)
            all_subjects[self.best_overall['subject_mask']] = labels
            
            assignments = pd.DataFrame({
                'Participant': base_names,
                'Included': self.best_overall['subject_mask'],
                'Cluster': all_subjects,
                'Group': ['Group_' + str(l) if i else 'Excluded' 
                        for i, l in zip(self.best_overall['subject_mask'], all_subjects)]
            })
            
            assignments.to_csv(os.path.join(output_dir, 'cluster_assignments.csv'), index=False)
            print(f"Saved final cluster assignments to {output_dir}/cluster_assignments.csv")
        
        # Step 9: Standard clustering analysis (for comparison)
        print("\nRunning standard clustering analysis for comparison...")
        clustering_results = self.run_clustering_analysis(preprocessed_datasets)
        
        # Step 10: Analyze feature importance
        feature_importance = self.analyze_feature_importance()
        
        print("\n" + "=" * 40)
        print("CLUSTERING OPTIMIZATION COMPLETE!")
        print(f"Results saved to {output_dir}/")
        
        # Compare standard clustering vs optimized clustering
        if hasattr(self, 'best_model') and self.best_model and hasattr(self, 'best_overall') and self.best_overall:
            print("\nStandard vs Optimized Clustering Comparison:")
            print(f"Standard Best:  {self.best_model['silhouette_score']:.4f} silhouette (no exclusions)")
            print(f"Optimized Best: {self.best_overall['silhouette_score']:.4f} silhouette (with exclusions)")
            print(f"Improvement:    {self.best_overall['silhouette_score'] - self.best_model['silhouette_score']:.4f}")
        
        # Return all results
        return {
            'best_model': self.best_model,
            'best_optimal': self.best_overall,
            'clustering_results': self.clustering_results,
            'optimization_results': all_results,
            'feature_importance': self.feature_importance
        }


# Example usage
if __name__ == "__main__":
    # Initialize optimizer
    optimizer = ClusteringOptimizer(
        balance_tolerance=0.15,  # 15% max deviation from 50-50
        random_state=42,
        min_subjects=70
    )
    
    # Run complete analysis
    results = optimizer.run_complete_analysis(
        data_dir='processed_data',
        output_dir='clustering_results'
    )
    
    # Print summary
    if results['best_model']:
        print("\nFINAL SUMMARY:")
        print(f"Best silhouette score: {results['best_model']['silhouette_score']:.4f}")
        print(f"Balance achieved: {results['best_model']['balance_ratio']:.4f}")
        print(f"Participants in clusters: {np.bincount(results['best_model']['labels'])}")