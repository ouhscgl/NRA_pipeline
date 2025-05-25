"""
Data Processing Module for NAD+ Supplement Study

This module handles the loading, merging, contrasting, and normalization of NAD+ supplement study data.
It processes both pre-contrasted and uncontrasted datasets to prepare them for clustering analysis.

Author: [Your Name]
Date: May 21, 2025
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """
    Processor for NAD+ supplement study data.
    
    This class handles:
    1. Loading uncontrasted and pre-contrasted data
    2. Generating contrasts using multiple methods
    3. Merging all datasets
    4. Normalizing and preparing data for clustering
    """
    
    def __init__(self, alpha_threshold: float = 0.05, memory_efficient: bool = True):
        """
        Initialize the DataProcessor.
        
        Args:
            alpha_threshold: Threshold for statistical significance (default: 0.05)
            memory_efficient: Whether to use memory-efficient processing (default: True)
        """
        self.alpha_threshold = alpha_threshold
        self.memory_efficient = memory_efficient
        self.modality_weights = {}
        self.modality_presence = {}
        self.difference_methods = {}
        self.feature_p_values = {}
    
    def load_and_merge_modalities(self, file_paths: List[str], 
                                  precomputed_contrasts: Optional[List[str]] = None) -> Tuple:
        """
        Load and merge modalities from uncontrasted data files.
        
        Args:
            file_paths: List of file paths to uncontrasted data
            precomputed_contrasts: List of file paths to pre-contrasted data
            
        Returns:
            Tuple containing:
                - merged_df: Merged dataframe
                - modality_presence: Dictionary mapping participants to their available modalities
                - modality_weights: Dictionary mapping modalities to their weights
        """
        print("Loading and merging multimodal data...")
        
        # Load all modalities
        modalities = {}
        for file_path in file_paths:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.strip()
                modality_name = Path(file_path).stem
                modalities[modality_name] = df
                print(f"Loaded {modality_name}: {df.shape}")
        
        # Find all unique participant base names across all modalities
        all_base_names = set()
        for modality_name, df in modalities.items():
            names = df['Name'].dropna().unique()
            base_names = [name.replace('_V1', '').replace('_V2', '') for name in names]
            all_base_names.update(base_names)
        
        print(f"Found {len(all_base_names)} unique participants across all modalities")
        
        # Create comprehensive merged dataset with missing value tracking
        merged_data = []
        modality_presence = {}
        
        for base_name in all_base_names:
            v1_name = f"{base_name}_V1"
            v2_name = f"{base_name}_V2"
            
            participant_data = {'Name': base_name}
            modality_count = 0
            
            for modality_name, df in modalities.items():
                # Check if both V1 and V2 exist for this participant
                v1_data = df[df['Name'] == v1_name]
                v2_data = df[df['Name'] == v2_name]
                
                if not v1_data.empty and not v2_data.empty:
                    # Add V1 and V2 data with modality prefix
                    for col in df.columns:
                        if col != 'Name':
                            participant_data[f"{modality_name}_V1_{col}"] = v1_data[col].iloc[0]
                            participant_data[f"{modality_name}_V2_{col}"] = v2_data[col].iloc[0]
                    modality_count += 1
                    
                    # Track modality presence
                    if base_name not in modality_presence:
                        modality_presence[base_name] = set()
                    modality_presence[base_name].add(modality_name)
            
            if modality_count > 0:  # Include if participant has at least one modality
                merged_data.append(participant_data)
        
        merged_df = pd.DataFrame(merged_data)
        
        # Load pre-computed contrasts if provided
        if precomputed_contrasts:
            for file_path in precomputed_contrasts:
                if os.path.exists(file_path):
                    contrast_df = pd.read_csv(file_path)
                    contrast_df.columns = contrast_df.columns.str.strip()
                    modality_name = f"precontrast_{Path(file_path).stem}"
                    
                    # Merge pre-contrasted data
                    contrast_df['Name'] = contrast_df['Name'].str.replace('_V1', '').str.replace('_V2', '')
                    
                    # Add prefix to distinguish pre-contrasted features
                    contrast_cols = {col: f"{modality_name}_{col}" for col in contrast_df.columns if col != 'Name'}
                    contrast_df = contrast_df.rename(columns=contrast_cols)
                    
                    merged_df = merged_df.merge(contrast_df, on='Name', how='left')
                    print(f"Added pre-contrasted data from {modality_name}: {contrast_df.shape[1]-1} features")
        
        # Calculate modality weights (inverse of missing data proportion)
        modality_weights = {}
        for modality_name in modalities.keys():
            modality_cols = [col for col in merged_df.columns if col.startswith(modality_name)]
            if modality_cols:
                missing_prop = merged_df[modality_cols].isnull().mean().mean()
                modality_weights[modality_name] = 1 / (missing_prop + 0.01)  # Avoid division by zero
        
        # Normalize weights
        total_weight = sum(modality_weights.values())
        modality_weights = {k: v/total_weight for k, v in modality_weights.items()}
        
        self.modality_weights = modality_weights
        self.modality_presence = modality_presence
        
        print(f"Final merged dataset: {merged_df.shape}")
        print("Modality weights:", {k: round(v, 3) for k, v in modality_weights.items()})
        
        return merged_df, modality_presence, modality_weights
    
    def calculate_differences_multiple_methods(self, merged_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Calculate visit differences using multiple methods.
        
        Args:
            merged_df: Merged dataset containing V1 and V2 columns
            
        Returns:
            Dictionary mapping method names to dataframes with calculated differences
        """
        print("\nCalculating differences using multiple methods...")
        
        # Identify V1/V2 column pairs (excluding pre-contrasted data)
        v1_cols = [col for col in merged_df.columns if '_V1_' in col]
        v2_cols = [col for col in merged_df.columns if '_V2_' in col]
        
        # Match V1 and V2 columns
        paired_cols = []
        for v1_col in v1_cols:
            base_col = v1_col.replace('_V1_', '_V2_')
            if base_col in v2_cols:
                paired_cols.append((v1_col, base_col))
        
        print(f"Found {len(paired_cols)} V1-V2 paired measures")
        
        difference_methods = {}
        
        # Method 1: Simple Difference (V2 - V1)
        print("  Calculating simple differences...")
        simple_diff = {}
        for v1_col, v2_col in paired_cols:
            measure_name = v1_col.replace('_V1_', '_').replace('_V1', '')
            simple_diff[f"simple_diff_{measure_name}"] = merged_df[v2_col] - merged_df[v1_col]
        
        # Create DataFrame with Name column
        diff_df = pd.DataFrame(simple_diff, index=merged_df.index)
        if 'Name' in merged_df.columns:
            diff_df['Name'] = merged_df['Name'].values
        difference_methods['simple_difference'] = diff_df
        
        # Method 2: Percent Change
        print("  Calculating percent changes...")
        percent_change = {}
        for v1_col, v2_col in paired_cols:
            measure_name = v1_col.replace('_V1_', '_').replace('_V1', '')
            # Handle division by zero
            v1_vals = merged_df[v1_col].replace(0, np.nan)
            percent_change[f"pct_change_{measure_name}"] = ((merged_df[v2_col] - v1_vals) / np.abs(v1_vals)) * 100
        
        # Create DataFrame with Name column
        pct_df = pd.DataFrame(percent_change, index=merged_df.index)
        if 'Name' in merged_df.columns:
            pct_df['Name'] = merged_df['Name'].values
        difference_methods['percent_change'] = pct_df
        
        # Method 3: Standardized Effect Size (Cohen's d style)
        print("  Calculating standardized effect sizes...")
        effect_sizes = {}
        for v1_col, v2_col in paired_cols:
            measure_name = v1_col.replace('_V1_', '_').replace('_V1', '')
            v1_vals = merged_df[v1_col]
            v2_vals = merged_df[v2_col]
            pooled_std = np.sqrt((v1_vals.var() + v2_vals.var()) / 2)
            if pooled_std > 0:
                effect_sizes[f"effect_size_{measure_name}"] = (v2_vals - v1_vals) / pooled_std
            else:
                effect_sizes[f"effect_size_{measure_name}"] = 0
        
        # Create DataFrame with Name column
        effect_df = pd.DataFrame(effect_sizes, index=merged_df.index)
        if 'Name' in merged_df.columns:
            effect_df['Name'] = merged_df['Name'].values
        difference_methods['effect_size'] = effect_df
        
        # Method 4: Robust Effect Size (using median and MAD) - CORRECTED
        print("  Calculating robust effect sizes...")
        robust_effect_sizes = {}
        for v1_col, v2_col in paired_cols:
            measure_name = v1_col.replace('_V1_', '_').replace('_V1', '')
            v1_vals = merged_df[v1_col]
            v2_vals = merged_df[v2_col]
            
            # Calculate difference for each participant
            diff = v2_vals - v1_vals
            
            # Calculate robust standardization using overall median and MAD
            median_diff = diff.median()
            mad = diff.sub(median_diff).abs().median() * 1.4826  # Scale factor for normal distribution
            
            if mad > 0:
                # Robust z-score: (individual_diff - median_diff) / MAD
                # This gives each participant their own robust effect size
                robust_effect_sizes[f"robust_effect_{measure_name}"] = (diff - median_diff) / mad
            else:
                # If MAD is zero, all differences are the same - use simple standardization
                std_diff = diff.std()
                if std_diff > 0:
                    robust_effect_sizes[f"robust_effect_{measure_name}"] = (diff - median_diff) / std_diff
                else:
                    robust_effect_sizes[f"robust_effect_{measure_name}"] = np.zeros_like(diff)
        
        # Create DataFrame with Name column
        robust_df = pd.DataFrame(robust_effect_sizes, index=merged_df.index)
        if 'Name' in merged_df.columns:
            robust_df['Name'] = merged_df['Name'].values
        difference_methods['robust_effect_size'] = robust_df
        
        # Method 5: Change Magnitude/Direction
        print("  Calculating change magnitude/direction...")
        magnitude_direction = {}
        for v1_col, v2_col in paired_cols:
            measure_name = v1_col.replace('_V1_', '_').replace('_V1', '')
            v1_vals = merged_df[v1_col]
            v2_vals = merged_df[v2_col]
            
            # Magnitude: absolute change normalized by baseline
            baseline_norm = np.abs(v1_vals).replace(0, np.nan)
            magnitude = np.abs(v2_vals - v1_vals) / baseline_norm
            
            # Direction: sign of change
            direction = np.sign(v2_vals - v1_vals)
            
            # Combined: magnitude * direction
            magnitude_direction[f"mag_dir_{measure_name}"] = magnitude * direction
        
        # Create DataFrame with Name column
        mag_dir_df = pd.DataFrame(magnitude_direction, index=merged_df.index)
        if 'Name' in merged_df.columns:
            mag_dir_df['Name'] = merged_df['Name'].values
        difference_methods['magnitude_direction'] = mag_dir_df
        
        # Method 6: Robust Z-score (using median and MAD)
        print("  Calculating robust z-scores...")
        robust_z = {}
        for v1_col, v2_col in paired_cols:
            measure_name = v1_col.replace('_V1_', '_').replace('_V1', '')
            diff = merged_df[v2_col] - merged_df[v1_col]
            median_diff = diff.median()
            mad = np.median(np.abs(diff - median_diff))
            if mad > 0:
                robust_z[f"robust_z_{measure_name}"] = (diff - median_diff) / (1.4826 * mad)  # 1.4826 for normal consistency
            else:
                robust_z[f"robust_z_{measure_name}"] = 0
        
        # Create DataFrame with Name column
        robust_z_df = pd.DataFrame(robust_z, index=merged_df.index)
        if 'Name' in merged_df.columns:
            robust_z_df['Name'] = merged_df['Name'].values
        difference_methods['robust_z_score'] = robust_z_df
        
        # Method 7: Log Ratio with sign preservation
        print("  Calculating sign-preserving log ratios...")
        log_ratio_sign = {}
        for v1_col, v2_col in paired_cols:
            measure_name = v1_col.replace('_V1_', '_').replace('_V1', '')
            v1_vals = merged_df[v1_col]
            v2_vals = merged_df[v2_col]
            
            # Initialize with NaN
            log_ratios = np.full(len(v1_vals), np.nan)
            
            # Case 1: Both positive
            both_pos = (v1_vals > 0) & (v2_vals > 0)
            if np.any(both_pos):
                log_ratios[both_pos] = np.log2(v2_vals[both_pos] / v1_vals[both_pos])
            
            # Case 2: Both negative
            both_neg = (v1_vals < 0) & (v2_vals < 0)
            if np.any(both_neg):
                # Preserve meaning: decrease in negative value = improvement
                log_ratios[both_neg] = np.log2(np.abs(v2_vals[both_neg]) / np.abs(v1_vals[both_neg]))
            
            # Case 3: Positive to negative (deterioration)
            pos_to_neg = (v1_vals > 0) & (v2_vals < 0)
            if np.any(pos_to_neg):
                # Encode direction change: use negative log ratio
                magnitude = np.log2(np.abs(v2_vals[pos_to_neg]) / v1_vals[pos_to_neg])
                log_ratios[pos_to_neg] = -magnitude  # Negative sign indicates direction change
            
            # Case 4: Negative to positive (improvement)
            neg_to_pos = (v1_vals < 0) & (v2_vals > 0)
            if np.any(neg_to_pos):
                # Encode direction change with positive log ratio
                magnitude = np.log2(v2_vals[neg_to_pos] / np.abs(v1_vals[neg_to_pos]))
                log_ratios[neg_to_pos] = magnitude  # Positive sign for improvement
            
            # Store result
            log_ratio_sign[f"log_ratio_sign_{measure_name}"] = log_ratios
        
        # Create DataFrame with Name column
        log_ratio_sign_df = pd.DataFrame(log_ratio_sign, index=merged_df.index)
        if 'Name' in merged_df.columns:
            log_ratio_sign_df['Name'] = merged_df['Name'].values
        difference_methods['log_ratio_sign'] = log_ratio_sign_df
        
        # Method 8: Winsorized Differences
        print("  Calculating winsorized differences...")
        winsorized_diff = {}
        for v1_col, v2_col in paired_cols:
            measure_name = v1_col.replace('_V1_', '_').replace('_V1', '')
            v1_vals = merged_df[v1_col]
            v2_vals = merged_df[v2_col]
            
            # Calculate raw difference
            raw_diff = v2_vals - v1_vals
            
            # Handle positive and negative values separately for more meaningful winsorization
            pos_diffs = raw_diff[raw_diff >= 0]
            neg_diffs = raw_diff[raw_diff < 0]
            
            # Initialize with original values
            winsorized = raw_diff.copy()
            
            # Winsorize positive differences if we have enough
            if len(pos_diffs) > 5:
                pos_low, pos_high = np.percentile(pos_diffs, [5, 95])
                winsorized[raw_diff >= 0] = np.clip(raw_diff[raw_diff >= 0], pos_low, pos_high)
                
            # Winsorize negative differences if we have enough
            if len(neg_diffs) > 5:
                neg_low, neg_high = np.percentile(neg_diffs, [5, 95])
                winsorized[raw_diff < 0] = np.clip(raw_diff[raw_diff < 0], neg_low, neg_high)
            
            # Store winsorized difference
            winsorized_diff[f"winsorized_{measure_name}"] = winsorized
        
        # Create DataFrame with Name column
        winsorized_df = pd.DataFrame(winsorized_diff, index=merged_df.index)
        if 'Name' in merged_df.columns:
            winsorized_df['Name'] = merged_df['Name'].values
        difference_methods['winsorized_difference'] = winsorized_df
        
        # Method 9: Sigmoid Transformation
        print("  Calculating sigmoid transformations...")
        sigmoid_transform = {}
        for v1_col, v2_col in paired_cols:
            measure_name = v1_col.replace('_V1_', '_').replace('_V1', '')
            v1_vals = merged_df[v1_col]
            v2_vals = merged_df[v2_col]
            
            # Calculate raw difference
            diff = v2_vals - v1_vals
            
            # Calculate standard deviation (ignoring NaNs)
            std_dev = np.nanstd(diff)
            if std_dev > 0:
                # Apply hyperbolic tangent transformation
                # This compresses extreme values while preserving directionality
                sigmoid = np.tanh(diff / std_dev)
                sigmoid_transform[f"sigmoid_{measure_name}"] = sigmoid
        
        # Create DataFrame with Name column
        sigmoid_df = pd.DataFrame(sigmoid_transform, index=merged_df.index)
        if 'Name' in merged_df.columns:
            sigmoid_df['Name'] = merged_df['Name'].values
        difference_methods['sigmoid_transform'] = sigmoid_df
        
        # Method 10: Normalized Gain (for bounded measures)
        print("  Calculating normalized gains...")
        normalized_gain = {}
        for v1_col, v2_col in paired_cols:
            measure_name = v1_col.replace('_V1_', '_').replace('_V1', '')
            v1_vals = merged_df[v1_col]
            v2_vals = merged_df[v2_col]
            
            # Calculate theoretical max for each variable (heuristic: max observed + 20%)
            v1_max = np.nanmax(v1_vals)
            v2_max = np.nanmax(v2_vals)
            theoretical_max = max(v1_max, v2_max) * 1.2
            
            # Calculate normalized gain for values below theoretical max
            gain = np.full(len(v1_vals), np.nan)
            
            # Get indices where this calculation makes sense:
            # - v1 is not at ceiling
            # - v1 is positive (for positive metrics)
            valid_idx = (v1_vals < theoretical_max) & (v1_vals > 0)
            
            if np.any(valid_idx):
                # Normalized gain formula: (V2 - V1) / (MAX - V1)
                denominator = theoretical_max - v1_vals[valid_idx]
                # Avoid division by zero
                denominator[denominator == 0] = np.nan
                gain[valid_idx] = (v2_vals[valid_idx] - v1_vals[valid_idx]) / denominator
            
            # Handle negative metrics differently - invert the logic
            neg_metrics_idx = (v1_vals < 0) & (v2_vals < 0)
            if np.any(neg_metrics_idx):
                # For negative metrics, theoretical min is more relevant
                theoretical_min = min(np.nanmin(v1_vals), np.nanmin(v2_vals)) * 1.2
                denominator = v1_vals[neg_metrics_idx] - theoretical_min
                # Avoid division by zero
                denominator[denominator == 0] = np.nan
                # Note negative sign: for negative metrics, more negative = worse
                gain[neg_metrics_idx] = -(v2_vals[neg_metrics_idx] - v1_vals[neg_metrics_idx]) / denominator
            
            normalized_gain[f"norm_gain_{measure_name}"] = gain
        
        # Create DataFrame with Name column
        gain_df = pd.DataFrame(normalized_gain, index=merged_df.index)
        if 'Name' in merged_df.columns:
            gain_df['Name'] = merged_df['Name'].values
        difference_methods['normalized_gain'] = gain_df
        
        # Method 11: Polynomial Transformation
        print("  Calculating polynomial transformations...")
        poly_transform = {}
        for v1_col, v2_col in paired_cols:
            measure_name = v1_col.replace('_V1_', '_').replace('_V1', '')
            v1_vals = merged_df[v1_col]
            v2_vals = merged_df[v2_col]
            
            # Calculate difference
            diff = v2_vals - v1_vals
            
            # Apply composite polynomial transformation
            # sign(diff) × |diff|^0.75 - emphasizes small changes while compressing large ones
            sign_diff = np.sign(diff)
            abs_diff = np.abs(diff)
            
            # Apply transformation with alpha=0.75 (adjustable)
            alpha = 0.75
            poly = sign_diff * np.power(abs_diff, alpha)
            
            poly_transform[f"poly_{measure_name}"] = poly
        
        # Create DataFrame with Name column
        poly_df = pd.DataFrame(poly_transform, index=merged_df.index)
        if 'Name' in merged_df.columns:
            poly_df['Name'] = merged_df['Name'].values
        difference_methods['polynomial_transform'] = poly_df
        
        # Method 12: Rank-Based Percentile Contrasts
        print("  Calculating rank-based percentile contrasts...")
        rank_based = {}
        for v1_col, v2_col in paired_cols:
            measure_name = v1_col.replace('_V1_', '_').replace('_V1', '')
            v1_vals = merged_df[v1_col]
            v2_vals = merged_df[v2_col]
            
            # Calculate raw difference
            diff = v2_vals - v1_vals
            
            # Convert to percentile ranks (0-100)
            # This handles positive/negative automatically through ranking
            valid_indices = ~np.isnan(diff)
            if sum(valid_indices) > 3:  # Need at least a few values
                ranks = np.zeros_like(diff)
                ranks[valid_indices] = pd.Series(diff[valid_indices]).rank(pct=True) * 100
                rank_based[f"rank_pct_{measure_name}"] = ranks
        
        # Create DataFrame with Name column
        rank_df = pd.DataFrame(rank_based, index=merged_df.index)
        if 'Name' in merged_df.columns:
            rank_df['Name'] = merged_df['Name'].values
        difference_methods['rank_percentile'] = rank_df        
        
        # Add pre-contrasted features (already differences)
        precontrast_cols = [col for col in merged_df.columns if col.startswith('precontrast_')]
        if precontrast_cols:
            precontrast_df = merged_df[['Name'] + precontrast_cols].copy()
            difference_methods['precontrasted'] = precontrast_df
            print(f"  Added {len(precontrast_cols)} pre-contrasted features")

        # Add Name for tracking if missing
        for method_name, method_df in difference_methods.items():
            if 'Name' not in method_df.columns and 'Name' in merged_df.columns:
                method_df['Name'] = merged_df['Name'].values
        
        self.difference_methods = difference_methods
        
        print(f"Created {len(difference_methods)} difference calculation methods")
        for method_name, method_df in difference_methods.items():
            print(f"  {method_name}: {method_df.shape[1]-1} features")
        
        return difference_methods
    
    def test_feature_significance(self, difference_methods: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """
        Test which features show significant changes from zero.
        
        Args:
            difference_methods: Results from calculate_differences_multiple_methods
            
        Returns:
            Dictionary mapping method names to lists of significant feature names
        """
        print("\nTesting feature significance...")
        
        significant_features = {}
        
        for method_name, method_df in difference_methods.items():
            if method_name == 'precontrasted':
                # For pre-contrasted data, assume all features are already significant
                feature_cols = [col for col in method_df.columns if col != 'Name']
                significant_features[method_name] = feature_cols
                print(f"  {method_name}: {len(feature_cols)} features (pre-selected)")
                continue
            
            feature_cols = [col for col in method_df.columns if col != 'Name']
            significant_cols = []
            p_values = {}
            
            for col in feature_cols:
                values = method_df[col].dropna()
                if len(values) > 3:  # Need minimum samples for test
                    # Test if mean is significantly different from zero
                    from scipy import stats
                    t_stat, p_val = stats.ttest_1samp(values, 0)
                    p_values[col] = p_val
                    
                    if p_val < self.alpha_threshold:
                        significant_cols.append(col)
            
            significant_features[method_name] = significant_cols
            print(f"  {method_name}: {len(significant_cols)}/{len(feature_cols)} significant features")
            
            # Save p-values for this method
            if not hasattr(self, 'feature_p_values'):
                self.feature_p_values = {}
            self.feature_p_values[method_name] = p_values
        
        return significant_features
    
    def create_composite_weighted_datasets(self, difference_methods: Dict[str, pd.DataFrame], 
                                         significant_features: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
        """
        Create composite datasets with modality weighting.
        
        Args:
            difference_methods: Difference calculation results
            significant_features: Significant features for each method
            
        Returns:
            Dictionary mapping method names to weighted composite datasets
        """
        print("\nCreating modality-weighted composite datasets...")
        
        composite_datasets = {}
        
        for method_name, method_df in difference_methods.items():
            if method_name not in significant_features:
                continue
                
            sig_features = significant_features[method_name]
            if len(sig_features) == 0:
                continue
            
            # Check for required columns and handle missing ones
            available_cols = []
            for col in sig_features:
                if col in method_df.columns:
                    available_cols.append(col)
                else:
                    print(f"  Warning: Feature '{col}' not found in {method_name} dataset")
            
            if len(available_cols) == 0:
                print(f"  Warning: No significant features found in {method_name} dataset")
                continue
            
            # Create a copy with available features
            if 'Name' in method_df.columns:
                weighted_features = method_df[['Name'] + available_cols].copy()
            else:
                # Handle missing Name column
                print(f"  Warning: 'Name' column missing in {method_name} dataset, creating index")
                weighted_features = method_df[available_cols].copy()
                weighted_features['Name'] = np.arange(len(weighted_features)).astype(str)
            
            # Apply weights to available features
            for feature in available_cols:
                # Determine modality from feature name
                modality = None
                for mod_name in self.modality_weights.keys():
                    if mod_name in feature:
                        modality = mod_name
                        break
                
                # Apply modality weight
                if modality and modality in self.modality_weights:
                    weight = self.modality_weights[modality]
                    weighted_features[feature] = weighted_features[feature] * weight
            
            composite_datasets[method_name] = weighted_features
            print(f"  {method_name}: {len(available_cols)} weighted features")
        
        return composite_datasets
    
    def normalize_datasets(self, composite_datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Normalize each dataset to ensure compatibility between different scales.
        
        Args:
            composite_datasets: Weighted composite datasets
            
        Returns:
            Dictionary mapping method names to normalized datasets
        """
        print("\nNormalizing datasets...")
        
        normalized_datasets = {}
        
        for method_name, dataset in composite_datasets.items():
            # Get feature columns (excluding 'Name')
            feature_cols = [col for col in dataset.columns if col != 'Name']
            
            if len(feature_cols) == 0:
                continue
                
            # Create copy with only feature columns
            features_only = dataset[feature_cols].copy()
            
            # Apply robust normalization (min-max scaling based on quantiles)
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            normalized_features = scaler.fit_transform(features_only)
            
            # Create new dataframe with normalized features
            normalized_df = pd.DataFrame(normalized_features, columns=feature_cols, index=dataset.index)
            
            # Add Name column back
            if 'Name' in dataset.columns:
                normalized_df['Name'] = dataset['Name'].values
            
            normalized_datasets[method_name] = normalized_df
            print(f"  Normalized {method_name}: {normalized_df.shape[1]-1} features")
        
        return normalized_datasets
    
    def save_processed_datasets(self, datasets: Dict[str, pd.DataFrame], output_dir: str = 'processed_data') -> None:
        """
        Save all processed datasets to CSV files.
        
        Args:
            datasets: Dictionary of datasets to save
            output_dir: Directory to save files
        """
        print(f"\nSaving processed datasets to {output_dir}/...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        for method_name, dataset in datasets.items():
            filename = f"{output_dir}/{method_name}_processed.csv"
            dataset.to_csv(filename, index=False)
            print(f"  Saved {method_name} to {filename}")
    
    def process_all_data(self, file_paths: List[str], precomputed_contrasts: Optional[List[str]] = None, 
                        output_dir: str = 'processed_data') -> Dict:
        """
        Run complete data processing pipeline.
        
        Args:
            file_paths: List of file paths to uncontrasted data
            precomputed_contrasts: List of file paths to pre-contrasted data
            output_dir: Directory to save processed data
            
        Returns:
            Dictionary containing all processed data
        """
        # Step 1: Load and merge modalities
        merged_df, modality_presence, modality_weights = self.load_and_merge_modalities(
            file_paths, precomputed_contrasts
        )
        
        # Step 2: Calculate differences using multiple methods
        difference_methods = self.calculate_differences_multiple_methods(merged_df)
        
        # Step 3: Test feature significance
        significant_features = self.test_feature_significance(difference_methods)
        
        # Step 4: Create weighted composite datasets
        composite_datasets = self.create_composite_weighted_datasets(
            difference_methods, significant_features
        )
        
        # Step 5: Normalize datasets
        normalized_datasets = self.normalize_datasets(composite_datasets)
        
        # Step 6: Save processed datasets
        self.save_processed_datasets(normalized_datasets, output_dir)
        
        # Return all results
        return {
            'merged_df': merged_df,
            'modality_presence': modality_presence,
            'modality_weights': modality_weights,
            'difference_methods': difference_methods,
            'significant_features': significant_features,
            'composite_datasets': composite_datasets,
            'normalized_datasets': normalized_datasets
        }


# Example usage
if __name__ == "__main__":
    # Initialize the processor
    processor = DataProcessor(alpha_threshold=0.05)
    
    # Define file paths
    file_paths = [
        'statistics/lci_stats.csv',
        'statistics/pwa_stats.csv',
        'statistics/nih_stats.csv', 
        'statistics/nir_ftp_stat.csv',
        'statistics/nir_nbk_stat.csv',
        'statistics/combined_nback_results.csv',
        #'statistics/gly_stats.csv'
    ]
    
    # Pre-contrasted data
    precomputed_contrasts = [
        'statistics/nir_ftp_cntr.csv',
        'statistics/nir_nbk_cntr.csv'
    ]
    
    # Process all data
    results = processor.process_all_data(
        file_paths=file_paths,
        precomputed_contrasts=precomputed_contrasts,
        output_dir='processed_data'
    )
    
    print("\nData processing complete!")
    print(f"Processed {len(results['normalized_datasets'])} contrast methods")