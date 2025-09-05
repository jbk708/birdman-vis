#!/usr/bin/env python3
"""
Log2 Ratio Analysis Module

Calculates log2 ratios between top positive and negative tumor type effects
and generates box plots with scatter overlay for visualization.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse

from utils import setup_logging, simplify_feature_name

plt.rcParams.update({
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "savefig.bbox": "tight"
})


class Log2RatioError(Exception):
    """Custom exception for log2 ratio analysis errors."""
    pass


def load_biom_table(biom_path: Path) -> pd.DataFrame:
    """
    Load BIOM table and convert to DataFrame.
    
    Parameters:
    -----------
    biom_path : Path
        Path to BIOM table file
        
    Returns:
    --------
    pd.DataFrame : Feature abundance table
    """
    try:
        import biom
    except ImportError:
        raise ImportError("biom-format package required. Install with: pip install biom-format")
    
    table = biom.load_table(str(biom_path))
    
    # Convert to dense DataFrame
    df = pd.DataFrame(
        table.matrix_data.toarray() if sparse.issparse(table.matrix_data) else table.matrix_data,
        index=table.ids('observation'),
        columns=table.ids('sample')
    )
    
    return df


def validate_inputs(beta_var_path: Path, metadata_path: Path, biom_path: Path, 
                   logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Validate and load all input files.
    
    Parameters:
    -----------
    beta_var_path : Path
        Path to BIRDMAn beta_var.tsv file
    metadata_path : Path
        Path to metadata.tsv file  
    biom_path : Path
        Path to BIOM table file
    logger : logging.Logger
        Logger instance
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] : beta_var, metadata, biom DataFrames
    
    Raises:
    -------
    FileNotFoundError : If any input file doesn't exist
    ValueError : If file format is invalid
    """
    # Check file existence
    for path in [beta_var_path, metadata_path, biom_path]:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
    
    logger.info("Loading input files...")
    
    # Load beta_var file
    try:
        beta_var_df = pd.read_csv(beta_var_path, sep='\t', index_col=0)
        logger.info(f"Loaded beta_var with shape: {beta_var_df.shape}")
    except Exception as e:
        raise ValueError(f"Failed to read beta_var file: {e}")
    
    # Load metadata
    try:
        metadata_df = pd.read_csv(metadata_path, sep='\t')
        logger.info(f"Loaded metadata with shape: {metadata_df.shape}")
    except Exception as e:
        raise ValueError(f"Failed to read metadata file: {e}")
    
    # Load BIOM table
    try:
        biom_df = load_biom_table(biom_path)
        logger.info(f"Loaded BIOM table with shape: {biom_df.shape}")
    except Exception as e:
        raise ValueError(f"Failed to read BIOM file: {e}")
    
    # Validate required columns
    if 'tumor_type' not in metadata_df.columns:
        raise ValueError("metadata.tsv must contain 'tumor_type' column")
    
    # Check for tumor_type columns in beta_var
    tumor_cols = [col for col in beta_var_df.columns if 'tumor_type' in col]
    if not tumor_cols:
        raise ValueError("beta_var.tsv must contain tumor_type effect columns")
    
    logger.info("Input validation successful")
    return beta_var_df, metadata_df, biom_df


def extract_tumor_types_from_columns(beta_var_df: pd.DataFrame, 
                                    logger: logging.Logger = None) -> Dict[str, str]:
    """
    Extract tumor types from BIRDMAn column names.
    
    Parameters:
    -----------
    beta_var_df : pd.DataFrame
        BIRDMAn beta_var output
    logger : logging.Logger
        Logger instance
        
    Returns:
    --------
    Dict[str, str] : Mapping of tumor_type -> column_name
    """
    import re
    
    # Pattern to match: C(tumor_type, Treatment('cancer'))[T.TUMOR_TYPE]_mean
    pattern = r"C\(tumor_type,\s*Treatment\('([^']+)'\)\)\[T\.([^\]]+)\]_mean"
    
    tumor_type_cols = {}
    
    for col in beta_var_df.columns:
        match = re.match(pattern, col)
        if match:
            reference_level = match.group(1)  # Should be 'cancer'
            tumor_type = match.group(2)       # e.g., 'adjacent', 'healthy'
            tumor_type_cols[tumor_type] = col
    
    if logger:
        logger.info(f"Found tumor type columns: {list(tumor_type_cols.keys())}")
        logger.info(f"Reference level appears to be: {reference_level if 'reference_level' in locals() else 'cancer'}")
    
    return tumor_type_cols


def get_top_features_by_tumor_effect(beta_var_df: pd.DataFrame, tumor_type_cols: Dict[str, str],
                                    n_features: int = 10, logger: logging.Logger = None) -> Dict[str, Tuple[List[str], List[str]]]:
    """
    Extract top positive and negative features for each tumor type comparison.
    
    Parameters:
    -----------
    beta_var_df : pd.DataFrame
        BIRDMAn beta_var output
    tumor_type_cols : Dict[str, str]
        Mapping of tumor_type -> column_name
    n_features : int
        Number of top features to select per comparison
    logger : logging.Logger
        Logger instance
        
    Returns:
    --------
    Dict[str, Tuple[List[str], List[str]]] : For each tumor type, (positive_features, negative_features)
    """
    results = {}
    
    for tumor_type, col_name in tumor_type_cols.items():
        if logger:
            logger.info(f"Processing {tumor_type} vs cancer comparison...")
        
        try:
            # Get effect sizes (cancer is reference, so positive = higher in cancer)
            effects = pd.to_numeric(beta_var_df[col_name], errors='coerce').dropna()
            
            if effects.empty:
                logger.warning(f"No numeric data for {tumor_type} comparison")
                continue
            
            # Get top features where cancer > tumor_type (positive effects)
            positive_effects = effects[effects > 0]
            positive_features = positive_effects.nlargest(min(n_features, len(positive_effects))).index.tolist()
            
            # Get top features where tumor_type > cancer (negative effects)  
            negative_effects = effects[effects < 0]
            negative_features = negative_effects.nsmallest(min(n_features, len(negative_effects))).index.tolist()
            
            results[tumor_type] = (positive_features, negative_features)
            
            if logger:
                logger.info(f"  Found {len(positive_features)} cancer-enriched, {len(negative_features)} {tumor_type}-enriched features")
                
        except Exception as e:
            logger.error(f"Error processing {tumor_type}: {e}")
            continue
    
    return results


def calculate_log2_ratios_per_sample(biom_df: pd.DataFrame, metadata_df: pd.DataFrame,
                                   tumor_type: str, cancer_features: List[str], 
                                   other_features: List[str], logger: logging.Logger) -> Tuple[pd.DataFrame, List[str]]:
    """
    Calculate log2 ratios for each sample for a specific tumor type comparison.
    
    Parameters:
    -----------
    biom_df : pd.DataFrame
        Feature abundance table
    metadata_df : pd.DataFrame
        Sample metadata
    tumor_type : str
        The tumor type being compared to cancer (e.g., 'adjacent', 'healthy')
    cancer_features : List[str]
        Features enriched in cancer samples
    other_features : List[str]
        Features enriched in the other tumor type
    logger : logging.Logger
        Logger instance
        
    Returns:
    --------
    Tuple[pd.DataFrame, List[str]] : DataFrame with log2 ratios and list of excluded studies
    """
    # Ensure features exist in biom table
    available_cancer = [f for f in cancer_features if f in biom_df.index]
    available_other = [f for f in other_features if f in biom_df.index]
    
    if len(available_cancer) < len(cancer_features):
        missing = set(cancer_features) - set(available_cancer)
        logger.warning(f"Missing cancer-enriched features in BIOM table: {missing}")
    
    if len(available_other) < len(other_features):
        missing = set(other_features) - set(available_other)
        logger.warning(f"Missing {tumor_type}-enriched features in BIOM table: {missing}")
    
    if not available_cancer or not available_other:
        raise Log2RatioError("Insufficient features available in BIOM table")
    
    logger.info(f"Using {len(available_cancer)} cancer-enriched and {len(available_other)} {tumor_type}-enriched features")
    
    # Calculate mean abundances per sample
    cancer_means = biom_df.loc[available_cancer].mean(axis=0)
    other_means = biom_df.loc[available_other].mean(axis=0)
    
    # Debug sample ID formats
    logger.info(f"BIOM sample IDs (first 5): {list(cancer_means.index[:5])}")
    logger.info(f"Metadata shape: {metadata_df.shape}")
    logger.info(f"Metadata columns: {list(metadata_df.columns)}")
    logger.info(f"Metadata index: {list(metadata_df.index[:5])}")
    
    # Try different approaches to match sample IDs
    sample_metadata = None
    common_samples = []
    
    # Approach 1: Use first column as sample ID
    if len(metadata_df.columns) > 0:
        first_col_values = list(metadata_df[metadata_df.columns[0]][:5])
        logger.info(f"First column values: {first_col_values}")
        
        temp_metadata = metadata_df.set_index(metadata_df.columns[0])
        temp_common = cancer_means.index.intersection(temp_metadata.index)
        logger.info(f"Common samples using first column: {len(temp_common)}")
        
        if len(temp_common) > 0:
            sample_metadata = temp_metadata
            common_samples = temp_common
    
    # Approach 2: Use index if first column didn't work
    if len(common_samples) == 0:
        temp_common = cancer_means.index.intersection(metadata_df.index)
        logger.info(f"Common samples using index: {len(temp_common)}")
        
        if len(temp_common) > 0:
            sample_metadata = metadata_df.copy()
            common_samples = temp_common
    
    # Approach 3: Try string matching (in case of slight format differences)
    if len(common_samples) == 0:
        logger.info("Trying string-based matching...")
        biom_samples = set(str(s) for s in cancer_means.index)
        
        # Check first column
        if len(metadata_df.columns) > 0:
            metadata_samples = set(str(s) for s in metadata_df[metadata_df.columns[0]])
            intersection = biom_samples.intersection(metadata_samples)
            logger.info(f"String matching with first column: {len(intersection)} matches")
            
            if len(intersection) > 0:
                # Create mapping
                sample_metadata = metadata_df.set_index(metadata_df.columns[0])
                common_samples = [s for s in cancer_means.index if str(s) in metadata_samples]
        
        # Check index  
        if len(common_samples) == 0:
            metadata_samples = set(str(s) for s in metadata_df.index)
            intersection = biom_samples.intersection(metadata_samples)
            logger.info(f"String matching with index: {len(intersection)} matches")
            
            if len(intersection) > 0:
                sample_metadata = metadata_df.copy()
                common_samples = [s for s in cancer_means.index if str(s) in metadata_samples]
    
    if len(common_samples) == 0 or sample_metadata is None:
        raise Log2RatioError("No common samples found between BIOM table and metadata")
    
    logger.info(f"Processing {len(common_samples)} common samples")
    
    # Calculate log2 ratios
    results = []
    excluded_studies = []
    
    for sample_id in common_samples:
        if sample_id not in sample_metadata.index:
            continue
            
        sample_tumor_type = sample_metadata.loc[sample_id, 'tumor_type']
        study_id = sample_metadata.loc[sample_id, 'qiita_study_id'] if 'qiita_study_id' in sample_metadata.columns else 'unknown'
        cancer_abundance = cancer_means[sample_id]
        other_abundance = other_means[sample_id]
        
        # Check for zero denominators
        if other_abundance == 0 or np.isnan(other_abundance):
            excluded_studies.append(f"{sample_id} (tumor_type: {sample_tumor_type})")
            continue
            
        # Calculate log2 ratio: log2(cancer_abundance / other_abundance)
        log2_ratio = np.log2(cancer_abundance / other_abundance)
        
        results.append({
            'sample_id': sample_id,
            'tumor_type': sample_tumor_type,
            'qiita_study_id': study_id,
            'comparison': f'cancer_vs_{tumor_type}',
            'log2_ratio': log2_ratio,
            'cancer_abundance': cancer_abundance,
            'other_abundance': other_abundance
        })
    
    if excluded_studies:
        logger.warning(f"Excluded {len(excluded_studies)} samples due to zero denominators: {excluded_studies}")
    
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        raise Log2RatioError("No valid samples for log2 ratio calculation")
    
    logger.info(f"Calculated log2 ratios for {len(results_df)} samples comparing cancer vs {tumor_type}")
    
    return results_df, excluded_studies


def plot_individual_comparison(results_df: pd.DataFrame, tumor_type: str, 
                              output_dir: Path, logger: logging.Logger, 
                              figsize: Tuple[int, int] = (14, 8)) -> Optional[Path]:
    """
    Create box plot for a single tumor type comparison with two boxes per study (cancer vs comparator).
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with log2 ratios per sample
    tumor_type : str
        The tumor type being compared (e.g., 'adjacent')
    output_dir : Path
        Output directory for plot
    logger : logging.Logger
        Logger instance
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    Optional[Path] : Path to saved plot file
    """
    try:
        # Filter to only include cancer and the specific comparator tumor type
        filtered_df = results_df[results_df['tumor_type'].isin(['cancer', tumor_type])]
        
        if filtered_df.empty:
            logger.warning(f"No cancer or {tumor_type} samples found for comparison")
            return None
        
        logger.info(f"Filtered to {len(filtered_df)} samples with tumor types: cancer, {tumor_type}")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique study IDs and sort them
        study_ids = sorted(filtered_df['qiita_study_id'].unique())
        
        # Prepare data for paired box plots (side-by-side)
        box_positions = []
        box_data = []
        box_labels = []
        colors = []
        study_labels = []
        
        # Colors for cancer and comparator
        cancer_color = 'lightcoral'
        comparator_color = 'lightblue'
        
        base_position = 0
        box_width = 0.4
        spacing_between_pairs = 0.2
        spacing_between_studies = 1.0
        
        for i, study_id in enumerate(study_ids):
            study_data = filtered_df[filtered_df['qiita_study_id'] == study_id]
            
            # Calculate positions for this study (side-by-side)
            cancer_pos = base_position - box_width/2
            comparator_pos = base_position + box_width/2
            
            study_has_data = False
            
            # Cancer samples for this study
            cancer_data = study_data[study_data['tumor_type'] == 'cancer']['log2_ratio'].values
            if len(cancer_data) > 0:
                box_data.append(cancer_data)
                box_positions.append(cancer_pos)
                colors.append(cancer_color)
                study_has_data = True
            
            # Comparator samples for this study  
            comparator_data = study_data[study_data['tumor_type'] == tumor_type]['log2_ratio'].values
            if len(comparator_data) > 0:
                box_data.append(comparator_data)
                box_positions.append(comparator_pos)
                colors.append(comparator_color)
                study_has_data = True
            
            # Only add study label if we have data
            if study_has_data:
                study_labels.append((base_position, f'Study {study_id}'))
            
            # Move to next study position
            base_position += spacing_between_studies
        
        if not box_data:
            logger.warning(f"No valid data found for {tumor_type} comparison")
            return None
        
        # Create box plots
        bp = ax.boxplot(box_data, positions=box_positions, patch_artist=True,
                       boxprops=dict(alpha=0.7),
                       medianprops=dict(color='red', linewidth=2),
                       widths=box_width)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        # Overlay scatter points (one point per sample)
        for i, (pos, data) in enumerate(zip(box_positions, box_data)):
            # Add small jitter to x-coordinates for visibility
            x_jitter = np.random.normal(pos, 0.02, size=len(data))
            ax.scatter(x_jitter, data, c='black', alpha=0.6, s=20, edgecolors='white', linewidth=0.5)
        
        # Create custom legend showing only cancer and comparator
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor=cancer_color, alpha=0.7, label='Cancer'),
            plt.Rectangle((0,0),1,1, facecolor=comparator_color, alpha=0.7, label=tumor_type.title())
        ]
        ax.legend(handles=legend_elements, title='Tumor Type', 
                 bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Formatting
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Set x-axis labels to show study IDs at center of each pair
        study_positions = [pos for pos, label in study_labels]
        study_names = [label for pos, label in study_labels]
        ax.set_xticks(study_positions)
        ax.set_xticklabels(study_names, fontsize=10)
        
        ax.set_xlabel('Study ID', fontweight='bold')
        ax.set_ylabel('Log2 Ratio (Cancer Features / Other Features)', fontweight='bold')
        ax.set_title(f'Cancer vs {tumor_type.title()} Feature Abundance Ratios by Study', 
                    fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        output_path = output_dir / f"log2_ratio_cancer_vs_{tumor_type}.svg"
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.3, format='svg')
        plt.close()
        
        logger.info(f"Individual comparison plot saved: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating individual comparison plot: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None


def plot_combined_comparison(all_results: Dict[str, pd.DataFrame], 
                           output_dir: Path, logger: logging.Logger,
                           figsize: Tuple[int, int] = (16, 10)) -> Optional[Path]:
    """
    Create combined bar plot showing mean log2 ratios across all comparisons grouped by study.
    
    Parameters:
    -----------
    all_results : Dict[str, pd.DataFrame]
        Dictionary mapping tumor_type -> results DataFrame
    output_dir : Path
        Output directory for plot
    logger : logging.Logger
        Logger instance
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    Optional[Path] : Path to saved plot file
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data for combined plot grouped by study
        plot_data = []
        
        for comparison_type, results_df in all_results.items():
            # Calculate mean log2 ratio by study ID
            means = results_df.groupby('qiita_study_id')['log2_ratio'].mean()
            stds = results_df.groupby('qiita_study_id')['log2_ratio'].std()
            
            for study_id, mean_ratio in means.items():
                plot_data.append({
                    'comparison': f'cancer_vs_{comparison_type}',
                    'qiita_study_id': study_id,
                    'mean_log2_ratio': mean_ratio,
                    'std_log2_ratio': stds[study_id] if not pd.isna(stds[study_id]) else 0
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create grouped bar plot
        comparisons = sorted(plot_df['comparison'].unique())
        study_ids = sorted(plot_df['qiita_study_id'].unique())
        
        x = np.arange(len(study_ids))
        width = 0.8 / len(comparisons)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(comparisons)))
        
        for i, comparison in enumerate(comparisons):
            data = plot_df[plot_df['comparison'] == comparison]
            means = []
            stds = []
            
            for study_id in study_ids:
                study_data = data[data['qiita_study_id'] == study_id]
                if len(study_data) > 0:
                    means.append(study_data['mean_log2_ratio'].iloc[0])
                    stds.append(study_data['std_log2_ratio'].iloc[0])
                else:
                    means.append(0)
                    stds.append(0)
            
            ax.bar(x + i * width, means, width, yerr=stds, 
                  label=comparison.replace('cancer_vs_', '').replace('_', ' ').title(), 
                  alpha=0.8, capsize=3, color=colors[i])
        
        # Formatting
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Qiita Study ID', fontweight='bold')
        ax.set_ylabel('Mean Log2 Ratio (Cancer / Other)', fontweight='bold')
        ax.set_title('Combined Analysis: Cancer vs Other Tumor Types by Study', 
                    fontweight='bold', pad=20)
        ax.set_xticks(x + width * (len(comparisons) - 1) / 2)
        ax.set_xticklabels([f'Study {sid}' for sid in study_ids], rotation=45, ha='right')
        ax.legend(title='Comparison Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = output_dir / "log2_ratio_combined_analysis.svg"
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.3, format='svg')
        plt.close()
        
        logger.info(f"Combined comparison plot saved: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating combined comparison plot: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None


def run_log2_ratio_analysis(beta_var_path: Path, metadata_path: Path, biom_path: Path,
                           output_dir: Path, n_features: int = 10, 
                           figsize: Tuple[int, int] = (14, 8)) -> bool:
    """
    Run complete log2 ratio analysis pipeline with individual and combined plots.
    
    Parameters:
    -----------
    beta_var_path : Path
        Path to BIRDMAn beta_var.tsv file
    metadata_path : Path
        Path to metadata.tsv file
    biom_path : Path
        Path to BIOM table file
    output_dir : Path
        Output directory
    n_features : int
        Number of top features to use per comparison
    figsize : Tuple[int, int]
        Figure size for plots
        
    Returns:
    --------
    bool : Success status
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logging(output_dir)
        
        logger.info("Starting log2 ratio analysis pipeline")
        
        # Load and validate inputs
        beta_var_df, metadata_df, biom_df = validate_inputs(
            beta_var_path, metadata_path, biom_path, logger
        )
        
        # Extract tumor types from column names
        tumor_type_cols = extract_tumor_types_from_columns(beta_var_df, logger)
        
        if not tumor_type_cols:
            logger.error("No tumor type columns found in beta_var file")
            return False
        
        # Get top features for each comparison
        all_features = get_top_features_by_tumor_effect(
            beta_var_df, tumor_type_cols, n_features, logger
        )
        
        if not all_features:
            logger.error("No valid tumor type comparisons found")
            return False
        
        # Process each tumor type comparison
        all_results = {}
        generated_plots = []
        
        for tumor_type, (cancer_features, other_features) in all_features.items():
            logger.info(f"Processing {tumor_type} comparison...")
            logger.info(f"  Cancer-enriched features: {[simplify_feature_name(f, 30) for f in cancer_features[:3]]}...")
            logger.info(f"  {tumor_type.title()}-enriched features: {[simplify_feature_name(f, 30) for f in other_features[:3]]}...")
            
            try:
                # Calculate log2 ratios for this comparison
                results_df, excluded_studies = calculate_log2_ratios_per_sample(
                    biom_df, metadata_df, tumor_type, cancer_features, other_features, logger
                )
                
                if results_df.empty:
                    logger.warning(f"No valid samples for {tumor_type} comparison")
                    continue
                
                all_results[tumor_type] = results_df
                
                # Save detailed results
                results_path = output_dir / f"log2_ratio_results_cancer_vs_{tumor_type}.tsv"
                results_df.to_csv(results_path, sep='\t', index=False)
                logger.info(f"Detailed results saved: {results_path}")
                
                # Create individual plot
                individual_plot = plot_individual_comparison(
                    results_df, tumor_type, output_dir, logger, figsize
                )
                if individual_plot:
                    generated_plots.append(individual_plot)
                
                # Save summary statistics by study ID
                summary_path = output_dir / f"log2_ratio_summary_cancer_vs_{tumor_type}.tsv"
                summary_stats = results_df.groupby('qiita_study_id')['log2_ratio'].agg([
                    'count', 'mean', 'std', 'min', 'median', 'max'
                ]).round(4)
                summary_stats.to_csv(summary_path, sep='\t')
                logger.info(f"Summary statistics saved: {summary_path}")
                
                # Log the specific features used
                logger.info(f"Features used for {tumor_type} comparison:")
                logger.info(f"  Cancer-enriched features: {cancer_features}")
                logger.info(f"  {tumor_type.title()}-enriched features: {other_features}")
                
            except Exception as e:
                logger.error(f"Failed to process {tumor_type} comparison: {e}")
                continue
        
        # Create combined plot if multiple comparisons
        if len(all_results) > 1:
            combined_plot = plot_combined_comparison(
                all_results, output_dir, logger, figsize
            )
            if combined_plot:
                generated_plots.append(combined_plot)
        
        if generated_plots:
            logger.info(f"Log2 ratio analysis completed successfully. Generated {len(generated_plots)} plots:")
            for plot_path in generated_plots:
                logger.info(f"  - {plot_path}")
            return True
        else:
            logger.error("No plots were generated")
            return False
            
    except Exception as e:
        logger.error(f"Log2 ratio analysis failed: {e}")
        return False


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for standalone usage."""
    parser = argparse.ArgumentParser(
        description="Log2 Ratio Analysis for BIRDMAn Results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument("beta_var", type=Path, help="BIRDMAn beta_var.tsv file")
    parser.add_argument("metadata", type=Path, help="Sample metadata.tsv file")
    parser.add_argument("biom_table", type=Path, help="Feature abundance BIOM table")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    
    # Optional arguments
    parser.add_argument("--top-n", type=int, default=10,
                       help="Number of top positive/negative features to use")
    parser.add_argument("--figsize", nargs=2, type=int, default=[14, 8],
                       metavar=("W", "H"), help="Figure size (width, height)")
    
    return parser.parse_args()


def main():
    """Main execution function for standalone usage."""
    args = parse_arguments()
    
    success = run_log2_ratio_analysis(
        args.beta_var,
        args.metadata,
        args.biom_table,
        args.output_dir,
        args.top_n,
        tuple(args.figsize)
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()