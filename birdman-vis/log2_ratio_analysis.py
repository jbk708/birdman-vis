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


def get_top_features_by_tumor_effect(beta_var_df: pd.DataFrame, n_features: int = 10) -> Tuple[List[str], List[str]]:
    """
    Extract top positive and negative features based on tumor_type effects.
    
    Parameters:
    -----------
    beta_var_df : pd.DataFrame
        BIRDMAn beta_var output
    n_features : int
        Number of top features to select
        
    Returns:
    --------
    Tuple[List[str], List[str]] : Lists of top positive and negative feature names
    """
    # Find tumor_type effect columns
    tumor_cols = [col for col in beta_var_df.columns if 'tumor_type' in col and not col.endswith('_var')]
    
    if not tumor_cols:
        raise Log2RatioError("No tumor_type effect columns found in beta_var data")
    
    # Calculate mean effect across all tumor types for each feature
    mean_effects = beta_var_df[tumor_cols].mean(axis=1)
    
    # Get top positive effects
    positive_features = mean_effects[mean_effects > 0].nlargest(n_features).index.tolist()
    
    # Get top negative effects (most negative)
    negative_features = mean_effects[mean_effects < 0].nsmallest(n_features).index.tolist()
    
    return positive_features, negative_features


def calculate_log2_ratios_per_sample(biom_df: pd.DataFrame, metadata_df: pd.DataFrame,
                                   positive_features: List[str], negative_features: List[str],
                                   logger: logging.Logger) -> Tuple[pd.DataFrame, List[str]]:
    """
    Calculate log2 ratios for each sample based on tumor type.
    
    Parameters:
    -----------
    biom_df : pd.DataFrame
        Feature abundance table
    metadata_df : pd.DataFrame
        Sample metadata
    positive_features : List[str]
        Top positive effect features
    negative_features : List[str]
        Top negative effect features
    logger : logging.Logger
        Logger instance
        
    Returns:
    --------
    Tuple[pd.DataFrame, List[str]] : DataFrame with log2 ratios and list of excluded studies
    """
    # Ensure features exist in biom table
    available_positive = [f for f in positive_features if f in biom_df.index]
    available_negative = [f for f in negative_features if f in biom_df.index]
    
    if len(available_positive) < len(positive_features):
        missing = set(positive_features) - set(available_positive)
        logger.warning(f"Missing positive features in BIOM table: {missing}")
    
    if len(available_negative) < len(negative_features):
        missing = set(negative_features) - set(available_negative)
        logger.warning(f"Missing negative features in BIOM table: {missing}")
    
    if not available_positive or not available_negative:
        raise Log2RatioError("Insufficient features available in BIOM table")
    
    logger.info(f"Using {len(available_positive)} positive and {len(available_negative)} negative features")
    
    # Calculate mean abundances per sample
    positive_means = biom_df.loc[available_positive].mean(axis=0)
    negative_means = biom_df.loc[available_negative].mean(axis=0)
    
    # Merge with metadata
    sample_metadata = metadata_df.set_index(metadata_df.columns[0])  # Assume first column is sample ID
    
    # Filter to samples present in both datasets
    common_samples = positive_means.index.intersection(sample_metadata.index)
    
    if len(common_samples) == 0:
        raise Log2RatioError("No common samples found between BIOM table and metadata")
    
    logger.info(f"Processing {len(common_samples)} common samples")
    
    # Calculate log2 ratios
    results = []
    excluded_studies = []
    
    for sample_id in common_samples:
        if sample_id not in sample_metadata.index:
            continue
            
        tumor_type = sample_metadata.loc[sample_id, 'tumor_type']
        pos_abundance = positive_means[sample_id]
        neg_abundance = negative_means[sample_id]
        
        # Check for zero denominators
        if neg_abundance == 0 or np.isnan(neg_abundance):
            excluded_studies.append(f"{sample_id} (tumor_type: {tumor_type})")
            continue
            
        # Calculate log2 ratio (negative in numerator as requested)
        log2_ratio = np.log2(neg_abundance / pos_abundance)
        
        results.append({
            'sample_id': sample_id,
            'tumor_type': tumor_type,
            'log2_ratio': log2_ratio,
            'positive_abundance': pos_abundance,
            'negative_abundance': neg_abundance
        })
    
    if excluded_studies:
        logger.warning(f"Excluded {len(excluded_studies)} samples due to zero denominators: {excluded_studies}")
    
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        raise Log2RatioError("No valid samples for log2 ratio calculation")
    
    logger.info(f"Calculated log2 ratios for {len(results_df)} samples across {results_df['tumor_type'].nunique()} tumor types")
    
    return results_df, excluded_studies


def plot_log2_ratios(results_df: pd.DataFrame, output_dir: Path, 
                    logger: logging.Logger, figsize: Tuple[int, int] = (14, 8)) -> Optional[Path]:
    """
    Create box and whisker plot with scatter overlay for log2 ratios.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with log2 ratios per sample
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
        
        # Get unique tumor types and sort them
        tumor_types = sorted(results_df['tumor_type'].unique())
        
        # Create box plot
        box_data = [results_df[results_df['tumor_type'] == tt]['log2_ratio'].values for tt in tumor_types]
        
        bp = ax.boxplot(box_data, labels=tumor_types, patch_artist=True, 
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
        
        # Overlay scatter points
        colors = plt.cm.Set1(np.linspace(0, 1, len(tumor_types)))
        
        for i, tumor_type in enumerate(tumor_types):
            tumor_data = results_df[results_df['tumor_type'] == tumor_type]['log2_ratio']
            
            # Add some jitter to x-coordinates for better visibility
            x_jitter = np.random.normal(i + 1, 0.04, size=len(tumor_data))
            
            ax.scatter(x_jitter, tumor_data.values, 
                      c=[colors[i]], alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        
        # Formatting
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Tumor Type', fontweight='bold')
        ax.set_ylabel('Log2 Ratio (Negative Features / Positive Features)', fontweight='bold')
        ax.set_title('Log2 Ratios of Top Tumor Effect Features by Sample', fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels if many tumor types
        if len(tumor_types) > 8:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        output_path = output_dir / "log2_ratio_analysis.svg"
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.3, format='svg')
        plt.close()
        
        logger.info(f"Log2 ratio plot saved: {output_path}")
        
        # Save summary statistics
        summary_path = output_dir / "log2_ratio_summary.tsv"
        summary_stats = results_df.groupby('tumor_type')['log2_ratio'].agg([
            'count', 'mean', 'std', 'min', 'median', 'max'
        ]).round(4)
        summary_stats.to_csv(summary_path, sep='\t')
        logger.info(f"Summary statistics saved: {summary_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating log2 ratio plot: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None


def run_log2_ratio_analysis(beta_var_path: Path, metadata_path: Path, biom_path: Path,
                           output_dir: Path, n_features: int = 10, 
                           figsize: Tuple[int, int] = (14, 8)) -> bool:
    """
    Run complete log2 ratio analysis pipeline.
    
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
        Number of top features to use
    figsize : Tuple[int, int]
        Figure size for plot
        
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
        
        # Get top features
        positive_features, negative_features = get_top_features_by_tumor_effect(
            beta_var_df, n_features
        )
        
        logger.info(f"Top {len(positive_features)} positive features: {[simplify_feature_name(f, 30) for f in positive_features[:3]]}...")
        logger.info(f"Top {len(negative_features)} negative features: {[simplify_feature_name(f, 30) for f in negative_features[:3]]}...")
        
        # Calculate log2 ratios
        results_df, excluded_studies = calculate_log2_ratios_per_sample(
            biom_df, metadata_df, positive_features, negative_features, logger
        )
        
        # Save detailed results
        results_path = output_dir / "log2_ratio_results.tsv"
        results_df.to_csv(results_path, sep='\t', index=False)
        logger.info(f"Detailed results saved: {results_path}")
        
        # Create visualization
        plot_path = plot_log2_ratios(results_df, output_dir, logger, figsize)
        
        if plot_path:
            logger.info("Log2 ratio analysis completed successfully")
            return True
        else:
            logger.error("Plot generation failed")
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