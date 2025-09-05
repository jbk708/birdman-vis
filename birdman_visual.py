#!/usr/bin/env python3
# birdman_visual.py
"""
BIRDMAn Output Visualization Tool

Generates forest plots, heatmaps, and comparative analyses for differential abundance results.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional

from utils import (
    get_available_cancer_types,
    process_birdman_output,
    setup_logging,
    validate_input_file,
)

from plots import (
    plot_condition_effects,
    plot_combined_effects, 
    plot_single_condition_effects,
    plot_primer_effects,
    plot_upset_analysis,
)


class PlotConfig:
    """Configuration for plot generation."""
    def __init__(self, name: str, func: Callable, condition_check: Callable = None, 
                 significant_version: bool = False, **kwargs):
        self.name = name
        self.func = func
        self.condition_check = condition_check or (lambda df: True)
        self.significant_version = significant_version
        self.kwargs = kwargs


def create_plot_configs(args, cancer_types: List[str]) -> List[PlotConfig]:
    """Create plot configurations based on arguments."""
    
    def has_tumor_cols(df):
        return any("tumor_type" in col and col.endswith("_mean") for col in df.columns)
    
    def has_cancer_cols(df):
        return any("cancer_type[T." in col and col.endswith("_mean") for col in df.columns)
    
    configs = [
        # Regular plots
        PlotConfig("tumor_effects", plot_condition_effects, has_tumor_cols,
                  condition_type="tumor_type", top_n=args.top_n, 
                  figsize=tuple(args.tumor_figsize)),
        
        PlotConfig("combined_tumor_effects", plot_combined_effects, has_tumor_cols,
                  condition_type="tumor_type", top_n=args.top_n,
                  figsize=tuple(args.tumor_figsize)),
        
        PlotConfig("primer_effects", plot_primer_effects,
                  top_n=args.top_n, figsize=tuple(args.primer_figsize)),
        
        PlotConfig("upset_analysis", plot_upset_analysis, has_cancer_cols,
                  min_effect=0.9),
        
        # Significant-only plots
        PlotConfig("tumor_effects_significant", plot_condition_effects, has_tumor_cols,
                  significant_version=True, condition_type="tumor_type", 
                  top_n=args.top_n, figsize=tuple(args.tumor_figsize), 
                  significant_only=True),
        
        PlotConfig("combined_tumor_effects_significant", plot_combined_effects, has_tumor_cols,
                  significant_version=True, condition_type="tumor_type",
                  top_n=args.top_n, figsize=tuple(args.tumor_figsize),
                  significant_only=True),
    ]
    
    # Individual cancer plots
    for cancer_type in cancer_types:
        configs.extend([
            PlotConfig(f"individual_cancer_{cancer_type}", plot_single_condition_effects,
                      has_cancer_cols, condition_type="cancer_type", 
                      condition_name=cancer_type, top_n=args.top_n,
                      figsize=tuple(args.single_cancer_figsize)),
            
            PlotConfig(f"individual_cancer_{cancer_type}_significant", 
                      plot_single_condition_effects, has_cancer_cols,
                      significant_version=True, condition_type="cancer_type",
                      condition_name=cancer_type, top_n=args.top_n,
                      figsize=tuple(args.single_cancer_figsize), significant_only=True),
        ])
    
    return configs


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="BIRDMAn Output Visualization Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("input_file", type=Path, help="BIRDMAn output TSV file")
    parser.add_argument("output_dir", type=Path, help="Output directory")

    # General options
    parser.add_argument("--top-n", type=int, default=100, 
                       help="Number of top features to display")
    parser.add_argument("--effect-threshold", type=float, default=0.5,
                       help="Minimum absolute effect size threshold")
    parser.add_argument("--cancer-types", nargs="*", default=None,
                       help="Specific cancer types (default: auto-detect)")

    # Figure sizes
    size_group = parser.add_argument_group("Figure sizes")
    size_group.add_argument("--tumor-figsize", nargs=2, type=int, default=[24, 16],
                           metavar=("W", "H"), help="Tumor effects figure size")
    size_group.add_argument("--primer-figsize", nargs=2, type=int, default=[14, 10],
                           metavar=("W", "H"), help="Primer effects figure size")  
    size_group.add_argument("--single-cancer-figsize", nargs=2, type=int, default=[14, 16],
                           metavar=("W", "H"), help="Single cancer figure size")

    # Plot selection
    plot_group = parser.add_argument_group("Plot selection")
    plots = ["tumor-effects", "combined-tumor-effects", "primer-effects", 
             "upset-analysis", "individual-cancers"]
    sig_plots = ["tumor-effects-significant", "combined-tumor-effects-significant",
                 "individual-cancers-significant"]
    
    for plot in plots + sig_plots:
        plot_group.add_argument(f"--{plot}", action="store_true",
                               help=f"Generate {plot.replace('-', ' ')} plot")
    
    plot_group.add_argument("--include-significant", action="store_true",
                           help="Include significant-only versions of all plots")

    return parser.parse_args()


def get_requested_plots(args) -> set:
    """Get set of requested plot names from arguments."""
    requested = set()
    
    plot_mapping = {
        'tumor_effects': args.tumor_effects,
        'combined_tumor_effects': args.combined_tumor_effects,
        'primer_effects': args.primer_effects,
        'upset_analysis': args.upset_analysis,
        'individual_cancers': args.individual_cancers,
        'tumor_effects_significant': args.tumor_effects_significant,
        'combined_tumor_effects_significant': args.combined_tumor_effects_significant,
        'individual_cancers_significant': args.individual_cancers_significant,
    }
    
    for plot_name, flag in plot_mapping.items():
        if flag:
            requested.add(plot_name)
    
    return requested


def should_generate_plot(config: PlotConfig, requested_plots: set, 
                        generate_all: bool, include_significant: bool) -> bool:
    """Determine if a plot should be generated based on configuration and flags."""
    
    # Handle individual cancer plots
    if config.name.startswith("individual_cancer_"):
        base_name = "individual_cancers_significant" if config.significant_version else "individual_cancers"
        return (generate_all and (not config.significant_version or include_significant)) or base_name in requested_plots
    
    # Handle regular plots
    if config.significant_version:
        return (generate_all and include_significant) or config.name in requested_plots
    else:
        return generate_all or config.name in requested_plots


def generate_plot(config: PlotConfig, df, output_dir, logger) -> Optional[Path]:
    """Generate a single plot with error handling."""
    try:
        if not config.condition_check(df):
            logger.warning(f"Skipping {config.name}: data requirements not met")
            return None
            
        result = config.func(df, output_dir=output_dir, logger=logger, **config.kwargs)
        if result:
            logger.info(f"Generated: {result.name}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to generate {config.name}: {e}")
        return None


def main():
    """Main execution function."""
    args = parse_arguments()
    
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logging(args.output_dir)
        
        logger.info("Starting BIRDMAn visualization pipeline")
        
        # Load and process data
        df = validate_input_file(args.input_file, logger)
        df_processed = process_birdman_output(df, logger)
        
        # Determine cancer types
        if args.cancer_types is None:
            cancer_types = get_available_cancer_types(df_processed)
            logger.info(f"Auto-detected cancer types: {cancer_types}")
        else:
            cancer_types = args.cancer_types
            logger.info(f"Using specified cancer types: {cancer_types}")
        
        # Log data structure info
        tumor_cols = [col for col in df_processed.columns 
                     if "tumor_type" in col and col.endswith("_mean")]
        cancer_cols = [col for col in df_processed.columns 
                      if "cancer_type[T." in col and col.endswith("_mean")]
        
        logger.info(f"Data contains {len(tumor_cols)} tumor types, {len(cancer_cols)} cancer types")
        
        # Determine plot generation strategy
        requested_plots = get_requested_plots(args)
        generate_all = not requested_plots
        
        # Create and filter plot configurations
        all_configs = create_plot_configs(args, cancer_types)
        configs_to_run = [
            config for config in all_configs
            if should_generate_plot(config, requested_plots, generate_all, args.include_significant)
        ]
        
        # Generate plots
        generated_files = []
        for config in configs_to_run:
            result = generate_plot(config, df_processed, args.output_dir, logger)
            if result:
                generated_files.append(result)
        
        logger.info(f"Pipeline completed. Generated {len(generated_files)} visualizations:")
        for file_path in generated_files:
            logger.info(f"  - {file_path}")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()