#!/usr/bin/env python3
"""
BIRDMAn Output Visualization Tool

Generates forest plots, heatmaps, and comparative analyses for differential abundance results.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import upsetplot
from utils import get_reference_level, simplify_feature_name

plt.rcParams.update({
    "figure.dpi": 400,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})


class PlotError(Exception):
    """Custom exception for plotting errors."""
    pass


def _calculate_plot_dimensions(num_features: int, base_height: float = 8) -> Tuple[float, float]:
    """Calculate optimal plot dimensions based on feature count."""
    spacing = max(0.4, min(1.2, 10 / max(num_features, 1)))
    height = max(base_height, num_features * spacing + 2)
    return spacing, height


def _extract_condition_from_column(column: str) -> str:
    """Extract condition name from BIRDMAn column name."""
    match = re.search(r"\[T\.(.*?)\]", column)
    return match.group(1) if match else column


def _get_condition_columns(df: pd.DataFrame, condition_type: str) -> List[str]:
    """Get all columns for a specific condition type."""
    pattern = f"{condition_type}[T." if condition_type != "tumor_type" else "tumor_type"
    return [col for col in df.columns if pattern in col and col.endswith("_mean")]


def _filter_significant_effects(df: pd.DataFrame, mean_col: str) -> pd.DataFrame:
    """Filter dataframe to significant effects only."""
    hdi_lower = mean_col.replace("_mean", "_hdi_lower")
    hdi_upper = mean_col.replace("_mean", "_hdi_upper")
    
    if hdi_lower not in df.columns or hdi_upper not in df.columns:
        raise PlotError(f"Missing HDI columns for {mean_col}")
    
    mask = (df[hdi_lower] > 0) | (df[hdi_upper] < 0)
    return df[mask]


def _select_top_features(df: pd.DataFrame, effect_col: str, top_n: int, 
                        significant_only: bool = False) -> pd.DataFrame:
    """Select and sort top features by effect size."""
    if significant_only:
        df = _filter_significant_effects(df, effect_col)
        if df.empty:
            return df
    
    top_indices = np.abs(df[effect_col]).argsort()[::-1][:top_n]
    df_selected = df.iloc[top_indices]
    return df_selected.iloc[df_selected[effect_col].argsort()[::-1]]


def _plot_forest_subplot(ax, df_sorted: pd.DataFrame, mean_col: str, condition: str, 
                        reference: str, spacing: float, significant_only: bool = False):
    """Create a single forest plot subplot."""
    hdi_lower = mean_col.replace("_mean", "_hdi_lower")
    hdi_upper = mean_col.replace("_mean", "_hdi_upper")
    
    y_pos = np.arange(len(df_sorted)) * spacing
    
    # Determine colors
    if significant_only:
        colors = ["red"] * len(df_sorted)
        line_colors = ["red"] * len(df_sorted)
    else:
        sig_mask = (df_sorted[hdi_lower] > 0) | (df_sorted[hdi_upper] < 0)
        colors = ["red" if sig else "gray" for sig in sig_mask]
        line_colors = ["gray"] * len(df_sorted)
    
    # Plot confidence intervals and points
    ax.hlines(y_pos, df_sorted[hdi_lower], df_sorted[hdi_upper], 
              colors=line_colors, alpha=0.6, linewidth=2)
    ax.scatter(df_sorted[mean_col], y_pos, c=colors, s=80, zorder=3, 
               edgecolors="black", linewidth=0.5)
    
    # Formatting
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel(f"Log-fold change", fontweight="bold")
    ax.set_title(f'{reference.title()} vs {condition.replace("_", " ").title()}', 
                fontweight="bold", pad=10)
    
    # Y-axis labels
    labels = [simplify_feature_name(f, 45) for f in df_sorted["Feature"]]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(axis='y', pad=5, length=0)
    
    if significant_only:
        for label in ax.get_yticklabels():
            label.set_weight('bold')
    else:
        sig_mask = (df_sorted[hdi_lower] > 0) | (df_sorted[hdi_upper] < 0)
        for i, is_sig in enumerate(sig_mask):
            if is_sig:
                ax.get_yticklabels()[i].set_weight('bold')
    
    ax.set_ylabel("Features", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_ylim(-spacing/2, (len(df_sorted)-1) * spacing + spacing/2)


def plot_condition_effects(df: pd.DataFrame, condition_type: str, top_n: int, 
                          figsize: Tuple[int, int], output_dir: Path, 
                          logger: logging.Logger, significant_only: bool = False) -> Optional[Path]:
    """Generate forest plots for condition effects (tumor_type, cancer_type, etc.)."""
    try:
        condition_cols = _get_condition_columns(df, condition_type)
        if not condition_cols:
            logger.warning(f"No {condition_type} columns found")
            return None

        reference_level = get_reference_level(df, condition_type)
        n_conditions = len(condition_cols)
        
        # Calculate subplot layout
        n_cols = min(3, n_conditions)
        n_rows = (n_conditions + n_cols - 1) // n_cols
        
        spacing, plot_height = _calculate_plot_dimensions(top_n)
        fig_width = figsize[0] * 1.2
        fig_height = plot_height * n_rows
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        axes = np.atleast_2d(axes)
        if axes.shape[0] == 1:
            axes = axes.reshape(n_rows, n_cols)
        
        for idx, col in enumerate(condition_cols):
            row, col_idx = divmod(idx, n_cols)
            condition = _extract_condition_from_column(col)
            
            df_plot = _select_top_features(df, col, top_n, significant_only)
            if df_plot.empty:
                logger.warning(f"No {'significant ' if significant_only else ''}effects for {condition}")
                fig.delaxes(axes[row, col_idx])
                continue
            
            _plot_forest_subplot(axes[row, col_idx], df_plot, col, condition, 
                                reference_level, spacing, significant_only)
        
        # Remove unused subplots
        for idx in range(n_conditions, n_rows * n_cols):
            row, col_idx = divmod(idx, n_cols)
            fig.delaxes(axes[row, col_idx])
        
        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(left=0.20, right=0.95, top=0.92, bottom=0.08)
        
        suffix = "_significant" if significant_only else ""
        filename = f"{condition_type}_effects{suffix}.svg"
        output_path = output_dir / filename
        
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.5, format='svg')
        plt.close()
        
        logger.info(f"{condition_type.title()} effects plot saved: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating {condition_type} effects plot: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None


def plot_combined_effects(df: pd.DataFrame, condition_type: str, top_n: int, 
                         figsize: Tuple[int, int], output_dir: Path, 
                         logger: logging.Logger, significant_only: bool = False) -> Optional[Path]:
    """Generate combined forest plot of all condition effects."""
    try:
        condition_cols = _get_condition_columns(df, condition_type)
        if not condition_cols:
            logger.warning(f"No {condition_type} columns found")
            return None

        reference_level = get_reference_level(df, condition_type)
        
        # Calculate total height needed
        total_features = sum(len(_select_top_features(df, col, top_n, significant_only)) 
                           for col in condition_cols)
        if total_features == 0:
            logger.warning(f"No {'significant ' if significant_only else ''}effects found")
            return None
        
        _, fig_height = _calculate_plot_dimensions(total_features, figsize[1])
        fig, ax = plt.subplots(figsize=(figsize[0], fig_height))
        
        y_offset = 0
        feature_spacing = 1.0
        group_spacing = 2.0
        colors_palette = plt.cm.Set1(np.linspace(0, 1, len(condition_cols)))
        
        all_labels, all_positions, all_significance = [], [], []
        
        for idx, col in enumerate(condition_cols):
            condition = _extract_condition_from_column(col)
            df_plot = _select_top_features(df, col, top_n, significant_only)
            
            if df_plot.empty:
                continue
            
            n_features = len(df_plot)
            y_pos = np.arange(y_offset, y_offset + n_features * feature_spacing, feature_spacing)
            
            hdi_lower = col.replace("_mean", "_hdi_lower")
            hdi_upper = col.replace("_mean", "_hdi_upper")
            
            # Plot lines and points
            ax.hlines(y_pos, df_plot[hdi_lower], df_plot[hdi_upper], 
                     colors=colors_palette[idx], alpha=0.6, linewidth=2,
                     label=condition.replace("_", " ").title())
            
            point_colors = ["red"] * n_features if significant_only else [
                "red" if (lower > 0 or upper < 0) else "lightgray"
                for lower, upper in zip(df_plot[hdi_lower], df_plot[hdi_upper])
            ]
            
            ax.scatter(df_plot[col], y_pos, c=point_colors, s=60, zorder=3,
                      edgecolors=colors_palette[idx], linewidth=1)
            
            # Collect labels and positions
            labels = [f"{simplify_feature_name(f, 40)} ({condition})" for f in df_plot["Feature"]]
            significance = [True] * n_features if significant_only else [
                (lower > 0 or upper < 0) for lower, upper in zip(df_plot[hdi_lower], df_plot[hdi_upper])
            ]
            
            all_labels.extend(labels)
            all_positions.extend(y_pos)
            all_significance.extend(significance)
            
            y_offset += n_features * feature_spacing + group_spacing
        
        # Final formatting
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.7, linewidth=2)
        ax.set_xlabel(f'{reference.title()} vs {condition.replace("_", " ").title()}')
        
        title_suffix = "Significant " if significant_only else ""
        ax.set_title(f"Combined {title_suffix}{condition_type.replace('_', ' ').title()} Effects", 
                    fontweight="bold", pad=20)
        
        ax.set_yticks(all_positions)
        ax.set_yticklabels(all_labels, fontsize=8)
        ax.tick_params(axis='y', pad=8, length=0)
        
        for i, is_sig in enumerate(all_significance):
            if is_sig:
                ax.get_yticklabels()[i].set_weight('bold')
        
        ax.set_ylabel(f"Features by {condition_type.replace('_', ' ').title()}", fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_ylim(-feature_spacing, max(all_positions) + feature_spacing)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.30, right=0.82)
        
        suffix = "_significant" if significant_only else ""
        filename = f"combined_{condition_type}_effects{suffix}.svg"
        output_path = output_dir / filename
        
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.5, format='svg')
        plt.close()
        
        logger.info(f"Combined {condition_type} effects plot saved: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating combined {condition_type} effects plot: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None


def plot_single_condition_effects(df: pd.DataFrame, condition_type: str, condition_name: str,
                                 top_n: int, figsize: Tuple[int, int], output_dir: Path,
                                 logger: logging.Logger, significant_only: bool = False) -> Optional[Path]:
    """Generate detailed effects plot for a single condition."""
    try:
        condition_col = None
        for col in df.columns:
            if f"{condition_type}[T.{condition_name}]" in col and col.endswith("_mean"):
                condition_col = col
                break
        
        if not condition_col:
            logger.warning(f"{condition_type} '{condition_name}' not found")
            return None
        
        reference_level = get_reference_level(df, condition_type)
        df_plot = _select_top_features(df, condition_col, top_n, significant_only)
        
        if df_plot.empty:
            logger.warning(f"No {'significant ' if significant_only else ''}effects for {condition_name}")
            return None
        
        spacing, fig_height = _calculate_plot_dimensions(len(df_plot))
        fig, ax = plt.subplots(figsize=(figsize[0], fig_height))
        
        _plot_forest_subplot(ax, df_plot, condition_col, condition_name, 
                            reference_level, spacing, significant_only)
        
        title_suffix = "Significant " if significant_only else ""
        ax.set_title(f"{condition_name.title()} {title_suffix}Effects on Microbiome", 
                    fontsize=16, fontweight="bold")
        ax.set_ylabel("Microbiome Features", fontweight="bold")
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.35, right=0.95)
        
        suffix = "_significant" if significant_only else ""
        filename = f"{condition_name}_detailed_effects{suffix}.svg"
        output_path = output_dir / filename
        
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.5, format='svg')
        plt.close()
        
        logger.info(f"Detailed {condition_name} plot saved: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating {condition_name} detailed plot: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None


def plot_primer_effects(df: pd.DataFrame, top_n: int, figsize: Tuple[int, int], 
                       output_dir: Path, logger: logging.Logger) -> Optional[Path]:
    """Generate primer region effects plot."""
    try:
        primer_cols = [col for col in df.columns if "primer_region[T." in col and col.endswith("_mean")]
        if not primer_cols:
            logger.warning("No primer region columns found")
            return None
        
        reference_level = get_reference_level(df, "primer_region")
        primer_effects = df[primer_cols].values
        primer_variance = np.var(primer_effects, axis=1)
        top_indices = np.argsort(primer_variance)[::-1][:top_n]
        
        fig, ax = plt.subplots(figsize=figsize)
        primer_names = [_extract_condition_from_column(col) for col in primer_cols]
        
        x = np.arange(len(primer_names))
        width = 0.8 / top_n
        
        for i, idx in enumerate(top_indices):
            feature_name = simplify_feature_name(df.iloc[idx]["Feature"], 35)
            effects = [df.iloc[idx][col] for col in primer_cols]
            ax.bar(x + i * width, effects, width, label=feature_name)
        
        ax.set_xlabel("Primer Region")
        ax.set_ylabel(f"Log-fold change vs {reference_level}")
        ax.set_title("Primer Effects on Most Variable Features")
        ax.set_xticks(x + width * (top_n - 1) / 2)
        ax.set_xticklabels(primer_names)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.7)
        
        output_path = output_dir / "primer_effects.svg"
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.3, format='svg')
        plt.close()
        
        logger.info(f"Primer effects plot saved: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating primer effects plot: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None


def plot_upset_analysis(df: pd.DataFrame, min_effect: float, output_dir: Path, 
                       logger: logging.Logger) -> Optional[Path]:
    """Generate UpSet plot for significant feature overlap."""
    try:
        cancer_cols = [col for col in df.columns if "cancer_type[T." in col and col.endswith("_mean")]
        if not cancer_cols:
            logger.warning("No cancer type columns found for UpSet plot")
            return None
        
        significant_features = {}
        
        for col in cancer_cols:
            cancer_type = _extract_condition_from_column(col)
            hdi_lower = col.replace("_mean", "_hdi_lower")
            hdi_upper = col.replace("_mean", "_hdi_upper")
            
            sig_mask = ((df[hdi_lower] > 0) | (df[hdi_upper] < 0)) & (np.abs(df[col]) > min_effect)
            sig_features = set(df.loc[sig_mask, "Feature"].apply(lambda x: simplify_feature_name(x, 50)))
            significant_features[cancer_type] = sig_features
        
        if not any(significant_features.values()):
            logger.warning("No significant features found for UpSet plot")
            return None
        
        upset_df = upsetplot.from_contents(significant_features)
        fig = plt.figure(figsize=(16, 10))
        upsetplot.plot(upset_df, fig=fig, show_counts=True, element_size=40)
        plt.suptitle("Significant Feature Overlap Across Cancer Types", fontweight="bold")
        
        output_path = output_dir / "upset_analysis.svg"
        fig.savefig(output_path, format='svg')
        plt.close()
        
        # Save data
        data_path = output_dir / "upset_data.json"
        with open(data_path, "w") as f:
            json.dump({k: list(v) for k, v in significant_features.items()}, f, indent=2)
        
        logger.info(f"UpSet plot saved: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating UpSet plot: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None


# Convenience wrapper functions
def plot_tumor_type_effects(df: pd.DataFrame, top_n: int, figsize: Tuple[int, int], 
                           output_dir: Path, logger: logging.Logger) -> Optional[Path]:
    """Generate tumor type effects plots."""
    return plot_condition_effects(df, "tumor_type", top_n, figsize, output_dir, logger)


def plot_tumor_type_effects_significant(df: pd.DataFrame, top_n: int, figsize: Tuple[int, int], 
                                      output_dir: Path, logger: logging.Logger) -> Optional[Path]:
    """Generate significant tumor type effects plots."""
    return plot_condition_effects(df, "tumor_type", top_n, figsize, output_dir, logger, True)


def plot_combined_tumor_effects(df: pd.DataFrame, top_n: int, figsize: Tuple[int, int], 
                               output_dir: Path, logger: logging.Logger) -> Optional[Path]:
    """Generate combined tumor type effects plot."""
    return plot_combined_effects(df, "tumor_type", top_n, figsize, output_dir, logger)


def plot_single_cancer_effects(df: pd.DataFrame, cancer_type: str, top_n: int, 
                              figsize: Tuple[int, int], output_dir: Path, 
                              logger: logging.Logger) -> Optional[Path]:
    """Generate single cancer type effects plot."""
    return plot_single_condition_effects(df, "cancer_type", cancer_type, top_n, 
                                        figsize, output_dir, logger)