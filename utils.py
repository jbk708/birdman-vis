import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd


def setup_logging(output_dir: Path) -> logging.Logger:
    """
    Configure logging system with file and console output.

    Parameters:
    -----------
    output_dir : Path
        Directory for log file output

    Returns:
    --------
    logging.Logger : Configured logger instance
    """
    log_file = output_dir / "birdman_visualization.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    return logging.getLogger(__name__)


def validate_input_file(file_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Validate and load BIRDMAn output file.

    Parameters:
    -----------
    file_path : Path
        Path to input TSV file
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    pd.DataFrame : Loaded and validated dataframe

    Raises:
    -------
    FileNotFoundError : If input file doesn't exist
    ValueError : If file format is invalid
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    logger.info(f"Loading input file: {file_path}")

    try:
        df = pd.read_csv(file_path, sep="\t")
        logger.info(f"Loaded dataframe with shape: {df.shape}")
    except Exception as e:
        raise ValueError(f"Failed to read input file: {e}")

    required_patterns = ["_mean", "_hdi"]
    found_patterns = [
        any(col.endswith(pattern) for col in df.columns)
        for pattern in required_patterns
    ]

    if not all(found_patterns):
        missing = [
            pattern
            for pattern, found in zip(required_patterns, found_patterns)
            if not found
        ]
        raise ValueError(f"Missing required column patterns: {missing}")

    logger.info("Input file validation successful")
    return df


def parse_hdi_column(hdi_str: Union[str, float]) -> Tuple[float, float]:
    """
    Parse HDI string format '(-25.1282, -13.1903)' to tuple of floats.

    Parameters:
    -----------
    hdi_str : Union[str, float]
        HDI string in format '(lower, upper)' or NaN

    Returns:
    --------
    Tuple[float, float] : Lower and upper bounds
    """
    if pd.isna(hdi_str) or hdi_str == "":
        return np.nan, np.nan

    hdi_str = str(hdi_str)
    values = hdi_str.strip("()").split(",")

    if len(values) != 2:
        return np.nan, np.nan

    try:
        return float(values[0].strip()), float(values[1].strip())
    except ValueError:
        return np.nan, np.nan


def process_birdman_output(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Process BIRDMAn output dataframe for visualization.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw BIRDMAn output
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    pd.DataFrame : Processed dataframe with parsed HDI values
    """
    df = df.copy()

    if "Feature" not in df.columns:
        if df.index.name == "Feature" or (
            len(df) > 0 and str(df.index[0]).startswith("d__")
        ):
            logger.info("Feature column found in index, resetting index")
            df = df.reset_index()
            if "index" in df.columns:
                df = df.rename(columns={"index": "Feature"})
        else:
            raise ValueError("Feature column not found in data")

    hdi_cols = [col for col in df.columns if col.endswith("_hdi")]
    logger.info(f"Processing {len(hdi_cols)} HDI columns")

    for col in hdi_cols:
        param_name = col.replace("_hdi", "")
        hdi_values = df[col].apply(parse_hdi_column)

        df[f"{param_name}_hdi_lower"] = hdi_values.apply(
            lambda x: x[0] if isinstance(x, tuple) else np.nan
        )
        df[f"{param_name}_hdi_upper"] = hdi_values.apply(
            lambda x: x[1] if isinstance(x, tuple) else np.nan
        )

    return df


def estimate_figure_sizes(
    df: pd.DataFrame, logger: logging.Logger
) -> Dict[str, Tuple[int, int]]:
    """
    Estimate appropriate figure sizes based on data dimensions.

    Parameters:
    -----------
    df : pd.DataFrame
        Processed BIRDMAn output
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    Dict[str, Tuple[int, int]] : Figure sizes for different plot types
    """
    n_features = len(df)

    tumor_cols = [
        col for col in df.columns if "tumor_type" in col and col.endswith("_mean")
    ]
    cancer_cols = [
        col for col in df.columns if "cancer_type[T." in col and col.endswith("_mean")
    ]

    n_tumor_types = len(tumor_cols)
    n_cancer_types = len(cancer_cols)

    # Conservative sizing to prevent matplotlib memory issues
    tumor_width = min(30, max(20, n_tumor_types * 6))
    tumor_height = min(25, max(16, min(50, n_features * 0.01)))  # Cap based on features

    # Much more conservative heatmap sizing
    heatmap_width = min(
        25, max(12, min(100, n_features * 0.03))
    )  # Cap at reasonable size
    heatmap_height = min(20, max(8, n_cancer_types * 1.2))

    sizes = {
        "tumor_figsize": (tumor_width, tumor_height),
        "heatmap_figsize": (heatmap_width, heatmap_height),
        "primer_figsize": (14, 10),
        "study_figsize": (16, 8),
        "single_cancer_figsize": (14, 16),
    }

    logger.info(
        f"Estimated figure sizes for {n_features} features, {n_cancer_types} cancer types: {sizes}"
    )
    return sizes


def simplify_feature_name(feature: str, max_length: int = 35) -> str:
    """
    Simplify taxonomic feature names for display.

    Parameters:
    -----------
    feature : str
        Full taxonomic string
    max_length : int
        Maximum display length

    Returns:
    --------
    str : Simplified feature name
    """
    parts = feature.split(";")

    if len(parts) >= 2:
        # Start with the most specific level (species)
        name = parts[-1].replace("g__", "").replace("s__", "")

        # If species level is empty or unclassified, use genus
        if not name or name in ["unclassified", ""]:
            if len(parts) >= 2:
                name = parts[-2].replace("g__", "").replace("f__", "")

        # If genus is also empty, try family
        if not name or name in ["unclassified", ""]:
            if len(parts) >= 3:
                name = parts[-3].replace("f__", "").replace("o__", "")

        # Clean up underscores and replace with spaces
        name = name.replace("_", " ").strip()

        # If still empty, use a fallback
        if not name:
            name = "Unknown taxon"

    else:
        # Single part, just clean it up
        name = feature.replace("_", " ").strip()

    # Apply length limit consistently
    if len(name) > max_length:
        name = name[: max_length - 3] + "..."

    return name


def get_available_cancer_types(df: pd.DataFrame) -> List[str]:
    """
    Extract all available cancer types from the dataframe columns.

    Parameters:
    -----------
    df : pd.DataFrame
        Processed BIRDMAn output

    Returns:
    --------
    List[str] : List of available cancer types
    """
    cancer_cols = [
        col for col in df.columns if "cancer_type[T." in col and col.endswith("_mean")
    ]
    cancer_types = []

    for col in cancer_cols:
        match = re.search(r"\[T\.(.*?)\]", col)
        if match:
            cancer_types.append(match.group(1))

    return sorted(cancer_types)


def get_reference_level(df: pd.DataFrame, effect_type: str = "tumor_type") -> str:
    """
    Determine the reference level from column names.

    Parameters:
    -----------
    df : pd.DataFrame
        Processed BIRDMAn output
    effect_type : str
        Type of effect to analyze

    Returns:
    --------
    str : Reference level name
    """
    reference_mapping = {
        "tumor_type": "cancer",
        "cancer_type": "reference",
        "primer_region": "reference",
    }
    return reference_mapping.get(effect_type, "reference")
