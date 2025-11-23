"""
ESPEI Integration Module v2 - Multi-Phase Support
Converts Materials Project CSV data to ESPEI-format datasets and generates TDB files
Now supports multiple phases (LIQUID, FCC, BCC, HCP)
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np


def create_espei_dataset_from_mp_row(
    row: pd.Series,
    elements: List[str],
    phase: str = 'LIQUID',
    reference_state: str = 'SGTE91',
    temperature: float = 298.15
) -> Dict[str, Any]:
    """
    Create an ESPEI-format dataset entry from a Materials Project data row.

    Parameters
    ----------
    row : pd.Series
        Row from Materials Project CSV containing composition and formation energy
    elements : List[str]
        List of elements in the system
    phase : str
        Phase name (LIQUID, FCC_A1, BCC_A2, HCP_A3, etc.)
    reference_state : str
        Reference state for the calculations (default: SGTE91)
    temperature : float
        Temperature in K for the data point (default: 298.15 K)

    Returns
    -------
    Dict[str, Any]
        ESPEI-format dataset dictionary
    """
    # Extract composition
    composition = {}
    for elem in elements:
        # Try case-insensitive column matching
        for col in row.index:
            if col.upper() == elem.upper():
                val = row[col]
                if pd.notna(val) and val > 0:
                    composition[elem.upper()] = float(val)
                break

    # Normalize composition
    if len(composition) > 0:
        total = sum(composition.values())
        composition = {k: v/total for k, v in composition.items()}

    # Get formation energy (try different possible column names)
    formation_energy = None
    for col_name in ['formation_energy_per_atom', 'formation_energy', 'e_form', 'energy_per_atom']:
        if col_name in row.index and pd.notna(row[col_name]):
            formation_energy = float(row[col_name])
            break

    if formation_energy is None:
        return None

    # Convert eV/atom to J/mol (1 eV/atom ≈ 96485.3 J/mol)
    formation_energy_j_mol = formation_energy * 96485.3

    # Get sorted elements and their occupancies
    sorted_elements = sorted(composition.keys())
    occupancies = [composition[elem] for elem in sorted_elements]

    # Phase-specific sublattice structure
    # Different phases have different sublattice models
    # For multi-sublattice phases, occupancies structure: [[[subl1_occs], [subl2_occs], ...]]
    phase_sublattice_info = {
        'LIQUID': {
            'site_ratios': [1.0],
            'configurations': [[sorted_elements]],
            'occupancies': [[occupancies]]
        },
        'FCC_A1': {
            'site_ratios': [1.0, 1.0],
            'configurations': [[sorted_elements, ['VA']]],
            'occupancies': [[occupancies, [1.0]]]
        },
        'BCC_A2': {
            'site_ratios': [1.0, 3.0],
            'configurations': [[sorted_elements, ['VA']]],
            'occupancies': [[occupancies, [1.0]]]
        },
        'HCP_A3': {
            'site_ratios': [1.0, 0.5],
            'configurations': [[sorted_elements, ['VA']]],
            'occupancies': [[occupancies, [1.0]]]
        },
        'BCC_B2': {
            'site_ratios': [0.5, 0.5, 1.0],
            'configurations': [[sorted_elements, sorted_elements, ['VA']]],
            'occupancies': [[occupancies, occupancies, [1.0]]]
        }
    }

    # Get sublattice info for this phase (default to LIQUID if unknown)
    subl_info = phase_sublattice_info.get(phase, phase_sublattice_info['LIQUID'])

    # For solid phases with VA, need to include VA in components
    components = sorted_elements.copy()
    if phase in ['FCC_A1', 'BCC_A2', 'HCP_A3', 'BCC_B2']:
        components = sorted(sorted_elements + ['VA'])

    dataset = {
        "components": components,
        "phases": [phase],
        "solver": {
            "mode": "manual",
            "sublattice_site_ratios": subl_info['site_ratios'],
            "sublattice_configurations": subl_info['configurations'],
            "sublattice_occupancies": subl_info['occupancies']
        },
        "conditions": {
            "P": 101325,
            "T": temperature
        },
        "output": "HM_FORM",
        "values": [[[formation_energy_j_mol]]],
        "reference": row.get('material_id', 'Materials_Project'),
        "comment": f"Formation energy data from Materials Project for {row.get('formula', 'unknown')} in {phase} phase"
    }

    return dataset


def create_phase_models_json(elements: List[str], output_path: Path, phases: List[str] = None) -> Path:
    """
    Create a phase models JSON file for ESPEI with multiple phases.

    Parameters
    ----------
    elements : List[str]
        List of elements in the system
    output_path : Path
        Directory to save the phase models file
    phases : List[str], optional
        List of phases to include. If None, uses ['LIQUID', 'FCC_A1', 'BCC_A2']

    Returns
    -------
    Path
        Path to the created phase models file
    """
    if phases is None:
        phases = ['LIQUID', 'FCC_A1', 'BCC_A2']

    sorted_elements = sorted(elements)

    # Check if we need to include VA (for phases with interstitial sublattices)
    needs_va = any(p in phases for p in ['FCC_A1', 'BCC_A2', 'HCP_A3', 'BCC_B2'])

    # Components list - include VA if any solid phase is requested
    components_list = sorted(sorted_elements + ['VA']) if needs_va else sorted_elements

    # Define phase models for common crystal structures
    # Note: For solid phases with interstitial sublattices, we include VA (vacancy)
    phase_definitions = {
        "LIQUID": {
            "sublattice_model": [sorted_elements],
            "sublattice_site_ratios": [1.0]
        },
        "FCC_A1": {
            "sublattice_model": [sorted_elements, ["VA"]],
            "sublattice_site_ratios": [1.0, 1.0]
        },
        "BCC_A2": {
            "sublattice_model": [sorted_elements, ["VA"]],
            "sublattice_site_ratios": [1.0, 3.0]
        },
        "HCP_A3": {
            "sublattice_model": [sorted_elements, ["VA"]],
            "sublattice_site_ratios": [1.0, 0.5]
        },
        "BCC_B2": {
            "sublattice_model": [sorted_elements, sorted_elements, ["VA"]],
            "sublattice_site_ratios": [0.5, 0.5, 1.0]
        }
    }

    # Build phase models for requested phases
    phase_models = {
        "components": components_list,  # Include VA if needed
        "phases": {}
    }

    for phase in phases:
        if phase in phase_definitions:
            phase_models["phases"][phase] = phase_definitions[phase]
        else:
            # Default to single sublattice model for unknown phases
            phase_models["phases"][phase] = {
                "sublattice_model": [sorted_elements],
                "sublattice_site_ratios": [1.0]
            }

    output_file = output_path / "phase_models.json"
    with open(output_file, 'w') as f:
        json.dump(phase_models, f, indent=2)

    return output_file


def detect_element_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Detect element composition columns in a DataFrame.

    Handles multiple naming conventions:
    - Simple: Fe, Cr, Ni
    - Fraction prefix: frac_Fe, frac_Cr
    - Fraction suffix: Fe_fraction, Cr_fraction
    - Lowercase: fe_fraction, cr_fraction

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to scan for element columns

    Returns
    -------
    Dict[str, str]
        Mapping of element symbol to column name
    """
    import re

    element_map = {}

    # Common element symbols (extend as needed)
    common_elements = [
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
        'Pa', 'U', 'Np', 'Pu'
    ]

    for col in df.columns:
        col_clean = col.strip()

        # Try to match various patterns
        for elem in common_elements:
            # Pattern 1: Exact match (case-insensitive)
            if col_clean.upper() == elem.upper():
                if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                    if (df[col] > 0).any():
                        element_map[elem.upper()] = col
                        break

            # Pattern 2: frac_Element (e.g., frac_Fe, frac_Cr)
            if re.match(rf'^frac[_\-]?{elem}$', col_clean, re.IGNORECASE):
                if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                    if (df[col] > 0).any():
                        element_map[elem.upper()] = col
                        break

            # Pattern 3: Element_fraction (e.g., Fe_fraction, cr_fraction)
            if re.match(rf'^{elem}[_\-]?fraction$', col_clean, re.IGNORECASE):
                if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                    if (df[col] > 0).any():
                        element_map[elem.upper()] = col
                        break

    return element_map


def convert_mp_csv_to_espei_datasets(
    csv_path: str,
    output_dir: Path,
    element_columns: Dict[str, str] = None,
    phases: List[str] = None,
    max_materials: int = None,
    temperature: float = 298.15
) -> tuple:
    """
    Convert Materials Project CSV to ESPEI dataset files.

    Parameters
    ----------
    csv_path : str
        Path to the Materials Project CSV file
    output_dir : Path
        Directory to save ESPEI dataset files
    element_columns : Dict[str, str], optional
        Mapping of element symbol to column name (e.g., {'FE': 'fe_fraction'})
        If None, will auto-detect
    phases : List[str], optional
        List of phases to generate datasets for. If None, creates datasets for
        LIQUID, FCC_A1, and BCC_A2
    max_materials : int, optional
        Maximum number of materials to process
    temperature : float
        Temperature for the data points (K)

    Returns
    -------
    tuple
        (num_datasets_created, list_of_elements, datasets_dir, element_column_map, phases_used)
    """
    # Load CSV
    df = pd.read_csv(csv_path)

    if max_materials:
        df = df.head(max_materials)

    # Create output directory for datasets
    datasets_dir = output_dir / "espei_datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    # Detect or use provided element columns
    if element_columns is None:
        element_columns = detect_element_columns(df)

    if len(element_columns) == 0:
        raise ValueError(
            "No element composition columns detected in CSV. "
            "Expected columns like 'Fe', 'Cr', 'frac_Fe', or 'Fe_fraction'. "
            f"Available columns: {', '.join(df.columns.tolist())}"
        )

    elements = sorted(list(element_columns.keys()))

    # Default phases if not specified
    if phases is None:
        phases = ['LIQUID', 'FCC_A1', 'BCC_A2']

    # Create datasets for each phase
    datasets_created = 0
    for idx, row in df.iterrows():
        # Create a modified row with standardized element columns
        row_modified = row.copy()
        for elem, col_name in element_columns.items():
            row_modified[elem] = row[col_name]

        # Create a dataset for each phase
        # Note: Same formation energy is used for all phases initially
        # In reality, different phases have different formation energies
        for phase in phases:
            dataset = create_espei_dataset_from_mp_row(
                row_modified, elements, phase=phase, temperature=temperature
            )

            if dataset is not None:
                # Save each dataset as a separate JSON file
                mat_id = row.get('material_id', f'material_{idx}')
                filename = f"{mat_id.replace('-', '_')}_{phase}.json"
                filepath = datasets_dir / filename

                with open(filepath, 'w') as f:
                    json.dump(dataset, f, indent=2)

                datasets_created += 1

    return datasets_created, elements, datasets_dir, element_columns, phases


def create_espei_input_yaml(
    phase_models_path: Path,
    datasets_dir: Path,
    output_tdb_path: Path,
    output_yaml_path: Path
) -> Path:
    """
    Create ESPEI input YAML file.

    Parameters
    ----------
    phase_models_path : Path
        Path to phase models JSON file
    datasets_dir : Path
        Path to datasets directory
    output_tdb_path : Path
        Path where the output TDB file should be saved
    output_yaml_path : Path
        Path where the YAML file should be saved

    Returns
    -------
    Path
        Path to the created YAML file
    """
    yaml_content = f"""# ESPEI Configuration for Materials Project Data
system:
  phase_models: {phase_models_path.as_posix()}
  datasets: {datasets_dir.as_posix()}

generate_parameters:
  excess_model: linear
  ref_state: SGTE91
  ridge_alpha: null
  aicc_penalty_factor: null

output:
  verbosity: 2
  output_db: {output_tdb_path.as_posix()}
"""

    with open(output_yaml_path, 'w') as f:
        f.write(yaml_content)

    return output_yaml_path


def run_espei_generate_parameters(yaml_path: Path) -> tuple:
    """
    Run ESPEI parameter generation.

    Parameters
    ----------
    yaml_path : Path
        Path to ESPEI input YAML file

    Returns
    -------
    tuple
        (success: bool, message: str, tdb_path: Path or None)
    """
    try:
        from espei.espei_script import main
        import sys

        # Save original argv
        original_argv = sys.argv.copy()

        try:
            # Set argv for ESPEI
            sys.argv = ['espei', '--input', str(yaml_path)]

            # Run ESPEI
            main()

            return True, "ESPEI parameter generation completed successfully", None

        finally:
            # Restore original argv
            sys.argv = original_argv

    except Exception as e:
        return False, f"ESPEI error: {str(e)}", None
