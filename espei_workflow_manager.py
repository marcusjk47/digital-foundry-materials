"""
ESPEI Workflow Manager for GNN Integration
Automates TDB generation from Materials Project data for use in GNN feature extraction

Author: Digital Foundry Materials Science Toolkit
"""

import json
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np

from espei_integration import (
    convert_mp_csv_to_espei_datasets,
    create_phase_models_json,
    create_espei_input_yaml,
    run_espei_generate_parameters,
    detect_element_columns
)

try:
    from pycalphad import Database
    PYCALPHAD_AVAILABLE = True
except ImportError:
    PYCALPHAD_AVAILABLE = False


class ESPEIWorkflowManager:
    """
    Manage ESPEI TDB generation workflow for GNN integration.

    This class orchestrates the process of:
    1. Converting MP CSV data to ESPEI format
    2. Generating phase models
    3. Running ESPEI parameter generation
    4. Validating the generated TDB
    5. Making the TDB available for GNN feature extraction
    """

    def __init__(self, output_dir: Path):
        """
        Initialize ESPEI workflow manager.

        Args:
            output_dir: Directory for ESPEI outputs (datasets, TDB files, etc.)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.tdb_dir = self.output_dir / "tdb_files"
        self.tdb_dir.mkdir(exist_ok=True)

    def generate_tdb_from_csv(
        self,
        csv_path: str,
        elements: List[str] = None,
        phases: List[str] = None,
        system_name: str = None,
        max_materials: int = None,
        temperature: float = 298.15
    ) -> Tuple[bool, str, Optional[Path]]:
        """
        Generate TDB file from Materials Project CSV data.

        This is the main entry point for TDB generation. It handles the complete
        workflow from CSV to validated TDB file.

        Args:
            csv_path: Path to Materials Project CSV file
            elements: List of elements (if None, auto-detect from CSV)
            phases: List of phases to include (default: ['LIQUID', 'FCC_A1', 'BCC_A2'])
            system_name: Name for the system (e.g., 'Fe-Ni', auto-generated if None)
            max_materials: Maximum number of materials to process (None = all)
            temperature: Temperature for data points in K (default: 298.15)

        Returns:
            Tuple of (success: bool, message: str, tdb_path: Path or None)
        """
        try:
            # Step 1: Validate inputs
            if not Path(csv_path).exists():
                return False, f"CSV file not found: {csv_path}", None

            # Load CSV to detect elements if needed
            df = pd.read_csv(csv_path)

            if elements is None:
                element_columns = detect_element_columns(df)
                if not element_columns:
                    return False, "Could not detect element columns in CSV", None
                elements = sorted(list(element_columns.keys()))
            else:
                elements = sorted([e.upper() for e in elements])

            if len(elements) < 2:
                return False, f"Need at least 2 elements, found: {elements}", None

            # Generate system name
            if system_name is None:
                system_name = '-'.join(elements)

            # Default phases
            if phases is None:
                phases = ['LIQUID', 'FCC_A1', 'BCC_A2']

            print(f"\n{'='*70}")
            print(f"ESPEI TDB GENERATION WORKFLOW")
            print(f"{'='*70}")
            print(f"System: {system_name}")
            print(f"Elements: {', '.join(elements)}")
            print(f"Phases: {', '.join(phases)}")
            print(f"Materials: {max_materials if max_materials else 'all'}")
            print(f"Temperature: {temperature} K")

            # Step 2: Create ESPEI datasets
            print(f"\nStep 1/4: Converting MP data to ESPEI datasets...")

            espei_output_dir = self.output_dir / f"espei_{system_name.replace('-', '_')}"
            espei_output_dir.mkdir(parents=True, exist_ok=True)

            num_datasets, detected_elements, datasets_dir, elem_cols, phases_used = \
                convert_mp_csv_to_espei_datasets(
                    csv_path=csv_path,
                    output_dir=espei_output_dir,
                    element_columns=None,  # Auto-detect
                    phases=phases,
                    max_materials=max_materials,
                    temperature=temperature
                )

            if num_datasets == 0:
                return False, "No datasets created from CSV", None

            print(f"  [OK] Created {num_datasets} ESPEI datasets")

            # Step 3: Create phase models
            print(f"\nStep 2/4: Creating phase models...")

            phase_models_path = create_phase_models_json(
                elements=detected_elements,
                output_path=espei_output_dir,
                phases=phases_used
            )

            print(f"  [OK] Phase models: {phase_models_path}")

            # Step 4: Create ESPEI input YAML
            print(f"\nStep 3/4: Creating ESPEI input configuration...")

            output_tdb_path = self.tdb_dir / f"{system_name.replace('-', '_')}.tdb"
            yaml_path = espei_output_dir / "espei_input.yaml"

            create_espei_input_yaml(
                phase_models_path=phase_models_path,
                datasets_dir=datasets_dir,
                output_tdb_path=output_tdb_path,
                output_yaml_path=yaml_path
            )

            print(f"  [OK] ESPEI config: {yaml_path}")

            # Step 5: Run ESPEI parameter generation
            print(f"\nStep 4/4: Running ESPEI parameter generation...")
            print(f"  (This may take 1-5 minutes depending on dataset size)")

            success, message, _ = run_espei_generate_parameters(yaml_path)

            if not success:
                return False, f"ESPEI generation failed: {message}", None

            # Step 6: Validate TDB
            print(f"\nValidating generated TDB...")

            if not output_tdb_path.exists():
                return False, "TDB file was not created", None

            validation_result = self.validate_tdb(output_tdb_path, elements)

            if not validation_result['valid']:
                return False, f"TDB validation failed: {validation_result['message']}", None

            print(f"  [OK] TDB validated successfully")
            print(f"\n{'='*70}")
            print(f"TDB GENERATION COMPLETE")
            print(f"{'='*70}")
            print(f"Output TDB: {output_tdb_path}")
            print(f"Size: {output_tdb_path.stat().st_size / 1024:.1f} KB")
            print(f"Elements: {', '.join(validation_result['elements'])}")
            print(f"Phases: {', '.join(validation_result['phases'])}")

            return True, f"Successfully generated TDB for {system_name}", output_tdb_path

        except Exception as e:
            import traceback
            error_msg = f"Error during TDB generation: {str(e)}\n{traceback.format_exc()}"
            return False, error_msg, None

    def validate_tdb(self, tdb_path: Path, required_elements: List[str] = None) -> Dict:
        """
        Validate a TDB file to ensure it can be used for feature extraction.

        Args:
            tdb_path: Path to TDB file
            required_elements: Elements that must be present (None = don't check)

        Returns:
            Dictionary with validation results:
            {
                'valid': bool,
                'message': str,
                'elements': List[str],
                'phases': List[str],
                'file_size': int
            }
        """
        result = {
            'valid': False,
            'message': '',
            'elements': [],
            'phases': [],
            'file_size': 0
        }

        try:
            # Check file exists
            if not tdb_path.exists():
                result['message'] = f"TDB file not found: {tdb_path}"
                return result

            result['file_size'] = tdb_path.stat().st_size

            # Check file is not empty
            if result['file_size'] == 0:
                result['message'] = "TDB file is empty"
                return result

            # Try to load with PyCalphad
            if not PYCALPHAD_AVAILABLE:
                result['message'] = "PyCalphad not available for validation"
                result['valid'] = True  # Assume valid if we can't check
                return result

            try:
                db = Database(str(tdb_path))

                # Extract elements
                result['elements'] = sorted([str(e) for e in db.elements if str(e) != 'VA'])

                # Extract phases
                result['phases'] = sorted([str(p) for p in db.phases.keys()])

                # Check required elements
                if required_elements:
                    required_set = set(e.upper() for e in required_elements)
                    available_set = set(e.upper() for e in result['elements'])

                    missing = required_set - available_set
                    if missing:
                        result['message'] = f"Missing required elements: {', '.join(missing)}"
                        return result

                # Check has at least one phase
                if len(result['phases']) == 0:
                    result['message'] = "No phases defined in TDB"
                    return result

                result['valid'] = True
                result['message'] = "TDB validation successful"

            except Exception as e:
                result['message'] = f"Failed to load TDB with PyCalphad: {str(e)}"
                return result

        except Exception as e:
            result['message'] = f"Validation error: {str(e)}"
            return result

        return result

    def check_tdb_compatibility(
        self,
        tdb_path: Path,
        elements: List[str],
        phases: List[str] = None
    ) -> Tuple[bool, str]:
        """
        Check if a TDB file is compatible with a given materials system.

        Args:
            tdb_path: Path to TDB file
            elements: Required elements
            phases: Required phases (if None, any phase is OK)

        Returns:
            Tuple of (compatible: bool, message: str)
        """
        validation = self.validate_tdb(tdb_path, elements)

        if not validation['valid']:
            return False, validation['message']

        # Check phases if specified
        if phases:
            required_phases = set(p.upper() for p in phases)
            available_phases = set(p.upper() for p in validation['phases'])

            missing_phases = required_phases - available_phases
            if missing_phases:
                return False, f"Missing required phases: {', '.join(missing_phases)}"

        return True, "TDB is compatible"

    def copy_tdb_to_project(
        self,
        source_tdb: Path,
        system_name: str
    ) -> Path:
        """
        Copy an external TDB file to the project TDB directory.

        Args:
            source_tdb: Path to source TDB file
            system_name: Name for the system (used in filename)

        Returns:
            Path to copied TDB file in project directory
        """
        dest_filename = f"{system_name.replace('-', '_')}.tdb"
        dest_path = self.tdb_dir / dest_filename

        shutil.copy2(source_tdb, dest_path)

        return dest_path

    def list_available_tdbs(self) -> List[Dict]:
        """
        List all TDB files in the project directory.

        Returns:
            List of dictionaries with TDB information:
            [{
                'path': Path,
                'name': str,
                'size': int,
                'elements': List[str],
                'phases': List[str]
            }]
        """
        tdbs = []

        for tdb_file in self.tdb_dir.glob("*.tdb"):
            validation = self.validate_tdb(tdb_file)

            tdbs.append({
                'path': tdb_file,
                'name': tdb_file.stem,
                'size': tdb_file.stat().st_size,
                'elements': validation.get('elements', []),
                'phases': validation.get('phases', []),
                'valid': validation['valid']
            })

        return tdbs

    def get_tdb_info(self, tdb_path: Path) -> Dict:
        """
        Get detailed information about a TDB file.

        Args:
            tdb_path: Path to TDB file

        Returns:
            Dictionary with TDB information
        """
        validation = self.validate_tdb(tdb_path)

        info = {
            'path': str(tdb_path),
            'name': tdb_path.name,
            'size_kb': tdb_path.stat().st_size / 1024 if tdb_path.exists() else 0,
            'valid': validation['valid'],
            'message': validation['message'],
            'elements': validation['elements'],
            'phases': validation['phases'],
            'num_elements': len(validation['elements']),
            'num_phases': len(validation['phases'])
        }

        return info


# Convenience functions for direct use

def generate_tdb_for_dataset(
    csv_path: str,
    output_dir: str = "espei_output",
    elements: List[str] = None,
    phases: List[str] = None,
    system_name: str = None,
    max_materials: int = None
) -> Tuple[bool, str, Optional[Path]]:
    """
    Convenience function to generate TDB from MP CSV.

    Args:
        csv_path: Path to Materials Project CSV
        output_dir: Directory for outputs
        elements: Elements in system (auto-detect if None)
        phases: Phases to include (default: LIQUID, FCC_A1, BCC_A2)
        system_name: System name (auto-generate if None)
        max_materials: Max materials to process (None = all)

    Returns:
        Tuple of (success, message, tdb_path)
    """
    manager = ESPEIWorkflowManager(Path(output_dir))

    return manager.generate_tdb_from_csv(
        csv_path=csv_path,
        elements=elements,
        phases=phases,
        system_name=system_name,
        max_materials=max_materials
    )


def validate_tdb_for_elements(
    tdb_path: str,
    elements: List[str]
) -> Tuple[bool, str]:
    """
    Quick validation that TDB contains required elements.

    Args:
        tdb_path: Path to TDB file
        elements: Required elements

    Returns:
        Tuple of (valid, message)
    """
    manager = ESPEIWorkflowManager(Path("espei_output"))
    validation = manager.validate_tdb(Path(tdb_path), elements)

    return validation['valid'], validation['message']


if __name__ == "__main__":
    """Test ESPEI workflow manager"""

    print("="*70)
    print("ESPEI WORKFLOW MANAGER TEST")
    print("="*70)

    # Test with a sample CSV (if available)
    test_csv = Path("datasets") / "test_fe_ni.csv"

    if test_csv.exists():
        print(f"\nTesting with: {test_csv}")

        manager = ESPEIWorkflowManager(Path("espei_test_output"))

        success, message, tdb_path = manager.generate_tdb_from_csv(
            csv_path=str(test_csv),
            elements=['Fe', 'Ni'],
            phases=['LIQUID', 'FCC_A1'],
            system_name='Fe-Ni',
            max_materials=50  # Limit for testing
        )

        print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
        print(f"Message: {message}")

        if tdb_path:
            print(f"TDB Path: {tdb_path}")

            # Test validation
            info = manager.get_tdb_info(tdb_path)
            print(f"\nTDB Info:")
            print(f"  Size: {info['size_kb']:.1f} KB")
            print(f"  Elements: {', '.join(info['elements'])}")
            print(f"  Phases: {', '.join(info['phases'])}")
    else:
        print(f"\nTest CSV not found: {test_csv}")
        print("Workflow manager created successfully")

        # Demo listing TDBs
        manager = ESPEIWorkflowManager(Path("espei_output"))
        tdbs = manager.list_available_tdbs()

        if tdbs:
            print(f"\nFound {len(tdbs)} TDB files:")
            for tdb in tdbs:
                print(f"  - {tdb['name']}: {', '.join(tdb['elements'])}")
        else:
            print("\nNo TDB files found in espei_output/tdb_files")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
