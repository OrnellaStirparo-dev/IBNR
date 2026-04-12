import os

import pandas as pd
from pathlib import Path


def _is_zip_xlsx(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 64:
        return False
    with open(path, "rb") as f:
        return f.read(2) == b"PK"


def list_excel_sheets(filepath: str) -> list[str]:
    """List all sheet names in an Excel file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Determine engine based on file extension
    if path.suffix.lower() == '.xlsx':
        engine = 'openpyxl'
    elif path.suffix.lower() == '.xls':
        engine = 'xlrd'
    else:
        engine = 'openpyxl'
    
    try:
        excel_file = pd.ExcelFile(filepath, engine=engine)
        return excel_file.sheet_names
    except Exception as e:
        raise ValueError(f"Failed to read Excel file {filepath}: {e}")


def load_siniestros_data(filepath: str, sheet_name: int | str = 0) -> pd.DataFrame:
    """
    Load a generic siniestros Excel file given an explicit path.
    
    Args:
        filepath: Path to the Excel file
        sheet_name: Sheet to load. Can be:
            - An integer (0-based index): 0 for first sheet, 1 for second sheet, etc.
            - A string: Name of the sheet (e.g., "Sheet2", "Data")
            - Default: 0 (first sheet)
    
    Returns:
        DataFrame with the loaded data
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Check file size - Excel files should be at least a few KB
    file_size = path.stat().st_size
    if file_size < 1024:  # Less than 1 KB is suspicious
        raise ValueError(f"File {filepath} is too small ({file_size} bytes). "
                        f"It may be corrupted or empty. Please provide a valid Excel file.")

    # Determine engine based on file extension
    if path.suffix.lower() == '.xlsx':
        engine = 'openpyxl'
    elif path.suffix.lower() == '.xls':
        engine = 'xlrd'
    else:
        # Try openpyxl first, fall back to xlrd
        engine = 'openpyxl'

    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name, engine=engine)
        sheet_info = f"sheet '{sheet_name}'" if isinstance(sheet_name, str) else f"sheet {sheet_name}"
        print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns from {path.name} ({sheet_info})")
        return df
    except Exception as e:
        # If sheet not found, list available sheets
        try:
            available_sheets = list_excel_sheets(filepath)
            raise ValueError(f"Failed to read Excel file {filepath}: {e}\n"
                           f"Available sheets: {available_sheets}\n"
                           f"File size: {file_size} bytes")
        except:
            raise ValueError(f"Failed to read Excel file {filepath}: {e}\n"
                           f"File size: {file_size} bytes\n"
                           f"Please ensure the file is a valid Excel file (.xlsx or .xls).")


def load_claims_triangle(sheet_name: int | str = 0) -> pd.DataFrame:
    """
    Load insurance policy / claims data from ``data/raw/*.xlsx``.

    Resolution order:

    1. Environment variable ``IBNR_RAW_XLSX`` (absolute or relative path).
    2. ``data/raw/ssn_2025_siniestro_casco.xlsx`` (project root).
    3. First ``*.xlsx`` file in ``data/raw/``.

    If the file is missing or not a valid ``.xlsx`` zip archive, a small demo
    workbook is written so the pipeline can run end-to-end.
    """
    from src.data.sample_policies import write_demo_policies_excel

    project_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = project_root / "data" / "raw"
    default_name = "ssn_2025_siniestro_casco.xlsx"

    env_path = os.environ.get("IBNR_RAW_XLSX", "").strip()
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(raw_dir / default_name)
    if raw_dir.is_dir():
        candidates.extend(sorted(raw_dir.glob("*.xlsx")))

    default_path: Path | None = None
    for path in candidates:
        path = path.resolve() if path.is_absolute() else (project_root / path).resolve()
        if path.is_file() and _is_zip_xlsx(path):
            default_path = path
            break

    demo_path = raw_dir / default_name
    if default_path is None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        write_demo_policies_excel(demo_path)
        default_path = demo_path.resolve()
    elif not _is_zip_xlsx(default_path):
        raw_dir.mkdir(parents=True, exist_ok=True)
        write_demo_policies_excel(demo_path)
        default_path = demo_path.resolve()

    return load_siniestros_data(str(default_path), sheet_name=sheet_name)
