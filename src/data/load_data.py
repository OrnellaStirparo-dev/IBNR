import pandas as pd
from pathlib import Path


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
    Convenience loader used by main.py.

    For now it just loads the provided raw SSN siniestros file and returns
    the raw DataFrame. You can later transform it into a proper triangle
    in src/data/transformer.py.
    
    Args:
        sheet_name: Sheet to load (0-based index or sheet name). Default: 0 (first sheet)
    
    Returns:
        DataFrame with the loaded data
    """
    # Try multiple possible locations for the Excel file
    # 1. Project root (where the actual 265KB file is located)
    project_root = Path(__file__).resolve().parent.parent.parent
    possible_paths = [
        project_root / "ssn_2025_siniestro_casco.xlsx",  # Project root
        project_root / "data" / "raw" / "ssn_2025_siniestro_casco.xlsx",  # data/raw subdirectory
        Path("ssn_2025_siniestro_casco.xlsx"),  # Current working directory
        Path("data/raw/ssn_2025_siniestro_casco.xlsx"),  # Relative to CWD
    ]
    
    # Find the first existing file that's actually an Excel file (not a placeholder)
    default_path = None
    for path in possible_paths:
        if path.exists():
            # Check if it's a real Excel file (not the 83B placeholder)
            file_size = path.stat().st_size
            if file_size > 1024:  # At least 1KB to be a real Excel file
                default_path = path.resolve()
                break
    
    if default_path is None:
        raise FileNotFoundError(
            f"Excel file not found. Checked:\n"
            f"  {project_root / 'ssn_2025_siniestro_casco.xlsx'}\n"
            f"  {project_root / 'data' / 'raw' / 'ssn_2025_siniestro_casco.xlsx'}\n"
            f"  {Path('ssn_2025_siniestro_casco.xlsx').resolve()}\n"
            f"  {Path('data/raw/ssn_2025_siniestro_casco.xlsx').resolve()}\n"
            f"Please ensure the Excel file exists and is at least 1KB in size."
        )
    
    return load_siniestros_data(str(default_path), sheet_name=sheet_name)
