"""
EcoVision Lite - Dataset Acquisition and Extraction Module.

This script manages dataset retrieval for the Garbage Classification dataset:
1. Detects existing extracted dataset in data/garbage_classification/
2. Detects local zip archives (e.g., Unconfirmed *.crdownload, archive.zip, etc.)
3. Downloads directly from Kaggle via kagglehub / kaggle API if not available locally
4. Provides clear authentication troubleshooting instructions if credentials are required
"""

import os
import sys
import glob
import zipfile
import shutil
from pathlib import Path

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

DATASET_HANDLE = "mostafaabla/garbage-classification"
DATA_DIR = Path("data")
TARGET_DIR = DATA_DIR / "garbage_classification"


def get_kaggle_credentials_guide() -> str:
    """Return instructions for configuring Kaggle authentication."""
    return """
========================================================================
                      KAGGLE SETUP INSTRUCTIONS
========================================================================
To download datasets directly from Kaggle using the API:
1. Log in to your Kaggle account at https://www.kaggle.com
2. Navigate to your Account Settings: https://www.kaggle.com/settings
3. Scroll to the "API" section and click "Create New Token".
4. A file named 'kaggle.json' will be downloaded to your computer.
5. Place 'kaggle.json' in:
   - Windows: C:\\Users\\<Your-Username>\\.kaggle\\kaggle.json
   - Linux/Mac: ~/.kaggle/kaggle.json

Alternatively:
- Download the dataset manually from:
  https://www.kaggle.com/mostafaabla/garbage-classification
- Place the downloaded .zip file into the project's 'data/' folder or root folder.
========================================================================
"""

def extract_selected_folder_from_zip(zip_filepath: Path, subfolder_prefix: str, output_dir: Path) -> int:
    """
    Extract only files from a specific subfolder (e.g. 'standardized_256/') 
    from the zip file directly into output_dir to save disk space and time.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_count = 0
    
    print(f"📦 Extracting '{subfolder_prefix}' from {zip_filepath.name} into {output_dir}...")
    
    with zipfile.ZipFile(zip_filepath, 'r') as z:
        members = z.namelist()
        # Find members that match subfolder_prefix
        matching_members = [
            m for m in members 
            if m.startswith(subfolder_prefix) and not m.endswith('/')
        ]
        
        total = len(matching_members)
        if total == 0:
            # Fallback to any image files in the zip
            print(f"Prefix '{subfolder_prefix}' not found. Inspecting root entries...")
            matching_members = [m for m in members if not m.endswith('/') and m.lower().endswith(('.jpg', '.jpeg', '.png'))]
            total = len(matching_members)
            prefix_len = 0
        else:
            prefix_len = len(subfolder_prefix)

        print(f"Found {total} images to extract...")
        
        for i, member in enumerate(matching_members):
            # Extract to target directory without leading prefix
            rel_path = member[prefix_len:] if prefix_len > 0 else member
            target_file = output_dir / rel_path
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            with z.open(member) as source, open(target_file, "wb") as target:
                shutil.copyfileobj(source, target)
            
            extracted_count += 1
            if (i + 1) % 1500 == 0 or (i + 1) == total:
                print(f"  Extracted {i + 1}/{total} files ({int((i+1)/total*100)}%)...")

    print(f"✅ Successfully extracted {extracted_count} images to {output_dir}!")
    return extracted_count

def find_local_archive() -> Path | None:
    """Check for existing archives in the project directory or data directory."""
    patterns = [
        "*.crdownload",
        "*.zip",
        "data/*.zip",
        "data/*.crdownload",
        "../archive.zip"
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        for m in matches:
            p = Path(m)
            if p.is_file() and p.stat().st_size > 10_000_000: # larger than 10MB
                return p
    return None

def is_dataset_ready(dir_path: Path) -> bool:
    """Check if the target directory exists and has image categories."""
    if not dir_path.is_dir():
        return False
    categories = [d for d in dir_path.iterdir() if d.is_dir()]
    if not categories:
        return False
    # Check total image count
    img_count = sum(len(list(c.glob("*.*"))) for c in categories)
    return img_count >= 100

def download_or_prepare_dataset() -> Path:
    """Ensure dataset is present, downloaded, or extracted."""
    print("=" * 60)
    print("🌿 EcoVision Lite - Dataset Manager")
    print("=" * 60)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Check if already extracted
    if is_dataset_ready(TARGET_DIR):
        print(f"✅ Dataset already available at: {TARGET_DIR.resolve()}")
        return TARGET_DIR

    # Check if nested garbage_classification exists
    nested_path = DATA_DIR / "garbage-classification"
    if is_dataset_ready(nested_path):
        print(f"✅ Found dataset at: {nested_path.resolve()}")
        return nested_path

    # 2. Check for local zip or crdownload archive
    local_archive = find_local_archive()
    if local_archive:
        print(f"📦 Detected local dataset archive: {local_archive}")
        try:
            # Check if standardized_256 or original exists in the zip
            with zipfile.ZipFile(local_archive, 'r') as z:
                names = z.namelist()
                has_std256 = any(n.startswith("standardized_256/") for n in names)
                prefix = "standardized_256/" if has_std256 else "original/"
                if not has_std256 and not any(n.startswith("original/") for n in names):
                    prefix = ""
            
            extract_selected_folder_from_zip(local_archive, prefix, TARGET_DIR)
            if is_dataset_ready(TARGET_DIR):
                return TARGET_DIR
        except Exception as e:
            print(f"⚠️ Failed extracting local archive: {e}")

    # 3. Download via kagglehub
    print(f"🌐 Attempting direct download from Kaggle: {DATASET_HANDLE}...")
    try:
        import kagglehub
        downloaded_path = kagglehub.dataset_download(DATASET_HANDLE)
        print(f"✅ Downloaded to: {downloaded_path}")
        dl_path = Path(downloaded_path)
        
        # Check if standardized_256 is in the downloaded directory
        std_path = dl_path / "standardized_256"
        orig_path = dl_path / "original"
        
        source_dir = None
        if std_path.exists() and is_dataset_ready(std_path):
            source_dir = std_path
        elif orig_path.exists() and is_dataset_ready(orig_path):
            source_dir = orig_path
        elif is_dataset_ready(dl_path):
            source_dir = dl_path
            
        if source_dir:
            print(f"Copying/linking {source_dir} to {TARGET_DIR}...")
            # Create a symlink or copy
            shutil.copytree(source_dir, TARGET_DIR, dirs_exist_ok=True)
            return TARGET_DIR
        else:
            return dl_path

    except Exception as e:
        print(f"\n❌ Direct Kaggle download encountered an issue: {e}")
        print(get_kaggle_credentials_guide())
        raise RuntimeError(f"Could not automatically download dataset: {e}")

if __name__ == "__main__":
    try:
        path = download_or_prepare_dataset()
        print(f"\n🎉 Dataset setup complete! Ready at: {path}")
    except Exception as exc:
        print(f"\nError: {exc}")
        sys.exit(1)
