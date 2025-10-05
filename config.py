from pathlib import Path

# Use absolute path for the whole project
PROJECT_ROOT = Path(__file__).resolve().parent
# Directory to store all prescriptions
DATA_DIR = PROJECT_ROOT / 'data'
# Directory to store original prescription files
ORIGINAL_DATA_DIR = DATA_DIR / 'original'
# Directory to store aligned prescription files (oblique images corrected)
ALIGNED_DATA_DIR = DATA_DIR / 'aligned'

# Target image file format
TARGET_IMAGE_FORMAT = '.png'