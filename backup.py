import os
import shutil
import datetime


def backup_python_files(source_root, dest_root, exclude_dirs=None):
    """
    Backup all .py files from source_root to dest_root preserving directory structure.
    :param source_root: The root directory of the source project.
    :param dest_root: The root directory where backups will be stored.
    :param exclude_dirs: A list of directory names to exclude from backup.
    """
    if exclude_dirs is None:
        exclude_dirs = []

    # Check if source directory exists
    if not os.path.isdir(source_root):
        print(f"Error, source directory does not exist -> {source_root}")
        return

    # Check and create destination directory
    try:
        os.makedirs(dest_root, exist_ok=True)
    except OSError as e:
        print(f"Error, could not create/access destination directory -> {dest_root}")
        print(f"Reason: {e}")
        return

    print("=" * 60)
    print("Starting backup of .py files...")
    print(f"Source directory: {source_root}")
    print(f"Target directory: {dest_root}")
    if exclude_dirs:
        print(f"Excluding directories: {', '.join(exclude_dirs)}")
    print("=" * 60)

    copied_count = 0
    # Iterate through the source directory
    for dirpath, dirnames, filenames in os.walk(source_root):
        # Modify dirnames in-place to skip excluded directories
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for filename in filenames:
            # Check for .py files
            if filename.endswith(".py"):

                # Construct the full source file path
                source_file_path = os.path.join(dirpath, filename)

                relative_path = os.path.relpath(source_file_path, source_root)
                dest_file_path = os.path.join(dest_root, relative_path)

                dest_file_dir = os.path.dirname(dest_file_path)
                os.makedirs(dest_file_dir, exist_ok=True)

                # Copy the file
                try:
                    shutil.copy2(source_file_path, dest_file_path)
                    print(f"Backup finished: {relative_path}")
                    copied_count += 1
                except Exception as e:
                    print(f"!!! Copy failed: {relative_path}")
                    print(f"    Reason: {e}")

    print("-" * 60)
    print("Backup finished!")
    print(f"Total .py files backed up: {copied_count}")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    SOURCE_PROJECT_ROOT = r'E:\FYP\TcmPrescriptionOcr'

    DESTINATION_BACKUP_ROOT = r'C:\Users\YFshadaow\OneDrive\CG4001\project_backup'

    backup_python_files(
        SOURCE_PROJECT_ROOT,
        DESTINATION_BACKUP_ROOT,
        exclude_dirs=['.git', '__pycache__', 'venv']
    )
