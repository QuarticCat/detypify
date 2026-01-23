import shutil
import sys
import tarfile
from pathlib import Path


def main():
    archive_name = "mathwriting-2024.tgz"
    archive_path = Path(archive_name)

    # If not in current dir, check external/ assuming we are at root
    if not archive_path.exists():
        external_path = Path("external") / archive_name
        if external_path.exists():
            archive_path = external_path
        else:
            print(f"Error: {archive_name} not found in current directory or external/.")
            sys.exit(1)

    print(f"Found archive at: {archive_path}")

    # We will extract into the same directory as the archive
    extract_base = archive_path.parent

    # Define what we want to extract
    target_member_prefix = "mathwriting-2024/symbols"

    print("Extracting symbols (this may take a while)...")
    with tarfile.open(archive_path, "r:gz") as tar:
        # Filter members to extract only the symbols folder
        members = [
            m for m in tar.getmembers() if m.name.startswith(target_member_prefix)
        ]

        if not members:
            print(
                f"Error: No members found starting with '{target_member_prefix}' in archive."
            )
            sys.exit(1)

        tar.extractall(path=extract_base, members=members)

    # The extraction creates extract_base/mathwriting-2024/symbols
    extracted_source = extract_base / "mathwriting-2024" / "symbols"
    final_destination = extract_base / "mathwriting"

    if not extracted_source.exists():
        print(f"Error: Extraction seemed successful but {extracted_source} is missing.")
        sys.exit(1)

    print(f"Renaming {extracted_source} to {final_destination}...")

    if final_destination.exists():
        print(f"Warning: {final_destination} already exists. Merging/Overwriting...")
        # shutil.move will fail if dest is a dir and exists, usually moves *inside* it
        # We want to rename `symbols` to `mathwriting`. If `mathwriting` exists, we probably want to replace it or warn.
        # For safety, let's remove dest if it exists (fresh extraction)
        shutil.rmtree(final_destination)

    shutil.move(extracted_source, final_destination)

    # Cleanup the now empty/partial parent folder 'mathwriting-2024'
    parent_folder = extract_base / "mathwriting-2024"
    if parent_folder.exists():
        # Only remove if empty or contains only directories we don't need?
        # Since we only extracted symbols, it should be empty of files, but might contain empty dirs
        # The user's 'mv' command moves the subdir out, leaving the parent.
        # We can try to remove it.
        try:
            # shutil.rmtree(parent_folder) # This might be too aggressive if user had other stuff there?
            # But we just extracted it.
            # To be safe and mimic 'mv' side effects (leaving the empty shell), we can try rmdir or rmtree
            # Given the context of a fresh extraction, rmtree is probably fine for the wrapper folder.
            shutil.rmtree(parent_folder)
        except Exception as e:
            print(
                f"Warning: Could not remove temporary parent folder {parent_folder}: {e}"
            )

    print("Done.")


if __name__ == "__main__":
    main()
