#!/bin/bash
# This script extracts video files from zip archives whose filenames start with
# 'demonstrations' and moves all .MP4 files into the same directory as the zip files.

# Set the destination directory to the current working directory.
DEST_DIR=$(pwd)
echo "Using destination directory: $DEST_DIR"

# Loop over all zip files starting with the prefix.
for zip_file in demonstrations*.zip; do
    # Check if the file exists (in case no files match the pattern).
    if [ -f "$zip_file" ]; then
        echo "Processing $zip_file..."

        # Create a temporary directory for extraction.
        TEMP_DIR=$(mktemp -d)

        # Extract the zip file quietly into the temporary directory.
        unzip -q "$zip_file" -d "$TEMP_DIR"

        # Find and move only .MP4 files (case-insensitive) to DEST_DIR.
        find "$TEMP_DIR" -type f -iname "*.MP4" -exec mv {} "$DEST_DIR" \;

        # Remove the temporary directory.
        rm -rf "$TEMP_DIR"
    else
        echo "No zip file found matching: $zip_file"
    fi
done

echo "All .MP4 videos have been extracted to '$DEST_DIR'."