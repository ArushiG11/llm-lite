# Import pathlib for cross-platform path handling
import pathlib
# Import urllib.request for downloading files from URLs
import urllib.request

# URL to the Simple English Wikipedia dump file (compressed bz2 format)
# Simple English Wikipedia uses simpler language, making it better for training
URL = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"

# Define the output path where we'll save the downloaded dump file
OUT_PATH = pathlib.Path("data/wiki/dump/simplewiki-latest-pages-articles.xml.bz2")
# Create the parent directory (data/wiki/dump/) if it doesn't exist
# parents=True creates all parent directories, exist_ok=True doesn't error if it exists
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Print status messages to inform user about the download
print("Downloading Wikipedia dump...")
print("From:", URL)
print("To:", OUT_PATH)

# Download the file from URL and save it to OUT_PATH
# This is a blocking operation that will take time depending on file size and connection
urllib.request.urlretrieve(URL, OUT_PATH)

# Confirm that the download completed successfully
print("Done! File saved at:", OUT_PATH)