# Import bz2 for decompressing bzip2 compressed files
import bz2
# Import pathlib for cross-platform path handling
import pathlib
# Import mwxml for parsing Wikipedia XML dump files
import mwxml
# Import mwparserfromhell for cleaning Wikipedia markup syntax
import mwparserfromhell
# Import tqdm for showing progress bars during processing
from tqdm import tqdm

# Path to the compressed Wikipedia dump file (downloaded by download_wiki_dump.py)
DUMP_PATH = pathlib.Path("data/wiki/dump/simplewiki-latest-pages-articles.xml.bz2")
# Path where we'll save the extracted clean text (one article per line)
OUT_PATH = pathlib.Path("data/wiki/extracted/wiki_text.txt")
# Create the output directory if it doesn't exist
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def clean_wiki_markup(text: str) -> str:
    """
    Takes Wikipedia markup and returns readable plain text.
    Removes wiki syntax like [[links]], {{templates}}, etc.
    """
    # If text is empty or None, return empty string
    if not text:
        return ""
    # Parse the Wikipedia markup text into a structured format
    wikicode = mwparserfromhell.parse(text)
    # Strip all wiki markup code, leaving only plain text
    plain = wikicode.strip_code()
    # Remove extra whitespace: split by whitespace then join with single spaces
    # This normalizes multiple spaces/tabs/newlines into single spaces
    plain = " ".join(plain.split())
    return plain

def main():
    # Print status messages
    print("Reading dump:", DUMP_PATH)
    print("Writing clean text to:", OUT_PATH)

    # Counter to track how many articles we've written
    written = 0
    # Open both files: the compressed dump (read binary) and output (write text)
    # Using context managers ensures files are properly closed
    with bz2.open(DUMP_PATH, "rb") as f, OUT_PATH.open("w", encoding="utf-8") as out:
        # Parse the Wikipedia XML dump file
        dump = mwxml.Dump.from_file(f)

        # Iterate through all pages in the dump with a progress bar
        for page in tqdm(dump.pages):
            # Skip non-article pages (namespace 0 = main articles)
            # Other namespaces include talk pages, user pages, etc.
            if page.namespace != 0:
                continue

            # Get the latest revision text (page is iterable and yields revisions)
            # Wikipedia pages have multiple revisions; we want the most recent one
            revision = None
            # Iterate through all revisions to get the last one
            for rev in page:
                revision = rev
            
            # If no revision found, skip this page
            if revision is None:
                continue

            # Get the raw text content from the revision
            text = revision.text
            # Clean the text to remove wiki markup and normalize whitespace
            cleaned = clean_wiki_markup(text)

            # Skip tiny pages (less than 200 characters) as they're not useful for training
            if len(cleaned) < 200:
                continue

            # Write the cleaned article text to output file, one per line
            out.write(cleaned + "\n")
            # Increment counter
            written += 1

            # Safety limit so it doesn't get too big for CPU learning
            # Stop after extracting 50,000 articles
            if written >= 50000:
                break

    # Print final count of articles extracted
    print("Done. Articles written:", written)

# Only run main() if this script is executed directly (not imported as a module)
if __name__ == "__main__":
    main()
