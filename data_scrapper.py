import time
import csv
import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import shutil

def scrape_valmiki_ramayana():
    """
    Scraping the Valmiki Ramayana and save all verses to a single CSV file
    with the format: Kanda/Book, Sarga/Chapter, Shloka/Verse Number, English Translation
    """
    # Base URL
    base_url = "https://www.valmikiramayan.net"
    
    # Define the kandas with their URL patterns and details
    kandas = [
        {
            "name": "Bala Kanda", 
            "url_prefix": "utf8/baala", 
            "file_prefix": "bala",
            "total_sargas": 77
        },
        {
            "name": "Ayodhya Kanda", 
            "url_prefix": "utf8/ayodhya", 
            "file_prefix": "ayodhya",
            "total_sargas": 119
        },
        {
            "name": "Aranya Kanda", 
            "url_prefix": "utf8/aranya", 
            "file_prefix": "aranya",
            "total_sargas": 75
        },
        {
            "name": "Kishkindha Kanda", 
            "url_prefix": "utf8/kish", 
            "file_prefix": "kishkindha",
            "total_sargas": 67
        },
        {
            "name": "Sundara Kanda", 
            "url_prefix": "utf8/sundara", 
            "file_prefix": "sundara",
            "total_sargas": 68
        },
        {
            "name": "Yuddha Kanda", 
            "url_prefix": "utf8/yuddha", 
            "file_prefix": "yuddha",
            "total_sargas": 128
        }
    ]
    
    # Create a directory to store the scraped data
    os.makedirs("ramayana_data", exist_ok=True)
    
    # Set headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    # Create the main output CSV file with the desired format
    output_csv_path = os.path.join("ramayana_data", "merged_ramayana.csv")
    
    # Initialize the CSV file with headers
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        csv_writer.writerow(["Kanda/Book", "Sarga/Chapter", "Shloka/Verse Number", "English Translation"])
    
    print(f"Starting to scrape Valmiki Ramayana...")
    print(f"Output will be saved to: {output_csv_path}")
    
    total_verses_scraped = 0
    
    # Process each kanda
    for kanda in kandas:
        kanda_name = kanda["name"]
        url_prefix = kanda["url_prefix"]
        file_prefix = kanda["file_prefix"]
        total_sargas = kanda["total_sargas"]
        
        print(f"\nProcessing {kanda_name}...")
        kanda_verses = 0
        
        # Process each sarga in this kanda
        for sarga_num in range(1, total_sargas + 1):
            print(f"  Processing {kanda_name} Sarga {sarga_num}...")
            
            # Try all possible content URLs directly, prioritizing the "roman" version
            content_urls = [
                f"{base_url}/{url_prefix}/sarga{sarga_num}/{file_prefix}roman{sarga_num}.htm",
                f"{base_url}/{url_prefix}/sarga{sarga_num}/{file_prefix}_roman_{sarga_num}.htm",
                f"{base_url}/{url_prefix}/sarga{sarga_num}/{file_prefix}itrans{sarga_num}.htm",
                f"{base_url}/{url_prefix}/sarga{sarga_num}/{file_prefix}sans{sarga_num}.htm",
                f"{base_url}/{url_prefix}/sarga{sarga_num}/{file_prefix}_{sarga_num}.htm",
                f"{base_url}/{url_prefix}/sarga{sarga_num}/{file_prefix}{sarga_num}.htm"
            ]
            
            content_html = None
            content_url = None
            
            # Try each content URL
            for url in content_urls:
                try:
                    print(f"    Trying content URL: {url}")
                    response = requests.get(url, headers=headers, timeout=10)
                    if response.status_code == 200 and len(response.text) > 500:
                        content_html = response.text
                        content_url = url
                        print(f"    Successfully accessed: {url}")
                        break
                except Exception as e:
                    print(f"    Error accessing {url}: {str(e)}")
            
            # If direct content URLs didn't work, try the frame URL as a fallback
            if not content_html:
                try:
                    frame_url = f"{base_url}/{url_prefix}/sarga{sarga_num}/{file_prefix}_{sarga_num}_frame.htm"
                    print(f"    Trying frame URL: {frame_url}")
                    response = requests.get(frame_url, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        frame_soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Look for frame elements
                        frames = frame_soup.find_all('frame')
                        if frames:
                            # Try to find the content frame (usually named "content")
                            content_frame = None
                            for frame in frames:
                                if frame.get('name', '').lower() == 'content' or 'content' in frame.get('src', '').lower():
                                    content_frame = frame
                                    break
                            
                            if not content_frame and frames:
                                # If can't identify specific content frame, just use the first one
                                content_frame = frames[0]
                            
                            if content_frame and content_frame.get('src'):
                                # Get the content URL from the frame
                                frame_content_url = urljoin(frame_url, content_frame['src'])
                                try:
                                    print(f"    Following frame to: {frame_content_url}")
                                    content_response = requests.get(frame_content_url, headers=headers, timeout=10)
                                    if content_response.status_code == 200:
                                        content_html = content_response.text
                                        content_url = frame_content_url
                                except Exception as e:
                                    print(f"    Error accessing frame content: {str(e)}")
                        else:
                            # The frame page might directly contain content
                            content_html = response.text
                            content_url = frame_url
                except Exception as e:
                    print(f"    Error accessing frame URL: {str(e)}")
            
            if not content_html:
                print(f"  Could not access content for {kanda_name} Sarga {sarga_num}. Skipping.")
                continue
            
            # Parse the content
            soup = BeautifulSoup(content_html, 'html.parser')
            
            # Extract verses using multiple methods
            verses = extract_verses_from_content(soup)
            
            # Write verses to main CSV file
            if verses:
                with open(output_csv_path, 'a', encoding='utf-8', newline='') as f:
                    csv_writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                    for verse in verses:
                        csv_writer.writerow([
                            kanda_name,                    # Kanda/Book
                            sarga_num,                     # Sarga/Chapter
                            verse['verse_num'],            # Shloka/Verse Number
                            verse['text']                  # English Translation
                        ])
                
                sarga_verse_count = len(verses)
                kanda_verses += sarga_verse_count
                total_verses_scraped += sarga_verse_count
                print(f"  Successfully extracted {sarga_verse_count} verses from {kanda_name} Sarga {sarga_num}")
            else:
                print(f"  No verses found in {kanda_name} Sarga {sarga_num}")
            
            # Be nice to the server
            time.sleep(2)
        
        print(f"Finished processing {kanda_name} - Total verses: {kanda_verses}")
    
    print(f"\nScraping completed successfully!")
    print(f"Total verses scraped: {total_verses_scraped}")
    print(f"All data has been saved to: {output_csv_path}")
    
    return output_csv_path, total_verses_scraped

def extract_verses_from_content(soup):
    """
    Extract verses from BeautifulSoup content using multiple methods.
    Returns a list of dictionaries with verse_num and text.
    """
    verses = []
    
    # Method 1: Find paragraphs with class "tat" (most reliable)
    tat_paragraphs = soup.select("p.tat")
    
    if tat_paragraphs:
        print(f"    Found {len(tat_paragraphs)} paragraphs with class 'tat'")
                 
        for i, p in enumerate(tat_paragraphs, 1):
            text = p.text.strip()
            
            # Skip very short or empty paragraphs
            if len(text) < 5:
                continue
                           
            # Clean up the text
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Try to extract verse number from content if available
            verse_ref_match = re.search(r'\[(\d+)-(\d+)-(\d+)\]', text)
            if verse_ref_match:
                verse_num = int(verse_ref_match.group(3))
                # Remove the reference from the text
                text = re.sub(r'\s*\[\d+-\d+-\d+\].*$', '', text).strip()
            else:
                verse_num = i
            
            verses.append({
                'verse_num': verse_num,
                'text': text
            })
    
    # Method 2: Look for translation markers
    if not verses:
        # Try to find elements with IDs like "sl10", "sl11" (seen in some pages)
        verse_elements = soup.select('[id^="sl"]')
        if verse_elements:
            print(f"    Found {len(verse_elements)} elements with 'sl' IDs")
            for elem in verse_elements:
                # Extract verse number from ID if possible
                verse_id = elem.get('id', '')
                verse_num_match = re.search(r'sl(\d+)', verse_id)
                verse_num = int(verse_num_match.group(1)) if verse_num_match else 0
                
                # Get text from this element and next sibling
                text = elem.text.strip()
                next_elem = elem.find_next_sibling()
                if next_elem and not next_elem.get('id', '').startswith('sl'):
                    next_text = next_elem.text.strip()
                    if len(next_text) > len(text):
                        text = next_text
                
                if len(text) > 20:  # Skip headers and short text
                    # Clean up the text
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    verses.append({
                        'verse_num': verse_num,
                        'text': text
                    })
    
    # Method 3: Find via pattern matching
    if not verses:
        # Look for all paragraphs with decent length
        all_paragraphs = soup.find_all(['p', 'div'])
        
        verse_pattern = re.compile(r'(\d+\.\d+)\s+(.+)', re.DOTALL)
        verse_num = 1
        
        for p in all_paragraphs:
            text = p.text.strip()
            
            # Skip very short paragraphs and navigation/menu text
            if len(text) < 20 or 'next' in text.lower() or 'previous' in text.lower():
                continue
            
            # Check for numbered verse pattern
            verse_match = verse_pattern.match(text)
            if verse_match:
                verse_num_text = verse_match.group(1)
                verse_text = verse_match.group(2).strip()
                
                try:
                    verse_num = int(verse_num_text.split('.')[-1])
                except:
                    verse_num += 1
                
                verses.append({
                    'verse_num': verse_num,
                    'text': verse_text
                })
                continue
            
            # Look for paragraphs with translation characteristics
            if ((text.startswith('"') or text[0].isupper()) and 
                len(text) > 50 and
                '.' in text and
                not text.startswith('Copyright') and
                not 'valmiki' in text.lower() and
                not 'ramayana' in text.lower()):
                
                # Extract verse number if possible from reference [#-#-#]
                verse_ref_match = re.search(r'\[(\d+)-(\d+)-(\d+)\]', text)
                if verse_ref_match:
                    verse_num = int(verse_ref_match.group(3))
                    # Remove the reference from the text
                    text = re.sub(r'\s*\[\d+-\d+-\d+\].*$', '', text).strip()
                
                # Clean up the text
                if text.startswith('"') and text.endswith('"'):
                    text = text[1:-1].strip()
                
                text = re.sub(r'\s+', ' ', text).strip()
                
                verses.append({
                    'verse_num': verse_num,
                    'text': text
                })
                verse_num += 1
    
    return verses

def remove_first_two_rows(input_file="ramayana_data/merged_ramayana.csv"):
    """
    Remove the first two data rows from a CSV file (keeping the header).
    
    Args:
        input_file (str): Path to the input CSV file
    """
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found!")
        return False
    
    temp_file = input_file + ".temp"
    
    try:
        # Read the original file and write to temp file, skipping first two data rows
        with open(input_file, 'r', encoding='utf-8', newline='') as infile:
            with open(temp_file, 'w', encoding='utf-8', newline='') as outfile:
                reader = csv.reader(infile)
                writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
                
                # Read and write the header
                header = next(reader)
                writer.writerow(header)
                print(f"\nHeader: {header}")
                
                # Skip the first two data rows
                row_count = 0
                skipped_rows = []
                
                for i, row in enumerate(reader):
                    if i < 2:  # Skip first two data rows (index 0 and 1)
                        skipped_rows.append(row)
                        print(f"Skipping row {i+1}: {row}")
                        continue
                    
                    # Write all remaining rows
                    writer.writerow(row)
                    row_count += 1
        
        # Replace the original file with the temp file
        shutil.move(temp_file, input_file)
        print(f"\n✓ Successfully cleaned '{input_file}'")
        print(f"✓ Removed 2 rows from the beginning")
        print(f"✓ Remaining data rows: {row_count}")
        
        # Show what was removed
        print(f"\nRemoved rows:")
        for i, row in enumerate(skipped_rows):
            print(f"  Row {i+1}: {row}")
        
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return False

def verify_output_file(file_path):
    """Verify the output file and print some statistics."""
    if not os.path.exists(file_path):
        print(f"Error: Output file {file_path} was not created!")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Skip header row
            
            verse_count = 0
            kanda_stats = {}
            
            for row in reader:
                if len(row) >= 4:
                    kanda = row[0]
                    verse_count += 1
                    
                    if kanda not in kanda_stats:
                        kanda_stats[kanda] = 0
                    kanda_stats[kanda] += 1
        
        print(f"\n=== Output File Verification ===")
        print(f"File: {file_path}")
        print(f"Total verses: {verse_count}")
        print(f"File size: {os.path.getsize(file_path)} bytes")
        print(f"Headers: {headers}")
        
        print(f"\nVerses per Kanda:")
        for kanda, count in kanda_stats.items():
            print(f"  {kanda}: {count} verses")
        
        # Show first 5 data rows for verification
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            
            print(f"\nFirst 5 data rows:")
            for i, row in enumerate(reader):
                if i >= 5:
                    break
                print(f"  Row {i+1}: {row}")
        
        return True
        
    except Exception as e:
        print(f"Error verifying output file: {e}")
        return False

def main():
    """Main function to run the scraper and clean the output."""
    print("=== Valmiki Ramayana Scraper with Auto-Cleanup ===")
    print("This will:")
    print("1. Scrape the entire Valmiki Ramayana")
    print("2. Save it in format: Kanda/Book, Sarga/Chapter, Shloka/Verse Number, English Translation")
    print("3. Remove the first two non-verse rows automatically")
    print()
    
    try:
        # Step 1: Scrape the Ramayana
        output_file, total_verses = scrape_valmiki_ramayana()
        
        # Step 2: Verify the initial output
        print(f"\n=== Initial Scraping Results ===")
        if verify_output_file(output_file):
            print(f"✓ Initial scraping completed successfully!")
        
        # Step 3: Remove first two rows (non-verse rows)
        print(f"\n=== Cleaning Non-Verse Rows ===")
        if remove_first_two_rows(output_file):
            print(f"✓ Successfully removed first two non-verse rows!")
            
            # Step 4: Final verification
            print(f"\n=== Final Results ===")
            verify_output_file(output_file)
            
            print(f"\n🎉 COMPLETE! 🎉")
            print(f"✓ Ramayana scraped and cleaned successfully!")
            print(f"✓ Output file: {output_file}")
            print(f"✓ Ready for use in your fact verification system!")
        else:
            print(f"✗ Failed to clean the output file")
            
    except KeyboardInterrupt:
        print(f"\n\nScraping interrupted by user.")
    except Exception as e:
        print(f"\n\nError during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()