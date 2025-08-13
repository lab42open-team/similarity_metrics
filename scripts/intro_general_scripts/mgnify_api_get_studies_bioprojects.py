import os
import requests
import time
import csv

OUTPUT_DIR = "/ccmri/similarity_metrics/data/ena_study_mappings"
FAILED_LOG = os.path.join(OUTPUT_DIR, "failed_pages.txt")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ena_study_accession_mappings.csv")
PAGE_SIZE = 1000
PAGE = 0

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Clear failed log
open(FAILED_LOG, 'w').close()

def fetch_study_page(limit):
    url = f"https://www.ebi.ac.uk/ena/portal/api/search"
    params = {
        "result": "study",
        "fields": "accession,secondary_accession",
        "format": "json",
        "limit": str(limit)
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code != 200:
            print(f"Error: Failed to fetch page {PAGE}, Status Code: {response.status_code}")
            return None
        return response.json()
    except Exception as e:
        print(f"Request failed on page {PAGE}: {e}")
        return None

def save_to_csv(data, output_file):
    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Accession", "Secondary Accession"])
        writer.writerows(data)
    print(f"Saved {len(data)} records to {output_file}")

def main():
    all_records = []
    seen_accessions = set()

    while True:
        print(f"Fetching page {PAGE}...")
        data = fetch_study_page(PAGE_SIZE)
        
        if data is None or len(data) == 0:
            print("No more data or failed request. Stopping.")
            break

        page_records = []
        for entry in data:
            acc = entry.get("accession", "NA")
            sec_acc = entry.get("secondary_accession", "NA")

            if acc not in seen_accessions:
                page_records.append([acc, sec_acc])
                seen_accessions.add(acc)

        if not page_records:
            print("No new records on this page. Stopping.")
            break

        all_records.extend(page_records)
        PAGE += 1
        time.sleep(2)

    if all_records:
        save_to_csv(all_records, OUTPUT_FILE)
    else:
        print("No data fetched.")

if __name__ == "__main__":
    main()
