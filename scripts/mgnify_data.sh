#!/bin/bash

OUTPUT_DIR="/ccmri/similarity_metrics/data/experiment_type/runs_v1"
FAILED_LOG="/ccmri/similarity_metrics/data/experiment_type/failed_ids.txt"
PAGE=1
PAGE_SIZE=10

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Clear the previous failed IDs log if it exists
> "$FAILED_LOG"

while true; do
    OUTPUT_FILE="${OUTPUT_DIR}/runs_page_${PAGE}.json"
    echo "Fetching page ${PAGE}..."

    # Fetch the data and save the response
    RESPONSE=$(curl -s -w "%{http_code}" -o "$OUTPUT_FILE" \
        -X 'GET' \
        "https://www.ebi.ac.uk/metagenomics/api/v1/experiment-types/metagenomic/runs?page=${PAGE}&page_size=${PAGE_SIZE}" \
        -H 'accept: application/json')

    # Extract the status code correctly
    STATUS_CODE=$(echo "$RESPONSE" | tail -n1)

        # Stop fetching if we hit a 404 error
    if [ "$STATUS_CODE" -eq 404 ]; then
        echo "Reached the end of available data at page ${PAGE}. Stopping."
        rm -f "$OUTPUT_FILE"
        break
    fi

    # Check if the request was successful
    if [ "$STATUS_CODE" -ne 200 ]; then
        echo "Error occurred on page ${PAGE}. Status code: $STATUS_CODE"
        echo "$PAGE" >> "$FAILED_LOG"  # Log the failed page number
        rm -f "$OUTPUT_FILE"  # Remove the empty file if the request failed
        PAGE=$((PAGE + 1))
        continue
    fi

    # Check if the output file is empty
    if [ ! -s "$OUTPUT_FILE" ]; then
        echo "No data received for page ${PAGE}, stopping..."
        echo "$PAGE" >> "$FAILED_LOG"
        rm -f "$OUTPUT_FILE"
        break
    fi

    echo "Downloaded page ${PAGE} to ${OUTPUT_FILE}"
    PAGE=$((PAGE + 1))

    # Add a delay to prevent rate limiting
    sleep 2
done

echo "Script completed. Failed pages are logged in $FAILED_LOG"



