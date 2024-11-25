import re

input_file = '/home/nefeli/tag_data/pmids_output'
output_file = '/home/nefeli/tag_data/formatted_output.tsv'

def reformat_data(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # Split the line by tabs
            parts = line.strip().split('\t')

            # Handle lines with fewer than 6 columns by filling missing fields
            while len(parts) < 6:
                parts.append("")

            # If there are more than 6 parts, join the extra parts into the "text" field
            if len(parts) > 6:
                identifier = parts[0]
                authors = parts[1]
                journal = parts[2]
                year = parts[3]
                title = parts[4]
                text = ' '.join(parts[5:])  # Combine all remaining parts into the text field
            else:
                identifier, authors, journal, year, title, text = parts

            # Write the reformatted row to the output file
            outfile.write(f"{identifier}\t{authors}\t{journal}\t{year}\t{title}\t{text}\n")

reformat_data(input_file, output_file)
print(f"Reformatted data saved to {output_file}")
