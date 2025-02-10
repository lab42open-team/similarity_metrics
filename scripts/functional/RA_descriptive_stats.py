import pandas as pd

def calculate_statistics(dataframe, output_txt_file):
    """Calculate descriptive statistics and save results to a .txt file."""
    # Extract relative abundances
    relative_abundances = dataframe["Count"]

    # Descriptive statistics
    stats = {
        "Min": relative_abundances.min(),
        "Max": relative_abundances.max(),
        "Mean": relative_abundances.mean(),
        "Median": relative_abundances.median(),
        "Standard Deviation": relative_abundances.std(),
        "Most Common Value (Mode)": relative_abundances.mode().values[0] if not relative_abundances.mode().empty else None,
    }

    # Count samples within specific ranges
    #bins = [0, 0.01, 0.05, 0.1, 0.5, 1.0]
    #range_counts = pd.cut(relative_abundances, bins=bins).value_counts(sort=False)

    # Write statistics to a file
    with open(output_txt_file, "a") as f:
        f.write("\nVersion v4.1:\n")
        f.write("Descriptive Statistics:\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
        
        #f.write("\nSamples in Relative Abundance Ranges:\n")
        #for range_, count in range_counts.items():
            #f.write(f"{range_}: {count}\n")
    
    print(f"Statistics saved to {output_txt_file}")

if __name__ == "__main__":
    # Input file with normalized data
    input_file = "/ccmri/similarity_metrics/data/functional/raw_data/GO-slim_abundances/super_table/lf_v4.1_super_table.tsv"
    output_txt_file = "/ccmri/similarity_metrics/data/functional/raw_data/GO-slim_abundances/super_table/counts_statistics_output.txt"
    
    # Load the normalized data
    normalized_df = pd.read_csv(input_file, sep="\t")
    
    # Calculate statistics and write to file
    calculate_statistics(normalized_df, output_txt_file)
