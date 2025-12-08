#!/bin/bash

# Get the current date and time in the desired format
LOGFILE="/_full_path_in_your_server_to_/output_$(date '+%Y-%m-%d_%H-%M-%S').log"

# Redirect both stdout and stderr to the log file
exec > >(tee -a "$LOGFILE") 2>&1

# Capture the start time
start_time=$(date +%s)

# Define models array
models=("phi4:14b-q8_0")
#models=("qwen3:30b-a3b-q8_0" "phi4:14b-q8_0")


# Define dataset path
dataset="/_full_path_in_your_server_to_/all_mgnify_studies.tsv"
studypairs="/_full_path_in_your_server_to_/merged_total_studies_similarity.tsv"

# Count dataset lines (datasets have no header)
total_lines=$(wc -l < "$dataset")

# Adjust number of processes dynamically
if [ "$total_lines" -lt 50 ]; then
    process_size=1
elif [ "$total_lines" -lt 500 ]; then
    process_size=2
else
    process_size=8
fi

echo "Dataset has $total_lines lines → Using $process_size parallel processes."

# Loop through each model
for model_choice in "${models[@]}"; do
    model_start_time=$(date +%s)  # Start time for the model

    run_start_time=$(date +%s)  # Start time for the run

    # Generate filenames
    model_base=$(basename "$model_choice" | tr ':' '_')  # Replace ':' with '_'

    # Create output directory
    output_dir="/_full_path_in_your_server_to_/"
    mkdir -p "$output_dir"

    # Execute scripts
    echo "Running model: $model_choice"
    for ((process_index=0;process_index<process_size;process_index++)); do
        output="$output_dir/LLM_pairwise_output_${model_base}_part_${process_index}.json"
        /usr/bin/python3 ./pairwise_comparisons_LLM_invoker_V2.py \
            "$model_choice" "$dataset" "$studypairs" "$output" \
            $process_index $process_size &
    done
    wait $(jobs -p)

    # Check if all expected part-files exist and are non-empty
    missing_parts=false
    for ((process_index=0;process_index<process_size;process_index++)); do
        part_file="$output_dir/LLM_pairwise_output_${model_base}_part_${process_index}.json"
        if [ ! -s "$part_file" ]; then
            echo "Warning: Missing or empty output file: $part_file" >> "$LOGFILE"
            missing_parts=true
        fi
    done

    if [ "$missing_parts" = false ]; then
        # Merge safely
        cat $output_dir/LLM_pairwise_output_${model_base}_part_*.json > \
            $output_dir/LLM_pairwise_output_${model_base}.json
        rm $output_dir/LLM_pairwise_output_${model_base}_part_*.json
        echo "Successfully merged output for $model_choice" >> "$LOGFILE"
    else
        echo "Skipping merge due to missing part files for $model_choice" >> "$LOGFILE"
    fi

    # Capture runtime per run
    run_end_time=$(date +%s)
    run_duration=$((run_end_time - run_start_time))
    run_minutes=$((run_duration / 60))
    run_seconds=$((run_duration % 60))
    echo "Model: $model_choice | Run completed in $run_minutes min $run_seconds sec" >> "$LOGFILE"

    # Capture total runtime per model
    model_end_time=$(date +%s)
    model_duration=$((model_end_time - model_start_time))
    model_hours=$((model_duration / 3600))
    model_minutes=$(( (model_duration % 3600) / 60 ))
    model_seconds=$((model_duration % 60))
    echo "Model: $model_choice | Total runtime for 5 runs: $model_hours hours $model_minutes min $model_seconds sec" >> "$LOGFILE"
done

# Capture total script execution time
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

echo "Script execution completed in $hours hours, $minutes minutes, and $seconds seconds." >> "$LOGFILE"
