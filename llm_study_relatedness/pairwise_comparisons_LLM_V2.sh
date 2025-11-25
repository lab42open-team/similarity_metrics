#!/bin/bash

# Get the current date and time in the desired format
LOGFILE="../logfiles/output_$(date '+%Y-%m-%d_%H-%M-%S').log"

# Redirect both stdout and stderr to the log file
exec > >(tee -a "$LOGFILE") 2>&1

# Capture the start time
start_time=$(date +%s)

# Define models and prompts arrays
models=("phi4:14b-q8_0")
#models=("qwen3:30b-a3b-q8_0")
#"qwen3:30b-a3b-q8_0" "phi4:14b-q8_0" "llama3.1:8b-instruct-q8_0" "qwen2.5:14b-instruct-q8_0" "llama3.3:70b-instruct-q8_0" "qwen2.5:72b-instruct-q8_0" "qwen2.5:32b-instruct-q8_0" "mistral-small:22b-instruct-2409-q8_0" "mistral-nemo:12b-instruct-2407-q8_0"
#prompts=("/home/loukas/5_run_LLM_classifier/prompts/prompt4.tsv" "/home/loukas/5_run_LLM_classifier/prompts/prompt5.tsv")
#"/home/loukas/5_run_LLM_classifier/prompts/prompt4.tsv" "/home/loukas/5_run_LLM_classifier/prompts/prompt5.tsv"
# "C:/Users/Alex/Documents/5_run_structured_output_LLM_classifier/prompts/prompt5_structured_V1.tsv")  # Example prompts


# Define dataset path
#dataset="../datasets/all_mgnify_studies.tsv"
dataset="../datasets/1074.tsv"
studypairs="../datasets/merged_total_studies_similarity.tsv"

# Loop through each model and prompt
for model_choice in "${models[@]}"; do
    model_start_time=$(date +%s)  # Start time for the model

    run_start_time=$(date +%s)  # Start time for the run

    # Generate filenames
    model_base=$(basename "$model_choice" | tr ':' '_')  # Replace ':' with '_'

    # Create a directory for the model and prompt combination if it doesn't exist
    #output_dir="../results/${model_base}"
    output_dir="../results/1074"
    mkdir -p "$output_dir"  # Create the directory if it doesn't exist

    #metrics_output="$output_dir/metrics_LLM_output_${model_base}_${prompt_base}_run_${run}.tsv"

    # Execute scripts
    echo "Running model: $model_choice"
    process_size=8
    for ((process_index=0;process_index<process_size;process_index++)); do
        output="$output_dir/LLM_pairwise_output_${model_base}_part_${process_index}.json"
        /usr/bin/python3 ./pairwise_comparisons_LLM_invoker_V2.py "$model_choice" "$dataset" "$studypairs" "$output" $process_index $process_size &
    done
    wait $(jobs -p)
    cat $output_dir/LLM_pairwise_output_${model_base}_part_*.json > $output_dir/LLM_pairwise_output_${model_base}.json
    rm $output_dir/LLM_pairwise_output_${model_base}_part_*.json
    #/usr/bin/python3 ./structured_final_metrics_5_runs_LLM_classifier.py "$output" "$metrics_output"
    #/c/Users/Alex/AppData/Local/Programs/Python/Python35/python
    #/usr/bin/python3

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
