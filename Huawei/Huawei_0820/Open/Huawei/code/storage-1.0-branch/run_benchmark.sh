#!/bin/bash
###########################################################################################
# Our current test environment does not have a complete training and storage cluster,     #
# as well as the task scheduling module. Therefore, this demo script is used to simulate  #
# the data pre-loading and training process. It contains two steps. The first step        #
# is to preload data, and the second step is to invoke benchmark.sh provided by the       #
# MLPerf Storage benchmark to start training.                                             #
###########################################################################################
# Function to display usage information
usage() {
    echo "Usage: $0 run {args}"
    echo "Args:"
    echo "  --hosts                 Comma-separated list of host IPs"
    echo "  --num-accelerators      Number of accelerators"
    echo "  --param                 Additional parameters"
    echo "Examples:"
    echo "  $0 run --hosts 10.117.61.121,10.117.61.165 --num-accelerators 16 --param dataset.num_files_train=8000 --param dataset.data_folder=unet3d_data --param dataset.num_subfolders_train=48"
}

# Function to validate that a required parameter is not empty
validate_non_empty() {
    local name=$1
    local value=$2
    if [ -z "$value" ]; then
        echo "Error: $name is required"
        usage
        exit 1
    fi
}

# Function to distribute node indices across IPs
distribute_nodes() {
    local hosts=$1
    local num_accelerators=$2
    local -a ips
    IFS=',' read -ra ips <<< "$hosts"

    local total_ips=${#ips[@]}
    local -A node_distribution

    for ((i=0; i<num_accelerators; i++)); do
        local ip_index=$((i % total_ips))
        local ip=${ips[$ip_index]}
        node_distribution["$ip"]+="$i,"
    done

    for ip in "${ips[@]}"; do
        node_distribution["$ip"]="${node_distribution["$ip"]%,}" # remove trailing comma
        echo "$ip: ${node_distribution[$ip]}"
    done
}

# Function to prepare data on multiple hosts
preparedata() {
    local hosts=$1
    local num_accelerators=$2
    shift 2
    local params=("$@")

    local dataturbo_command="preparedata"
    local required_params=("dataset.num_files_train" "dataset.data_folder" "dataset.num_subfolders_train")

    for param in "${params[@]}"; do
        case $param in
            dataset.num_files_train=*|dataset.data_folder=*|dataset.num_subfolders_train=*)
                dataturbo_command+=" $param"
                ;;
            *)
                echo "Unknown parameter $param"
                usage
                exit 1
                ;;
        esac
    done

    # Check if required parameters are present
    for required_param in "${required_params[@]}"; do
        if [[ ! " ${params[@]} " =~ "${required_param}=" ]]; then
            echo "Error: Missing required parameter $required_param"
            usage
            exit 1
        fi
    done

    echo "Preparing data..."

    # Distribute node indices across IPs and create commands
    declare -A node_distribution
    while IFS= read -r line; do
        IFS=': ' read -r ip indices <<< "$line"
        node_distribution["$ip"]="$indices"
    done < <(distribute_nodes "$hosts" "$num_accelerators")

    # Execute data preparation command on each host with rank_id
    IFS=',' read -ra ADDR <<< "$hosts"
    for host in "${ADDR[@]}"; do
        rank_ids=${node_distribution["$host"]}
        full_command="$dataturbo_command rank_id=$rank_ids"
        echo "Executing data preparation command on host $host with rank_ids $rank_ids"
        echo "$full_command"
        ssh "$host" "$full_command" &

        if [ $? -ne 0 ]; then
            echo "Data preparation on host $host with rank_ids $rank_ids failed."
            exit 1
        fi
    done
    wait

    echo "Data preparation completed successfully."
}

# Function to run benchmark using ./benchmark.sh
benchmark_run() {
    local cmd="./benchmark.sh run $@"
    echo "Executing benchmark command:"
    echo "$cmd"
    # execute benchmark command
    eval "$cmd"

    if [ $? -ne 0 ]; then
        echo "Benchmark execution failed."
        exit 1
    fi

    echo "Benchmark completed successfully."
}

main() {
    local mode=$1; shift

    if [[ "$mode" != "run" ]]; then
        echo "Invalid mode: $mode"
        usage
        exit 1
    fi

    # Validate and parse command line arguments
    local hosts=""
    local num_accelerators=""
    local results_dir=""
    local params=()

    while [ $# -gt 0 ]; do
        case "$1" in
            --hosts ) hosts="$2"; shift 2 ;;
            --num-accelerators ) num_accelerators="$2"; shift 2 ;;
            --param ) params+=("$2"); shift 2 ;;
            --results-dir ) results_dir="$2"; shift 2 ;;
            * ) echo "Invalid option $1"; usage; exit 1 ;;
        esac
    done

    # Validate required parameters
    validate_non_empty "hosts" "$hosts"
    validate_non_empty "num-accelerators" "$num_accelerators"

    # Call preparedata function
    preparedata "$hosts" "$num_accelerators" "${params[@]}"

    # Call benchmark_run function
    if [ -n "$results_dir" ]; then
        benchmark_run "${params[@]}" --results-dir="$results_dir"
    else
        benchmark_run "${params[@]}"
    fi
}

main "$@"

