#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'
ARGS_DIR="official_samples/args"
TRACES_DIR="official_samples/traces"
SASS_DIR="official_samples/sass_code"
RESULTS_DIR="official_samples_results"

mkdir -p "$RESULTS_DIR"

SUMMARY_FILE="$RESULTS_DIR/official_samples_summary.txt"
DETAILED_LOG="$RESULTS_DIR/official_samples_detailed.log"

total_samples=0
successful_matches=0
failed_matches=0

log_message() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        "INFO")
            echo -e "${BLUE}[INFO]${NC} $message"
            echo "[$timestamp] [INFO] $message" >> "$DETAILED_LOG"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} $message"
            echo "[$timestamp] [SUCCESS] $message" >> "$DETAILED_LOG"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} $message"
            echo "[$timestamp] [ERROR] $message" >> "$DETAILED_LOG"
            ;;
        "WARNING")
            echo -e "${YELLOW}[WARNING]${NC} $message"
            echo "[$timestamp] [WARNING] $message" >> "$DETAILED_LOG"
            ;;
    esac
}

find_sass_file() {
    local kernel_name=$1
    local sample_name=$2
    
    local sass_file=$(find "$SASS_DIR" -name "*${kernel_name}*" -type f | head -1)
    
    if [[ -z "$sass_file" ]]; then
        sass_file=$(find "$SASS_DIR" -name "*${sample_name}*" -type f | head -1)
    fi
    
    if [[ -z "$sass_file" ]]; then
        local first_word=$(echo "$kernel_name" | cut -d'(' -f1)
        sass_file=$(find "$SASS_DIR" -name "*${first_word}*" -type f | head -1)
    fi
    
    echo "$sass_file"
}

extract_kernel_names() {
    local args_file=$1
    grep "Intercepted kernel launch:" "$args_file" | sed 's/Intercepted kernel launch: //' | cut -d'(' -f1 | sort -u
}

process_sample() {
    local sample_name=$1
    local args_file="$ARGS_DIR/${sample_name}_args.txt"
    local trace_file="$TRACES_DIR/${sample_name}_trace.txt"
    
    log_message "INFO" "Processing sample: $sample_name"
    
    if [[ ! -f "$args_file" ]]; then
        log_message "ERROR" "Args file not found: $args_file"
        return 1
    fi
    if [[ ! -f "$trace_file" ]]; then
        log_message "ERROR" "Trace file not found: $trace_file"
        return 1
    fi
    
    if ! grep -q "Intercepted kernel launch:" "$args_file"; then
        log_message "WARNING" "No kernel launch data found in: $args_file"
        return 1
    fi
    
    local kernel_names=$(extract_kernel_names "$args_file")
    local kernel_count=$(echo "$kernel_names" | wc -l)
    
    log_message "INFO" "Found $kernel_count unique kernel(s) in $sample_name"
    
    local sample_success=true
    
    while IFS= read -r kernel_name; do
        [[ -z "$kernel_name" ]] && continue
        
        log_message "INFO" "Processing kernel: $kernel_name"
        
        local sass_file=$(find_sass_file "$kernel_name" "$sample_name")
        
        if [[ -z "$sass_file" ]]; then
            log_message "WARNING" "No SASS file found for kernel: $kernel_name"
            sample_success=false
            continue
        fi
        
        log_message "INFO" "Using SASS file: $(basename "$sass_file")"
        
        local output_file="$RESULTS_DIR/${sample_name}_$(echo "$kernel_name" | tr '/' '_' | tr '(' '_' | tr ')' '_' | tr '*' '_' | tr ',' '_').json"
        
        if timeout 60 python check_memory_ranges.py \
            -k "$args_file" \
            -t "$trace_file" \
            -s "$sass_file" \
            -n "$kernel_name" > "$RESULTS_DIR/${sample_name}_output.log" 2>&1; then
            
            if grep -q "✅ Memory analysis matches trace data" "$RESULTS_DIR/${sample_name}_output.log"; then
                log_message "SUCCESS" "Analysis successful for $kernel_name"
                ((successful_matches++))
            else
                log_message "ERROR" "Analysis failed for $kernel_name"
                sample_success=false
                ((failed_matches++))
            fi
        else
            log_message "ERROR" "Analysis timed out or crashed for $kernel_name"
            sample_success=false
            ((failed_matches++))
        fi
        
    done <<< "$kernel_names"
    
    if $sample_success; then
        log_message "SUCCESS" "Sample $sample_name completed successfully"
        return 0
    else
        log_message "ERROR" "Sample $sample_name had failures"
        return 1
    fi
}

main() {
    log_message "INFO" "Starting official samples analysis"
    log_message "INFO" "Results will be saved to: $RESULTS_DIR"
    
    > "$SUMMARY_FILE"
    > "$DETAILED_LOG"
    
    local samples=()
    for args_file in "$ARGS_DIR"/*_args.txt; do
        if [[ -f "$args_file" ]]; then
            local sample_name=$(basename "$args_file" "_args.txt")
            samples+=("$sample_name")
        fi
    done
    
    log_message "INFO" "Found ${#samples[@]} samples to process"
    
    for sample in "${samples[@]}"; do
        ((total_samples++))
        echo
        echo "=" * 80
        process_sample "$sample"
        echo "=" * 80
        echo
    done
    
    echo
    echo "============================================================"
    echo "OFFICIAL SAMPLES ANALYSIS SUMMARY"
    echo "============================================================"
    echo "Total samples processed: $total_samples"
    echo "Successful matches: $successful_matches"
    echo "Failed matches: $failed_matches"
    echo "Success rate: $(awk "BEGIN {printf \"%.2f\", $successful_matches * 100 / ($successful_matches + $failed_matches)}")%"
    echo "Results saved to: $RESULTS_DIR"
    echo "============================================================"
    
    {
        echo "OFFICIAL SAMPLES ANALYSIS SUMMARY"
        echo "Generated on: $(date)"
        echo "============================================================"
        echo "Total samples processed: $total_samples"
        echo "Successful matches: $successful_matches"  
        echo "Failed matches: $failed_matches"
        echo "Success rate: $(awk "BEGIN {printf \"%.2f\", $successful_matches * 100 / ($successful_matches + $failed_matches)}")%"
        echo "============================================================"
    } > "$SUMMARY_FILE"
    
    log_message "INFO" "Analysis complete. Check $SUMMARY_FILE for summary."
}

check_dependencies() {
    local missing_deps=()
    
    if ! command -v python &> /dev/null; then
        missing_deps+=("python")
    fi
    
    if [[ ! -f "check_memory_ranges.py" ]]; then
        missing_deps+=("check_memory_ranges.py script")
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_message "ERROR" "Missing dependencies: ${missing_deps[*]}"
        exit 1
    fi
}

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Check official CUDA samples with the memory analysis tool"
    echo
    echo "OPTIONS:"
    echo "  -h, --help     Show this help message"
    echo "  -s, --sample   Process only specific sample (e.g., 'BlackScholes')"
    echo "  -l, --list     List available samples"
    echo "  -v, --verbose  Enable verbose output"
    echo
    echo "EXAMPLES:"
    echo "  $0                           # Process all samples"
    echo "  $0 -s BlackScholes          # Process only BlackScholes sample"
    echo "  $0 -l                       # List available samples"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -s|--sample)
            SINGLE_SAMPLE="$2"
            shift 2
            ;;
        -l|--list)
            echo "Available samples:"
            for args_file in "$ARGS_DIR"/*_args.txt; do
                if [[ -f "$args_file" ]]; then
                    basename "$args_file" "_args.txt"
                fi
            done | sort
            exit 0
            ;;
        -v|--verbose)
            set -x
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

check_dependencies

if [[ -n "$SINGLE_SAMPLE" ]]; then
    log_message "INFO" "Processing single sample: $SINGLE_SAMPLE"
    total_samples=1
    process_sample "$SINGLE_SAMPLE"
    echo
    echo "Single sample analysis complete."
else
    main
fi
