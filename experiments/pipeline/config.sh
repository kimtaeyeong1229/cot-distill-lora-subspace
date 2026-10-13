
PAPER_CELLS="pairA/qwen0.5b:Qwen2.5-0.5B-Instruct pairA/llama1b:Llama-3.2-1B-Instruct pairA/olmo1b:OLMo-2-1B-Instruct
pairB/qwen0.5b:Qwen2.5-0.5B-Instruct pairB/llama1b:Llama-3.2-1B-Instruct pairB/olmo1b:OLMo-2-1B-Instruct
pairC/qwen0.5b:Qwen2.5-0.5B-Instruct pairC/llama1b:Llama-3.2-1B-Instruct pairC/olmo1b:OLMo-2-1B-Instruct"

PAPER_SWITCH="pairA/qwen0.5b:0.4 pairA/llama1b:0.4 pairA/olmo1b:0.4 pairB/qwen0.5b:0.6 pairB/llama1b:0.6 pairB/olmo1b:0.6
pairC/qwen0.5b:0.6 pairC/llama1b:0.6 pairC/olmo1b:0.6"

PAPER_LAMBDAS="0.0 0.2 0.4 0.6 0.8 1.0"
PAPER_SEEDS="0 1 2"

cell_splits() {
  case "$1" in
    pairA/*) echo experiments/pipeline/pairA/splits ;;
    pairB/*) echo experiments/pipeline/pairB/splits ;;
    pairC/*) echo experiments/pipeline/pairC/splits ;;
    *) echo "unknown cell $1" >&2; return 1 ;;
  esac
}

gen_limits() {
  MAXLEN=8192; NEWTOK=7680
  if [[ "$1" == *OLMo* ]]; then MAXLEN=4096; NEWTOK=3840; fi
}
