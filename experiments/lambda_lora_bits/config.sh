# Shell twin of config.py: the cells, switch lambdas and generation limits used by drivers/ and reproduce/.
# Source it; it defines variables and one function and changes nothing else.
#
# A cell spec is "<cell dir>:<student model>"; a switch spec is "<cell dir>:<switch lambda>".
# The paper's pair C lives in pairD/.

PAPER_CELLS="sweep:Qwen2.5-0.5B-Instruct sweep_llama:Llama-3.2-1B-Instruct sweep_olmo2:OLMo-2-1B-Instruct
pairB/qwen0.5b:Qwen2.5-0.5B-Instruct pairB/llama1b:Llama-3.2-1B-Instruct pairB/olmo1b:OLMo-2-1B-Instruct
pairD/qwen0.5b:Qwen2.5-0.5B-Instruct pairD/llama1b:Llama-3.2-1B-Instruct pairD/olmo1b:OLMo-2-1B-Instruct"

PAPER_SWITCH="sweep:0.4 sweep_llama:0.4 sweep_olmo2:0.4 pairB/qwen0.5b:0.6 pairB/llama1b:0.6 pairB/olmo1b:0.6
pairD/qwen0.5b:0.6 pairD/llama1b:0.6 pairD/olmo1b:0.6"

PAPER_LAMBDAS="0.0 0.2 0.4 0.6 0.8 1.0"
PAPER_SEEDS="0 1 2"

# cell_splits <cell dir>: prints the pair's splits directory (relative to the repository root).
cell_splits() {
  case "$1" in
    sweep*) echo experiments/lambda_lora_bits/sweep/splits ;;
    pairB/*) echo experiments/lambda_lora_bits/pairB/splits ;;
    pairD/*) echo experiments/lambda_lora_bits/pairD/splits ;;
    *) echo "unknown cell $1" >&2; return 1 ;;
  esac
}

# gen_limits <student>: sets MAXLEN (vLLM context) and NEWTOK (max new tokens). OLMo-2-1B has a 4,096 context.
gen_limits() {
  MAXLEN=8192; NEWTOK=7680
  if [[ "$1" == *OLMo* ]]; then MAXLEN=4096; NEWTOK=3840; fi
}
