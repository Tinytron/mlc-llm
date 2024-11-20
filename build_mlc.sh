#!/bin/bash
  
# model name
#qwen2
MODEL_BASE_PATH="/mnt/kaiwu-group-pp-sh/edge_llm/mlc/1029-qwen2-2.9b-student-3.1b-instruct-teacher-1e-5-finetune-fix-load"
MODEL_NAME="iter_20000"
model_type=qwen2

# phi2
# MODEL_BASE_PATH=/mnt/kaiwu-group-pp-sh/edge_llm/distill_hf/1023-phi-test-seperate/
# MODEL_NAME=iter_24000
# llama3.1
# MODEL_BASE_PATH="/mnt/kaiwu-group-pp-sh/edge_llm/distill_hf/1110-llama3-iterative-round3-pruned-3.8b-v2-data-gen-by-distill-14k/pp1"
# MODEL_NAME="iter_24000"

# cauchy
MODEL_BASE_PATH=/mnt/kaiwu-group-pp-sh/edge_llm/track2_pretrained_cauchy/converted_hf/cauchy_3b_stage2_bs960
MODEL_BASE_PATH=/root
model_type=cauchy
MODEL_NAME=iter_0022000

# phi2 tie emb
MODEL_BASE_PATH=/mnt/kaiwu-group-pp-sh/edge_llm/distill_hf/1118-phi2-tie-emb-unfreeze-decoder-use-qwen-vocab-1.5e-4-2k-seq-tp1-qwen2-7b-inst-as-teacher/
MODEL_BASE_PATH=/root
MODEL_NAME="iter_5000"
model_type=phi
conv_template=phi-2
conv_template=qwen2




# paths
MODEL_DIR="$MODEL_BASE_PATH/$MODEL_NAME"
BUNDLE_DIR="$MODEL_BASE_PATH/dist/mlc_$MODEL_PATH"
LIB_PATH="$MODEL_BASE_PATH/dist/libs"
mkdir -p $LIB_PATH

# convert_weight
# mlc_llm convert_weight $MODEL_DIR --quantization q0f16 -o $BUNDLE_DIR --model-type $model_type

# # # gen_config
# # #mlc_llm gen_config $MODEL_DIR --conv-template chatml --quantization q0f16 --context-window-size 2048 --prefill-chunk-size=1 --max-batch-size=1 -o $BUNDLE_DIR
# # # mlc_llm gen_config $MODEL_DIR --conv-template llama-3_1 --quantization q0f16 --context-window-size 2048 --prefill-chunk-size=1 --max-batch-size=1 -o $BUNDLE_DIR
# mlc_llm gen_config $MODEL_DIR --model-type $model_type --conv-template $conv_template --quantization q0f16 --context-window-size 2048 --prefill-chunk-size=1 --max-batch-size=1 -o $BUNDLE_DIR

# # #compile model
# mlc_llm compile $BUNDLE_DIR/mlc-chat-config.json --device cuda -o $LIB_PATH/mlc.so --model-type $model_type --debug-dump /root/mlc_debug

#chat
mlc_llm chat $BUNDLE_DIR --model-lib $LIB_PATH/mlc.so
