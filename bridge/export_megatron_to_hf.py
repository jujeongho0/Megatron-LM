import torch
import shutil
import argparse
from pathlib import Path

from megatron.core import parallel_state
from megatron.core.enums import ModelType

from megatron.bridge.training.model_load_save import load_model_config, temporary_distributed_context
from megatron.bridge.training.mlm_compat.arguments import _tokenizer_config_from_args
from megatron.bridge.training.checkpointing import _load_model_weights_from_checkpoint
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size
from megatron.bridge import AutoBridge

from pretrain_gpt_for_wbl import model_provider_with_args

from bridge.wbl_bridge import WBLBridge # register bridge


def load_megatron_model(megatron_path, hf_model):
    _, mlm_args = load_model_config(megatron_path)
    mlm_args.use_cpu_initialization = args.use_cpu_initialization

    mlm_args.sliding_window_size = 512
    mlm_args.sliding_window_interleave_k = 6

    # TODO: parallel conversion
    mlm_args.context_parallel_size = 1
    mlm_args.expert_model_parallel_size = 1
    mlm_args.expert_tensor_parallel_size = 1
    mlm_args.pipeline_model_parallel_size = 1
    mlm_args.tensor_model_parallel_size = 1
    mlm_args.sequence_parallel = False
    mlm_args.context_parallel_size = 1
    mlm_args.transformer_pipeline_model_parallel_size = 1

    # with torch.device("meta"):
    pre_process = parallel_state.is_pipeline_first_stage()
    post_process = parallel_state.is_pipeline_last_stage()
    model = model_provider_with_args(mlm_args, pre_process=pre_process, post_process=post_process)
    model.model_type = ModelType.encoder_or_decoder
    _load_model_weights_from_checkpoint(megatron_path, [model])
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", type=str)
    parser.add_argument("--megatron-path", type=str)
    parser.add_argument("--hf-path", type=str)
    parser.add_argument('--no-use-cpu-initialization', action='store_false', dest='use_cpu_initialization')
    args = parser.parse_args()

    bridge = AutoBridge.from_hf_pretrained(args.hf_model, trust_remote_code=True)

    backend = "gloo" if args.use_cpu_initialization else "nccl"
    with temporary_distributed_context(backend):
        megatron_model = load_megatron_model(args.megatron_path, args.hf_model)
        bridge.save_hf_pretrained([megatron_model], args.hf_path)
        shutil.copy(f"{args.hf_model}/modeling_wbl.py", args.hf_path)

    print(f"✅ Successfully exported model to: {args.hf_path}")

    export_path = Path(args.hf_path)
    if export_path.exists():
        print("📁 Export structure:")
        for item in export_path.iterdir():
            if item.is_dir():
                print(f"   📂 {item.name}/")
            else:
                print(f"   📄 {item.name}")

    print("🔍 You can now load this model with:")
    print("   from transformers import AutoModelForCausalLM")
    print(f"   model = AutoModelForCausalLM.from_pretrained('{args.hf_path}')")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
