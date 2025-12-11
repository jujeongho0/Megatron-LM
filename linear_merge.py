import torch

from examples.inference.gpt.utils import add_common_inference_args
from megatron.training import get_args, get_model, print_rank_0
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.initialize import initialize_megatron

from pretrain_gpt_for_wbl import model_provider


def add_static_inference_args(parser):

    add_common_inference_args(parser)
    group = parser.add_argument_group()

    return parser


def validate_merge_args(args):

    if args.num_merge_models is None:
        raise ValueError("num_merge_models must not be None")

    n = args.num_merge_models

    if not (1 <= n <= 10):
        raise ValueError("num_merge_models must be between 1 and 10")

    for i in range(1, 11):
        model_i  = getattr(args, f"model{i}")
        weight_i = getattr(args, f"weight{i}")

        if i <= n:
            if model_i is None:
                raise ValueError(f"model{i} must not be None because num_merge_models is {n}")
        else:
            if model_i is not None:
                raise ValueError(f"model{i} must be None because num_merge_models is {n}")
            if weight_i is not None:
                raise ValueError(f"weight{i} must be None because num_merge_models is {n}")


@torch.inference_mode()
def main():

    initialize_megatron(
        extra_args_provider=add_static_inference_args,
        args_defaults={
            'no_load_rng': True,
            'no_load_optim': True,
            'micro_batch_size': 1,
            'exit_on_missing_checkpoint': True,
        },
    )

    args = get_args()
    validate_merge_args(args)

    model_indices, weights = [], []
    for i in range(1, args.num_merge_models + 1):
        model_indices.append(f"model{i}")
        wi = getattr(args, f"weight{i}")
        if wi is not None:
            weights.append(wi)

    if not weights:
        weights = [1.0 / args.num_merge_models] * args.num_merge_models

    total = sum(weights)
    weights = [w / total for w in weights] # normalize

    base_model = get_model(model_provider, wrap_with_ddp=False)
    load_checkpoint(base_model, None, None, load_arg=model_indices[0], strict=False)

    for chunk_idx in range(len(base_model)):
        base_state = base_model[chunk_idx].state_dict()
        for k, v in base_state.items():
            if v is not None and torch.is_floating_point(v):
                v.mul_(weights[0])

        for model_index, w in zip(model_indices[1:], weights[1:]):
            model = get_model(model_provider, wrap_with_ddp=False)
            load_checkpoint(model, None, None, load_arg=model_index, strict=False)

            other_state = model[chunk_idx].state_dict()
            for k, v_other in other_state.items():
                if v_other is not None and torch.is_floating_point(v_other):
                    base_state[k].add_(v_other, alpha=w)

            del model, other_state
            torch.cuda.empty_cache()

    common_state_dict = dist_checkpointing.load_common_state_dict(getattr(args, model_indices[0]))

    latest_checkpointed_iteration = 0
    save_checkpoint(latest_checkpointed_iteration, base_model, None, None, common_state_dict['num_floating_point_operations_so_far'])

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()

