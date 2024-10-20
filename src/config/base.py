from typing import Optional

import torch

import distributed


def none_or_str(value):
    if value == "None":
        return None
    return value


def parse_args(base_parser, args, namespace):
    parser = base_parser

    # General training params
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--acc_steps", default=4, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--data_seed", default=1337, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--iterations", default=25000, type=int)
    parser.add_argument(
        "--warmup_steps", default=300, type=int
    )  # it was only warmup_percent before
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument(
        "--wsd_final_lr_scale", default=0.0, type=float
    )  # wsd scheduler
    parser.add_argument("--wsd_fract_decay", default=0.1, type=float)  # wsd scheduler
    parser.add_argument(
        "--decay_type",
        default="linear",
        choices=["linear", "cosine", "exp", "miror_cosine", "square", "sqrt"],
    )
    parser.add_argument(
        "--warmup_percent", default=0.05, type=float
    )  # leave it anyway, warmup_steps / iterations
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument(
        "--scheduler",
        default="cos",
        choices=["linear", "cos", "wsd", "cos_inf", "none"],
    )
    parser.add_argument("--cos_inf_steps", default=0, type=int)
    parser.add_argument(
        "--opt",
        default="adamw",
        choices=[
            "adamw",
            "sgd",
            "muon",
            "soap",
            "ademamix",
            "lion",
            "sf-adamw",
            "sf-sgd",
            "adam-mini",
        ],
    )
    parser.add_argument("--eval_freq", default=200, type=int)  # in iterations
    parser.add_argument("--results_base_folder", default="./exps", type=str)
    parser.add_argument(
        "--grad_clip", default=0.0, type=float
    )  # default value is 1.0 in NanoGPT
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--shampoo_beta", default=-1.0, type=float)
    parser.add_argument("--precondition_frequency", default=10, type=int)
    parser.add_argument("--max_precond_dim", default=10000, type=int)
    parser.add_argument("--merge_dims", default=False, type=bool)
    parser.add_argument("--precondition_1d", default=False, type=bool)
    parser.add_argument("--normalize_grads", default=False, type=bool)
    parser.add_argument("--soap_data_format", default="channels_first", type=str)
    parser.add_argument("--correct_bias", default=True, type=bool)
    parser.add_argument("--nesterov", default=True, type=bool)
    parser.add_argument("--muon_backend", default="newtonschulz5", type=str)
    parser.add_argument("--muon_backend_steps", default=5, type=int)
    parser.add_argument("--adema_beta3", default=0.9, type=float)
    parser.add_argument("--adema_alpha", default=2.0, type=float)
    parser.add_argument("--adema_beta3_warmup", default=None, type=int)
    parser.add_argument("--adema_alpha_warmup", default=None, type=int)
    parser.add_argument("--schedulefree_r", default=0.0, type=float)
    parser.add_argument("--weight_lr_power", default=2.0, type=float)
    parser.add_argument("--model_sharding", default=None, type=bool)
    parser.add_argument("--adam_mini_verbose", default=False, type=bool)

    # Dataset params
    parser.add_argument(
        "--dataset",
        default="slimpajama",
        choices=[
            "slimpajama",
            "wikitext",
            "shakespeare-char",
            "arxiv",
            "arxiv2000",
            "arxiv+wiki",
            "openwebtext2",
        ],
    )
    parser.add_argument("--vocab_size", default=50304, type=int)
    parser.add_argument(
        "--data_in_ram", action="store_true"
    )  # force the data to RAM, mostly useless except for openwebtext2

    # Model params
    parser.add_argument("--model", default="base", choices=["base", "llama2"])
    parser.add_argument(
        "--use_pretrained", default="auto", type=none_or_str
    )  # 'none', 'gpt-2' or a path to the pretraind model
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--n_head", default=12, type=int)
    parser.add_argument("--n_layer", default=12, type=int)  # depths in att + ff blocks
    parser.add_argument(
        "--n_embd", default=768, type=int
    )  # embedding size / hidden size ...
    parser.add_argument("--sequence_length", default=512, type=int)
    parser.add_argument("--dtype", default=torch.bfloat16, type=torch.dtype)
    parser.add_argument("--bias", default=False, type=bool)
    parser.add_argument(
        "--compile", action="store_true"
    )  # if true then model is compiled
    parser.add_argument("--rmsnorm_eps", default=1e-5, type=float)
    parser.add_argument(
        "--multiple_of",  # make SwiGLU hidden layer size multiple of large power of 2
        default=256,
        type=int,
    )
    parser.add_argument("--n_kv_head", default=None, type=Optional[int])
    parser.add_argument(
        "--run_prefix", default=None, type=str, required=False
    )  # is added before the autogenerated experiment name
    parser.add_argument("--exp_name", default=None, type=str, required=False)

    # logging params (WandB)
    parser.add_argument("--wandb", action="store_true")  # whether to use wandb or not
    parser.add_argument("--wandb_project", default="my-project", type=str)
    parser.add_argument("--wandb_entity", default=None, type=none_or_str)
    parser.add_argument(
        "--wandb_run_prefix", default="none", type=str
    )  # is added before the autogenerated experiment name
    parser.add_argument(
        "--eval_seq_prefix", default="Once upon a time", type=str
    )  # prefix used to generate sequences

    # Distributed args
    parser.add_argument(
        "--distributed_backend",
        default=None,
        type=str,
        required=False,
        choices=distributed.registered_backends(),
    )  # distributed backend type
    parser.add_argument(
        "--save_checkpoint_freq", default=None, type=int, required=False
    )

    args = parser.parse_args(args, namespace)

    if args.exp_name is None:
        special_name_handle_fields = {
            "model",
            "lr",
            "batch_size",
            "acc_steps",
            "seed",
            "exp_name",
            "wandb",
            "wandb_project",
            "eval_seq_prefix",
            "run_prefix",
            "distributed_backend",
            "config_format",
            "sequence_length",
        }
        overriden_values = []
        for key in vars(args):
            if key in special_name_handle_fields:
                continue
            if getattr(args, key) != parser.get_default(key):
                overriden_values.append((key, getattr(args, key)))
        chunk_len = 10
        overriden_values_str_parts = []
        for chunk_id in range(0, len(overriden_values), chunk_len):
            overriden_values_str = "_".join(
                [
                    "{}={}".format(key, value)
                    for key, value in overriden_values[chunk_id : chunk_id + chunk_len]
                ]
            )
            overriden_values_str_parts.append(overriden_values_str)
        overriden_values_str = "/".join(overriden_values_str_parts)
        exp_name = ""
        if args.run_prefix is not None:
            exp_name += f"{args.run_prefix}_"
        exp_name += f"{args.model}_lr{args.lr}_bs{args.batch_size}x{args.acc_steps}_seqlen{args.sequence_length}/{overriden_values_str}_seed={args.seed}"
        args.exp_name = exp_name

    if args.dtype == "torch.bfloat16":
        args.dtype = torch.bfloat16
    elif args.dtype == "torch.float16":
        args.dtype = torch.float16
    elif args.dtype == "torch.float32":
        args.dtype = torch.float32

    return args
