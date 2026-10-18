"""One-off HF -> Megatron conversion of the policy model, run before GRPO (see brev/run_on_brev.sh `convert`).
Used by the 2-GPU active_space_candidates runs; the 8-GPU node of training/launch_grpo.sh has enough host RAM without it.

NeMo RL converts the HF checkpoint inside MegatronPolicyWorker.__init__ on first use, but by then the colocated vLLM
engine has already put its ~30 GB/GPU weight backup in host RAM, and both workers loading the 30B model on top of that
exceeds a 147 GB node (Ray kills a worker at 95%). Converting here, with nothing else running, writes the same cache
NeMo RL looks for ($HF_HOME/nemo_rl/<model>/iter_0000000), so training skips its own import.

Run with the Megatron worker venv's python inside the NeMo RL container:
    python training/convert_hf_to_megatron.py nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
"""
import os
import sys

from megatron.bridge.training.model_load_save import temporary_distributed_context
from nemo_rl.models.megatron.community_import import import_model_from_hf_name
from nemo_rl.models.policy.utils import get_megatron_checkpoint_dir


def main():
    hf_model = sys.argv[1]
    out = os.path.join(get_megatron_checkpoint_dir(), hf_model)   # same path megatron/setup.py checks before importing
    # Single process, no parallelism: the checkpoint is saved in torch_dist format, which training reshards to TP2/EP2 on load.
    # Same settings as NeMo RL's own tests/functional/test_converter_roundtrip.py; parallel sizes are not saved in the config.
    megatron_config = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "num_layers_in_first_pipeline_stage": None,
        "num_layers_in_last_pipeline_stage": None,
        "pipeline_dtype": "float32",
        "sequence_parallel": False,
        "gradient_accumulation_fusion": False,
    }
    with temporary_distributed_context(backend="gloo"):
        import_model_from_hf_name(hf_model, out, megatron_config=megatron_config)
    assert os.path.isdir(os.path.join(out, "iter_0000000")), f"conversion wrote no iter_0000000 under {out}"
    # Marker so run_on_brev.sh can tell a finished conversion from one that was killed halfway (which still leaves iter_0000000).
    open(os.path.join(out, ".conversion_complete"), "w").close()
    print("converted", hf_model, "->", out)


# The checkpoint saver starts a helper process with the "spawn" method, which re-imports this file; without the guard the
# helper would rerun the conversion and die, and the save fails with EOFError.
if __name__ == "__main__":
    main()
