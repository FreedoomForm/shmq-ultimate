import sysconfig
from pathlib import Path


def load_cuda_extension() -> None:
    """Load the compiled extension only when the CUDA runtime is requested."""
    import torch

    current_dir = Path(__file__).resolve().parent
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    extension = current_dir / f"kernels{ext_suffix}"
    if not extension.is_file():
        raise RuntimeError(
            f"MixLLM CUDA extension is not built: {extension}. "
            "Reference quantization and capability modules remain available."
        )
    torch.ops.load_library(str(extension))


def __getattr__(name):
    """Preserve the legacy package API without eager CUDA side effects."""
    if name == "LinearMixLLM":
        load_cuda_extension()
        from mixllm.nn.modules.linear import LinearMixLLM
        return LinearMixLLM
    if name == "LinearMixLLM4vLLM":
        load_cuda_extension()
        from mixllm.nn.modules.linear_for_vllm import LinearMixLLM4vLLM
        return LinearMixLLM4vLLM
    if name == "MixLLMConfig":
        from mixllm.nn.modules.mixllm_config import MixLLMConfig
        return MixLLMConfig
    raise AttributeError(name)
