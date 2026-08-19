# Qwen2.5-0.5B Quality-Gate Source Notes

The official Hugging Face model card identifies `Qwen/Qwen2.5-0.5B` as the base causal language model, with approximately 0.49B parameters, 24 layers, GQA attention, tied word embeddings, and a 32,768-token context length. The card provides the Transformers loading path and explicitly distinguishes the base model from the instruction-tuned variant.

The official Kaggle model page for `qwen-lm/qwen2.5` exposes the Transformers `0.5b` variation and documents the local Kaggle path `/kaggle/input/qwen2.5/transformers/0.5b/1`, including loading through `AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained`. The listed model files include a roughly 999.6 MB package with `model.safetensors`, tokenizer files, and configuration files.

These sources support adding an honest optional full-model quality gate that uses the exact base model requested by the project, without substituting Qwen2.5-0.5B-Instruct or another model. The gate must report `not_run` if the Kaggle input is unavailable rather than silently falling back to a different model.

References:

1. [Qwen/Qwen2.5-0.5B model card](https://huggingface.co/Qwen/Qwen2.5-0.5B)
2. [Kaggle Qwen2.5 model, Transformers 0.5b variation](https://www.kaggle.com/models/qwen-lm/qwen2.5/Transformers/0.5b)
