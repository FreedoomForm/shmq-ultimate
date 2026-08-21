# Official Colab Operator Skill Findings

Source: [googlecolab/google-colab-cli `skills/colab-operator/SKILL.md`](https://raw.githubusercontent.com/googlecolab/google-colab-cli/main/skills/colab-operator/SKILL.md), read 2026-08-22.

The official workflow defines `colab run --gpu T4 -s <name> script.py` as an ephemeral one-shot job that provisions a fresh VM, executes a local script with native `__main__` and exit-code semantics, and stops the VM automatically. It separates CLI chatter on stderr from the script’s stdout. For incremental execution on one persistent kernel, the preferred command is `colab exec -s <name> -f <script.py>`; kernel state persists across exec calls, while `colab stop` or `restart-kernel` resets it. The skill requires explicit session names, recommends absolute `/content/...` paths, and requires stopping any persistent session when finished.

The current SHMQ runner already follows the main one-shot pattern by using `colab run --gpu T4` with a local Python script and confirmed teardown. Its remaining mismatch is scope: it runs the 91 contract tests only, while the full gate notebook separately contains benchmark, timing-integrity, memory, vLLM, and Qwen quality cells. The next safe runner change should reuse the exact gate logic and thresholds in a Colab-executable script, without changing model, inputs, precision, thresholds, or benchmark settings.
