#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

set -euo pipefail

VLLM_COMMIT="5fbbfe9a4c13094ad72ed3d6b4ef208a7ddc0fd7"

echo "Applying vllm patches for version v0.9.0..."
cd vllm
git checkout --detach "$VLLM_COMMIT"
test "$(git rev-parse HEAD)" = "$VLLM_COMMIT"
for patch in ../vllm_v0.9.0_patch/*.patch; do
  git am "$patch"
done
echo "Patches applied successfully!"
