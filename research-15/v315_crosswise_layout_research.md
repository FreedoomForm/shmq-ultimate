# v315–v316 native iterator layout research

## External evidence

NVIDIA’s CUTLASS 2.x layout documentation explains that a layout maps logical coordinates to physical offsets, that `TensorRef` combines a pointer with a layout, and that `Layout::packed(extent)` constructs a tightly packed layout. Source: [1] https://docs.nvidia.com/cutlass/latest/media/docs/cpp/layout.html (NVIDIA CUTLASS, “Layouts and Tensors”, accessed during this iteration).

The vendored CUTLASS SM75 implementation defines `TensorOpMultiplicandCongruous` and `TensorOpMultiplicandCrosswise` as distinct shared-memory layouts. The production `DefaultMmaCore` uses `RowMajorTensorOpMultiplicandCrosswise<sizeof_bits<ElementA>, Shape::kK>` for A and `ColumnMajorTensorOpMultiplicandCrosswise<sizeof_bits<ElementB>, Shape::kK>` for B. Its warp-level default aliases pass MatrixShape A=`<M,K>` and B=`<K,N>`, while the row/column-major Crosswise wrappers internally convert these to pitch-linear shapes for the base iterator.

## Implication

The v309/v310 research aliases proved that the generic explicit iterator can compile, but the probe incorrectly used a congruous layout and then loaded a guessed backing tile. v311–v315 showed that changing raw storage, logical conversion, tile geometry, and backing extent independently did not produce a valid logical load: the probe returned zeros or triggered an illegal memory access. The next experiment must use the exact production Crosswise wrappers and their separate A/B MatrixShape contracts, with no production dispatch enabled.

## References

[1]: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/layout.html "NVIDIA CUTLASS — Layouts and Tensors"
[2]: https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/layout/tensor_op_multiplicand_sm75.h "NVIDIA CUTLASS — SM75 tensor multiplicand layouts"
