# comfyui_ccip_adapter

A minimal ComfyUI custom node plugin to load and run a CCIP adapter.

Before using,you need to install [comfyui-ccip](https://github.com/spawner1145/comfyui-ccip.git) and [comfyui-spawner-nodes](https://github.com/spawner1145/comfyui-spawner-nodes.git) first

## Nodes

- **CCIP Adapter Loader**
  - Loads adapter weights from a single file.
  - Parameters:
    - `adapter_name` dropdown (from `models/ccip_adapter/`)
    - `in_dim` (default 768)
    - `out_dim` (default 2304)
    - `tokens_per_ref` (default 32)

- **CCIP Adapter Infer**
  - Takes a `TENSOR` feature tensor and outputs:
    - `out_hidden` (`TENSOR`): `(B, K, out_dim)` float32 (returned on CPU)
    - `out_mask` (`TENSOR`): `(B, K)` int32 (all ones, returned on CPU)

## Install

Copy the whole folder `comfyui_ccip_adapter` into your ComfyUI `custom_nodes/` directory, for example:

- `ComfyUI/custom_nodes/comfyui_ccip_adapter`

Restart ComfyUI.

## Notes

- `CCIP Adapter Infer` uses ComfyUI built-in `TENSOR` sockets for both input and outputs.
- Place adapter weight files under `ComfyUI/models/ccip_adapter/` to get a dropdown list.
