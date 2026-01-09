from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn


try:
    import folder_paths  # type: ignore
except Exception:
    folder_paths = None

class CCIPToGemmaAdapter(nn.Module):
    def __init__(self, in_dim: int = 768, out_dim: int = 2304, tokens_per_ref: int = 32):
        super().__init__()
        if tokens_per_ref <= 0:
            raise ValueError("tokens_per_ref must be >= 1")

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.tokens_per_ref = int(tokens_per_ref)

        self.base_proj = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim),
        )

        self.token_embed = nn.Parameter(torch.randn(self.tokens_per_ref, self.out_dim) * 0.002)

        self.post = nn.Sequential(
            nn.LayerNorm(self.out_dim),
            nn.Linear(self.out_dim, self.out_dim * 4),
            nn.GELU(),
            nn.Linear(self.out_dim * 4, self.out_dim),
        )
        nn.init.zeros_(self.post[-1].weight)
        nn.init.zeros_(self.post[-1].bias)

        for m in self.base_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, in_dim) -> (N, K, out_dim)
        if x.ndim != 2:
            raise ValueError(f"expected 2D (N,{self.in_dim}) but got shape={tuple(x.shape)}")
        if x.shape[1] != self.in_dim:
            raise ValueError(f"expected feature dim={self.in_dim} but got {x.shape[1]}")

        base = self.base_proj(x)  # (N, D)
        tokens = base[:, None, :] + self.token_embed[None, :, :]
        return tokens + self.post(tokens)


@dataclass
class CCIPAdapterHandle:
    model: CCIPToGemmaAdapter
    in_dim: int
    out_dim: int
    tokens_per_ref: int


def _to_tensor_features(x: Any, *, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.tensor(x)

    if t.ndim == 1:
        t = t[None, :]
    if t.ndim != 2:
        raise ValueError(f"features must be 1D or 2D, got shape={tuple(t.shape)}")

    return t.to(device=device, dtype=torch.float32)


def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        from safetensors.torch import load_file

        return load_file(path)
    return torch.load(path, map_location="cpu")

class CCIPAdapterLoader:
    @staticmethod
    def _models_base_from_folder_paths() -> Optional[str]:
        if folder_paths is None:
            return None
        try:
            base = os.path.join(folder_paths.models_dir, "ccip_adapter")
            return os.path.abspath(base)
        except Exception:
            return None

    @staticmethod
    def _list_adapter_files(base_dir: str) -> List[str]:
        exts = {".safetensors", ".pt", ".pth", ".bin"}
        if not base_dir or not os.path.isdir(base_dir):
            return []
        try:
            files = []
            for f in sorted(os.listdir(base_dir)):
                p = os.path.join(base_dir, f)
                if os.path.isfile(p) and os.path.splitext(f)[1].lower() in exts:
                    files.append(f)
            return files
        except Exception:
            return []

    @classmethod
    def INPUT_TYPES(cls):
        base_dir = cls._models_base_from_folder_paths()
        choices = cls._list_adapter_files(base_dir) if base_dir else []

        if not choices:
            choices = ["(no adapter found in models/ccip_adapter)"]

        required = {
            "in_dim": ("INT", {"default": 768, "min": 1, "max": 16384}),
            "out_dim": ("INT", {"default": 2304, "min": 1, "max": 16384}),
            "tokens_per_ref": ("INT", {"default": 32, "min": 1, "max": 4096}),
            "adapter_name": (choices, {"default": choices[0]}),
        }
        return {"required": required}

    RETURN_TYPES = ("CCIP_ADAPTER",)
    RETURN_NAMES = ("adapter",)
    FUNCTION = "load"
    CATEGORY = "CCIP"

    def load(
        self,
        in_dim: int = 768,
        out_dim: int = 2304,
        tokens_per_ref: int = 32,
        adapter_name: Optional[str] = None,
    ):
        if not adapter_name or adapter_name.startswith("(no adapter"):
            raise ValueError("No adapter file found. Put your adapter weights into ComfyUI/models/ccip_adapter/ and restart ComfyUI.")

        base = self._models_base_from_folder_paths()
        if base is None:
            raise FileNotFoundError(
                "folder_paths.models_dir not available; cannot load local models. "
                "Please ensure ComfyUI folder_paths is available."
            )

        adapter_path = os.path.join(base, adapter_name)
        if not os.path.isfile(adapter_path):
            raise FileNotFoundError(adapter_path)
        adapter = CCIPToGemmaAdapter(in_dim=in_dim, out_dim=out_dim, tokens_per_ref=tokens_per_ref)
        sd = _load_state_dict(adapter_path)
        adapter.load_state_dict(sd, strict=True)
        adapter.eval()

        handle = CCIPAdapterHandle(model=adapter, in_dim=int(in_dim), out_dim=int(out_dim), tokens_per_ref=int(tokens_per_ref))
        return (handle,)


class CCIPAdapterInfer:
    """Run adapter on an input feature tensor and output out_hidden/out_mask."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "adapter": ("CCIP_ADAPTER",),
                "features": ("TENSOR",),
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR")
    RETURN_NAMES = ("out_hidden", "out_mask")
    FUNCTION = "infer"
    CATEGORY = "CCIP"

    def infer(self, adapter: CCIPAdapterHandle, features: Any):
        if not isinstance(adapter, CCIPAdapterHandle):
            raise TypeError(f"adapter must be CCIPAdapterHandle, got {type(adapter)!r}")
        device = None
        try:
            import comfy.model_management as mm  # type: ignore

            device = mm.get_torch_device()
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        feat = _to_tensor_features(features, device=device)
        if feat.shape[1] != adapter.in_dim:
            raise ValueError(f"features last dim must be {adapter.in_dim}, got {feat.shape[1]}")

        model = adapter.model.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            out_hidden = model(feat)  # (B, K, D)

        # out_mask: (B, K)
        bsz, k, _ = out_hidden.shape
        out_mask = torch.ones((bsz, k), device=out_hidden.device, dtype=torch.int32)

        out_hidden = out_hidden.detach().to(device="cpu")
        out_mask = out_mask.detach().to(device="cpu")

        return (out_hidden, out_mask)


NODE_CLASS_MAPPINGS = {
    "CCIPAdapterLoader": CCIPAdapterLoader,
    "CCIPAdapterInfer": CCIPAdapterInfer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CCIPAdapterLoader": "CCIP Adapter Loader",
    "CCIPAdapterInfer": "CCIP Adapter Infer",
}
