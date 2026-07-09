# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF weights adapter for the Qwen3.5/Qwen3.6 family (GGUF arch "qwen35").

The default adapter cannot handle these hybrid GDN/full-attention models:

- model_type "qwen3_5(_text)" / "qwen3_5_moe(_text)" maps to GGUF arch
  "qwen35" / "qwen35moe" (the generic reverse lookup fails because the HF
  and GGUF names differ),
- gguf-py's tensor map misses ``linear_attn.dt_bias`` (it only knows the
  older "dt_proj" spelling) and suffix-less params such as
  ``linear_attn.A_log`` need exact tensor names,
- llama.cpp's conversion transforms several weights: Gemma-style RMSNorm
  gammas are stored with +1 baked in, ``ssm_a`` holds ``-exp(A_log)``, and
  when ``num_v_heads != num_k_heads`` all GDN v-head dimensions are
  re-tiled from grouped to tiled order,
- the MTP (multi-token-prediction) draft layer is stored as
  ``blk.<num_hidden_layers>`` with ``nextn.*`` tensors and must be mapped
  to the HF-style ``mtp.*`` names vLLM's Qwen3_5MTP expects,
- vLLM creates ``embed_tokens`` without quant_config, so a quantized
  ``token_embd`` must be dequantized on the fly.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import gguf
import torch
from gguf.quants import dequantize
from transformers import AutoModelForCausalLM
from vllm.logger import init_logger

from .default import GGUFWeightsAdapter

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

_MODEL_TYPE_TO_GGUF_ARCH = {
    "qwen3_5": gguf.MODEL_ARCH.QWEN35,
    "qwen3_5_text": gguf.MODEL_ARCH.QWEN35,
    "qwen3_5_moe": gguf.MODEL_ARCH.QWEN35MOE,
    "qwen3_5_moe_text": gguf.MODEL_ARCH.QWEN35MOE,
}

_MTP_ARCHITECTURES = ("Qwen3_5MTP", "Qwen3_5MoeMTP")

# GGUF tensor names local to one MTP block -> HF names inside one
# Qwen3_5MultiTokenPredictor decoder layer (always full attention).
_MTP_LAYER_TENSORS = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

# nextn.* tensors of the first MTP block -> top-level predictor params
# (naming follows vllm/model_executor/models/qwen3_5_mtp.py).
_MTP_NEXTN_TENSORS = {
    "nextn.eh_proj.weight": "mtp.fc.weight",
    "nextn.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
    "nextn.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
    "nextn.shared_head_norm.weight": "mtp.norm.weight",
}

_LAYER_IDX_RE = re.compile(r"\.layers\.(\d+)\.")


class Qwen35GGUFAdapter(GGUFWeightsAdapter):
    """qwen35/qwen35moe GGUF adapter (main model and MTP draft)."""

    # RMSNorm gammas that vLLM implements Gemma-style (x * (1 + w),
    # zero-centered checkpoint weights).  llama.cpp bakes the +1 into the
    # GGUF at conversion time, so it must be subtracted again.
    # linear_attn.norm is RMSNormGated and stored raw in both formats.
    _GEMMA_NORM_SUFFIXES = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
    )
    _GEMMA_NORM_NAMES = (
        "model.norm.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    )

    @classmethod
    def matches(cls, config) -> bool:
        model_type = getattr(config, "model_type", None)
        if model_type == "qwen3_5_mtp":
            return True
        return model_type in _MODEL_TYPE_TO_GGUF_ARCH

    def _is_mtp(self) -> bool:
        if getattr(self.config, "model_type", None) == "qwen3_5_mtp":
            return True
        archs = getattr(self.config, "architectures", None) or []
        return any(a in _MTP_ARCHITECTURES for a in archs)

    # ------------------------------------------------------------------
    # Name mapping
    # ------------------------------------------------------------------

    def build_name_map(self, model_config: ModelConfig) -> dict[str, str]:
        if self._is_mtp():
            return self._build_mtp_name_map()
        return self._build_main_name_map(model_config)

    def _build_mtp_name_map(self) -> dict[str, str]:
        text_config = self.config.get_text_config()
        num_layers = text_config.num_hidden_layers
        num_mtp = getattr(text_config, "mtp_num_hidden_layers", 1) or 1
        # Embeddings and lm_head are shared with the target model and read
        # from the same GGUF file.
        name_map = {
            "token_embd.weight": "model.embed_tokens.weight",
            "output.weight": "lm_head.weight",
        }
        for gguf_local, hf_name in _MTP_NEXTN_TENSORS.items():
            name_map[f"blk.{num_layers}.{gguf_local}"] = hf_name
        for i in range(num_mtp):
            blk = num_layers + i
            for gguf_local, hf_local in _MTP_LAYER_TENSORS.items():
                name_map[f"blk.{blk}.{gguf_local}"] = f"mtp.layers.{i}.{hf_local}"
        logger.info(
            "qwen35 GGUF MTP name map: %d tensors (blk.%d..%d)",
            len(name_map),
            num_layers,
            num_layers + num_mtp - 1,
        )
        return name_map

    def _build_main_name_map(self, model_config: ModelConfig) -> dict[str, str]:
        config = model_config.hf_config
        text_config = config.get_text_config()
        model_type = getattr(text_config, "model_type", None) or config.model_type
        arch = _MODEL_TYPE_TO_GGUF_ARCH[model_type]
        num_layers = text_config.num_hidden_layers

        gguf_to_hf_name_map: dict[str, str] = {}
        sideload_params: list[re.Pattern] = []

        if arch == gguf.MODEL_ARCH.QWEN35MOE:
            # Same fused-experts convention as qwen3moe: GGUF stores all
            # experts stacked in one tensor mapped to expert 0.
            for idx in range(num_layers):
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
                )
                sideload_params.append(
                    re.compile(
                        f"model\\.layers\\.{idx}"
                        r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
                    )
                )

        text_name_map = gguf.get_tensor_name_map(arch, num_layers)

        with torch.device("meta"):
            # transformers' Qwen3_5ForCausalLM expects the text config,
            # not the multimodal wrapper.
            dummy_model = AutoModelForCausalLM.from_config(
                text_config, trust_remote_code=model_config.trust_remote_code
            )
        state_dict = dummy_model.state_dict()

        def find_hf_name_in_tensor_map(hf_name: str) -> str | None:
            # The dummy model is built from the text config, so names are
            # always plain "model.layers..." style.
            if hf_name.endswith((".weight", ".bias")):
                base_name, suffix = hf_name.rsplit(".", 1)
            else:
                base_name, suffix = hf_name, ""
            gguf_name = text_name_map.get_name(base_name)
            if gguf_name is None and suffix == "":
                # gguf-py knows dt_proj but not the dt_bias spelling used
                # by transformers' Qwen3_5 GDN layers.
                if base_name.endswith("linear_attn.dt_bias"):
                    m = _LAYER_IDX_RE.search(base_name)
                    if m is not None:
                        return f"blk.{m.group(1)}.ssm_dt.bias"
            if gguf_name is None:
                return None
            if suffix == "":
                # Suffix-less params (A_log, dt_bias) map to GGUF tensors
                # without a ".weight"/".bias" suffix.
                return gguf_name
            return gguf_name + "." + suffix

        unmapped_params = []
        for hf_name in state_dict:
            gguf_name_with_suffix = find_hf_name_in_tensor_map(hf_name)
            if gguf_name_with_suffix is not None:
                gguf_to_hf_name_map[gguf_name_with_suffix] = hf_name
            elif hf_name not in gguf_to_hf_name_map.values():
                unmapped_params.append(hf_name)

        if unmapped_params:
            unmapped_params = [
                x
                for x in unmapped_params
                if not any(re.fullmatch(p, x) for p in sideload_params)
            ]
        if unmapped_params:
            raise RuntimeError(
                f"Failed to map qwen35 GGUF parameters "
                f"({len(unmapped_params)}): {unmapped_params}"
            )
        logger.info(
            "qwen35 GGUF name map: %d tensors for %d layers (arch %s)",
            len(gguf_to_hf_name_map),
            num_layers,
            gguf.MODEL_ARCH_NAMES[arch],
        )
        return gguf_to_hf_name_map

    # ------------------------------------------------------------------
    # Weight-value transformations
    # ------------------------------------------------------------------

    def transform_weight(
        self,
        hf_name: str,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        # GGUF stores the GDN conv1d weight as (channels, kernel); HF
        # checkpoints and vLLM's mamba loader expect (channels, 1, kernel).
        if hf_name.endswith("conv1d.weight") and weight.dim() == 2:
            return weight.unsqueeze(1)
        # llama.cpp stores A as -exp(A_log); vLLM expects A_log.
        if hf_name.endswith("linear_attn.A_log"):
            return torch.log(torch.neg(weight.float()))
        if hf_name.endswith(self._GEMMA_NORM_SUFFIXES) or (
            hf_name in self._GEMMA_NORM_NAMES
        ):
            return weight.float() - 1.0
        return weight

    def _undo_v_tiling(
        self,
        weight: torch.Tensor,
        dim: int,
        head_units: int,
    ) -> torch.Tensor:
        """Invert llama.cpp's GDN v-head reorder (grouped -> tiled).

        llama.cpp's conversion (_LinearAttentionVReorderBase) retiles the
        v-head dimension of all GDN tensors when
        num_v_heads != num_k_heads.  vLLM/HF expect grouped order, so the
        inverse permutation must be applied when loading GGUF.
        """
        text_config = self.config.get_text_config()
        num_k = getattr(text_config, "linear_num_key_heads", 0) or 0
        num_v = getattr(text_config, "linear_num_value_heads", 0) or 0
        if num_k <= 0 or num_v <= 0 or num_k == num_v:
            return weight
        num_v_per_k = num_v // num_k
        shape = list(weight.shape)
        d = dim if dim >= 0 else dim + len(shape)
        new_shape = shape[:d] + [num_v_per_k, num_k, head_units] + shape[d + 1 :]
        weight = weight.reshape(*new_shape)
        perm = list(range(len(new_shape)))
        perm[d], perm[d + 1] = perm[d + 1], perm[d]
        return weight.permute(*perm).contiguous().reshape(*shape)

    def _v_retiling_active(self) -> bool:
        text_config = self.config.get_text_config()
        num_k = getattr(text_config, "linear_num_key_heads", 0) or 0
        num_v = getattr(text_config, "linear_num_value_heads", 0) or 0
        return num_k > 0 and num_v > 0 and num_k != num_v

    def _out_proj_dequant_needed(self, qtype: int) -> bool:
        """out_proj columns must be un-tiled; raw-byte permutation is only
        exact when a v head spans whole quantization blocks.  Otherwise the
        tensor is dequantized and loaded unquantized."""
        if not self._v_retiling_active():
            return False
        text_config = self.config.get_text_config()
        head_v_dim = getattr(text_config, "linear_value_head_dim", 0)
        block_size, _ = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(qtype)]
        return head_v_dim % block_size != 0

    def get_unquantized_modules(self, weight_type_map: dict[str, str]) -> list[str]:
        unquantized = GGUFWeightsAdapter.get_unquantized_modules(weight_type_map)
        for name, type_name in weight_type_map.items():
            if not name.endswith("linear_attn.out_proj.weight"):
                continue
            qtype = int(gguf.GGMLQuantizationType[type_name])
            if self._out_proj_dequant_needed(qtype):
                unquantized.append(name.removesuffix(".weight"))
        return unquantized

    def _undo_gdn_reorder(
        self,
        hf_name: str,
        weight: torch.Tensor,
        qweight_types: dict[str, int],
    ) -> torch.Tensor:
        text_config = self.config.get_text_config()
        head_v_dim = getattr(text_config, "linear_value_head_dim", 0)
        head_k_dim = getattr(text_config, "linear_key_head_dim", 0)
        num_k = getattr(text_config, "linear_num_key_heads", 0)
        if hf_name.endswith(("in_proj_qkv.qweight", "in_proj_qkv.weight")):
            # Raw GGUF rows are quantized independently, so permuting whole
            # rows is exact for any quantization type.
            qk_rows = 2 * head_k_dim * num_k
            v_part = self._undo_v_tiling(weight[qk_rows:], 0, head_v_dim)
            return torch.cat([weight[:qk_rows], v_part], dim=0)
        if hf_name.endswith(("in_proj_z.qweight", "in_proj_z.weight")):
            return self._undo_v_tiling(weight, 0, head_v_dim)
        if hf_name.endswith(
            (
                "in_proj_b.qweight",
                "in_proj_b.weight",
                "in_proj_a.qweight",
                "in_proj_a.weight",
            )
        ):
            return self._undo_v_tiling(weight, 0, 1)
        if hf_name.endswith(("linear_attn.A_log", "linear_attn.dt_bias")):
            return self._undo_v_tiling(weight.unsqueeze(-1), 0, 1).squeeze(-1)
        if hf_name.endswith("conv1d.weight"):
            # (channels, 1, kernel) after transform_weight; channels are
            # [q | k | v] and only the v part is retiled.
            qk_channels = 2 * head_k_dim * num_k
            v_part = self._undo_v_tiling(weight[qk_channels:], 0, head_v_dim)
            return torch.cat([weight[:qk_channels], v_part], dim=0)
        if hf_name.endswith("out_proj.qweight"):
            # Columns (input dim) are v-ordered.  Permuting raw bytes is
            # only exact when a v head spans whole quantization blocks.
            qtype = qweight_types.get(hf_name)
            assert qtype is not None, "out_proj qweight_type not seen yet"
            block_size, type_size = gguf.GGML_QUANT_SIZES[
                gguf.GGMLQuantizationType(qtype)
            ]
            if head_v_dim % block_size != 0:
                raise ValueError(
                    f"Cannot undo GDN v-head reorder of {hf_name}: "
                    f"head_v_dim={head_v_dim} is not a multiple of the "
                    f"{gguf.GGMLQuantizationType(qtype).name} block size "
                    f"{block_size}. Re-quantize out_proj with a "
                    "block-aligned type (e.g. Q8_0)."
                )
            head_bytes = head_v_dim // block_size * type_size
            return self._undo_v_tiling(weight, 1, head_bytes)
        if hf_name.endswith("out_proj.weight"):
            return self._undo_v_tiling(weight, 1, head_v_dim)
        return weight

    # ------------------------------------------------------------------
    # Weight stream
    # ------------------------------------------------------------------

    def prepare_weights(
        self,
        model_config: ModelConfig,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        # Qwen3_5Model / Qwen3_5MultiTokenPredictor create embed_tokens
        # WITHOUT quant_config, so the module has a plain `weight`
        # parameter.  A quantized token_embd from the GGUF file must be
        # dequantized on the fly or it would be silently skipped and the
        # embedding would stay uninitialized.
        embed_prefix = "model.embed_tokens."
        embed_qtype: int | None = None
        qweight_types: dict[str, int] = {}
        for name, weight in super().prepare_weights(model_config):
            if name == embed_prefix + "qweight_type":
                embed_qtype = int(weight.item())
                continue
            if name == embed_prefix + "qweight":
                assert embed_qtype is not None
                dequantized = dequantize(
                    weight.numpy(), gguf.GGMLQuantizationType(embed_qtype)
                )
                logger.info(
                    "Dequantized GGUF token_embd (%s) to %s for the "
                    "unquantized embedding layer",
                    gguf.GGMLQuantizationType(embed_qtype).name,
                    model_config.dtype,
                )
                yield (
                    embed_prefix + "weight",
                    torch.from_numpy(dequantized).to(model_config.dtype),
                )
                continue
            if name.endswith(".qweight_type"):
                qweight_types[name.removesuffix("_type")] = int(weight.item())
                if name.endswith(
                    "linear_attn.out_proj.qweight_type"
                ) and self._out_proj_dequant_needed(int(weight.item())):
                    # Layer is created unquantized (see
                    # get_unquantized_modules); it has no qweight_type param.
                    continue
            elif name.endswith("linear_attn.out_proj.qweight"):
                qtype = qweight_types.get(name)
                assert qtype is not None, "out_proj qweight_type not seen yet"
                if self._out_proj_dequant_needed(qtype):
                    # Block-misaligned quantization (e.g. K-quants with
                    # head_v_dim=128): dequantize, un-tile columns
                    # element-wise, load as plain weight.
                    text_config = self.config.get_text_config()
                    dequantized = torch.from_numpy(
                        dequantize(
                            weight.numpy(), gguf.GGMLQuantizationType(qtype)
                        )
                    )
                    dequantized = self._undo_v_tiling(
                        dequantized, 1, text_config.linear_value_head_dim
                    )
                    logger.info_once(
                        "Dequantized GDN out_proj (%s) to %s to undo the "
                        "v-head reorder (head_v_dim not block-aligned)",
                        gguf.GGMLQuantizationType(qtype).name,
                        model_config.dtype,
                    )
                    yield (
                        name.removesuffix(".qweight") + ".weight",
                        dequantized.to(model_config.dtype),
                    )
                    continue
                weight = self._undo_gdn_reorder(name, weight, qweight_types)
            elif ".linear_attn." in name:
                weight = self._undo_gdn_reorder(name, weight, qweight_types)
            yield name, weight
