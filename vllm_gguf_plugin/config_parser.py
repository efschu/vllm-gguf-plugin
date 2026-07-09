# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from transformers import PretrainedConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from vllm.transformers_utils.config import HFConfigParser
from vllm.transformers_utils.config_parser_base import ConfigParserBase

from .gguf_utils import (
    check_gguf_file,
    detect_gguf_multimodal,
    is_gguf,
    is_remote_gguf,
    maybe_patch_hf_config_from_gguf,
    split_remote_gguf,
)


class GGUFConfigParser(ConfigParserBase):
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs,
    ) -> tuple[dict, PretrainedConfig]:
        original_model = model
        resolved_model = self._resolve_config_source(model)
        config_dict, config = HFConfigParser().parse(
            resolved_model,
            trust_remote_code=trust_remote_code,
            revision=revision,
            code_revision=code_revision,
            **kwargs,
        )

        if config.model_type == "qwen3_moe" and "norm_topk_prob" not in config_dict:
            config_dict["norm_topk_prob"] = True
            config.update({"norm_topk_prob": True})

        # Multimodal GGUF (mmproj file next to the model, multimodal
        # config): keep the original conditional-generation architecture
        # instead of forcing the text-only causal-LM class.  Note that
        # the engine-args patch may have already replaced the model path
        # with its config-source DIRECTORY, so check both spellings.
        mmproj_present = False
        if check_gguf_file(original_model):
            mmproj_present = detect_gguf_multimodal(str(original_model)) is not None
        elif Path(original_model).is_dir():
            mmproj_present = any(Path(original_model).glob("*mmproj*.gguf"))
        is_multimodal_gguf = (
            mmproj_present
            and getattr(config, "vision_config", None) is not None
            and config_dict.get("architectures")
        )
        if not is_multimodal_gguf:
            if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
                raise RuntimeError(f"Can't get gguf config for {config.model_type}.")

            model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
            config_dict["architectures"] = [model_type]
            config.update({"architectures": [model_type]})

        if is_gguf(original_model):
            config = maybe_patch_hf_config_from_gguf(str(original_model), config)

        return config_dict, config

    @staticmethod
    def _resolve_config_source(model: str | Path) -> str | Path:
        if check_gguf_file(model):
            return Path(model).parent
        if is_remote_gguf(model):
            repo_id, _ = split_remote_gguf(model)
            return repo_id
        return model
