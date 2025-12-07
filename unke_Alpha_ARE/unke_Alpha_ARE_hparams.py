from dataclasses import dataclass
from typing import List, Literal

from util.hparams import HyperParams


@dataclass
class unkeAlphaAREHyperParams(HyperParams):
    # Method
    model_name: str
    layers: List[int]
    layer_selection: Literal["all", "random"]
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ]
    lr: float
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    early_stop_patience: int
    ex_data_num: int
    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # NullSpaceLoRA-specific
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
    nullspace_threshold: float

    update_abs_floor: float
    update_abs_cap: float
    update_rel_frac: float

    L2: float
    reg_warmup_steps: int
    reg_abs_floor: float
    reg_abs_cap: float
    reg_rel_frac: float

    previous_scale: float
    prev_warmup_steps: int
    prev_abs_floor: float
    prev_abs_cap: float
    prev_rel_frac: float

    r: int
    lora_alpha: int
    lora_dropout: float

    # AnyEdit-specific
    window_size: int
    overlap: int
