import torch
import math
import torch.nn.functional as F
from torch import nn


def convert_linear_to_moe(
    name: str,
    config: dict,
    layer_idx: int,
    in_features: int,
    out_features: int,
    bias: bool = True,
):
    """Converts nn.Linear to MoeLayer
    Args:
        name (str): Layer Name
        config (dict): Composer config
        layer_idx (int): Transformer block id.
        in_features (int): Input features of Default nn.Linear layer.
        out_features (int): Output features of Default nn.Linear layer.
        bias (bool, optional): Defaults to True.
    """
    if (layer_idx in config.router_layers_index) and (name in config.router_layers):
        if hasattr(config, "adapter_configs"):
            return LoRAMoeLayer(
                config=config,
                in_features=in_features,
                out_features=out_features,
                bias=bias,
            )
        else:
            return MoeLayer(
                in_features=in_features,
                out_features=out_features,
                bias=bias,
                num_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
            )
    return nn.Linear(in_features, out_features, bias=bias)


class MoeLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        num_experts: int,
        num_experts_per_tok: int = 2,
    ):
        """Mixture of Expert Layer
        Args:
            in_features (int): Input Features
            out_features (int): Output Features
            bias (bool): bias
            num_experts (int): Total numbers of experts that Router Layer would handle
            num_experts_per_tok (int, optional): Number of Active Experts per token(step). Defaults to 2.
        """
        super().__init__()
        self.gate = nn.Linear(in_features, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [nn.Linear(in_features, out_features, bias) for _ in range(num_experts)]
        )
        self.num_experts_per_tok = num_experts_per_tok
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, inputs: torch.Tensor):
        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_tok)
        weights = F.softmax(weights, dim=2, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros(
            (inputs.shape[0], inputs.shape[1], self.out_features),
            device=inputs.device,
            dtype=inputs.dtype,
        )
        for ix, expert in enumerate(self.experts):
            batch_idx, tok_idx, expert_idx = torch.where(selected_experts == ix)
            results[batch_idx, tok_idx] += expert(inputs[batch_idx, tok_idx]) * weights[
                batch_idx, tok_idx, expert_idx
            ].unsqueeze(-1)
        return results


class LoRAMoeLayer(torch.nn.Module):
    def __init__(self, config, in_features, out_features, bias) -> None:
        super().__init__()

        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.in_features = in_features
        self.out_features = out_features

        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.use_dora = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.base_layer = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.gate = torch.nn.Linear(
            config.hidden_size, config.num_experts, bias=False
        )  # device="mps:0")# TODO FIXME
        self.active_adapters = []
        for ix, adapter_config in enumerate(self.config.adapter_configs):
            self.update_layer(
                adapter_name=str(ix),
                r=adapter_config["r"],
                lora_alpha=adapter_config["lora_alpha"],
                lora_dropout=adapter_config["lora_dropout"],
                init_lora_weights=adapter_config["init_lora_weights"],
                use_rslora=adapter_config["use_rslora"],
                use_dora=adapter_config["use_dora"],
            )

    def forward(self, x, *args, **kwargs):
        """
        This method is designed to be a drop-in-replacement for the peft LoRA layers' .forward method.
        To use it, a bound method must be created (bound to an instance of the LoRALayer class).
        """

        previous_dtype = x.dtype
        gate_logits = self.gate(x)  # b,s,N
        weights, selected_experts = torch.topk(
            gate_logits, self.num_experts_per_tok
        )  # b,s,n
        weights = F.softmax(weights, dim=2, dtype=torch.float).to(
            previous_dtype
        )  # b,s,n
        result = self.base_layer(x, *args, **kwargs)

        """TODO MAYBE
        - tensorize this loop add learnable weights here 
        - These are in my mind ( sigle embedding,  each lora layer with a gate,  lora gating loss similar to iclr )
        """

        for ix, active_adapter in enumerate(self.active_adapters):
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x = x.to(lora_A.weight.dtype)  # type: ignore

            batch_idx, tok_idx, expert_idx = torch.where(selected_experts == ix)
            x_adapter = x[
                batch_idx, tok_idx
            ]  # slicing uses the same tensor, whereas indexing will result in a copy. check the tensor address using tensor.storage().data_ptr()
            x_adapter = (
                lora_B(lora_A(dropout(x_adapter))) * scaling
            )  # * self.config.global_scaling_weight
            # maybe we require a small linear layer that we train here, not sure.
            result[batch_idx, tok_idx] += x_adapter * weights[
                batch_idx, tok_idx, expert_idx
            ].unsqueeze(-1)

            # apply nn.functional.silu ?? can pretrained lora be tweaked for this variation.
        result = result.to(previous_dtype)
        return result

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        use_dora: bool = False,
    ):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha

        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # check weight and qweight (for GPTQ)
        for weight_name in ("weight", "qweight"):
            weight = getattr(self.base_layer, weight_name, None)
            if weight is not None:
                # the layer is already completely initialized, this is an update
                if weight.dtype.is_floating_point or weight.dtype.is_complex:
                    self.to(weight.device, dtype=weight.dtype)
                else:
                    self.to(weight.device)
                break

        if use_dora:
            raise NotImplementedError
        self.use_dora[adapter_name] = False
        self.active_adapters.append(adapter_name)
 
    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                nn.init.kaiming_uniform_(
                    self.lora_A[adapter_name].weight, a=math.sqrt(5)
                )
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(
                    self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name]
                )
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        if hasattr(self, "lora_embedding_A"):
            if adapter_name in self.lora_embedding_A.keys():
                # initialize a the same way as the default for nn.linear and b to zero
                nn.init.zeros_(self.lora_embedding_A[adapter_name])
                nn.init.normal_(self.lora_embedding_B[adapter_name])
