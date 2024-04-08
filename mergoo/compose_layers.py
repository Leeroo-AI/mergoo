import torch
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
    if layer_idx in config.router_layers_index:
        if name in config.router_layers:
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
