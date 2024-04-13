import torch
from transformers import AutoModelForCausalLM
from mergoo.composers.composer_moe import ComposeMoeExperts
from mergoo.composers.composer_lora_moe import ComposeLoraMoeExperts


class ComposeExperts:
    def __init__(
        self,
        config,
        torch_dtype=torch.float16,
        device="cpu",
        device_map="auto",
        max_shard_size="9GB",
        model_cls=AutoModelForCausalLM,
    ):
        """
        Args:
            config (dict): Configuration required to setup the composer. Explore configs/ for examples/
            torch_dtype (torch.dtype, optional): Datatype for loading and saving the weights. Defaults to torch.float16.
            device (str, optional): Defaults to "cpu".
            device_map (str, optional): Defaults to "auto".
            max_shard_size (str, optional): Maximum Shard size checkpoint chuncks. Defaults to "9GB".
            model_cls (type, optional): Change this when using a architecture not registered with transformers. Defaults to AutoModelForCausalLM.
        """

        if "base_model" not in config:
            composer_class = ComposeMoeExperts
        else:
            composer_class = ComposeLoraMoeExperts

        self.composer = composer_class(
            config, torch_dtype, device, device_map, max_shard_size, model_cls
        )

    def __getattr__(self, attr: str):
        return getattr(self.composer, attr)
