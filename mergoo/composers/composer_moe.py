import os
import json
import torch
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from mergoo.safe_saving import save_pretrained


class ComposeMoeExperts:
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
        self.config = config
        self.model_configs = []
        self.torch_dtype = torch_dtype
        self._tied_weights_keys = []
        self.device = device
        self.device_map = device_map
        self.max_shard_size = max_shard_size
        self.model_cls = model_cls
        self.config["router_layers_index"] = self.config.get("router_layers_index", [])
        self.moe_layer_index = self.config["router_layers_index"]
        self.select_moe_model_config_idx = 0
        self._set_moe_layer_index()

    def _set_moe_layer_index(self):
        if len(self.moe_layer_index) == 0:
            self._check_moe_layers = lambda x: True
            print(f"MoE Layer Index : [*]")

        elif len(self.moe_layer_index) >= 1 and self.moe_layer_index[0] is not None:
            self._check_moe_layers = lambda x: x in self.moe_layer_index
            print(f"MoE Layer Index : {self.moe_layer_index}")

        else:
            self._check_moe_layers = lambda x: False
            print(f"No MoE layer indexes.")

    def _is_layer_suitable_for_router(self, layer_identifier, model_layer):
        model_layer_index = [int(x) for x in model_layer.split(".") if x.isdigit()]
        if not model_layer_index:
            valid_layer_index = False
        else:
            assert len(model_layer_index) == 1
            valid_layer_index = self._check_moe_layers(model_layer_index[0])

        if (layer_identifier in model_layer) and valid_layer_index:
            if self.config["model_type"] in ("llama", "mistral", "phi"):
                if "mlp" in model_layer or "self_attn" in model_layer:
                    return True
            elif self.config["model_type"] == "bert":
                if "attention" in model_layer:
                    return True
        return False

    def _load_base_model(self, model_id):
        try:
            mps = torch.backends.mps.is_available()
        except:
            mps = False
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if config.model_type == "bert":
            model = self.model_cls.from_pretrained(
                model_id, torch_dtype=self.torch_dtype, trust_remote_code=True
            )
        else:
            model = self.model_cls.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                device_map=None if mps else self.device_map,
                trust_remote_code=True,
            )
        if mps:
            return model.to(self.device)
        return model

    def compose(self):
        """
        Compose all the experts into a single unified checkpoint.
        """
        n = len(self.config["experts"])
        self.state_dict = {}
        count_total_router_layers = 0
        for ix, expert in enumerate(self.config["experts"]):
            model_id = expert["model_id"]
            model = self._load_base_model(model_id)
            print(f"merging expert : {model_id}")

            if hasattr(model, "_tied_weights_keys"):
                self._tied_weights_keys.extend(model._tied_weights_keys)
            self.model_configs.append(model.config.to_dict())
            router_layers = self.config["router_layers"]

            count_router_layers = 0
            count_averaged_layers = 0
            for layer_name, param in tqdm(model.state_dict().items()):
                is_merge_layer = True
                for router_layer in router_layers:
                    if self._is_layer_suitable_for_router(router_layer, layer_name):
                        is_merge_layer = False
                        wb = layer_name.split(".")[-1]
                        new_layer_name = layer_name.split(f"{wb}")[0]
                        new_layer_name = (
                            # TODO debug this
                            f"{new_layer_name}experts.{ix}.{wb}"
                        )
                        assert new_layer_name not in self.state_dict
                        self.state_dict[new_layer_name] = param.to("cpu")
                        count_total_router_layers += 1
                        count_router_layers += 1

                if is_merge_layer:  # average
                    prev_weight = self.state_dict.get(layer_name)
                    if prev_weight is None:
                        prev_weight = torch.tensor(0)
                    else:
                        if not prev_weight.shape == param.shape:
                            prev_weight, param = self._shape_adjuster(
                                prev_weight, param, ix
                            )

                    try:  # sometimes data is empty / non weights
                        self.state_dict[layer_name] = prev_weight + (param / n).to(
                            "cpu"
                        )
                    except Exception as e:
                        print(layer_name, param)
                        self.state_dict[layer_name] = param.to("cpu")

                    count_averaged_layers += 1

        print(f"count_averaged_layers : {count_averaged_layers}")
        print(f"count_router_layers : {count_router_layers}")
        print(f"count_total_router_layers : {count_total_router_layers}")
        del model
        gc.collect()

    def _shape_adjuster(self, tensor1, tensor2, ix):
        assert tensor1.ndim == tensor2.ndim
        if tensor1.shape[0] < tensor2.shape[0]:
            pad_tensor = torch.zeros_like(
                tensor2, dtype=self.torch_dtype, device=tensor1.device
            )
            pad_tensor[: tensor1.shape[0]] += tensor1
            tensor1 = pad_tensor
            self.select_moe_model_config_idx = ix
        else:
            pad_tensor = torch.zeros_like(
                tensor1, dtype=self.torch_dtype, device=tensor2.device
            )
            pad_tensor[: tensor2.shape[0]] += tensor2
            tensor2 = pad_tensor
        return tensor1, tensor2

    def save_checkpoint(self, checkpoint_path):
        """
        Save the composed Unified checkpoint.
        Checkpoints are saved as safe tensors in shards(chuncks).
        """
        os.makedirs(checkpoint_path, exist_ok=True)
        config = self.model_configs[self.select_moe_model_config_idx]
        config["num_experts"] = len(self.config["experts"])
        config["num_experts_per_tok"] = self.config["num_experts_per_tok"]
        config["router_layers"] = self.config["router_layers"]

        layer_indexes = list(range(config["num_hidden_layers"]))
        if not self.config["router_layers_index"]:
            config["router_layers_index"] = layer_indexes

        else:
            config["router_layers_index"] = list(
                set(layer_indexes).intersection(set(self.config["router_layers_index"]))
            )

        json.dump(config, open(f"{checkpoint_path}/config.json", "w"), indent=1)
        save_pretrained(
            save_directory=checkpoint_path,
            state_dict=self.state_dict,
            tied_weights_keys=self._tied_weights_keys,
            max_shard_size=self.max_shard_size,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.config["experts"][self.select_moe_model_config_idx]["model_id"]
        )
        tokenizer.save_pretrained(checkpoint_path)
        print(f"checkpoint saved at {checkpoint_path}")
