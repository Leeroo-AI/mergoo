import os
import re
import json
import torch
import collections
from transformers.pytorch_utils import id_tensor_storage
from transformers.modeling_utils import (
    _add_variant,
    shard_checkpoint,
    safe_save_file,
    is_safetensors_available,
)
from transformers.utils import (
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    WEIGHTS_INDEX_NAME,
)


def save_pretrained(
    save_directory,
    state_dict,
    is_main_process=True,
    save_function=torch.save,
    max_shard_size="4GB",
    safe_serialization: bool = True,
    variant=None,
    tied_weights_keys=None,
    _keys_to_ignore_on_save=None,
    **kwargs,
):
    """
    Save a model and its configuration file to a directory, so that it can be re-loaded using the
    [`~PreTrainedModel.from_pretrained`] class method.

    Arguments:
        save_directory (`str` or `os.PathLike`):
            Directory to which to save. Will be created if it doesn't exist.
        is_main_process (`bool`, *optional*, defaults to `True`):
            Whether the process calling this is the main process or not. Useful when in distributed training like
            TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
            the main process to avoid race conditions.
        state_dict (nested dictionary of `torch.Tensor`):
            The state dictionary of the model to save. Will default to `self.state_dict()`, but can be used to only
            save parts of the model or if special precautions need to be taken when recovering the state dictionary
            of a model (like when using model parallelism).
        save_function (`Callable`):
            The function to use to save the state dictionary. Useful on distributed training like TPUs when one
            need to replace `torch.save` by another method.
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
            repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
            namespace).
        max_shard_size (`int` or `str`, *optional*, defaults to `"5GB"`):
            The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
            lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).
            We default it to 5GB in order for models to be able to run easily on free-tier google colab instances
            without CPU OOM issues.

            <Tip warning={true}>

            If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
            which will be bigger than `max_shard_size`.

            </Tip>

        safe_serialization (`bool`, *optional*, defaults to `True`):
            Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
        variant (`str`, *optional*):
            If specified, weights are saved in the format pytorch_model.<variant>.bin.
        token (`str` or `bool`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
            the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
        save_peft_format (`bool`, *optional*, defaults to `True`):
            For backward compatibility with PEFT library, in case adapter weights are attached to the model, all
            keys of the state dict of adapters needs to be pre-pended with `base_model.model`. Advanced users can
            disable this behaviours by setting `save_peft_format` to `False`.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
    """
    if not safe_serialization:
        raise NotImplementedError

    if safe_serialization and not is_safetensors_available():
        raise ImportError(
            "`safe_serialization` requires the `safetensors library: `pip install safetensors`.")

    if os.path.isfile(save_directory):
        raise f"Provided path ({save_directory}) should be a directory, not a file"
        return

    os.makedirs(save_directory, exist_ok=True)

    # Handle the case where some state_dict keys shouldn't be saved
    if _keys_to_ignore_on_save is not None:
        for ignore_key in _keys_to_ignore_on_save:
            if ignore_key in state_dict.keys():
                del state_dict[ignore_key]

    if safe_serialization:
        # Safetensors does not allow tensor aliasing.
        # We're going to remove aliases before saving
        ptrs = collections.defaultdict(list)
        for name, tensor in state_dict.items():
            # Sometimes in the state_dict we have non-tensor objects.
            # e.g. in bitsandbytes we have some `str` objects in the state_dict
            if isinstance(tensor, torch.Tensor):
                ptrs[id_tensor_storage(tensor)].append(name)
            else:
                # In the non-tensor case, fall back to the pointer of the object itself
                ptrs[id(tensor)].append(name)

        # These are all the pointers of shared tensors.
        shared_ptrs = {ptr: names for ptr,
                       names in ptrs.items() if len(names) > 1}
        warn_names = set()
        for names in shared_ptrs.values():
            # Removing the keys which are declared as known duplicates on
            # load. This allows to make sure the name which is kept is consistent.
            if tied_weights_keys is not None:
                found = 0
                for name in sorted(names):
                    matches_pattern = any(re.search(pat, name)
                                          for pat in tied_weights_keys)
                    if matches_pattern and name in state_dict:
                        found += 1
                        if found < len(names):
                            del state_dict[name]

            # When not all duplicates have been cleaned, still remove those keys, but put a clear warning.
            # If the link between tensors was done at runtime then `from_pretrained` will not get
            # the key back leading to random tensor. A proper warning will be shown
            # during reload (if applicable), but since the file is not necessarily compatible with
            # the config, better show a proper warning.
            found = 0
            for name in names:
                if name in state_dict:
                    found += 1
                    if found > 1:
                        del state_dict[name]
                        warn_names.add(name)
        if len(warn_names) > 0:
            print(
                f"Removed shared tensor {warn_names} while saving. This should be OK, but check by verifying that you don't receive any warning while reloading")

    # Shard the model if it is too big.
    weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
    weights_name = _add_variant(weights_name, variant)

    shards, index = shard_checkpoint(
        state_dict, max_shard_size=max_shard_size, weights_name=weights_name)

    # Clean the folder from a previous save
    for filename in os.listdir(save_directory):
        full_filename = os.path.join(save_directory, filename)
        # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
        # in distributed settings to avoid race conditions.
        weights_no_suffix = weights_name.replace(
            ".bin", "").replace(".safetensors", "")

        # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
        filename_no_suffix = filename.replace(
            ".bin", "").replace(".safetensors", "")
        reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

        if (
            filename.startswith(weights_no_suffix)
            and os.path.isfile(full_filename)
            and filename not in shards.keys()
            and is_main_process
            and reg.fullmatch(filename_no_suffix) is not None
        ):
            os.remove(full_filename)

    # Save the model
    for shard_file, shard in shards.items():
        if safe_serialization:
            # At some point we will need to deal better with save_function (used for TPU and other distributed
            # joyfulness), but for now this enough.
            safe_save_file(shard, os.path.join(
                save_directory, shard_file), metadata={"format": "pt"})
        else:
            save_function(shard, os.path.join(save_directory, shard_file))

    if index is None:
        path_to_weights = os.path.join(save_directory, weights_name)
        print(f"Model weights saved in {path_to_weights}")
    else:
        save_index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
        save_index_file = os.path.join(
            save_directory, _add_variant(save_index_file, variant))
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
        print(
            f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
            f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
            f"index located at {save_index_file}."
        )
