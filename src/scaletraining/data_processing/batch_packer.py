from itertools import chain
from datasets import load_from_disk
import hydra
from omegaconf import DictConfig
from scaletraining.util.utils import write_metadata, tokenized_dir, packed_dir, _cfg_subset, flatten_cfg

def group_texts(examples, block_size: int):
    '''
    Group tokenized ids into fixed-size blocks for causal LM training.
    Steps:
      1. Flatten the batch of examples into one long sequence.
      2. Trim remainder so only full blocks are emitted.
      3. Return list of blocks as 'input_ids' only (labels computed during training).
    Args:
      examples: dict with key 'input_ids' -> list[list[int]].
      block_size: int maximum sequence length per block.
    Returns:
      dict with 'input_ids': list[list[int]] of length block_size.
    '''
    concatenated = list(chain.from_iterable(examples["input_ids"]))
    total_length = (len(concatenated) // block_size) * block_size
    if total_length == 0:
        return {"input_ids": []}
    blocks = [concatenated[i:i+block_size] for i in range(0, total_length, block_size)]
    return {"input_ids": blocks}

def pack_and_save(
    tokenized_path: str,
    packed_path: str,
    block_size: int,
    num_proc: int = 1,
    map_batch_size: int = 200,
    writer_batch_size: int = 1000,
    metadata: dict | None = None,
):
    train = load_from_disk(f"{tokenized_path}/train")
    packed_train = train.map(
        group_texts,
        fn_kwargs={"block_size": block_size},
        batched=True,
        batch_size=map_batch_size,
        num_proc=num_proc,
        remove_columns=train.column_names,
        load_from_cache_file=True,
        writer_batch_size=writer_batch_size,
        desc="Packing train",
    )
    packed_train.save_to_disk(f"{packed_path}/train")

    try:
        val = load_from_disk(f"{tokenized_path}/val")
        packed_val = val.map(
            group_texts,
            fn_kwargs={"block_size": block_size},
            batched=True,
            batch_size=map_batch_size,
            num_proc=num_proc,
            remove_columns=val.column_names,
            load_from_cache_file=True,
            writer_batch_size=writer_batch_size,
            desc="Packing val",
        )
        packed_val.save_to_disk(f"{packed_path}/val")
    except Exception as e:
        print(f"There is not validation split for packing: {e}")

    # Save metadata for compatibility checks
    if metadata is not None:
        write_metadata(packed_path, metadata)


@hydra.main(version_base=None, config_path='../../../conf', config_name='config')
def main(cfg: DictConfig) -> None:
    """Hydra console script entrypoint for dataset packing.

    Uses tokenized_dir(cfg) as input and writes packed blocks to packed_dir(cfg).
    """
    cfg = flatten_cfg(cfg)
    tok_dir = tokenized_dir(cfg)
    pk_dir = packed_dir(cfg)
    pack_and_save(
        tokenized_path=tok_dir,
        packed_path=pk_dir,
        block_size=int(cfg.max_seq_len),
        num_proc=int(cfg.pack_num_proc),
        map_batch_size=int(cfg.pack_map_batch_size),
        writer_batch_size=int(cfg.pack_writer_batch_size),
        metadata={"config": _cfg_subset(cfg)},
    )
