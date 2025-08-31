from itertools import chain
from datasets import load_from_disk

def group_texts(examples, block_size: int):
    # 1) concatenate within this map batch
    concatenated = list(chain.from_iterable(examples["input_ids"]))
    # 2) drop remainder so we only emit full blocks
    total_length = (len(concatenated) // block_size) * block_size
    if total_length == 0:
        return {"input_ids": [], "labels": []}
    blocks = [concatenated[i:i+block_size] for i in range(0, total_length, block_size)]
    return {"input_ids": blocks, "labels": [b[:] for b in blocks]}  # causal LM: labels==inputs

def pack_and_save(
    tokenized_path: str,
    packed_path: str,
    block_size: int,
    num_proc: int = 1,
    map_batch_size: int = 200,
    writer_batch_size: int = 1000,
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
    except Exception:
        pass  # no val split
