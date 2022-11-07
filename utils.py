import os
import pandas as pd
import tqdm


def read_parquet_dataset_from_local(path_to_dataset: str, start_from: int = 0,
                                     num_parts_to_read: int = 1, columns=None, verbose=False) -> pd.DataFrame:
    """
    Reads num_parts_to_read parquet partitions and returns the resulting pd.DataFrame

    :param path_to_dataset: directory with parquet partitions
    :param start_from: partition number to start with
    :param num_parts_to_read: amount of partitions to read
    :param columns: columns to read and include
    :return: pd.DataFrame
    """

    res = []
    dataset_paths = sorted([os.path.join(path_to_dataset, filename) for filename in os.listdir(path_to_dataset)
                              if filename.startswith('part')])

    start_from = max(0, start_from)
    if num_parts_to_read < 0:
        chunks = dataset_paths[start_from: ]
    else:
        chunks = dataset_paths[start_from: start_from + num_parts_to_read]
    if verbose:
        print('Reading chunks:\n')
        for chunk in chunks:
            print(chunk)
    for chunk_path in tqdm.tqdm_notebook(chunks, desc="Reading dataset with pandas"):
        chunk = pd.read_parquet(chunk_path, columns=columns)
        for col_name, col_type in [('amnt', 'float32'), ('hour_diff', 'int32')]:
            if col_name in chunk.columns:
                chunk[col_name] = chunk[col_name].astype(col_type)

        res.append(chunk)
    return pd.concat(res).reset_index(drop=True)
