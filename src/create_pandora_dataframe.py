import os
import sys
import torch
import tqdm
import pandas as pd
from torch.utils.data import DataLoader
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from src.utils.parser_args import parser
from src.utils.train_utils import (
    test_load,
)
from src.utils.train_utils import get_samples_steps_per_epoch
from src.layers.inference_oc_pandora import create_and_store_graph_output, store_at_epoch_end


def main():
    # parse arguments 
    args = parser.parse_args()
    args = get_samples_steps_per_epoch(args)
    args.local_rank = 0
    test_loaders, data_config = test_load(args)
    df_showers_pandora = []
    total_number_events = 0

    for name, get_test_loader in test_loaders.items():
        test_loader = get_test_loader()
        number_batch = 0
        with tqdm.tqdm(test_loader) as tq:
            for g,y in tq:
                df_batch_pandora,total_number_events = create_and_store_graph_output(g,y,number_batch, None,total_number_events)
                df_showers_pandora.append(df_batch_pandora)
                number_batch += 1
        df_showers_pandora = pd.concat(df_showers_pandora)
        store_at_epoch_end(path_save= args.model_prefix,
                    df_batch_pandora =df_showers_pandora, 
                    local_rank=0,
                    step=0)


if __name__ == "__main__":
    main()
