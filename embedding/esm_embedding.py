import os

import esm
# import matplotlib
import pandas as pd
import torch


model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
batch_convert = alphabet.get_batch_converter()
model.eval()
def get_fea():
    batch_labels, batch_strs, batch_tokens = batch_convert(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        ten = token_representations[i, 1: tokens_len - 1].mean(0)
        ten_ = ten.numpy()
        ttt.append(ten_)

def main():
    for f in os.listdir(args.path):
        if f.endswith('.csv'):
            file = pd.read_csv(path + f
                               , header=0, index_col=None)
            save_path = path + "esm_features_" + f.split(".")[0] + ".csv"
            data = []
            ttt = []
            for index, row in file.iterrows():
                seq = row["seq"]
                label = row["label"]
                data.append((label, seq))
                get_fea()
                data.clear()
                nnn = [x.tolist() for x in ttt]
                re = pd.DataFrame(nnn[0]).T
                re["seq"] = seq
                re["ddg"] = label

                re.to_csv(save_path, mode="a",
                          header=False,
                          index=False)
                ttt.clear()
            print(f, "finished!")

def get_finetune_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str,
                        default="./",
                        help='output path')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args=get_finetune_config()
    main()