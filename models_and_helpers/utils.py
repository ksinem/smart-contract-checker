import pandas as pd
import json
import os
import gensim.downloader as api
import numpy as np
from tqdm import tqdm



with open("../config.json", "r") as c:
    config = json.load(c)


def save_embedding_vectors(wv_model=api.load("glove-wiki-gigaword-300"),
                           file=pd.read_csv("../data_old/SC_Vuln_8label.csv", index_col=0)):
    for num, code in tqdm(enumerate(file['code']), total=len(file['code'])):
        code_encoded = []
        for word in code.split():
            try:
                code_encoded.append(wv_model[word])
                # per word per code: 300dimensional vector
            except KeyError:
                continue
        code_matrix = np.stack(code_encoded)
        final_embedding = np.mean(code_matrix, axis=0).astype(np.float32)
        np.save(f"../data_old/wv_encodings/code_{num}_encoded.npy", final_embedding)
    return 0




if __name__ == "__main__":
    save_embedding_vectors()
