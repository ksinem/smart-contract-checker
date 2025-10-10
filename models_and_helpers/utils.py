import pandas as pd
import json
import os
import gensim.downloader as api
import numpy as np
from tqdm import tqdm
from ydata_profiling import ProfileReport



with open("../config.json", "r") as c:
    config = json.load(c)


def save_embedding_vectors(wv_model=api.load("glove-wiki-gigaword-300"),
                           file=pd.read_csv("../data/SC_Vuln_8label.csv", index_col=0)):
    """
    Generates an average GloVe embedding vector for each code snippet
    and saves it to a local file.

    It loads pre-trained word vectors (default: GloVe 300d) and iterates through
    the code column of the input file. For each code, it computes the mean
    of the word vectors of all recognized words to create a fixed-size
    document embedding.

    Parameters
    ----------
    wv_model : gensim.models.keyedvectors.KeyedVectors, optional
        The pre-trained word embedding model to use (e.g., Word2Vec, GloVe).
        Defaults to 'glove-wiki-gigaword-300'.
    file : pandas.DataFrame, optional
        The DataFrame containing the code snippets. It must have a column
        named 'code'. Defaults to the data file specified in the config.

    Returns
    -------
    int
        Returns 0 upon successful completion.
    """
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
        np.save(f"../data/wv_encodings/code_{num}_encoded.npy", final_embedding)
    return 0


def generate_pandas_report(df, title):
    """
        Generates a comprehensive interactive HTML profiling report for a
        Pandas DataFrame using ydata-profiling.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame for which the report should be generated.
        title : str
            The title to be displayed in the report and used as the filename
            (e.g., "Smart contracts with 8 vulnerabilites.html").

        Returns
        -------
        int
            Returns 0 upon successful file generation.
        """
    profile = ProfileReport(df, title=title, explorative=True)
    profile.to_file(f"{title}.html")
    return 0


if __name__ == "__main__":
    save_embedding_vectors()

