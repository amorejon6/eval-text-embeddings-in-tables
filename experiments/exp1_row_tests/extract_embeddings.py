import joblib
import os
import pandas as pd
import traceback
from tqdm import tqdm
from experiments.shared.utils import get_model, find_delimiter, content_embeddings

# Extract base embeddings from the datasets
def extract_base_embeddings(args, dataset, files, models):
    for m in models:
        model, tokenizer, dimensions = get_model(m)
        model.max_seq_length = dimensions

        output_dir = os.path.join("cache_embeddings", "exp1", m, dataset, "base")
        os.makedirs(output_dir, exist_ok=True)

        for file in tqdm(files):
            try:
                # Read dataframe
                delimiter = find_delimiter(args.input + file)
                df = pd.read_csv(args.input + file, sep=delimiter)

                # Remove columns with all NaNs
                df = df.dropna(axis='columns', how='all')
                df.dropna(how='all', inplace=True)
                print(df)

                # Calculate embeddings
                embs = content_embeddings(model, df, dimensions, m, tokenizer)

                # Save embeddings as pickle for downstream tests
                file_base = os.path.splitext(file)[0]
                output_path = os.path.join(output_dir, f"{file_base}.pkl")
                joblib.dump(embs, output_path)
                
            except Exception as e:
                print('Error en archivo', file)
                print(e)
                print(traceback.format_exc())
