# Read a pickle file and count top-level keys.
import os
import pandas as pd


def read_pickle_files_in_directory(file_path):

    df = pd.read_pickle(file_path)

    # Count dictionary keys.
    key_counts = {key: len(value) for key, value in df.items()}

    print("Key counts in the pickle file:")
    for key, count in key_counts.items():
        print(f"{key}: {count}")    


if __name__ == "__main__":
    # Example usage.
    dir_path = './qwen_wiki.pkl'
    combined_df = read_pickle_files_in_directory(dir_path)
