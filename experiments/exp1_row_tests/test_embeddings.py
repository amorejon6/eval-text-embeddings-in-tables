import joblib
import pandas as pd
import numpy as np
import csv
import os
import traceback
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from experiments.shared.utils import get_model, find_delimiter, content_embeddings, random_words, load_clean_csv, load_embeddings

def test_random_reording_embeddings(args, dataset, files, models):
    for m in models:
        model, tokenizer, dimensions = get_model(m)
        model.max_seq_length = dimensions
        avg_similarities = np.empty(0)
        std_similarities = np.empty(0)

        path = os.path.join('cache_embeddings', 'exp1', m, dataset, 'base')
        shuffle_path = os.path.join('cache_embeddings', 'exp1', m, dataset, 'test1')
        os.makedirs(shuffle_path, exist_ok=True)
        
        # print('Number of files:', len(files))
        for file in tqdm(files):
            try:
                df = pd.read_csv(args.input + file, dtype=str)

                # Remove columns with all NaNs
                df = df.dropna(axis='columns', how='all')
                df.dropna(how='all', inplace=True)

                # Reorder the columns randomly
                df = shuffle(df)

                # Calculate embeddings for the shuffled table and save them in a pickle file
                embs = content_embeddings(model, df, dimensions, m, tokenizer)
                file_base = os.path.splitext(file)[0]
                output_path = os.path.join(shuffle_path, f"{file_base}.pkl")
                joblib.dump(embs, output_path)
                
                # Load original embeddings
                base_embs = load_embeddings(m, dataset, path, file_base)
                # print(base_embs)

                # Compare original embbedings with embeddings obtained after having mixed the table
                similarity_scores = cosine_similarity(base_embs, embs)

                # Average value of embeddings similarities
                avg_similarity = np.mean(similarity_scores)

                # Standard deviation of embeddings similarities
                std_similarity = np.std(similarity_scores)

                # Save these values
                avg_similarities = np.append(avg_similarities, avg_similarity)
                std_similarities = np.append(std_similarities, std_similarity)

            except Exception as e:
                print('Error en archivo', file)
                print(e)
                print(traceback.format_exc())
        
        # Average value of model similarities
        avg_similarity = np.mean(avg_similarities)

        # Average value of the standard deviations of the model similarities
        std_similarity = np.std(std_similarities)

        fields = np.array(["Average, Standard desviation"])
        values = np.append(avg_similarity, std_similarity)

        directory = 'results/exp1/test1/' + dataset + '/'
        if not os.path.isdir(directory):
            os.makedirs(directory)

        # Write values in a CSV file
        file_name = directory + m + '_' + dataset + '.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(fields)
            writer.writerow(values)

def test_random_deletion_of_columns(args, dataset, files, models):
    for m in models:
        model, tokenizer, dimensions = get_model(m)
        model.max_seq_length = dimensions
        avg_similarities = np.empty(0)
        std_similarities = np.empty(0)

        path = os.path.join('cache_embeddings', 'exp1', m, dataset, 'base')
        deletion_path = os.path.join('cache_embeddings', 'exp1', m, dataset, 'test2')
        os.makedirs(deletion_path, exist_ok=True)

        for file in tqdm(files):
            try:
                df = pd.read_csv(args.input + file, dtype=str)

                # Remove columns with all NaNs
                df = df.dropna(axis='columns', how='all')
                df.dropna(how='all', inplace=True)

                # Random deletion of columns
                columns_to_delete = np.random.choice(df.columns, size=int(df.shape[1] * 0.5), replace=False)
                df = df.drop(columns=columns_to_delete)

                # Calculate embeddings for the modified table and save them in a pickle file
                embs = content_embeddings(model, df, dimensions, m, tokenizer)
                file_base = os.path.splitext(file)[0]
                output_path = os.path.join(deletion_path, f"{file_base}.pkl")
                joblib.dump(embs, output_path)

                # Load original embeddings
                base_embs = load_embeddings(m, dataset, path, file_base)

                # Compare original embbedings with embeddings obtained after random deletion of columns
                similarity_scores = cosine_similarity(base_embs, embs)

                # Average value of embeddings similarities
                avg_similarity = np.mean(similarity_scores)

                # Standard deviation of embeddings similarities
                std_similarity = np.std(similarity_scores)

                # Save these values
                avg_similarities = np.append(avg_similarities, avg_similarity)
                std_similarities = np.append(std_similarities, std_similarity)

            except Exception as e:
                print('Error en archivo', file)
                print(e)
                print(traceback.format_exc())

        # Average value of model similarities
        avg_similarity = np.mean(avg_similarities)

        # Average value of the standard deviations of the model similarities
        std_similarity = np.std(std_similarities)

        fields = np.array(["Average, Standard desviation"])
        values = np.append(avg_similarity, std_similarity)

        directory = 'results/exp1/test2/' + dataset + '/'
        if not os.path.isdir(directory):
            os.makedirs(directory)

        # Write values in a CSV file
        file_name = directory + m + '_' + dataset + '.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(fields)
            writer.writerow(values)

def test_random_string(args, dataset, files, models):
    for m in models:
        model, tokenizer, dimensions = get_model(m)
        model.max_seq_length = dimensions
        avg_similarities = np.empty(0)
        std_similarities = np.empty(0)

        path = os.path.join('cache_embeddings', 'exp1', m, dataset, 'base')

        random_stream_path = os.path.join('cache_embeddings', 'exp1', m, dataset, 'test3')
        os.makedirs(random_stream_path, exist_ok=True)

        for file in tqdm(files):
            try:

                df = pd.read_csv(args.input + file, nrows=1)
                # Calculate random row embedding and save it in a pickle file
                embs = content_embeddings(model, df, dimensions, m, tokenizer)

                # Generate random row
                df = pd.DataFrame(np.random.choice(random_words, size=(1, df.shape[1])))

                # Calculate random row embedding and save it in a pickle file
                embs = content_embeddings(model, df, dimensions, m, tokenizer)
                output_path = os.path.join(random_stream_path, "random_string.pkl")
                joblib.dump(embs, output_path)

                # Load original embeddings
                file_base = os.path.splitext(file)[0]
                base_embs = load_embeddings(m, dataset, path, file_base)

                # Compare original embbedings with random string embedding
                similarity_scores = cosine_similarity(base_embs, embs)

                # Average value of embeddings similarities
                avg_similarity = np.mean(similarity_scores)

                # Standard deviation of embeddings similarities
                std_similarity = np.std(similarity_scores)

                # Save these values
                avg_similarities = np.append(avg_similarities, avg_similarity)
                std_similarities = np.append(std_similarities, std_similarity)

            except Exception as e:
                print('Error en archivo', file)
                print(e)
                print(traceback.format_exc())

        # Average value of model similarities
        avg_similarity = np.mean(avg_similarities)

        # Average value of the standard deviations of the model similarities
        std_similarity = np.std(std_similarities)

        fields = np.array(["Average, Standard desviation"])
        values = np.append(avg_similarity, std_similarity)

        directory = 'results/exp1/test3/' + dataset + '/'
        if not os.path.isdir(directory):
            os.makedirs(directory)

        # Write values in a CSV file
        file_name = directory + m + '_' + dataset + '.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(fields)
            writer.writerow(values)

def test_header_vector(args, dataset, files, models):
    for m in models:
        model, tokenizer, dimensions = get_model(m)
        model.max_seq_length = dimensions
        avg_similarities = np.empty(0)
        std_similarities = np.empty(0)

        path = os.path.join('cache_embeddings', 'exp1', m, dataset, 'base')
        header_path = os.path.join('cache_embeddings', 'exp1', m, dataset, 'test4')
        os.makedirs(header_path, exist_ok=True)

        for file in tqdm(files):
            try:
                df = pd.read_csv(args.input + file, dtype=str, nrows=1)

                # Calculate header embedding and save it in a pickle file
                embs = content_embeddings(model, df, dimensions, m, tokenizer, header=True)
                file_base = os.path.splitext(file)[0]
                output_path = os.path.join(header_path, f"{file_base}.pkl")
                joblib.dump(embs, output_path)

                # Load original embeddings
                base_embs = load_embeddings(m, dataset, path, file_base)

                # Compare original embbedings with header vector embedding 
                similarity_scores = cosine_similarity(base_embs, embs)

                # Average value of embeddings similarities
                avg_similarity = np.mean(similarity_scores)

                # Standard deviation of embeddings similarities
                std_similarity = np.std(similarity_scores)

                # Save these values
                avg_similarities = np.append(avg_similarities, avg_similarity)
                std_similarities = np.append(std_similarities, std_similarity)

            except Exception as e:
                print('Error en archivo', file)
                print(e)
                print(traceback.format_exc())

        # Average value of model similarities
        avg_similarity = np.mean(avg_similarities)

        # Average value of the standard deviations of the model similarities
        std_similarity = np.std(std_similarities)

        fields = np.array(["Average, Standard desviation"])
        values = np.append(avg_similarity, std_similarity)

        directory = 'results/exp1/test4/' + dataset + '/'
        if not os.path.isdir(directory):
            os.makedirs(directory)

        # Write values in a CSV file
        file_name = directory + m + '_' + dataset + '.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(fields)
            writer.writerow(values)
