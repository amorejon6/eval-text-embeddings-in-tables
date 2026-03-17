import argparse
import os
from experiments.exp1_row_tests.extract_embeddings import extract_base_embeddings
from experiments.exp1_row_tests.test_embeddings import test_random_reording_embeddings, test_random_deletion_of_columns, test_random_string, test_header_vector

def main():
    parser = argparse.ArgumentParser(description='Process Data')
    parser.add_argument('-i', '--input', default='sensors',
                        choices=['wikitables', 'dublin', 'sensors'],
                        help='Directorio de los datos')
    parser.add_argument('-m', '--model', default='nv-embed',
                        choices=['all', 'all-mini', 'all-mpnet', 'bge-large', 'snowflake', 'jasper', 'qwen'])
    parser.add_argument('-r', '--result', default='./indexs',
                        help='Name of the output folder that stores the indexs files')
    parser.add_argument('-t', '--test', default='test1',
                        choices=['all', 'test1', 'test2', 'test3', 'test4'],
                        help='Indicates which tests you want to run')
    parser.add_argument('-e', '--embeddings', default='yes',
                        choices=['yes', 'no'],
                        help='Indicates whether you want the original embeddings to be calculated or not.')
    args = parser.parse_args()

    dataset = args.input   
    args.input = 'datasets/' + args.input + '/'

    files = sorted(os.listdir(args.input))[:10]

    models = []
    if args.model == 'all':
        models = ['all-mpnet', 'bge-large', 'snowflake']
    else:
        models.append(args.model)

    # 1. Extract base embeddings
    if args.embeddings == 'yes':
        extract_base_embeddings(args, dataset, files, models)

    # 2. Random reordering of its columns test
    if args.test == 'test1' or args.test == 'all':
        print('Test 1')
        test_random_reording_embeddings(args, dataset, files, models)

    # 3. Random deleting 50% of columns test
    if args.test == 'test2' or args.test == 'all':
        print('Test 2')
        test_random_deletion_of_columns(args, dataset, files, models)
    
    # 4. Compare with random row test
    if args.test == 'test3' or args.test == 'all':
        print('Test 3')
        test_random_string(args, dataset, files, models)

    # 5. Compare with header row test
    if args.test == 'test4' or args.test == 'all':
        print('Test 4')
        test_header_vector(args, dataset, files, models)

if __name__ == "__main__":
    main()