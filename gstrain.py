from utils import GridSearch

main_file = 'train.py'
args = {'max_cycles': [100, 200, 500], 
        'n_hidden_layer': [1, 2],
        'n_rounds': [4, 8],
        'mutation_std': [0.0001, 0.01, 0.1]
}

# -------- create GridSearch object and run -------- #
myGridSearch = GridSearch(main_file, args, num_process=36)
myGridSearch.run()
