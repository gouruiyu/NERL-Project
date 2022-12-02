from utils import GridSearch

main_file = 'train.py'
args = {'max_cycles': [500],
        'n_rounds': [8, 16],
        'mutation_std': [0.01, 0.1, 1, 10],
        'init': ['xavier_normal', 'kaiming_normal'],
        'activation': ['relu', 'tanh']
}

# -------- create GridSearch object and run -------- #
myGridSearch = GridSearch(main_file, args, num_process=36)
myGridSearch.run()
