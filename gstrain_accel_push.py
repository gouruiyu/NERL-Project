""" Grid Search over env parameters: predator acceleration, eat_predation_ratio (incentive balance between eating and avoid predation) """


from utils import GridSearch

main_file = 'train_ne.py'
args = {'eat_predation_ratio': [100, 20, 1],
        'predator_acceleration': [10, 5],
        'altruism':[0, 0.5, 0.7]
}

# -------- create GridSearch object and run -------- #
myGridSearch = GridSearch(main_file, args, num_process=18)
myGridSearch.run()
