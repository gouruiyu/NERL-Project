""" Grid Search over env parameters: predator acceleration, eat_predation_ratio (incentive balance between eating and avoid predation) """


from utils import GridSearch

main_file = 'train_ne.py'
args = {'eat_predation_ratio': [20],
        'predator_acceleration': [5],
        'altruism':[0, 0.25, 0.5, 0.75, 1],
        'expr_name': ['different_prosociality'],
}

# -------- create GridSearch object and run -------- #
myGridSearch = GridSearch(main_file, args, num_process=5)
myGridSearch.run()
