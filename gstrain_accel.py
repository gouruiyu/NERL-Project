""" Grid Search over env parameters: predator acceleration, eat_predation_ratio (incentive balance between eating and avoid predation) """


from utils import GridSearch

main_file = 'train_ne.py'
args = {'eat_predation_ratio': [20, 10],
        'predator_acceleration': [6., 5., 4.],
        'altruism':[0, 0.5, 1]
}

# -------- create GridSearch object and run -------- #
myGridSearch = GridSearch(main_file, args, num_process=18)
myGridSearch.run()
