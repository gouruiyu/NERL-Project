from utils import GridSearch

main_file = 'testFair.py'
args = {'predator_accceleration': [1., 2., 3., 4., 5.], 
        'noise_regulator': [0.0, 0.1, 0.5, 1.0, 5.0],
        'eval_num_episodes':[100, 50, 200]}

# -------- create GridSearch object and run -------- #
myGridSearch = GridSearch(main_file, args, num_process=30)
myGridSearch.run()