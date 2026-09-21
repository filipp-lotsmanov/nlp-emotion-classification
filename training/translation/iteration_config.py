CURRENT_ITERATION = "iteration_3"

ITERATIONS = {
    "iteration_1": {
        "name": "First Iteration",
        "en_file": "data/english_first_iteration.tsv",
        "ru_file": "data/russian_first_iteration.tsv",
        "checkpoint_dir": "checkpoints/iteration_1",
        "log_dir": "logs/iteration_1",
        "data_dir": "data/iteration_1",
    },
    "iteration_2": {
        "name": "Second Iteration",
        "en_file": "data/english_second_iteration.tsv",
        "ru_file": "data/russian_second_iteration.tsv",
        "checkpoint_dir": "checkpoints/iteration_2",
        "log_dir": "logs/iteration_2",
        "data_dir": "data/iteration_2",
    },
    "iteration_3": {
        "name": "Third Iteration",
        "en_file": "data/english_third_iteration.tsv",
        "ru_file": "data/russian_third_iteration.tsv",
        "checkpoint_dir": "checkpoints/iteration_3",
        "log_dir": "logs/iteration_3",
        "data_dir": "data/iteration_3",
    },
}

ITER_CONFIG = ITERATIONS[CURRENT_ITERATION]
