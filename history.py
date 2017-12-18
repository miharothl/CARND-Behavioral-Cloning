import argparse

from tools.data_explorer import DataExplorer
from tools.train_history import TrainHistory

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training history viewer')
    parser.add_argument(
        'training_history',
        type=str,
        help='Path to training history .p file.'
    )
    args = parser.parse_args()


    history = TrainHistory()
    explorer = DataExplorer()

    history_objects = history.load_history(args.training_history)
    explorer.plot_training_history(history_objects)

    the = 'end'




