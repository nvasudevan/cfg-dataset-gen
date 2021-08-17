import numpy as np
from sklearn import model_selection
from stellargraph.mapper import PaddedGraphGenerator

from model import create_graph_classification_model, get_generators, train_fold


def repeated_k_fold(graphs, graph_labels, n_folds, n_repeats, es, epochs):
    test_accs = []
    stratified_folds = model_selection.RepeatedStratifiedKFold(
        n_splits=n_folds, n_repeats=n_repeats
    ).split(graph_labels, graph_labels)
    generator = PaddedGraphGenerator(graphs=graphs)

    for i, (train_index, test_index) in enumerate(stratified_folds):
        print(f"Training and evaluating on fold {i + 1} out of {n_folds * n_repeats}...")
        print("=> train_index: ", train_index)
        print("=> test_index: ", test_index)
        train_gen, test_gen = get_generators(generator, train_index=train_index,
                                             test_index=test_index,
                                             graph_labels=graph_labels,
                                             batch_size=10)
        model = create_graph_classification_model(generator)
        history, acc = train_fold(model, train_gen, test_gen, es, epochs)
        test_accs.append(acc)
        curr_mean = np.mean(test_accs) * 100
        curr_std = np.std(test_accs) * 100
        print(f"Accuracy over all folds mean: {curr_mean:.3}% and \
        std: {curr_std:.2}%")

    return test_accs
