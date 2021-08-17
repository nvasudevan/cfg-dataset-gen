import numpy as np
from sklearn import model_selection
from stellargraph.layer import GCNSupervisedGraphClassification
from stellargraph.mapper import PaddedGraphGenerator

from model import create_graph_classification_model


def train_fold(model, train_gen, test_gen, es, epochs):
    history = model.fit(train_gen,
                        validation_data=test_gen,
                        epochs=epochs,
                        verbose=0,
                        callbacks=[es])

    test_metrics = model.evaluate(test_gen, verbose=2)
    test_acc = test_metrics[model.metrics_names.index("acc")]

    return history, test_acc


# Get test and train generators
def get_generators(generator, train_index, test_index, graph_labels, batch_size):
    train_gen = generator.flow(train_index,
                               targets=graph_labels.iloc[train_index].values,
                               batch_size=batch_size)
    test_gen = generator.flow(test_index,
                              targets=graph_labels.iloc[test_index].values,
                              batch_size=batch_size)

    return train_gen, test_gen


def k_fold(graphs, graph_labels, n_splits, es, epochs, batch_size):
    test_accs = []
    kf = model_selection.KFold(n_splits, shuffle=True, random_state=999)
    generator = PaddedGraphGenerator(graphs=graphs)
    for train_index, test_index in kf.split(graphs, graph_labels):
        print("\n=>TRAIN:", train_index, "\nTEST:", test_index)
        train_gen, test_gen = get_generators(generator,
                                             train_index=train_index,
                                             test_index=test_index,
                                             graph_labels=graph_labels,
                                             batch_size=batch_size)
        model = create_graph_classification_model(generator)
        history, acc = train_fold(model, train_gen, test_gen, es, epochs)
        test_accs.append(acc)
        curr_mean = np.mean(test_accs) * 100
        curr_std = np.std(test_accs) * 100
        print(f"Accuracy over all folds mean: {curr_mean:.3}% and \
        std: {curr_std:.2}%")

    return test_accs
