import matplotlib.pyplot as plt
import stellargraph
from sklearn import model_selection
from stellargraph.mapper import PaddedGraphGenerator

from model import create_graph_classification_model
from datetime import datetime


def run(graphs, graph_labels, train_size, epochs, shuffle, layer1, layer2, adam):
    train_graphs, test_graphs = model_selection.train_test_split(
        graph_labels, train_size=train_size, test_size=None, stratify=graph_labels,
    )
    gen = PaddedGraphGenerator(graphs=graphs)

    train_gen = gen.flow(
        list(train_graphs.index - 1),
        targets=train_graphs.values,
        batch_size=50,
        symmetric_normalization=False,
    )

    test_gen = gen.flow(
        list(test_graphs.index - 1),
        targets=test_graphs.values,
        batch_size=1,
        symmetric_normalization=False,
    )

    generator = PaddedGraphGenerator(graphs=graphs)
    model = create_graph_classification_model(generator, layer1, layer2, adam)
    print("=> running model.fit ...")
    history = model.fit(
        train_gen, epochs=epochs, verbose=2, validation_data=test_gen, shuffle=shuffle,
    )
    # stellargraph.utils.plot_history(history)
    # plt_file = "./cfg_train_test_{}.png".format(datetime.now().strftime("%Y_%m_%d_%H_%M"))
    # plt.savefig(plt_file)

    test_metrics = model.evaluate(test_gen, verbose=2)
    test_acc = test_metrics[model.metrics_names.index("acc")]
    test_loss = test_metrics[model.metrics_names.index("loss")]
    #print("\nTest Set Metrics:")
    #for name, val in zip(model.metrics_names, test_metrics):
    #    print("\t{}: {:0.4f}".format(name, val))

    return history, test_acc, test_loss


def plot_run(history, train_size, epochs, shuffle, layer1, layer2, adam):
    plt_file = "./cfg_train_test_{}_ep_{}_shuffle_{}_layer_{}X{}_adam_{}_{}.png".format(
        train_size,
        epochs,
        shuffle,
        layer1,
        layer2,
        adam,
        datetime.now().strftime("%Y_%m_%d_%H_%M"))
    stellargraph.utils.plot_history(history)
    plt.savefig(plt_file)
    print(f"\n=> saved plot to {plt_file}")
