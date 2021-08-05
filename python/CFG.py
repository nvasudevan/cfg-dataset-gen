import pandas as pd
import numpy as np

from stellargraph.datasets.dataset_loader import DatasetLoader
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import GCNSupervisedGraphClassification
from stellargraph import StellarGraph

from sklearn import model_selection
from IPython.display import display, HTML

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime


class CFG(
    DatasetLoader,
    name="CFG",
    directory_name="CFG",
    url="http://localhost:8000/CFG.zip",
    url_archive_format="zip",
    expected_files=[
        "CFG_A.txt",
        "CFG_graph_indicator.txt",
        "CFG_node_labels.txt",
        "CFG_edge_labels.txt",
        "CFG_graph_labels.txt",
        "README.txt",
    ],
    description="Each graph represents a Context-Free Grammar and graph labels represent either a derivation, shifting of symbols or reduction of a rule."
                "The dataset includes 80 graphs with 97 nodes and 117 edges on average for each graph."
                "Graph nodes have a labels and each graph is labelled as belonging to 1 of 2 classes.",
    source="http://localhost:8000/CFG.zip",
):
    _edge_labels_as_weights = False
    _node_attributes = False

    def load(self):
        """
        Load this dataset into a list of StellarGraph objects with corresponding labels, downloading it if required.

        Note: Edges in CFG are labelled as one of 7 values: <eps>, a, c, r, s, u, and z indicated by integers
        0, 1, 2, 3 ,4 ,5 and 6 respectively. The edge labels are included in the  :class:`.StellarGraph` objects as edge weights in
        integer representation.

        Returns:
            A tuple that is a list of :class:`.StellarGraph` objects and a Pandas Series of labels one for each graph.
        """
        return _load_graph_kernel_dataset(self)


def _load_graph_kernel_dataset(dataset):
    dataset.download(ignore_cache=True)
    print("=> dataset downloaded.")

    def _load_from_txt_file(filename, names=None, dtype=None, index_increment=None):
        df = pd.read_csv(
            dataset._resolve_path(filename=f"{dataset.name}_{filename}.txt"),
            header=None,
            index_col=False,
            dtype=dtype,
            names=names,
        )
        # We optional increment the index by 1 because indexing, e.g. node IDs, for this dataset starts
        # at 1 whereas the Pandas DataFrame implicit index starts at 0 potentially causing confusion selecting
        # rows later on.
        if index_increment:
            df.index = df.index + index_increment
        return df

    # edge information:
    df_graph = _load_from_txt_file(filename="A", names=["source", "target"])
    print("=> processed A ...")

    if dataset._edge_labels_as_weights:
        # there's some edge labels, that can be used as edge weights
        df_edge_labels = _load_from_txt_file(
            filename="edge_labels", names=["weight"], dtype=int
        )
        df_graph = pd.concat([df_graph, df_edge_labels], axis=1)

    # node information:
    df_graph_ids = _load_from_txt_file(
        filename="graph_indicator", names=["graph_id"], index_increment=1
    )
    print("=> processed graph_indicator ...")

    # graph information:
    df_graph_labels = _load_from_txt_file(
        filename="graph_labels", dtype="category", names=["label"], index_increment=1
    )
    print("=> processed graph_labels ...")

    df_node_labels = _load_from_txt_file(
        filename="node_labels", dtype="category", index_increment=1
    )
    print("=> df_node_labels: \n", df_node_labels)
    # One-hot encode the node labels because these are used as node features in graph classification
    # tasks.
    df_node_features = pd.get_dummies(df_node_labels)
    print("=> processed node_labels ...")

    if dataset._node_attributes:
        # there's some actual node attributes
        df_node_attributes = _load_from_txt_file(
            filename="node_attributes", dtype=np.float32, index_increment=1
        )

        df_node_features = pd.concat([df_node_features, df_node_attributes], axis=1)

    # split the data into each of the graphs, based on the nodes in each one
    def graph_for_nodes(nodes):
        # each graph is disconnected, so the source is enough to identify the graph for an edge
        edges = df_graph[df_graph["source"].isin(nodes.index)]
        return StellarGraph(nodes, edges)

    groups = df_node_features.groupby(df_graph_ids["graph_id"])
    print("=> processed groups ...")
    graphs = [graph_for_nodes(nodes) for _, nodes in groups]
    print("=> processed graphs ...")

    return graphs, df_graph_labels["label"]


def create_graph_classification_model(gen):
    gc_model = GCNSupervisedGraphClassification(
        layer_sizes=[64, 64],
        activations=["relu", "relu"],
        generator=gen,
        dropout=0.5,
    )
    x_inp, x_out = gc_model.in_out_tensors()
    predictions = Dense(units=32, activation="relu")(x_out)
    predictions = Dense(units=16, activation="relu")(predictions)
    predictions = Dense(units=1, activation="sigmoid")(predictions)

    # create keras model and prepare for training
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(0.005), loss=binary_crossentropy, metrics=["acc"])

    return model


def train_fold(model, train_gen, test_gen, es, epochs):
    history = model.fit(train_gen,
                        epochs=epochs,
                        validation_data=test_gen,
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


def repeated_k_fold(generator, n_folds, n_repeats):
    test_accs = []
    stratified_folds = model_selection.RepeatedStratifiedKFold(
        n_splits=n_folds, n_repeats=n_repeats
    ).split(graph_labels, graph_labels)

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
        print(f"Accuracy over all folds mean: {np.mean(test_accs) * 100:.3}% and std: {np.std(test_accs) * 100:.2}%")

    return test_accs


def k_fold(n_splits, graphs, graph_labels):
    test_accs = []
    kf = model_selection.KFold(n_splits, shuffle=True, random_state=999)
    for train_index, test_index in kf.split(graphs, graph_labels):
        print("\n=>TRAIN:", train_index, "TEST:", test_index)
        train_gen, test_gen = get_generators(generator, train_index=train_index,
                                             test_index=test_index,
                                             graph_labels=graph_labels,
                                             batch_size=10)
        model = create_graph_classification_model(generator)
        history, acc = train_fold(model, train_gen, test_gen, es, epochs)
        test_accs.append(acc)
        print(f"Accuracy over all folds mean: {np.mean(test_accs) * 100:.3}% and std: {np.std(test_accs) * 100:.2}%")

    return test_accs


def plot_it(test_accs, plt_file):
    plt.figure(figsize=(8, 6))
    plt.hist(test_accs)
    plt.xlabel("Accuracy")
    plt.ylabel("Count")
    plt_name = "./cfg_run.png"
    plt.savefig(plt_name)
    print(f"\n=> saved plot to {plt_name}")


if __name__ == "__main__":
    print("=> running CFG ...")
    dataset = CFG()
    print("=> dataset: ", dataset)
    graphs, graph_labels = dataset.load()
    print("labels: ", graph_labels)

    summary = pd.DataFrame(
        [(g.number_of_nodes(), g.number_of_edges()) for g in graphs],
        columns=["nodes", "edges"],
    )
    print(summary.describe().round(1))
    print("\n", graph_labels.value_counts().to_frame())
    graph_labels = pd.get_dummies(graph_labels, drop_first=True)
    print("labels: ", graph_labels)
    generator = PaddedGraphGenerator(graphs=graphs)

    # configure training parameters
    epochs = 200
    n_repeats = 5
    n_folds = 10
    es = EarlyStopping(monitor="val_loss",
                       min_delta=0,
                       patience=25,
                       restore_best_weights=True)
    print(f"=> len: graphs: {len(graphs)}, graph_labels: {len(graph_labels)}")
    test_accs = k_fold(20, graphs, graph_labels)
    plot_file = "./cfg_{}.png".format(datetime.now().strftime("%Y_%m_%d_%H_%M"))
    plot_it(test_accs, plot_file)
