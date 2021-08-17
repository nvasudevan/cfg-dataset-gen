import pandas as pd
import numpy as np

from stellargraph.datasets.dataset_loader import DatasetLoader
from stellargraph.core.graph import StellarGraph

class MUTAG(
    DatasetLoader,
    name="MUTAG",
    directory_name="MUTAG",
    url="http://localhost:8000/MUTAG.zip",
    url_archive_format="zip",
    expected_files=[
        "MUTAG_A.txt",
        "MUTAG_graph_indicator.txt",
        "MUTAG_node_labels.txt",
        "MUTAG_edge_labels.txt",
        "MUTAG_graph_labels.txt",
        "README.txt",
    ],
    description="Each graph represents a Context-Free Grammar and graph labels represent either a derivation, shifting of symbols or reduction of a rule."
    "The dataset includes 80 graphs with 97 nodes and 117 edges on average for each graph."
    "Graph nodes have a labels and each graph is labelled as belonging to 1 of 2 classes.",
    source="http://localhost:8000/MUTAG.zip",
):
    _edge_labels_as_weights = False
    _node_attributes = False

    def load(self):
        """
        Load this dataset into a list of StellarGraph objects with corresponding labels, downloading it if required.

        Note: Edges in MUTAG are labelled as one of 7 values: <eps>, a, c, r, s, u, and z indicated by integers
        0, 1, 2, 3 ,4 ,5 and 6 respectively. The edge labels are included in the  :class:`.StellarGraph` objects as edge weights in
        integer representation.

        Returns:
            A tuple that is a list of :class:`.StellarGraph` objects and a Pandas Series of labels one for each graph.
        """
        return _load_graph_kernel_dataset(self)


def _load_graph_kernel_dataset(dataset):

    dataset.download()
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

    df_node_labels = _load_from_txt_file(
        filename="node_labels", dtype="category", index_increment=1
    )
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

    # graph information:
    df_graph_labels = _load_from_txt_file(
        filename="graph_labels", dtype="category", names=["label"], index_increment=1
    )
    print("=> processed graph_labels ...")

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


if __name__ == "__main__":
    print("=> running MUTAG ...")
    dataset = MUTAG()
    print("=> dataset: ", dataset)
    graphs, graph_labels = dataset.load()
    print("=> retrieved dataset: ", graphs)

    summary = pd.DataFrame(
        [(g.number_of_nodes(), g.number_of_edges()) for g in graphs],
        columns=["nodes", "edges"],
    )
    print(summary.describe().round(1))
