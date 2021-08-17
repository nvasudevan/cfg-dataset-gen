from stellargraph.layer import GCNSupervisedGraphClassification
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam


def create_graph_classification_model(gen, layer1, layer2, adam):
    gc_model = GCNSupervisedGraphClassification(
        layer_sizes=[layer1, layer2],
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
    model.compile(optimizer=Adam(adam), loss=binary_crossentropy, metrics=["acc"])

    return model

