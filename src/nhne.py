import numpy as np
import random

from keras.models import Model
from keras import optimizers
from keras import layers


def negative_sample(G, edge, edge_set, num_neg_samples=1):
    n_neg = 0
    neg_data = []
    while n_neg < num_neg_samples:
        neg_edge = tuple(sorted(random.choices(G.nodes(), k=len(edge))))

        if neg_edge in edge_set:
            continue
        n_neg += 1
        neg_data.append(neg_edge)
    return neg_data


def NHNE(G, embeddings, dual_embeddings, args):
    edges = G.edges()
    k = G.max_edge_degree
    edge_set = set(edges)
    n = len(G.nodes())

    embedding_matrix = np.zeros((n, args.dimensions))
    for _id, vect in embeddings.items():
        embedding_vector = embeddings.get(_id)
        if embedding_vector is not None:
            embedding_matrix[G.node_id(_id)] = embedding_vector

    dual_embedding_matrix = np.zeros((n, args.dimensions))
    for _id, vect in dual_embeddings.items():
        embedding_vector = dual_embeddings.get(_id)
        if embedding_vector is not None:
            dual_embedding_matrix[G.node_id(_id)] = embedding_vector

    inputs = layers.Input(shape=(k,), name='input', dtype='int32')
    EM = [embedding_matrix, dual_embedding_matrix]
    emb_layer = [layers.Embedding(n, args.dimensions, input_length=k, weights=[EM[i]],
                                  name='emb_{}'.format(i))(inputs) for i in range(2)]
    conv_layer = [layers.Conv1D(32, 3, activation='relu',
                                name='conv_{}'.format(i))(emb_layer[i]) for i in range(2)]
    pooling_layer = [layers.GlobalMaxPooling1D(
        name='pooling_{}'.format(i))(conv_layer[i]) for i in range(2)]
    merged = layers.concatenate(pooling_layer, axis=1)
    output_layer = layers.Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=inputs, outputs=output_layer)

    model.compile(optimizer=optimizers.RMSprop(lr=5e-4),
                  loss='binary_crossentropy', metrics=['acc'])

    X, y = [], []
    for edge in edges:
        _x = []
        j = 0
        for i in range(k - len(edge)):
            _x.append(0)
            j += 1
        for i, node in enumerate(edge):
            _x.append(G.node_id(node))
            j += 1
        X.append(_x)
        y.append(1)

        num_neg_samples = 15
        negs = negative_sample(G, edge, edge_set, num_neg_samples=num_neg_samples)
        for neg_edge in negs:
            _x = []
            j = 0
            for i in range(k - len(edge)):
                _x.append(0)
                j += 1
            for i, node in enumerate(neg_edge):
                _x.append(G.node_id(node))
                j += 1
            X.append(_x)
        for i in range(num_neg_samples):
            y.append(0)

    model.fit(np.array(X), y, epochs=args.epochs, batch_size=128)

    return model
