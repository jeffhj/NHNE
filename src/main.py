import argparse
from hyper2vec import *
from hypergraph import *
from nhne import *
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', nargs='?',
                        help='Input graph path')
    parser.add_argument('--save-model', nargs='?',
                        help='output model path')
    parser.add_argument('--dimensions', type=int, default=32,
                        help='Number of dimensions. Default is 32.')
    parser.add_argument('--walk-length', type=int, default=20,
                        help='Length of walk per source. Default is 20.')
    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')
    parser.add_argument('--window-size', type=int, default=5,
                        help='Context size for optimization. Default is 5.')
    parser.add_argument('--iter', default=20, type=int,
                        help='Number of epochs in Skipgram. Default is 20.')
    parser.add_argument('--epochs', default=20, type=int,
                        help='Number of epochs in RMSprop. Default is 20.')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')
    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')
    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')
    parser.add_argument('--r', type=float, default=0,
                        help='r. Default is 0.')
    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    return parser.parse_args()


def read_graph(filename):
    """
    Read the input hypergraph.
    """
    G = Hypergraph()

    f = open(filename, 'r', encoding='utf8')
    lines = f.readlines()
    if args.weighted:
        for line in lines:
            line = line.split()
            edge_name = line[0]
            weight = line[1]
            G.add_edge(edge_name, line[2:], float(weight))
    else:
        for line in lines:
            line = line.split()
            edge_name = line[0]
            G.add_edge(edge_name, line[1:])
    f.close()
    return G


def main(args):
    dataset = args.input.split('/')[1]  # extract dataset name
    print("Dataset:", dataset)

    print('\n##### reading hypergraph...')
    G = read_graph(args.input)
    embs = hyper2vec(G, args)

    fdual = "graph/" + dataset + "/dual_edgelist.txt"
    if not os.path.exists(fdual):
        print("\n##### generating dual hypergraph...")
        output_dual_hypergraph(G, fdual)

    # For simplification here. Training separately using hyper2vec is recommended.
    Gd = read_graph(fdual)
    embs_edge = hyper2vec(Gd, args)
    embs_dual = convert_edgeemb_to_nodeemb(G, embs_edge, args)

    print("\n##### training neural networks...")
    model = NHNE(G, embs, embs_dual, args)

    if args.save_model is None:
        dirs = f"model/{dataset}"
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        args.save_model = os.path.join(dirs, f"model_p{args.p}_q{args.q}_r{args.r}_e{args.epochs}.h5")
    model.save(args.save_model)


if __name__ == "__main__":
    args = parse_args()
    main(args)
