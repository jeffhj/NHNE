import numpy as np
import random
from gensim.models import Word2Vec
from hypergraph import *


class Walker(object):
    def __init__(self, G, p, q, r):
        self.G = G
        self.p = p
        self.q = q
        self.r = r
        self.Pr = get_Pr(G)

    def hyper2vec_walk(self, walk_length, start_node):
        """
        Simulate a random walk starting from start node.
        """
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    walk.append(cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])])
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length):
        """
        Repeatedly simulate random walks from each node.
        """
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print(str(walk_iter + 1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.hyper2vec_walk(walk_length=walk_length, start_node=node))
        return walks

    def get_alias_node(self, dst):
        """
        Get the node setup lists for a given node.
        """
        G = self.G
        Pr = self.Pr

        dst_id = G.node_id(dst)
        unnormalized_probs = []

        for dst_nbr in G.neighbors(dst):
            beta_ = beta(G._nodes[dst_nbr]['degree'], self.r)
            dst_nbr_id = G.node_id(dst_nbr)
            unnormalized_probs.append(beta_ * Pr[dst_id, dst_nbr_id])

        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
        return alias_setup(normalized_probs)

    def get_alias_edge(self, src, dst):
        """
        Get the alias edge setup lists for a given edge.
        """
        G = self.G
        p = self.p
        q = self.q
        Pr = self.Pr

        src_id = G.node_id(src)
        dst_id = G.node_id(dst)
        unnormalized_probs = []

        for dst_nbr in G.neighbors(dst):
            beta_ = beta(G._nodes[dst_nbr]['degree'], self.r)
            dst_nbr_id = G.node_id(dst_nbr)
            if dst_nbr == src:
                unnormalized_probs.append(beta_ * Pr[dst_id, dst_nbr_id] / p)
            elif Pr[dst_nbr_id, src_id] > 0:
                unnormalized_probs.append(beta_ * Pr[dst_id, dst_nbr_id])
            else:
                unnormalized_probs.append(beta_ * Pr[dst_id, dst_nbr_id] / q)

        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding random walks.
        """
        G = self.G
        Pr = self.Pr
        nodes = G.nodes()

        alias_nodes = {}
        for node in nodes:
            alias_nodes[node] = self.get_alias_node(node)

        alias_edges = {}
        for v1 in G.nodes():
            for v2 in G.neighbors(v1):
                alias_edges[(v1, v2)] = self.get_alias_edge(v1, v2)

        self.alias_nodes = alias_nodes  # J, q
        self.alias_edges = alias_edges


def beta(dx, r):
    if r > 0:
        return dx + r
    elif r < 0:
        return 1 / (dx - r)
    else:
        return 1


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/ for details.
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return J, q


def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def learn_embeddings(walks, G, args):
    """
    Learn embeddings by the Skip-gram model.
    """
    walks = [list(map(str, walk)) for walk in walks]
    word2vec = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers,
                        iter=args.iter, negative=5)

    embs = {}
    for word in map(str, list(G.nodes())):
        embs[word] = word2vec[word]

    return embs


def convert_edgeemb_to_nodeemb(G, embs_edge, args):
    embs_dual = {}

    for node in G.nodes():
        cnt = 0
        emb = [0] * args.dimensions
        for e in G.incident_edges(node):
            cnt += 1
            e_emb = embs_edge[e]
            for i in range(args.dimensions):
                emb[i] += float(e_emb[i])

        emb = np.divide(emb, cnt)
        embs_dual[node] = emb

    return embs_dual


def hyper2vec(G, args):
    print('\n##### initializing hypergraph...')
    walker = Walker(G, args.p, args.q, args.r)

    print('\n##### preprocessing transition probs...')
    walker.preprocess_transition_probs()

    print('\n##### walking...')
    walks = walker.simulate_walks(args.num_walks, args.walk_length)

    print("\n##### embedding...")
    embs = learn_embeddings(walks, G, args)
    return embs
