class Hypergraph(object):
    def __init__(self):
        self._edges = {}
        self._nodes = {}
        self.ID = 0
        self.node_ID = 0
        self.max_edge_degree = 0

    def nodes(self):
        return list(self._nodes.keys())

    def edges(self):
        es = []
        for e in self._edges.values():
            es.append(e['edge'])
        return es

    def add_edge(self, edge_name, edge, weight=1.):
        edge = tuple(sorted(edge))
        edge_dict = {}
        edge_dict['edge'] = edge
        edge_dict['weight'] = weight
        edge_dict['id'] = self.ID
        self.ID += 1
        self._edges[edge_name] = edge_dict
        self.max_edge_degree = max(len(edge), self.max_edge_degree)

        for v in edge:
            node_dict = self._nodes.get(v, {})
            edge_set = node_dict.get('edge', set())
            edge_set.add(edge_name)
            node_dict['edge'] = edge_set

            node_weight = node_dict.get('weight', 1.)
            node_dict['weight'] = node_weight

            node_degree = node_dict.get('degree', 0)
            node_degree += weight
            node_dict['degree'] = node_degree

            node_id = node_dict.get('id', -1)
            if node_id == -1:
                node_dict['id'] = self.node_ID
                self.node_ID += 1

            neighbors = node_dict.get('neighbors', set())
            for v0 in edge:
                if v0 != v:
                    neighbors.add(v0)
            node_dict['neighbors'] = neighbors

            self._nodes[v] = node_dict

    def neighbors(self, n):
        return self._nodes[n]['neighbors']

    def incident_edges(self, n):
        return self._nodes[n]['edge']

    def incident_nodes(self, e):
        return self._edges[e]['edge']

    def node_id(self, n):
        return self._nodes[n]['id']


def output_dual_hypergraph(G, fdual):
    f = open(fdual, 'w', encoding='utf8')
    for v1 in G.nodes():
        f.write(str(v1))
        for e in G.incident_edges(v1):
            f.write(' ' + e)
        f.write('\n')
    f.close()


def get_Pr(G):
    from scipy.sparse import lil_matrix
    n = len(G.nodes())
    P = lil_matrix((n, n))

    for v1 in G.nodes():
        for e in G.incident_edges(v1):
            for v2 in G.incident_nodes(e):
                if v1 != v2:
                    P[G.node_id(v1), G.node_id(v2)] += G._edges[e]['weight'] / (
                                G._nodes[v1]['degree'] * len(G.incident_nodes(e)))
    return P
