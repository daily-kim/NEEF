import random
import sys

import numpy as np
import torch
import dgl
from dgl.sampling import global_uniform_negative_sampling

import model as M
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f"Run {run + 1:02d}:")
            print(f"Highest Train: {result[:, 0].max():.2f}")
            print(f"Highest Valid: {result[:, 1].max():.2f}")
            print(f"  Final Train: {result[argmax, 0]:.2f}")
            print(f"   Final Test: {result[argmax, 2]:.2f}")
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f"All runs:")
            r = best_result[:, 0]
            print(f"Highest Train: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 1]
            print(f"Highest Valid: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 2]
            print(f"  Final Train: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 3]
            print(f"   Final Test: {r.mean():.2f} ± {r.std():.2f}")


def core_number_sparse(sparse_matrix):
    degrees = np.asarray(sparse_matrix.sum(axis=1)).squeeze()
    nodes = np.argsort(degrees)
    bin_boundaries = [0]
    curr_degree = 0

    for i, v in enumerate(nodes):
        if degrees[v] > curr_degree:
            bin_boundaries.extend([i] * (degrees[v] - curr_degree))
            curr_degree = degrees[v]

    node_pos = {v: pos for pos, v in enumerate(nodes)}
    core = degrees.copy()

    for v in nodes:
        neighbors = sparse_matrix[v].indices
        for u in neighbors:
            if core[u] > core[v]:
                pos = node_pos[u]
                bin_start = bin_boundaries[core[u]]
                node_pos[u] = bin_start
                node_pos[nodes[bin_start]] = pos
                nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
                bin_boundaries[core[u]] += 1
                core[u] -= 1

    return dict(zip(range(len(core)), core))



def get_core_num(g, split_edge):
    # valid, test edges are only used for getting nodes, not for training
    
    train_edge = split_edge['train']['edge']
    adj =  g.adj_external(scipy_fmt='csr')

    csr_train = csr_matrix((np.ones(len(train_edge)), (train_edge[:, 0], train_edge[:, 1])), shape=adj.shape, dtype=np.int64) 
    core_num = core_number_sparse(csr_train)
    return core_num


def get_edge_core(g, split_edge):

    def get_ecore(edges):
        e_core = []
        for i in range(len(edges[0])):
            e_core.append(
                min(core_num[int(edges[0][i])], core_num[int(edges[1][i])]))
        return torch.tensor(e_core).t()
    
    def get_is_warm(edges):
        e_is_warm = []
        for i in range(len(edges[0])):
            src = int(edges[0][i])
            dst = int(edges[1][i])
            src_core = core_num[src]
            dst_core = core_num[dst]
            if src_core >= 5 and dst_core >= 5: # warm-warm
                e_is_warm.append(2)
            elif src_core < 5 and dst_core < 5:# cold-cold
                e_is_warm.append(0)
            else:
                e_is_warm.append(1) # cold-warm

        return torch.tensor(e_is_warm).t()
    
    core_num = get_core_num(g,split_edge)
    g.edata['core'] = get_ecore(g.edges())
    g.edata['is_warm'] = get_is_warm(g.edges())

    g.ndata['core_num'] = torch.tensor(list(core_num.values()))

    return g, split_edge


def map_edges_to_subgraph(original_edges, new_g):
    mapper = {int(new_g.ndata['_ID'][i]): i for i in range(new_g.num_nodes())}
    newrow = []
    newcol = []
    row, col = original_edges.t()
    for i in range(len(row)):
        try:
            newrow.append(mapper[int(row[i])])
            newcol.append(mapper[int(col[i])])
        except Exception:
            continue
    return torch.tensor([newrow, newcol]).t()


def get_warmup_graph(args, g, split_edge):
    min_core_num = args.warmup_core
    core_num = get_core_num(g,split_edge)

    warm_node_ids = torch.tensor(
        [i for i in core_num if core_num[i] >= min_core_num])
    raw_train_edge = split_edge["train"]["edge"]

    row, col = raw_train_edge.t()
    pos_train_edge = raw_train_edge[torch.isin(
        row, warm_node_ids) & torch.isin(col, warm_node_ids), :]

    # get edge ids which edata['train_mask'] is True
    if 'train_mask'  in g.edata: # for collab dataset
        edges_no_val = g.edata['_ID'][g.edata['train_mask'].squeeze()]
        g_w = dgl.edge_subgraph(g, edges_no_val)
    else:
        g_w = g
    # get node ids which ndata['core_num']is more than k
    g_w = dgl.node_subgraph(g_w, warm_node_ids.tolist())
    # map edge ids to subgraph
    pos_train_edge = map_edges_to_subgraph(pos_train_edge, g_w)
    return g_w, pos_train_edge


def model_selector(model_name):
    model = None
    
    if model_name == "GCN":
        model = getattr(M, "GCN")  # Replace "GCN_function_name" with the actual function name for GCN model
    elif model_name == "SAGE":
        model = getattr(M, "SAGE")  # Replace "SAGE_function_name" with the actual function name for SAGE model
    
    # Add more elif conditions for other model names
    else:
        print("Invalid model name!")
    
    return model



def k_hop_subgraph(src, dst, num_hops, g, sample_ratio=1.0, directed=False):
    # Extract the k-hop enclosing subgraph around link (src, dst) from g
    nodes = [src, dst]
    visited = set([src, dst])
    fringe = set([src, dst])
    for _ in range(num_hops):
        if not directed:
            _, fringe = g.out_edges(list(fringe))
            fringe = fringe.tolist()
        else:
            _, out_neighbors = g.out_edges(list(fringe))
            in_neighbors, _ = g.in_edges(list(fringe))
            fringe = in_neighbors.tolist() + out_neighbors.tolist()
        fringe = set(fringe) - visited
        visited = visited.union(fringe)

        if sample_ratio < 1.0:
            fringe = random.sample(fringe, int(sample_ratio * len(fringe)))
        if len(fringe) == 0:
            break

        nodes = nodes + list(fringe)

    subg = g.subgraph(nodes, store_ids=True)

    return subg


def drnl_node_labeling(adj, src, dst):
    # Double Radius Node Labeling (DRNL).
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(
        adj_wo_dst, directed=False, unweighted=True, indices=src
    )
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(
        adj_wo_src, directed=False, unweighted=True, indices=dst - 1
    )
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = (
        torch.div(dist, 2, rounding_mode="floor"),
        dist % 2,
    )

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.0
    z[dst] = 1.0
    # shortest path may include inf values
    z[torch.isnan(z)] = 0.0

    return z.to(torch.long)


def get_pos_neg_edges(split, split_edge, g, percent=100):
    pos_edge = split_edge[split]["edge"]
    if split == "train":
        neg_edge = torch.stack(
            global_uniform_negative_sampling(
                g, num_samples=pos_edge.size(0), exclude_self_loops=True
            ),
            dim=1,
        )
    else:
        neg_edge = split_edge[split]["edge_neg"]

    # sampling according to the percent param
    np.random.seed(123)
    # pos sampling
    num_pos = pos_edge.size(0)
    perm = np.random.permutation(num_pos)
    perm = perm[: int(percent / 100 * num_pos)]
    pos_edge = pos_edge[perm]
    # neg sampling
    if neg_edge.dim() > 2:  # [Np, Nn, 2]
        neg_edge = neg_edge[perm].view(-1, 2)
    else:
        np.random.seed(123)
        num_neg = neg_edge.size(0)
        perm = np.random.permutation(num_neg)
        perm = perm[: int(percent / 100 * num_neg)]
        neg_edge = neg_edge[perm]

    return pos_edge, neg_edge  # ([2, Np], [2, Nn]) -> ([Np, 2], [Nn, 2])


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = {
            "valid": [[] for _ in range(runs)],
            "test": [[] for _ in range(runs)],
        }

    def add_result(self, run, result, split="valid"):
        assert run >= 0 and run < len(self.results["valid"])
        assert split in ["valid", "test"]
        self.results[split][run].append(result)

    def print_statistics(self, run=None, f=sys.stdout):
        if run is not None:
            result = torch.tensor(self.results["valid"][run])
            print(f"Run {run + 1:02d}:", file=f)
            print(f"Highest Valid: {result.max():.4f}", file=f)
            print(f"Highest Eval Point: {result.argmax().item()+1}", file=f)
            if not self.info.no_test:
                print(
                    f'   Final Test Point[1]: {self.results["test"][run][0][0]}',
                    f'   Final Valid: {self.results["test"][run][0][1]}',
                    f'   Final Test: {self.results["test"][run][0][2]}',
                    sep="\n",
                    file=f,
                )
        else:
            best_result = torch.tensor(
                [test_res[0] for test_res in self.results["test"]]
            )

            print(f"All runs:", file=f)
            r = best_result[:, 1]
            print(f"Highest Valid: {r.mean():.4f} ± {r.std():.4f}", file=f)
            if not self.info.no_test:
                r = best_result[:, 2]
                print(f"   Final Test: {r.mean():.4f} ± {r.std():.4f}", file=f)

if __name__ == '__main__': 
    ModelBase = model_selector("GCN")
