import dgl
import torch
from dgl.dataloading.negative_sampler import GlobalUniform
from torch.utils.data import DataLoader

def evaluate_hits(evaluator, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval(
            {
                "y_pred_pos": pos_train_pred,
                "y_pred_neg": neg_val_pred,
            }
        )[f"hits@{K}"]
        valid_hits = evaluator.eval(
            {
                "y_pred_pos": pos_val_pred,
                "y_pred_neg": neg_val_pred,
            }
        )[f"hits@{K}"]
        test_hits = evaluator.eval(
            {
                "y_pred_pos": pos_test_pred,
                "y_pred_neg": neg_test_pred,
            }
        )[f"hits@{K}"]

        results[f"Hits@{K}"] = (train_hits, valid_hits, test_hits)

    return results


def evaluate_mrr(evaluator, pos_train_pred,pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    print(
        pos_val_pred.size(),
        neg_val_pred.size(),
        pos_test_pred.size(),
        neg_test_pred.size(),
    )
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    results = {}
    train_mrr = (
        evaluator.eval(
            {
                "y_pred_pos": pos_train_pred,
                "y_pred_neg": neg_val_pred,
            }
        )["mrr_list"]
        .mean()
        .item()
    )

    valid_mrr = (
        evaluator.eval(
            {
                "y_pred_pos": pos_val_pred,
                "y_pred_neg": neg_val_pred,
            }
        )["mrr_list"]
        .mean()
        .item()
    )

    test_mrr = (
        evaluator.eval(
            {
                "y_pred_pos": pos_test_pred,
                "y_pred_neg": neg_test_pred,
            }
        )["mrr_list"]
        .mean()
        .item()
    )

    results["MRR"] = (train_mrr, valid_mrr, test_mrr)

    return results

def warmer(model, predictor, g, optimizer, batch_size, pos_train_edge):
    model.train()
    predictor.train()

    x = g.ndata['feat']

    neg_sampler = GlobalUniform(1)
    total_loss = total_examples = 0
    for perm in DataLoader(
        range(pos_train_edge.size(0)), batch_size, shuffle=True
    ):
        optimizer.zero_grad()

        h = model(g, x)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = neg_sampler(g, edge[0])

        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        if model.dataset == "ogbl-ddi":
            torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


def train(model, predictor, g, x, split_edge, optimizer, args):
    model.train()
    predictor.train()
    if args.use_valedges_as_input: # get edge ids which edata['train_mask'] is True
        train_eids = g.edata['_ID'][g.edata['train_mask'].squeeze()]
        g = dgl.edge_subgraph(g, train_eids)
    
    pos_train_edge = split_edge["train"]["edge"].to(x.device)
    neg_sampler = GlobalUniform(1)
    total_loss = total_examples = 0

    for perm in DataLoader(
        range(pos_train_edge.size(0)), args.batch_size, shuffle=True
    ):
        optimizer.zero_grad()

        h = model(g, x)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = neg_sampler(g, edge[0])

        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        if model.dataset == "ogbl-ddi":
            torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, g, x, split_edge, evaluator, args):

    def get_pred(test_edges, h):
        preds = []
        for perm in DataLoader(range(test_edges.size(0)), args.batch_size):
            edge = test_edges[perm].t()
            preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pred = torch.cat(preds, dim=0)
        return pred
    
 

    model.eval()
    predictor.eval()
    if args.use_valedges_as_input:
        train_eids = g.edata['_ID'][g.edata['train_mask'].squeeze()]
        g_train_only = dgl.edge_subgraph(g, train_eids)

        h_train_only = model(g_train_only, x)

        h_full = model(g, x)

    else:
        h = model(g, x)
        h_full = h
        h_train_only = h


    pos_train_edge = split_edge["eval_train"]["edge"].to(args.device)
    pos_valid_edge = split_edge["valid"]["edge"].to(args.device)
    neg_valid_edge = split_edge["valid"]["edge_neg"].to(args.device)
    pos_test_edge = split_edge["test"]["edge"].to(args.device)
    neg_test_edge = split_edge["test"]["edge_neg"].to(args.device)

    pos_train_pred = get_pred(pos_train_edge, h_train_only)
    pos_valid_pred = get_pred(pos_valid_edge, h_train_only)
    neg_valid_pred = get_pred(neg_valid_edge, h_train_only)
    pos_test_pred = get_pred(pos_test_edge, h_full)
    neg_test_pred = get_pred(neg_test_edge, h_full)


    if evaluator.name == 'citation2':
        results = evaluate_mrr(evaluator, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    else:
        results = evaluate_hits(evaluator, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    
    return results

@torch.no_grad()
def test_cold_seperate(model, predictor, g, x, split_edge, evaluator, args):

    def get_pred(test_edges, h):
        preds = []
        for perm in DataLoader(range(test_edges.size(0)), args.batch_size):
            edge = test_edges[perm].t()
            preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pred = torch.cat(preds, dim=0)
        return pred
    
    def seperate_pred(prediction,edges):
        ncore = g.ndata['core_num']
        cold_edge = []
        medium_edge = []
        warm_edge = []
        for idx, edge in enumerate(edges):
            if ncore[edge[0]] >= args.warmup_core and ncore[edge[1]] >= args.warmup_core:
                warm_edge.append(prediction[idx])
            elif ncore[edge[0]] <args.warmup_core and ncore[edge[1]] < args.warmup_core:
                cold_edge.append(prediction[idx])
            else:
                medium_edge.append(prediction[idx])
        return torch.tensor(cold_edge), torch.tensor(medium_edge), torch.tensor(warm_edge)
        

    model.eval()
    predictor.eval()
    if args.use_valedges_as_input:
        train_eids = g.edata['_ID'][g.edata['train_mask'].squeeze()]
        g_train_only = dgl.edge_subgraph(g, train_eids)

        h_train_only = model(g_train_only, x)

        h_full = model(g, x)

    else:
        h = model(g, x)
        h_full = h
        h_train_only = h


    pos_train_edge = split_edge["eval_train"]["edge"].to(args.device)
    pos_valid_edge = split_edge["valid"]["edge"].to(args.device)
    neg_valid_edge = split_edge["valid"]["edge_neg"].to(args.device)
    pos_test_edge = split_edge["test"]["edge"].to(args.device)
    neg_test_edge = split_edge["test"]["edge_neg"].to(args.device)

    pos_train_pred = get_pred(pos_train_edge, h_train_only)
    pos_valid_pred = get_pred(pos_valid_edge, h_train_only)
    neg_valid_pred = get_pred(neg_valid_edge, h_train_only)
    pos_test_pred = get_pred(pos_test_edge, h_full)
    neg_test_pred = get_pred(neg_test_edge, h_full)


    cold_pos_train_pred, medium_pos_train_pred, warm_pos_train_pred = seperate_pred(pos_train_pred, pos_train_edge)
    cold_pos_valid_pred, medium_pos_valid_pred, warm_pos_valid_pred = seperate_pred(pos_valid_pred, pos_valid_edge)
    cold_neg_valid_pred, medium_neg_valid_pred, warm_neg_valid_pred = seperate_pred(neg_valid_pred, neg_valid_edge)
    cold_pos_test_pred, medium_pos_test_pred, warm_pos_test_pred = seperate_pred(pos_test_pred, pos_test_edge)
    cold_neg_test_pred, medium_neg_test_pred, warm_neg_test_pred = seperate_pred(neg_test_pred, neg_test_edge)

    cold_results = evaluate_hits(evaluator, cold_pos_train_pred, cold_pos_valid_pred, cold_neg_valid_pred, cold_pos_test_pred, cold_neg_test_pred)
    medium_results = evaluate_hits(evaluator, medium_pos_train_pred, medium_pos_valid_pred, medium_neg_valid_pred, medium_pos_test_pred, medium_neg_test_pred)
    warm_results = evaluate_hits(evaluator, warm_pos_train_pred, warm_pos_valid_pred, warm_neg_valid_pred, warm_pos_test_pred, warm_neg_test_pred)
    
    results_sep = {'cold': cold_results, 'medium': medium_results, 'warm': warm_results}

    return results_sep
