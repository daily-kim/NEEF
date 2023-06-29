import argparse
import dgl

import torch
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from model import LinkPredictor
from utils import Logger,get_edge_core, get_warmup_graph,  model_selector
from train import warmer, train, test, test_cold_seperate


def main():
    parser = argparse.ArgumentParser(
        description="OGBL(Full Batch GCN/GraphSage + warmer)"
    )
    # dataset setting
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbl-collab",
        choices=["ogbl-ddi", "ogbl-collab"],
    )
    # device setting
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device ID. Use -1 for CPU training.",
    )
    # model structure settings
    parser.add_argument(
        "--model", type=str, default='SAGE', choices=['GCN', 'SAGE']
    )
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=64 * 1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=400)

    # training settings
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--runs", type=int, default=5)

    parser.add_argument("--use_warmer", action="store_true")
    parser.add_argument("--warmup_core", type=int, default=5)
    parser.add_argument("--warmup_epochs", type=int, default=200)

    parser.add_argument("--use_pretrain", action="store_true")

    # 1 for basic, 2 for seperate
    # parser.add_argument("--warmup_type", type=int, default=2)
    parser.add_argument("--warmup_channels", type=int, default=0)
    parser.add_argument("--warmup_model", type=str,
                        default='SAGE', choices=['GCN', 'SAGE'])
    parser.add_argument("--eval_seperate", action="store_true")

    args = parser.parse_args()
    print(args)

    device = (
        f"cuda:{args.device}"
        if args.device != -1 and torch.cuda.is_available()
        else "cpu"
    )
    print(device)

    dataset = DglLinkPropPredDataset(name=args.dataset)

    g = dataset[0]  # g only has training edge
    split_edge = dataset.get_edge_split()

    # re-format the data of citation2
    if args.dataset == "ogbl-citation2":
        for k in ["train", "valid", "test"]:
            src = split_edge[k]["source_node"]
            tgt = split_edge[k]["target_node"]
            split_edge[k]["edge"] = torch.stack([src, tgt], dim=1)
            if k != "train":
                tgt_neg = split_edge[k]["target_node_neg"]
                split_edge[k]["edge_neg"] = torch.stack(
                    [src[:, None].repeat(1, tgt_neg.size(1)), tgt_neg], dim=-1
                )  # [Ns, Nt, 2]
    
    g, split_edge = get_edge_core(g, split_edge)

    # We randomly pick some training samples that we want to evaluate on:
    idx = torch.randperm(split_edge["train"]["edge"].size(0))
    idx = idx[: split_edge["valid"]["edge"].size(0)]
    split_edge["eval_train"] = {"edge": split_edge["train"]["edge"][idx]}

    

    if dataset.name == "ogbl-ppa":
        g.ndata["feat"] = g.ndata["feat"].to(torch.float)
    if dataset.name == "ogbl-ddi":  # ddi dataset doesn't have node features
        emb = torch.nn.Embedding(
            g.num_nodes(), args.hidden_channels)
        in_channels = args.hidden_channels
        torch.nn.init.xavier_uniform_(emb.weight)
        g.ndata["feat"] = emb.weight

    else:  # ogbl-collab, ogbl-ppa
        in_channels = g.ndata["feat"].size(-1)

    if args.dataset == "ogbl-collab": # use validation edges as input
        g.edata['train_mask'] = torch.ones(size=(g.num_edges(), 1), dtype=torch.bool)

        val_edges = split_edge["valid"]["edge"]
        row, col = val_edges.t()
        val_weights = torch.ones(
            size=(2 * val_edges.size(0), 1), dtype=torch.int64)  # dummy weight
        train_mask = torch.zeros(
            size=(2 * val_edges.size(0), 1), dtype=torch.bool)
        g_include_valid = dgl.add_edges(g,
                                        torch.cat([row, col]),
                                        torch.cat([col, row]),
                                        {"weight": val_weights,
                                            "train_mask": train_mask},
                                        )
        g = g_include_valid.to_simple(copy_ndata=True, copy_edata=True)
    

    if args.use_warmer and not(args.use_pretrain):
        if args.warmup_model == 'GCN':
            g = dgl.add_self_loop(g) #for GCN
            g.edata['_ID'] = torch.arange(g.num_edges())
        
        ModelBase_w = model_selector(args.warmup_model)# select model by argument
        model_w = ModelBase_w(
            in_channels,
            args.hidden_channels,
            args.warmup_channels,
            args.num_layers,
            args.dropout,
            dataset.name,)
        predictor_w = LinkPredictor(
            args.warmup_channels, args.warmup_channels, 1, args.num_layers, args.dropout
        )

        model_w, predictor_w = map(
            lambda x: x.to(device), (model_w, predictor_w))
        # print(model_w)
    g.edata['_ID'] = torch.arange(g.num_edges())
    ModelBase= model_selector(args.model)# select model by argument
    model = ModelBase(
            in_channels + args.warmup_channels,
            args.hidden_channels,
            args.hidden_channels,
            args.num_layers,
            args.dropout,
            dataset.name,
        )
    if args.model == 'GCN':  # for GCN, diffent with warm-up model's one
        g = dgl.add_self_loop(g)
        g.edata['_ID'] = torch.arange(g.num_edges())

    predictor = LinkPredictor(
        args.hidden_channels, args.hidden_channels, 1, args.num_layers, args.dropout
    )
    g, model, predictor = map(lambda x: x.to(device), (g, model, predictor))

    if args.dataset.startswith("ogbl-citation"):
        args.eval_metric = "mrr"
    else:
        args.eval_metric = "hits"

    evaluator = Evaluator(name=args.dataset)
    if args.eval_metric == "hits":
        loggers = {
            "Hits@20": Logger(args.runs, args),
            "Hits@50": Logger(args.runs, args),
            "Hits@100": Logger(args.runs, args),
        }
    elif args.eval_metric == "mrr":
        loggers = {
            "MRR": Logger(args.runs, args),
        }
    
    if args.eval_seperate:
        sep_loggers = {
            is_warm: {
                "Hits@20": Logger(args.runs, args),
                "Hits@50": Logger(args.runs, args),
                "Hits@100": Logger(args.runs, args),
            } for is_warm in ['cold', 'medium','warm']}
    if args.use_warmer:
        if args.use_pretrain:
            dataset_path = f'dataset/{args.dataset}'.replace("-", "_")

            node_attribute = torch.load(f'{dataset_path}/warmup_embedding_{args.warmup_core}.pt', map_location = device)
            g.ndata['feat'] = node_attribute
            print('load pretrain embedding')
        else:
            print('Warm-up start')
            g.ndata['feat'] = g.ndata['feat'][:, :in_channels].detach()
            g_w, pos_train_edge = get_warmup_graph(args, g, split_edge)
            model_w.reset_parameters()
            predictor_w.reset_parameters()
            optimizer_w = torch.optim.Adam(
                list(model_w.parameters())
                + list(predictor_w.parameters()),
                lr=args.lr,
            )
            for pre_epoch in range(1, 1+args.warmup_epochs):
                warmup_loss = warmer(
                    model_w,
                    predictor_w,
                    g_w,
                    optimizer_w,
                    args.batch_size,
                    pos_train_edge
                )
                if pre_epoch % 10 == 0:
                    print(
                        f"[v2]pre_epoch: {pre_epoch:02d}, Loss: {warmup_loss:.4f}")

            emb_w = model_w(g_w, g_w.ndata['feat'])

            if args.dataset == 'ogbl-ddi':# just use warmup embedding
                emb_rand = torch.nn.Embedding(
                    g.ndata['feat'].shape[0], args.warmup_channels).to(device)
                torch.nn.init.xavier_uniform_(emb_rand.weight)
                g.ndata["feat"] = emb_rand.weight
                g.ndata['feat'][g_w.ndata['_ID']] = emb_w
            else:# concat warmup embedding
                emb_rand = torch.nn.Embedding(
                    g.ndata['feat'].shape[0], args.warmup_channels).to(device)
                torch.nn.init.xavier_uniform_(emb_rand.weight)
                g.ndata['feat'] = torch.cat(
                    [g.ndata['feat'], emb_rand.weight], dim=1)
                # replace warmup embedding
                g.ndata['feat'][:, -
                                args.warmup_channels:][g_w.ndata['_ID']] = emb_w
            g.ndata['feat'] = g.ndata['feat'].detach()

            if args.model == 'SAGE' and args.warmup_model == 'GCN':
                g = dgl.remove_self_loop(g)

    
    for run in range(args.runs):
        # print(g.ndata['feat'].shape)
        model.reset_parameters()
        predictor.reset_parameters()
        if dataset.name == "ogbl-ddi":
            emb = emb.to(device)
            torch.nn.init.xavier_uniform_(emb.weight)
            g.ndata["feat"] = emb.weight
        optimizer = torch.optim.Adam(
            list(model.parameters())
            + list(predictor.parameters())
            + (list(emb.parameters()) if dataset.name == "ogbl-ddi" else []),
            lr=args.lr,
        )
        
        for epoch in range(1, 1 + args.epochs):

            loss = train(
                model,
                predictor,
                g,
                g.ndata["feat"],
                split_edge,
                optimizer,
                args,
            )

            if epoch % args.eval_steps == 0:
                results = test(
                    model,
                    predictor,
                    g,
                    g.ndata["feat"],
                    split_edge,
                    evaluator,
                    args,
                )
                for key, result in results.items():
                    loggers[key].add_result(run, result)
                    train_hits, valid_hits, test_hits = result
                    print(key)
                    print(
                        f"Run: {run + 1:02d}, "
                        f"Epoch: {epoch:02d}, "
                        f"Loss: {loss:.4f}, "
                        f"Train: {100 * train_hits:.2f}%, "
                        f"Valid: {100 * valid_hits:.2f}%, "
                        f"Test: {100 * test_hits:.2f}%"
                    )

                if args.eval_seperate:
                    result_dict = test_cold_seperate(
                        model,
                        predictor,
                        g,
                        g.ndata["feat"],
                        split_edge,
                        evaluator,
                        args,
                    )
                    for type, results in result_dict.items():
                        for key, result in results.items():
                            sep_loggers[type][key].add_result(run, result)
                            # train_hits, valid_hits, test_hits = result
                            # print(key)
                            # print(
                            #     f"Run: {run + 1:02d}, "
                            #     f"Epoch: {epoch:02d}, "
                            #     f"Loss: {loss:.4f}, "
                            #     f"Train: {100 * train_hits:.2f}%, "
                            #     f"Valid: {100 * valid_hits:.2f}%, "
                            #     f"Test: {100 * test_hits:.2f}%"
                            # )
                print("---")
                

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)
        if args.eval_seperate:
            for type, results in sep_loggers.items():
                print("")
                print(type)
                for key, result in results.items():
                    print(key)
                    result.print_statistics(run)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()
    if args.eval_seperate:
        for type, results in sep_loggers.items():
            print("")
            print(type)
            for key, result in results.items():
                print(key)
                result.print_statistics()


if __name__ == "__main__":
    main()
