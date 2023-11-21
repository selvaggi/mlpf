        #     g = dgl.knn_graph(
        #         graph_coordinates,
        #         config.graph_config.get("k", 7),
        #         exclude_self=True,
        #     )
        #     if graph_coordinates.shape[0] < 10:
        #         print(graph_coordinates.shape)

        # i, j = g.edges()
        # edge_attr = torch.norm(
        #     graph_coordinates[i] - graph_coordinates[j], p=2, dim=1
        # ).view(-1, 1)
        # if n_noise > 0:
        #     noise = torch.zeros((p_hits.shape[0], n_noise)).float()
        #     noise.normal_(mean=0, std=1)
        #     hit_features_graph = torch.cat(
        #         (graph_coordinates, hit_type_one_hot, e_hits, p_hits, noise), dim=1
        #     )
        # else:


 # print("n hits:", number_hits, "number_part", number_part)
    # this builds fully connected graph
    # TODO build graph using the hit links (hit_particle_link) which assigns to each node the particle it belongs to
    # i, j = torch.tril_indices(number_hits, number_hits)
    # g = dgl.graph((i, j))
    # g = dgl.to_simple(g)
    # g = dgl.to_bidirected(g)


 # if config.graph_config.get("fully_connected", False):
        #     n_nodes = graph_coordinates.shape[0]
        #     if n_nodes > 1:
        #         i, j = torch.tril_indices(n_nodes, n_nodes, offset=-1)
        #         g = dgl.graph((i, j))  # create fully connected graph
        #         g = dgl.to_simple(g)  # remove repeated edges
        #         g = dgl.to_bidirected(g)
        #     else:
        #         g = dgl.knn_graph(graph_coordinates, 0, exclude_self=True)
        # else: