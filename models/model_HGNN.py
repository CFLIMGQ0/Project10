import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.fusion import AlignFusion
from models.layers.layers import *
from models.layers.sheaf_builder import *
from torch_scatter import scatter_mean
from .util import initialize_weights
from .util import SNN_Block
import dhg
from dhg.nn import HGNNPConv
from collections import defaultdict


class DynamicWeighting(nn.Module):
    """Dynamic modality weighting described by MRePath Eq. (7)-(9)."""

    def __init__(self, embedding_dim=256, num_pathways=6, num_patches=4096):
        super().__init__()
        self.num_patches = num_patches
        self.path_token_projection = nn.Linear(embedding_dim, 1)
        self.path_confidence = nn.Sequential(
            nn.Linear(num_patches, num_patches * 2),
            nn.Linear(num_patches * 2, num_patches),
            nn.Linear(num_patches, 1),
            nn.Sigmoid(),
        )
        genomic_dim = num_pathways * embedding_dim
        self.genomic_confidence = nn.Sequential(
            nn.Linear(genomic_dim, genomic_dim * 2),
            nn.Linear(genomic_dim * 2, genomic_dim),
            nn.Linear(genomic_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, pathology, genomics):
        if pathology.ndim != 3 or genomics.ndim != 3:
            raise ValueError("pathology and genomics must be [batch, tokens, dim]")

        path_scores = self.path_token_projection(pathology).squeeze(-1)
        if path_scores.shape[1] == self.num_patches:
            path_mono = self.path_confidence(path_scores)
        else:
            path_mono = torch.sigmoid(path_scores.mean(dim=1, keepdim=True))
        gene_mono = self.genomic_confidence(genomics.flatten(start_dim=1))

        eps = 1e-4
        path_mono = path_mono.clamp(eps, 1.0 - eps)
        gene_mono = gene_mono.clamp(eps, 1.0 - eps)
        log_path = torch.log(path_mono)
        log_gene = torch.log(gene_mono)
        log_joint = log_path + log_gene
        path_holo = log_path / log_joint
        gene_holo = log_gene / log_joint
        confidence = torch.cat(
            (path_mono + path_holo, gene_mono + gene_holo), dim=1
        )
        weights = torch.softmax(confidence, dim=1)
        return weights, (path_mono, gene_mono, path_holo, gene_holo)


class GeneGraphAggregator(nn.Module):
    """Aggregate the six genomic signatures on a fully connected graph."""

    def __init__(self, embedding_dim=256, method="default"):
        super().__init__()
        if method not in {"default", "gcn", "gat"}:
            raise ValueError(f"Unknown genomic aggregation method: {method}")
        self.method = method
        if method == "default":
            self.graph = None
        elif method == "gcn":
            from torch_geometric.nn.models import GCN

            self.graph = GCN(
                in_channels=embedding_dim,
                hidden_channels=embedding_dim,
                out_channels=embedding_dim,
                num_layers=3,
                dropout=0.25,
            )
        else:
            from torch_geometric.nn.models import GAT

            self.graph = GAT(
                in_channels=embedding_dim,
                hidden_channels=embedding_dim,
                out_channels=embedding_dim,
                num_layers=3,
                dropout=0.25,
            )

        fully_connected = [
            (source, target)
            for source in range(6)
            for target in range(6)
            if source != target
        ]
        self.register_buffer(
            "edge_index",
            torch.tensor(fully_connected, dtype=torch.long).t().contiguous(),
            persistent=False,
        )

    def forward(self, genomics):
        if self.graph is None:
            return genomics
        if genomics.ndim != 3 or genomics.shape[1] != 6:
            raise ValueError("genomics must be [batch, 6, embedding_dim]")
        return torch.stack(
            [self.graph(sample, self.edge_index) for sample in genomics], dim=0
        )

class MRePath(nn.Module):
    def __init__(self, omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4,
                 fusion="concat", model_size="small", graph_type="shgnn",
                 path_input_dim=1024, num_patches=4096,
                 hyperedge_mode="both", weighting_mode="dynamic",
                 fixed_pathology_weight=0.5, fixed_genomic_weight=0.5,
                 fusion_variant="ifa", gene_aggregation="default"):
        super(MRePath, self).__init__()

        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.fusion = fusion
        graph_aliases = {
            "HGNN": "shgnn",
            "SHGNN": "shgnn",
            "GCN": "gcn",
            "GAT": "gat",
            "MLP": "mlp",
        }
        graph_type = graph_aliases.get(graph_type, graph_type.lower())
        if graph_type not in {"mlp", "gat", "gcn", "hgnn", "shgnn"}:
            raise ValueError(f"Unknown pathology graph type: {graph_type}")
        if hyperedge_mode not in {"none", "topology", "feature", "both"}:
            raise ValueError(f"Unknown hyperedge mode: {hyperedge_mode}")
        if weighting_mode not in {"dynamic", "fixed"}:
            raise ValueError(f"Unknown modality weighting mode: {weighting_mode}")
        if fixed_pathology_weight < 0 or fixed_genomic_weight < 0:
            raise ValueError("Fixed modality weights must be non-negative")
        if abs(fixed_pathology_weight + fixed_genomic_weight - 1.0) > 1e-8:
            raise ValueError("Fixed modality weights must sum to one")
        self.graph_type = graph_type
        self.hyperedge_mode = hyperedge_mode
        self.weighting_mode = weighting_mode

        ###
        self.size_dict = {
            "pathomics": {"small": [path_input_dim, 256, 256], "large": [path_input_dim, 512, 256]},
            "genomics": {"small": [1024, 256], "large": [1024, 1024, 1024, 256]},
        }
        # Pathomics Embedding Network
        hidden = self.size_dict["pathomics"][model_size]
        fc = []
        for idx in range(len(hidden) - 1):
            fc.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            fc.append(nn.ReLU6())
            fc.append(nn.Dropout(0.25))
        self.pathomics_fc = nn.Sequential(*fc)
        if self.graph_type == "shgnn":
            self.sheaf_builder = SheafBuilderGeneral()
            self.convs=nn.ModuleList()
            # Sheaf Diffusion layers
            for _ in range(3):
                self.convs.append(HyperDiffusionGeneralSheafConv(256, 256, d=1, device='cuda'))
        elif self.graph_type == "hgnn":
            self.convs = nn.ModuleList(
                [
                    HGNNPConv(256, 256, drop_rate=0.25),
                    HGNNPConv(256, 256, drop_rate=0.25),
                    HGNNPConv(256, 256, drop_rate=0.0, is_last=True),
                ]
            )
        elif self.graph_type == "gcn":
            from torch_geometric.nn.models import GCN

            self.graph = GCN(
                in_channels=256,
                hidden_channels=512,
                out_channels=256,
                num_layers=3,
                dropout=0.25,
            )
        elif self.graph_type == "gat":
            from torch_geometric.nn.models import GAT

            self.graph = GAT(
                in_channels=256,
                hidden_channels=512,
                out_channels=256,
                num_layers=3,
                dropout=0.25,
            )
        
        # Genomic Embedding Network
        hidden = self.size_dict["genomics"][model_size]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.genomics_fc = nn.ModuleList(sig_networks)
        self.gene_aggregator = GeneGraphAggregator(
            embedding_dim=hidden[-1], method=gene_aggregation
        )
       
        
        # Modality rebalance from Eq. (7)-(9).
        g_dim = self.size_dict["genomics"][model_size][-1]
        g_num = 6
        if self.weighting_mode == "dynamic":
            self.dynamic_weighting = DynamicWeighting(
                embedding_dim=g_dim,
                num_pathways=g_num,
                num_patches=num_patches,
            )
        else:
            self.dynamic_weighting = None
        self.register_buffer(
            "fixed_modality_weights",
            torch.tensor(
                [[fixed_pathology_weight, fixed_genomic_weight]],
                dtype=torch.float32,
            ),
            persistent=True,
        )

        self.attention_fusion = AlignFusion(
            embedding_dim=g_dim,
            num_heads = 4,
            num_pathways = g_num,
            variant=fusion_variant,
        )

        # Classification Layer
        self.mm = nn.Sequential(
                *[nn.Linear(hidden[-1]*2, hidden[-1]//2), nn.ReLU()]
            )
        self.classifier = nn.Linear(hidden[-1]//2, self.n_classes)

        self.apply(initialize_weights)

    def forward(self, **kwargs):
        x_path = kwargs["x_path"]
        x_omic = [kwargs["x_omic%d" % i] for i in range(1, 7)]

        genomics_features = [self.genomics_fc[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]
        genomics_features = torch.stack(genomics_features).unsqueeze(0)  # [1, 6, 1024]
        genomics_features = self.gene_aggregator(genomics_features)
        pathomics_features = self.pathomics_fc(x_path)
        if pathomics_features.ndim == 3:
            if pathomics_features.shape[0] != 1:
                raise ValueError("MRePath currently expects one WSI per batch")
            pathomics_features = pathomics_features[0]
        
        # graph structure
        graph = kwargs["graph"]
        edge_index = graph.edge_index
        edge_latent = graph.edge_latent

        # Build the exact graph or hypergraph family selected by the paper
        # ablation. GAT/GCN consume ordinary pairwise edges; HGNN/SHGNN
        # convert the selected neighborhoods into hyperedges.
        has_hyperedges = False
        selected_edges = []
        if self.hyperedge_mode in {"topology", "both"}:
            selected_edges.append(edge_index)
        if self.hyperedge_mode in {"feature", "both"}:
            selected_edges.append(edge_latent)

        if self.graph_type in {"hgnn", "shgnn"} and selected_edges:
            hyperedges = []
            for edges in selected_edges:
                hyperedges.extend(self.get_hyperedge(edges))
            if hyperedges:
                hg = dhg.Hypergraph(
                    num_v=pathomics_features.shape[0],
                    e_list=hyperedges,
                    device=pathomics_features.device,
                )
                has_hyperedges = hg.num_e > 0
                if self.graph_type == "shgnn" and has_hyperedges:
                    H = hg.H.coalesce().indices().long().to(
                        pathomics_features.device
                    )
                    hyperedge_attr = self.init_hyperedge_attr(
                        x=pathomics_features, hyperedge_index=H
                    )
                    num_nodes = pathomics_features.shape[0]
                    num_edges = H[1].max().item() + 1
        elif self.graph_type in {"gcn", "gat"} and selected_edges:
            edge_total = torch.cat(selected_edges, dim=1).to(
                pathomics_features.device
            )
        else:
            edge_total = torch.empty(
                (2, 0), dtype=torch.long, device=pathomics_features.device
            )

        # Algorithm 1 computes weights from encoded raw pathology P and
        # genomics G, then applies them to high-order pathology Ph and G.
        if self.dynamic_weighting is not None:
            weights, confidence = self.dynamic_weighting(
                pathomics_features.unsqueeze(0), genomics_features
            )
        else:
            weights = self.fixed_modality_weights.to(
                device=pathomics_features.device,
                dtype=pathomics_features.dtype,
            )
            confidence = ()
        self.last_modality_weights = weights.detach()
        self.last_confidence = tuple(value.detach() for value in confidence)

        # Pathology aggregation.
        if self.graph_type == "shgnn":
            if has_hyperedges:
                for i, conv in enumerate(self.convs[:-1]):
                    if i == 0:
                        h_sheaf_index, h_sheaf_attributes = self.sheaf_builder(
                            pathomics_features, hyperedge_attr, H
                        )
                    pathomics_features = conv(
                        pathomics_features,
                        hyperedge_index=h_sheaf_index,
                        alpha=h_sheaf_attributes,
                        num_nodes=num_nodes,
                        num_edges=num_edges,
                    )
                    pathomics_features = F.dropout(
                        pathomics_features, p=0.25, training=self.training
                    )

                pathomics_features = self.convs[-1](
                    pathomics_features,
                    hyperedge_index=h_sheaf_index,
                    alpha=h_sheaf_attributes,
                    num_nodes=num_nodes,
                    num_edges=num_edges,
                )
            pathomics_features = pathomics_features.view(-1, 256).unsqueeze(0) # Nd x out_channels -> Nx(d*out_channels)
        elif self.graph_type == "hgnn":
            if has_hyperedges:
                for conv in self.convs:
                    pathomics_features = conv(pathomics_features, hg)
            pathomics_features = pathomics_features.unsqueeze(0)
        elif self.graph_type in {"gcn", "gat"}:
            if edge_total.numel() > 0:
                pathomics_features = self.graph(pathomics_features, edge_total)
            pathomics_features = pathomics_features.unsqueeze(0)
        else:
            pathomics_features = pathomics_features.unsqueeze(0)

        pathomics_features = pathomics_features * weights[:, 0].view(-1, 1, 1)
        genomics_features = genomics_features * weights[:, 1].view(-1, 1, 1)
        pathology_fused, genomics_fused = self.attention_fusion(
            pathology=pathomics_features,
            genomics=genomics_features,
        )

        paths_postSA_embed = torch.mean(genomics_fused, dim=1)
        wsi_postSA_embed = torch.mean(pathology_fused, dim=1)

        fusion = self.mm(
            torch.cat([paths_postSA_embed, wsi_postSA_embed], dim=1)
        )
        # predict
        logits = self.classifier(fusion)  
        
        return logits
        
    def get_hyperedge(self, edge):
        adj_matrix = edge.cpu().numpy()
        hyperedges = defaultdict(set)
        for start, end in adj_matrix.T:
            hyperedges[start].add(end)
        hypergraph_edges = []

        for start_node, end_nodes in hyperedges.items():
            edge = {start_node}.union(end_nodes)
            hypergraph_edges.append(list(edge))

        return hypergraph_edges
        
    def init_hyperedge_attr(self, type='avg', num_edges=None, x=None, hyperedge_index=None):
        #initialize hyperedge attributes either random or as the average of the node
        if type == 'rand':
            hyperedge_attr = torch.randn((num_edges, self.num_features)).to(self.device)
        elif type == 'avg':
            hyperedge_attr = scatter_mean(x[hyperedge_index[0]],hyperedge_index[1], dim=0)
        else:
            hyperedge_attr = None
        return hyperedge_attr
    def hyperedge_to_incidence_matrix(self,hyperedge_index, num_nodes, num_hyperedges):
        hyperedge_index = hyperedge_index.coalesce()

        # COO indices
        node_indices = hyperedge_index.indices()[0]
        edge_indices = hyperedge_index.indices()[1] 

        values = torch.ones(node_indices.size(0), dtype=torch.float32)

        incidence_matrix = torch.sparse_coo_tensor(
            indices=torch.stack([node_indices, edge_indices]),
            values=values,
            size=(num_nodes, num_hyperedges)
        )

        return incidence_matrix
        

    
