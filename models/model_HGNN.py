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
    
class MRePath(nn.Module):
    def __init__(self, omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4,
                 fusion="concat", model_size="small", graph_type="HGNN",
                 path_input_dim=1024, num_patches=4096):
        super(MRePath, self).__init__()

        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.fusion = fusion

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
        self.graph_type = graph_type
        
        if self.graph_type == "HGNN":
            self.sheaf_builder = SheafBuilderGeneral()
            self.convs=nn.ModuleList()
            # Sheaf Diffusion layers
            for _ in range(3):
                self.convs.append(HyperDiffusionGeneralSheafConv(256, 256, d=1, device='cuda'))
        elif self.graph_type == "GCN":
            from torch_geometric.nn.models import GCN
            self.graph = GCN(in_channels=256, hidden_channels=512, out_channels=256, num_layers=3, dropout=0.25).to("cuda")
        elif self.graph_type == "GAT":
            from torch_geometric.nn.models import GAT
            self.graph = GAT(in_channels=256, hidden_channels=512, out_channels=256, num_layers=3, dropout=0.25).to("cuda")                           
        
        # Genomic Embedding Network
        hidden = self.size_dict["genomics"][model_size]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.genomics_fc = nn.ModuleList(sig_networks)
       
        
        # Modality rebalance from Eq. (7)-(9).
        g_dim = self.size_dict["genomics"][model_size][-1]
        g_num = 6
        self.dynamic_weighting = DynamicWeighting(
            embedding_dim=g_dim,
            num_pathways=g_num,
            num_patches=num_patches,
        )
        
        self.attention_fusion = AlignFusion(
            embedding_dim=g_dim,
            num_heads = 4,
            num_pathways = g_num
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
        pathomics_features = self.pathomics_fc(x_path)
        if pathomics_features.ndim == 3:
            if pathomics_features.shape[0] != 1:
                raise ValueError("MRePath currently expects one WSI per batch")
            pathomics_features = pathomics_features[0]
        
        # graph structure
        graph = kwargs["graph"]
        edge_index = graph.edge_index
        edge_latent = graph.edge_latent

        # sheaf hypergraph
        has_hyperedges = False
        if self.graph_type == "HGNN":
          if edge_index.shape[1]+edge_latent.shape[1]>0:
              hyper_index = self.get_hyperedge(edge_index)
              hyper_latent = self.get_hyperedge(edge_latent)

              hyperedges = hyper_index + hyper_latent
              if hyperedges:
                  hg = dhg.Hypergraph(
                      num_v=pathomics_features.shape[0], e_list=hyperedges
                  )
                  H = hg.H.coalesce().indices().long().to(
                      pathomics_features.device
                  )
                  if H.numel() > 0:
                      hyperedge_attr = self.init_hyperedge_attr(
                          x=pathomics_features, hyperedge_index=H
                      )
                      num_nodes = pathomics_features.shape[0]
                      num_edges = H[1].max().item() + 1
                      has_hyperedges = True
        else:
            edge_total = torch.cat((edge_index, edge_latent), dim=1).to(
                pathomics_features.device
            )

        # Algorithm 1 computes weights from encoded raw pathology P and
        # genomics G, then applies them to high-order pathology Ph and G.
        weights, confidence = self.dynamic_weighting(
            pathomics_features.unsqueeze(0), genomics_features
        )
        self.last_modality_weights = weights.detach()
        self.last_confidence = tuple(value.detach() for value in confidence)
                
        # three layers hypergraph convolution
        if self.graph_type=="HGNN":
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
        else:
            pathomics_features = self.graph(pathomics_features, edge_total).unsqueeze(0)

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
        

    
