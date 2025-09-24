import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from vq import VectorQuantize
import dgl

class DualVQGNN(nn.Module):
    def __init__(
        self,
        input_dim,
        gnn_hidden_dim,
        output_dim_gnn,
        num_gnn_layers,
        codebook_size1,
        codebook_size2,
        vq_dim,
        activation_str="relu",
        norm_type="none",
        dropout_ratio=0.0,
        decay=0.8,
        commitment_weight=0.25,
        use_cosine_sim=True,
        kmeans_init=True,
        threshold_ema_dead_code=2,
        projection_head_hidden_dim=128,
        projection_head_output_dim=64
    ):
        super().__init__()
        self.num_gnn_layers = num_gnn_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        
        if activation_str == "relu":
            self.activation = F.relu
        elif activation_str == "leaky_relu":
            self.activation = F.leaky_relu
        else:
            self.activation = F.relu

        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()

        current_dim = input_dim
        for i in range(num_gnn_layers):
            out_dim = gnn_hidden_dim if i < num_gnn_layers - 1 else output_dim_gnn
            self.gnn_layers.append(GraphConv(current_dim, out_dim, activation=None, allow_zero_in_degree=True))
            if i < num_gnn_layers - 1:
                if self.norm_type == "batch":
                    self.gnn_norms.append(nn.BatchNorm1d(out_dim))
                elif self.norm_type == "layer":
                    self.gnn_norms.append(nn.LayerNorm(out_dim))
            current_dim = out_dim

        self.vq1 = VectorQuantize(
            dim=output_dim_gnn, codebook_dim=vq_dim, codebook_size=codebook_size1,
            decay=decay, commitment_weight=commitment_weight, use_cosine_sim=use_cosine_sim,
            kmeans_init=kmeans_init, threshold_ema_dead_code=threshold_ema_dead_code
        )
        self.vq2 = VectorQuantize(
            dim=output_dim_gnn, codebook_dim=vq_dim, codebook_size=codebook_size2,
            decay=decay, commitment_weight=commitment_weight, use_cosine_sim=use_cosine_sim,
            kmeans_init=kmeans_init, threshold_ema_dead_code=threshold_ema_dead_code
        )

        self.projection_head1 = nn.Sequential(
            nn.Linear(vq_dim, projection_head_hidden_dim), nn.ReLU(),
            nn.Linear(projection_head_hidden_dim, projection_head_output_dim)
        )
        self.projection_head2 = nn.Sequential(
            nn.Linear(vq_dim, projection_head_hidden_dim), nn.ReLU(),
            nn.Linear(projection_head_hidden_dim, projection_head_output_dim)
        )

    def forward(self, g, feats):
        h = feats
        for i, layer in enumerate(self.gnn_layers):
            h = layer(g, h)
            if i < self.num_gnn_layers - 1:
                if self.norm_type != "none" and i < len(self.gnn_norms):
                    h = self.gnn_norms[i](h)
                if self.activation:
                     h = self.activation(h)
                h = self.dropout(h)
        
        gnn_embeddings_for_vq = h

        quantized1, _, commit_loss1, _, _ = self.vq1(gnn_embeddings_for_vq)
        quantized2, _, commit_loss2, _, _ = self.vq2(gnn_embeddings_for_vq)

        projected1 = self.projection_head1(quantized1)
        projected2 = self.projection_head2(quantized2)

        return projected1, projected2, commit_loss1, commit_loss2

    @torch.no_grad()
    def get_gnn_representations(self, g, feats):
        self.eval()
        h = feats
        for i, layer in enumerate(self.gnn_layers):
            h = layer(g, h)
            if i < self.num_gnn_layers - 1:
                if self.norm_type != "none" and i < len(self.gnn_norms):
                    h = self.gnn_norms[i](h)
                if self.activation:
                     h = self.activation(h)
        return h

    @torch.no_grad()
    def get_codebook_tensors(self):
        self.eval()
        return self.vq1._codebook.embed.squeeze(), self.vq2._codebook.embed.squeeze()