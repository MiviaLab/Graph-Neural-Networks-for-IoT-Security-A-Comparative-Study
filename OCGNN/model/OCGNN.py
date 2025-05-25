import torch
import torch.nn as nn
from torch_geometric.nn import GCN


class OCGNN(nn.Module):
    """
    One-Class Graph Neural Networks for Anomaly Detection in
    Attributed Networks.

    OCGNN is an anomaly detector that measures the
    distance of anomaly to the centroid, in a similar fashion to the
    support vector machine, but in the embedding space after feeding
    towards several layers of GCN.


    Parameters
    ----------
    in_dim : int
        Input dimension of model.
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
        Total number of layers in model. Default: ``2``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    backbone : torch.nn.Module
        The backbone of the deep detector implemented in PyG.
        Default: ``torch_geometric.nn.GCN``.
    beta : float, optional
        The weight between the reconstruction loss and radius.
        Default: ``0.5``.
    warmup : int, optional
        The number of epochs for warm-up training. Default: ``2``.
    **kwargs
        Other parameters for the backbone model.
    """

    def __init__(self,
                 in_dim,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 backbone=GCN,
                 beta=0.5,
                 **kwargs):
        super(OCGNN, self).__init__()

        self.beta = beta

        self.gnn = backbone(in_channels=in_dim,
                            hidden_channels=hid_dim,
                            num_layers=num_layers,
                            out_channels=hid_dim,
                            dropout=dropout,
                            act=act,
                            **kwargs)

        self.emb = None

    def forward(self, x, edge_index):
        """
        Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings.
        edge_index : torch.Tensor
            Edge index.

        Returns
        -------
        emb : torch.Tensor
            Output embeddings.
        """

        self.emb = self.gnn(x, edge_index)

        return self.emb

    def loss_func(self, emb, center, radius):
        """
        Loss function for OCGNN

        Parameters
        ----------
        emb : torch.Tensor
            Embeddings.

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        score : torch.Tensor
            Outlier scores of shape :math:`N` with gradients.
        """

        dist = torch.sum(torch.pow(emb - center, 2), 1)
        score = dist - radius ** 2
        loss = radius ** 2 + (1 / self.beta) * torch.mean(torch.relu(score))

        return loss, score.detach().cpu(), dist.detach().cpu()
