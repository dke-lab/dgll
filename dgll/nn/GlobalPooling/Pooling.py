# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 04:46:07 2024

@author: Tariq
"""

from typing import List, Optional, Union

from dgll import backend as F
from torch_scatter import scatter

### This methods and class provides global pooling function for
# graph nueral network especially for graph classification task,
#It contains three popular global graph pooling methods such as sum or add, mean and max pooling.
# which reduce graph based on summation, mean and max of node features values
# date created = 28 Oct, 2021, Project name : GDBMs, created by: Tariq Habib Afridi, email address: afridi@khu.ac.kr
def sumPooling(x: F.Tensor, batch: Optional[F.Tensor],
                    size: Optional[int] = None) -> F.Tensor:
    r""" This method returns batch-wise graph-level-outputs by summation of node features
    across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` it is computed by
    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathbf{x}_n
    Args:
        x (Tensor): Node feature matrix contains number of nodes vs features
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific batch.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
    """
    if batch is None:
        return x.sum(dim=0, keepdim=True)
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='add')


def meanPooling(x: F.Tensor, batch: Optional[F.Tensor],
                     size: Optional[int] = None) -> F.Tensor:
    r""" This method returns batch-wise graph-level-outputs by averaging node features
    across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` it is computed by
    .. math::
        \mathbf{r}_i = \frac{1}{N_i} \sum_{n=1}^{N_i} \mathbf{x}_n
    Args:
        x (Tensor): Node feature matrix contains number of nodes vs features
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, that assigns each
            node to a specific batch.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
    """
    if batch is None:
        return x.mean(dim=0, keepdim=True)
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='mean')


def maxPooling(x: F.Tensor, batch: Optional[F.Tensor],
                    size: Optional[int] = None) -> F.Tensor:
    r"""This method returns batch-wise graph-level-outputs by taking the
    maximum values across the node features in a batch, so that for a single graph
    :math:`\mathcal{G}_i` it is computed by
    .. math::
        \mathbf{r}_i = \mathrm{max}_{n=1}^{N_i} \, \mathbf{x}_n
    Args:
        x (Tensor): Node feature matrix contains number of nodes vs features
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
    """
    if batch is None:
        return x.max(dim=0, keepdim=True)[0]
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='max')

class Pooling(F.nn.Module):
    r""" A general class for global pooling that wraps the usage of
    :meth:`~dgll.gnn.pooling.sumPooling`,
    :meth:`~dgll.gnn.pooling.meanPooling` and
    :meth:`~dgll.gnn.pooling.maxPooling` into a single module.
    Args:
        aggr (string or List[str]): The aggregation method to use such as
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            If provided with a list, it will apply multiple aggregations which will
            generate different outputs concatenated in the last dimension.
    """
    def __init__(self, aggr: Union[str, List[str]]):
        super().__init__()

        self.aggrs = [aggr] if isinstance(aggr, str) else aggr

        assert len(self.aggrs) > 0
        assert len(set(self.aggrs) | {'sum', 'add', 'mean', 'max'}) == 4

    def forward(self, x: F.Tensor, batch: Optional[F.Tensor],
                size: Optional[int] = None) -> F.Tensor:
        """"""
        xs: List[F.Tensor] = []

        for aggr in self.aggrs:
            if aggr == 'sum' or aggr == 'add':
                xs.append(sumPooling(x, batch, size))
            elif aggr == 'mean':
                xs.append(meanPooling(x, batch, size))
            elif aggr == 'max':
                xs.append(maxPooling(x, batch, size))

        return xs[0] if len(xs) == 1 else F.cat(xs, dim=-1)

    def __repr__(self) -> str:
        aggr = self.aggrs[0] if len(self.aggrs) == 1 else self.aggrs
        return f'{self.__class__.__name__}(aggr={aggr})'