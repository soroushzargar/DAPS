import torch
import scipy.sparse as sp
import torch_geometric

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Torch Graph Models are running on {device}")
from gnn_cp.cp.transformations import CPTransformation


class VertexMPTransformation(CPTransformation):
    def transform(self, logits, **kwargs):
        logits = logits.to(device)
        neigh_coef = kwargs.get("neigh_coef", 0)
        edge_index = kwargs.get("edge_index")
        n_vertices = kwargs.get("n_vertices")

        A = torch.sparse.FloatTensor(
            edge_index.to(device),
            torch.ones(edge_index.shape[1]).to(device),
                (n_vertices, n_vertices)
        )
        degs = torch.matmul(A, torch.ones((A.shape[0])).to(device))

        aggregated_logits = torch.linalg.matmul(A, logits) * (1 / (degs + 1e-10))[:, None]

        res = aggregated_logits * (neigh_coef) + logits * (1 - neigh_coef)
        return res

class KHopVertexMPTransformation(CPTransformation):
    def transform(self, logits, **kwargs):
        hop_coefs = kwargs.get("hop_coefs", [1])
        edge_index = kwargs.get("edge_index")
        n_vertices = kwargs.get("n_vertices")

        A = torch.sparse.FloatTensor(
            edge_index.to(device),
            torch.ones(edge_index.shape[1]).to(device),
                (n_vertices, n_vertices)
        )

        hop_adj = torch.sparse.FloatTensor(
            torch.stack([torch.arange(0, n_vertices)] * 2).to(device),
            torch.ones((n_vertices)).to(device)
        )

        res = torch.zeros_like(logits)

        for hop in range(len(hop_coefs)):
            degs = torch.matmul(hop_adj, torch.ones((A.shape[0])).to(device))
            level_aggregated_logits = torch.linalg.matmul(hop_adj, logits) * (1 / (degs + 1e-10))[:, None]
            res += level_aggregated_logits * hop_coefs[hop]

            hop_adj = torch.sparse.mm(A, hop_adj)

        return res


class PPRCVertexMPTransformation(CPTransformation):
    def transform(self, logits, **kwargs):
        k = kwargs.get("k", 10)
        alpha = kwargs.get("alpha", 0.85)
        edge_index = kwargs.get("edge_index")
        n_vertices = kwargs.get("n_vertices")

        adj = torch_geometric.utils.to_scipy_sparse_matrix(edge_index=edge_index)
        ppr = self.approx_ppr_product(adj, n=n_vertices, alpha=alpha, n_iter=k)
        ppr_t = torch_geometric.utils.from_scipy_sparse_matrix(ppr)

        ppr_ts = torch.sparse.FloatTensor(
            ppr_t[0].to(device),
            ppr_t[1].float().to(device)
        )

        edge_result = torch.matmul(ppr_ts, logits)

        result = edge_result
        return result

    @staticmethod
    def approx_ppr_product(adj, n, h=None, alpha=0.85, n_iter=10):
      pi = sp.diags(1/adj.sum(1).A1) @ adj

      if h is None:
        h = sp.eye(n)

      h0 = h.copy()

      for _ in range(n_iter):
        h = alpha * pi @ h + (1-alpha) * h0

      return h