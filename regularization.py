#  %%
import torch


class Compressor:
    def __init__(self, n_componenets=None):
        self.n_componenets = n_componenets

    def _center(self, emb):
        self.mean_ = emb.mean(0)
        return emb - self.mean_

    def fit(self, emb, n=None):
        n = n if n is not None else self.n_componenets
        *_, V = torch.svd(self._center(emb))
        V = V[:, :n]
        self.V = V
        return self

    def fit_compress(self, emb, n=None):
        return self.fit(emb, n).compress(emb)

    def compress(self, emb=None):
        emb = emb if emb is not None else self.emb
        return (emb - self.mean_) @ self.V

    def decompress(self, emb):
        return emb @ self.V.T + self.mean_

def get_projected(emb, n):
    comp = Compressor(n_componenets=n)
    emb_ = comp.fit_compress(emb)
    emb_ = comp.decompress(emb_)
    return emb_

def forward_with_reg(model, n, *model_args):
    # print(model(*model_args))
    original_emb = model.emb[0].weight.data
    emb_ = get_projected(model.emb[0].weight, n)
    model.emb[0].weight.data = emb_
    y = model(*model_args)
    # print(y)
    model.emb[0].weight.data = original_emb
    # print(model(*model_args))
    return y


if __name__ == "__main__":
    from sklearn.decomposition import PCA
    from matplotlib import pyplot as plt

    torch.manual_seed(0)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.ModuleList([torch.nn.Linear(2, 5)])
        def forward(self, x):
            return self.emb[0](x)
    model = Model()
    x = torch.randn(1, 2)
    y = model(x)
    # print(y)
    y_ = forward_with_reg(model, 1, x)
    loss = torch.nn.functional.mse_loss(y, torch.zeros_like(y))
    loss += torch.nn.functional.mse_loss(y_, torch.zeros_like(y_))
    loss.backward()
    # %%
    torch.manual_seed(1)
    emb = torch.randn(5, 10)
    comp = Compressor(n_componenets=2)
    emb_ = comp.fit_compress(emb)
    emb_ = comp.decompress(emb_)

    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(emb)
    emb_pca = pca.inverse_transform(emb_pca)

    # emb_ = emb.numpy()
    # mean = emb_.mean(0)
    # emb_ -= mean
    # emb_ = emb_ @ pca.components_.T
    # emb_ = emb_ @ pca.components_
    # emb_ += mean

    plt.scatter(emb_[:, 0], emb_[:, 1], label="compressed", alpha=0.75)
    plt.scatter(emb_pca[:, 0], emb_pca[:, 1], label="pca", alpha=0.75)
    plt.legend()
    plt.show()
# %%
