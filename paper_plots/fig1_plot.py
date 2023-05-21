# %%
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import sys, os
from mup import set_base_shapes
sys.path.append("../")
plt.style.use("../mystyle-bright.mplstyle")
model = torch.load("fig1_model.pt").cpu()
# logdir="/work/submit/kitouni/ai-nuclear/FULL/model_baseline/wd_0.01/lr_0.0001/epochs_50000/trainfrac_0.9/hiddendim_2048/depth_2/seed_0/batchsize_1024/targetsclassification_None/targetsregression_binding_semf:1-z:1-n:1-radius:1-qa:1-qbm:1-qbm_n:1-qec:1-sn:1-sp:1/sched_cosine/lipschitz_false"
# list all models that end in an integer
logdir = "./fig1"
models = [f for f in os.listdir(logdir) if f.split("_")[-1].split(".")[0].isdigit() and f.endswith(".pt")]
models = sorted(models, key=lambda x: int(x.split("_")[-1].split(".")[0]))
class config:
    n_components = 4
embedding_dict = {"Z": model.emb[0], "N": model.emb[1], "task": model.emb[2]}

pca = None
for model_idx in range(len(models)-1, -1, -1):
    for which in ["N"]:
        print(models[model_idx])
        model_dir = os.path.join(logdir, models[model_idx])
        shapes = None # os.path.join(logdir, "shapes.yaml")
        model.load_state_dict(torch.load(model_dir, map_location="cpu"))
        set_base_shapes(model, shapes, rescale_params=False, do_assert=False)

        pca_embedding_dict = {}
        for i, (key, emb) in enumerate(embedding_dict.items()):
            # if pca is None:
                pca = PCA(n_components=config.n_components)
                pca_embedding_dict[key] = pca.fit_transform(emb.detach().numpy())
            # else:
            #    pca_embedding_dict[key] = pca.transform(emb.detach().numpy()) 

        skip = 9 # these were not trained
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200)
        x = pca_embedding_dict[which][skip:, 0]
        y = pca_embedding_dict[which][skip:, 1]
        ax.scatter(x, y, 0.01, 'k', 'x')

        # z = torch.linspace(0, 1, len(x))
        z = pca_embedding_dict[which][skip:, 2]
        colors = plt.cm.viridis(z)
        z = pca_embedding_dict[which][skip:, 3]
        z = (z - z.min()) / (z.max() - z.min())
        sizes = ((z)*6).astype(int) + 3
        for n, (x_, y_) in enumerate(zip(x, y)):
            ax.annotate(str(n+skip), (x_, y_), fontsize=sizes[n], color=colors[n])
        # ax.set_axis_off()
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        # plt.savefig(f"fig1/fig1_{which}_{models[model_idx]}.pdf")
        plt.show()
        break


# %%
