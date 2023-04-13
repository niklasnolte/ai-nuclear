import torch
from config import Task
from config_utils import parse_arguments_and_get_name
from data import prepare_nuclear_data
from loss import loss_by_task, metric_by_task, weight_by_task
import math
import tqdm

# load TASK from env
TASK = Task.PN

args, name = parse_arguments_and_get_name(TASK)
args.LOG_FREQ = 1
torch.manual_seed(args.SEED)

data = prepare_nuclear_data(args)


def _interpret_nuclei_as_sequence(data):
    data = data._replace(
        X=data.X[:, :2].long(), vocab_size=3
    )  # invalid, proton, neutron
    return data


data = _interpret_nuclei_as_sequence(data)


class PNAttentionNetwork(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim,
        output_dim,
        max_pos_enc=294,
        nheads=1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.nheads = nheads
        self.hidden_dim = hidden_dim

        self.embedding = torch.nn.Embedding(vocab_size, self.hidden_dim)
        self.attn = torch.nn.MultiheadAttention(
            self.hidden_dim, self.nheads, batch_first=True
        )
        self.readout = torch.nn.Linear(self.hidden_dim, output_dim)

        # for positional encoding
        position = torch.arange(max_pos_enc).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim)
        )
        pe = torch.zeros(1, max_pos_enc, hidden_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("single_pe", pe)
        self.register_buffer("arange", torch.arange(max_pos_enc))

    def _positional_encoding(self, X, n_protons, n_neutrons):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        # TODO I think we should precalculate those
        seq_len = X.shape[1]
        pe = self.single_pe[:,:seq_len].repeat(X.shape[0], 1, 1)
        arange = self.arange[:seq_len]
        neutron_mask_for_total = (arange >= n_protons) & (arange < (n_protons + n_neutrons))
        neutron_mask_for_single = arange < n_neutrons
        pe[neutron_mask_for_total] = pe[neutron_mask_for_single]
        breakpoint()
        return X + pe

    @staticmethod
    def _create_seq(X):
        # X has entries [x,y] for each element
        # from that make [0,0,0 (x times), 1,1,1 (y times), padding to max]
        n_protons, n_neutrons = X[:, [0]], X[:, [1]]
        seq_len = X.sum(dim=1).max().item()
        seq = torch.zeros(X.shape[0], seq_len, dtype=torch.long, device=X.device)
        arange = torch.arange(seq_len, device=X.device)
        seq[arange < n_protons] = 1
        seq[(arange >= n_protons) & (arange < (n_protons + n_neutrons))] = 2
        return seq, n_protons, n_neutrons

    def forward(self, X):
        X, n_protons, n_neutrons = self._create_seq(X)
        X = self.embedding(X)
        X = self._positional_encoding(X, n_protons, n_neutrons)
        X, _ = self.attn(X, X, X, need_weights=False)
        X = X.sum(dim=1)
        return self.readout(X)


model = PNAttentionNetwork(
    data.vocab_size, args.HIDDEN_DIM, sum(data.output_map.values())
).to(args.DEV)
optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)

y_train = data.y[data.train_mask]
y_val = data.y[data.val_mask]

weights = weight_by_task(data.output_map, args)

bar = tqdm.trange(
    args.EPOCHS,
)
for epoch in range(args.EPOCHS):
    model.train()
    optimizer.zero_grad()
    out = torch.zeros(data.y.shape).to(args.DEV)
    out_col = 0
    # make a task specific X (task feature)
    out = model(data.X)

    train_losses = loss_by_task(out[data.train_mask], y_train, data.output_map, args)
    train_loss = (weights * train_losses).mean()

    train_losses = train_losses.mean(dim=1)
    train_loss.mean().backward()
    optimizer.step()

    if epoch % args.LOG_FREQ == 0:
        with torch.no_grad():
            model.eval()
            train_metrics = metric_by_task(
                out[data.train_mask],
                data.X[data.train_mask],
                y_train,
                data.output_map,
                args,
                qt=data.regression_transformer,
            )
            val_metrics = metric_by_task(
                out[data.val_mask],
                data.X[data.val_mask],
                y_val,
                data.output_map,
                args,
                qt=data.regression_transformer,
            )
            val_losses = loss_by_task(
                out[data.val_mask], y_val, data.output_map, args
            ).mean(dim=1)
            val_loss = (weights * val_losses).mean()

            msg = f"\nEpoch {epoch:<6} Train Losses | Metrics\n"
            for i, target in enumerate(data.output_map.keys()):
                msg += f"{target:>15}: {train_losses[i].item():.4e} | {train_metrics[i].item():.6f}\n"
            msg += f"\nEpoch {epoch:<8} Val Losses | Metrics\n"
            for i, target in enumerate(data.output_map.keys()):
                msg += f"{target:>15}: {val_losses[i].item():.4e} | {val_metrics[i].item():.6f}\n"

            print(msg)
            bar.update(args.LOG_FREQ)
