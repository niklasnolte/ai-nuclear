import torch
from functools import cached_property
from config import Task
from config_utils import parse_arguments_and_get_name
from data import semi_empirical_mass_formula, semi_empirical_radius_formula
from data import prepare_nuclear_data

# load TASK from env
TASK = Task.PN

args, name = parse_arguments_and_get_name(TASK)
args.LOG_FREQ = 1000
torch.manual_seed(args.SEED)


def _interpret_nuclei_as_sequence(data):
    data = data._replace(X=data.X[:200, :2].long())  # invalid, proton, neutron
    return data


data = _interpret_nuclei_as_sequence(prepare_nuclear_data(args))


def nanstd(tensor, dim, keepdim=False):
    mask = ~tensor.isnan()

    mean = (
        torch.nansum(tensor, dim=dim, keepdim=True)
        / mask.sum(dim=dim, keepdim=True).float()
    )

    centered_values = torch.where(mask, tensor - mean, torch.zeros_like(tensor))
    squared_diff = torch.where(mask, centered_values**2, torch.zeros_like(tensor))
    sum_squared_diff = torch.sum(squared_diff, dim=dim, keepdim=True)

    # Divide the sum by the number of non-NaN values and take the square root
    std = torch.sqrt(sum_squared_diff / (mask.sum(dim=dim, keepdim=True).float() - 1))

    if not keepdim:
        std = std.squeeze(dim)

    return std


class Simulator(torch.nn.Module):
    def __init__(self, parton_nums: torch.Tensor):
        super().__init__()
        self.device = parton_nums.device
        self.n_protons, self.n_neutrons = parton_nums[:, [0]], parton_nums[:, [1]]

        # learnable parameters
        self.register_parameter("energy_coeff", torch.nn.Parameter(torch.tensor(1.0)))
        self.register_parameter("energy_shift", torch.nn.Parameter(torch.tensor(1000.0)))
        self.register_parameter("energy_coeff_2", torch.nn.Parameter(torch.tensor(1.0)))
        self.register_parameter("energy_shift_2", torch.nn.Parameter(torch.tensor(1000.0)))
        self.register_parameter("radius_coeff", torch.nn.Parameter(torch.tensor(1.0)))
        self.register_parameter("radius_offset", torch.nn.Parameter(torch.tensor(1.0)))

        self.approx_max_seq_len = 300

    @cached_property
    def parton_types(self):
        # X has entries [x,y] for each element
        # from that make [0,0,0 (x times), 1,1,1 (y times), padding to max]
        seq_len = (self.n_protons + self.n_neutrons).max().item()
        seq = torch.zeros(
            self.n_protons.shape[0], seq_len, dtype=torch.long, device=self.device
        )
        arange = torch.arange(seq_len, device=self.device)
        seq[arange < self.n_protons] = -1
        seq[
            (arange >= self.n_protons) & (arange < (self.n_protons + self.n_neutrons))
        ] = 1
        return seq

    @cached_property
    def parton_positions(self):
        # initial parton locations
        # start from (0,0,0), smallest l2's first
        # so far one neutron and one proton can occupy the same position
        # but we also have spin, so later its 4 per position
        # TODO do that 4 position thing

        n_dims = 3
        side_len = (self.approx_max_seq_len ** (1 / n_dims)) // 2
        all_positions = torch.cartesian_prod(
            *[torch.arange(-side_len, side_len + 1)] * n_dims
        )
        all_positions = all_positions[torch.argsort(torch.norm(all_positions, dim=1))]

        positions = torch.empty(*self.parton_types.shape, n_dims, device=self.device)
        parton_idxs = torch.arange(self.parton_types.shape[1], device=self.device)
        alternating_mask = (parton_idxs % 2 == 0)
        for i, ps in enumerate(self.parton_types):
            poss = (
                torch.empty(self.parton_types.shape[1], n_dims, device=self.device)
                * torch.nan
            )

            proton_mask = ps == -1
            proton_spin_up = alternating_mask & proton_mask
            proton_spin_down = ~alternating_mask & proton_mask
            neutron_mask = ps == 1
            neutron_spin_up = alternating_mask & neutron_mask
            neutron_spin_down = ~alternating_mask & neutron_mask

            poss[proton_spin_up] = all_positions[:proton_spin_up.sum()]
            poss[proton_spin_down] = all_positions[:proton_spin_down.sum()]
            poss[neutron_spin_up] = all_positions[:neutron_spin_up.sum()]
            poss[neutron_spin_down] = all_positions[:neutron_spin_down.sum()]
            positions[i] = poss

        return positions

    @cached_property
    def distance_matrix(self) -> torch.Tensor:
        # positions is [batch, seq_len, 2]
        # make a matrix of distances between each element
        # [batch, seq_len, seq_len]
        p = self.parton_positions
        dists : torch.Tensor = torch.norm(p[:, None] - p[:, :, None], dim=-1)
        return dists

    def _get_pairwise_energy(self, dist: torch.Tensor) -> torch.Tensor:
        # need to do something else for the invalid type
        return self.energy_coeff / (dist + self.energy_shift) + self.energy_coeff_2 / (dist**2 + self.energy_shift_2)

    @cached_property
    def parton_positions_std(self) -> torch.Tensor:
        return nanstd(self.parton_positions, (1, 2)).view(-1, 1)

    def radius(self) -> torch.Tensor:
        return self.radius_coeff * self.parton_positions_std + self.radius_offset

    def energy(self) -> torch.Tensor:
        # binding energy is the sum of all the pairwise potentials
        # no force at infinite distance
        dists = self.distance_matrix.flatten(1).nan_to_num(nan=torch.inf)
        energy = self._get_pairwise_energy(dists)
        energy = energy.sum(dim=1, keepdim=True) / 2  # distance matrix is symmetric
        # make dense tensor
        return energy



radius = semi_empirical_radius_formula(data.X[:, 0], data.X[:, 1]).view(-1, 1)
energy = semi_empirical_mass_formula(data.X[:, 0], data.X[:, 1]).view(-1, 1) * (data.X.sum(1, keepdim=True)) / 1000 # MeV

sim = Simulator(data.X)
optimizer = torch.optim.SGD(sim.parameters(), lr=1e-3, momentum=.99)
breakpoint()

# train loop
loss_fn = torch.nn.MSELoss()
for i in range(100000):
    # get energy, radius
    e_pred = sim.energy()
    r_pred = sim.radius()
    # calculate loss
    loss_e = loss_fn(e_pred, energy)
    loss_r = loss_fn(r_pred, radius)

    # backprop
    (loss_e + loss_r).backward()

    # print
    if i % args.LOG_FREQ == 0:
        print(f"{i}: {(loss_e**.5).item()}")
        print(f"{i}: {(loss_r**.5).item()}")
        for name, param in sim.named_parameters():
          if param.grad is not None:
            print(f"{name}: {param.item():.3f} | Grad: {param.grad.norm().item():.3f}")

    optimizer.step()
    optimizer.zero_grad()

