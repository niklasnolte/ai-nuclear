from test_empirical import Empirical
import torch
from data import get_data

if __name__ == '__main__':
  sd = torch.load(f"empirical_sd.pt")
  model = torch.load('empirical_model.pt')
  model.load_state_dict(sd)