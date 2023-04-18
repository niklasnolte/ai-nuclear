import torch
class Config:
    def __init__(self):
        self.LIMIT = 100
        self.training_epsilon = .0001
        self.all_fn_dict = {
        r'$a+b$': lambda x, y: x + y,
        r'$|a-b|$': lambda x, y: abs(x - y),
        r'$(a+b)^{2/3}$': lambda x, y: (x + y) ** (2 / 3),
        r'$log(a+b+1)$': lambda x, y: torch.log(x + y + 1),
        r'$e^{-(a+b)^{1/2}/5}$': lambda x, y: torch.exp(-(x + y)**(1/2)/5),
    }