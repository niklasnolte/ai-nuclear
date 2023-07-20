import torch
class Config:
    def __init__(self):
        self.LIMIT = 20
        self.training_epsilon = .0001
        self.all_fn_dict = {
        r'$a+b$': lambda x, y: x + y,
        r'$|a-b|$': lambda x, y: abs(x - y),
        r'$(a+b)^{2/3}$': lambda x, y: (x + y) ** (2 / 3),
        r'$log(a+b+1)$': lambda x, y: torch.log(x + y + 1),
        r'$e^{-(a+b)^{1/2}/5}$': lambda x, y: torch.exp(-(x + y)**(1/2)/5),
        }

        self.all_fn_dict_aesthetic = {
        r'$a+b$': lambda x, y: x + y,
        r'$|a-b|$': lambda x, y: abs(x - y),
        r'$\sqrt[3]{(a + b)^2}$': lambda x, y: (x + y) ** (2 / 3),
        r'$\log (a+b+1)$': lambda x, y: torch.log(x + y + 1),
        r'$\exp{\frac{-\sqrt{a+b}}{5}}$': lambda x, y: torch.exp(-(x + y)**(1/2)/5),
        }
        # 0.3 training frac
        self.best_baseline_multi = 'BaselineModel_fn01234_hd128_nl2_optAdam_bs4__lr0.0001_wd0.01_epochs10000_seed1_20lim_ts0.11_0.5cos'
        #self.best_taskemb_multi = 'TaskEmbModel_fn01234_hd128_nl2_optAdam_bs32__lr0.0001_wd0.001_epochs1500_seed1_20lim_ts0.3_cos'
        self.best_taskemb_multi = 'TaskEmbModel_fn01234_hd128_nl2_optAdam_bs4__lr0.0001_wd0.001_epochs10000_seed1_20lim_ts0.13_0.5cos'
        self.best_baseline_single = {fn:f'BaselineModel_fn{fn}_hd128_nl2_optAdam_bs4__lr0.0001_wd0.1_epochs10000_seed1_20lim_ts0.11_0.5cos'
                                     for fn in range(5)}
        # self.best_baseline_single = {
        #                 0: 'BaselineModel_fn0_hd128_nl4_optAdam_bs32__lr0.0001_wd0.01_epochs1500_seed1_20lim_ts0.3_cos',
        #                 1: 'BaselineModel_fn1_hd64_nl3_optAdam_bs32__lr0.0001_wd0.01_epochs1500_seed1_20lim_ts0.3_cos',
        #                 2: 'BaselineModel_fn2_hd128_nl4_optAdam_bs32__lr0.0001_wd0.01_epochs1500_seed1_20lim_ts0.3_cos',
        #                 3: 'BaselineModel_fn3_hd64_nl3_optAdam_bs32__lr0.0001_wd0.01_epochs1500_seed1_20lim_ts0.3_cos',
        #                 4: 'BaselineModel_fn4_hd128_nl4_optAdam_bs32__lr0.0001_wd0.01_epochs1500_seed1_20lim_ts0.3_cos'}
    


        # 0.8 training frac
        # self.best_baseline_multi = 'BaselineModel_fn01234_hd128_nl1_optAdam_bs16__lr5e-05_wd0.0001_epochs5000_20lim_cos'
        # self.best_taskemb_multi = 'TaskEmbModel_fn01234_hd256_nl5_optAdam_bs32__lr5e-05_wd0.0001_epochs5000_seed1_20lim_cos'
        # self.best_baseline_single = {
        #                 0: 'BaselineModel_fn0_hd64_nl2_optAdam_bs16__lr0.0001_wd0.0001_epochs5000_20lim_cos',
        #                 1: 'BaselineModel_fn1_hd256_nl1_optAdam_bs16__lr0.0001_wd0.0001_epochs5000_20lim_cos',
        #                 2: 'BaselineModel_fn2_hd32_nl1_optAdam_bs16__lr0.0001_wd0.0001_epochs5000_20lim_cos',
        #                 3: 'BaselineModel_fn3_hd32_nl1_optAdam_bs16__lr0.0001_wd0.0001_epochs5000_20lim_cos',
        #                 4: 'BaselineModel_fn4_hd64_nl1_optAdam_bs16__lr0.0001_wd0.0001_epochs5000_20lim_cos'}
        