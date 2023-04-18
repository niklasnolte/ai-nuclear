import pandas as pd
def functions_to_names(functions):
    names = []
    for function in functions:
        names.append(function.replace('+','p').replace('-','m').replace('*','t'))
    # concatenate all names into one string

    return ''.join(names)

def run_models(modelclass, modelname, train_func, test_sizes, seeds, functions):
    results_dict = None
    for ts in test_sizes:
        for seed in seeds:
            ts = round(ts, 2)
            function_names  = functions_to_names(functions)
            title = f'{modelname}_{function_names}_ts{ts}_seed{seed}'
            res = train_func(modelclass = modelclass, 
                            functions = functions, 
                            lr = 1e-3, 
                            wd = 1e-4, 
                            embed_dim = 64, 
                            title = title,
                            basepath = f'models/{dir}/{title}/',
                            device = 'cuda:0',
                            seed = seed,
                            test_size = ts,
                        )
            if results_dict is None:
                results_dict = {m:[] for m in res.keys()}
                results_dict['title'] = []
                results_dict['test_size'] = []
                results_dict['seed'] = []
            for m in res.keys():
                results_dict[m].append(res[m])    
            results_dict['title'].append(title)
            results_dict['test_size'].append(ts)
            results_dict['seed'].append(seed)
            df = pd.DataFrame(results_dict)
            df.to_csv(f'full_results/{modelname}/{modelname}.csv')    
    return df  