import os
import torch
from pysr import PySRRegressor
import numpy as np
from data import get_data, binding_formula, yorig, rms

opt = 'data'

os.chdir("..")
if opt=='empirical':
    model_pred = torch.load("models/empirical/model.pt")
elif opt=='data':
    model_pred = torch.load("models/data/model.pt")
os.chdir("PySR")

#import data & Eb from the empirical formula (normalized)  
X_train, X_test, y_train, y_test, y_mean, y_std, vocab_size = get_data(opt,0)
_, _, y_train_emp, y_test_emp, y_mean_emp, y_std_emp, _ = get_data('empirical',0)
_, _, y_train_PySR, y_test_PySR, y_mean_PySR, y_std_PySR, _ = get_data('PySR',0)

#transform back to Eb (MeV)
y_test0 = yorig(y_test,y_mean,y_std)
y_train0 = yorig(y_train,y_mean,y_std)
y_test_emp0 = yorig(y_test_emp,y_mean_emp,y_std_emp)
y_train_emp0 = yorig(y_train_emp,y_mean_emp,y_std_emp)

#unite train and test 
X = torch.cat((X_train,X_test), 0)
y_dat0 = torch.cat((y_train0,y_test0), 0)
y_emp0 = torch.cat((y_train_emp0,y_test_emp0), 0)

#model predictions
y_pred = model_pred(X)
y_pred0 = yorig(y_pred,y_mean,y_std)

#y =(y_pred0-y_emp0)
y = y_dat0

#make np.array out of tensors
X = X.cpu().detach().numpy()
y =y.cpu().detach().numpy()

model = PySRRegressor(
    procs=4,
    populations=8,
    # ^ 2 populations per core, so one is always running.
    population_size=60,
    # ^ Slightly larger populations, for greater diversity.
    ncyclesperiteration=10000, 
    # Generations between migrations.
    niterations=100,  # Run forever
    # early_stop_condition=(
    #     "stop_if(loss, complexity) = loss < 0.0001 && complexity < 15"
    #     # Stop early if we find a good and simple equation
    # ),
    timeout_in_seconds=60 * 60 * 24,
    # ^ Alternatively, stop after 24 hours have passed.
    maxsize=35,
    # ^ Allow greater complexity.
    maxdepth=15,
    # ^ But, avoid deep nesting.
    binary_operators=["*","/","+"],
    # binary_operators=["+", "*", "sol1(x, y) = 1000f0*(15.8f0*(x+y))/(x+y)",
    #                   "sol2(x, y) = 1000f0*(- 18.3f0*abs(x+y)^(2f0/3f0))/(x+y)",
    #                   "sol3(x, y) = 1000f0*(- 0.714f0*x*(x-1f0)/(abs(x+y)^(1f0/3f0)))/(x+y)",
    #                   "sol4(x, y) = 1000f0*( - 23.2f0*(y-x)^2f0/(x+y))/(x+y)"],
    #, "A(x,y) = (abs(x+y))^(2/3)"
    # unary_operators=[
    #     "p2o3(x) = cbrt(square(abs(x)))",
    #     "pm1o3(x) = 1/(cbrt(abs(x)))",
    #     "square",       
    #     # ^ Custom operator (julia syntax)"square","cbrt",
    # ],
    unary_operators=["square", "cube","exp","sin","square","log","sqrt"],
    #unary_operators=["square", "cube","exp","sin","square","tan","tanh","log","sqrt"],
    # constraints={"/": (5,3),
    #               "square": 3,
    #               "sin": 3
    #               },
    # nested_constraints={
    # "sin": {"exp":0,"sin":0,"tan":0,"tanh":0,"log":0,"sqrt":0},
    # "exp": {"exp":0,"sin":0,"tan":0,"tanh":0,"log":0,"sqrt":0},
    # "tan": {"exp":0,"sin":0,"tan":0,"tanh":0,"log":0,"sqrt":0},
    # "tanh": {"exp":0,"sin":0,"tan":0,"tanh":0,"log":0,"sqrt":0},
    # "log": {"exp":0,"sin":0,"tan":0,"tanh":0,"log":0,"sqrt":0},
    # "sqrt": {"exp":0,"sin":0,"tan":0,"tanh":0,"log":0,"sqrt":0}},
    nested_constraints={
    "sin": {"exp":0,"sin":0,"log":0,"sqrt":0},
    "exp": {"exp":0,"sin":0,"log":0,"sqrt":0},
    "log": {"exp":0,"sin":0,"log":0,"sqrt":0},
    "sqrt": {"exp":0,"sin":0,"log":0,"sqrt":0}},
    # constraints={
    # "/": (5,3),
    # # "p2o3": 3,
    # # "pm1o3": 3,
    # # "square": 3
    # },
    # nested_constraints={
    # "sol1": {"+":0, "*":0, "sol1": 0, "sol2": 0, "sol3": 0, "sol4": 0},
    # "sol2": {"+":0, "*":0, "sol1": 0, "sol2": 0, "sol3": 0, "sol4": 0},
    # "sol3": {"+":0, "*":0, "sol1": 0, "sol2": 0, "sol3": 0, "sol4": 0},
    # "sol4": {"+":0, "*":0, "sol1": 0, "sol2": 0, "sol3": 0, "sol4": 0},
    # },
    # ^ Nesting constraints on operators. For example,
    # "square(exp(x))" is not allowed, since "square": {"exp": 0}.
    # extra_sympy_mappings={"p2o3": lambda x: (abs(x))**(2/3),"pm1o3": lambda x: (abs(x))**(-1/3),
    #                       "inv": lambda x: 1/abs(x)},
    # extra_sympy_mappings={"sol1": lambda x,y: 1000*((x+y))/(x+y),
    #                       "sol2": lambda x,y: 1000*(- 18.3*abs(x+y)**(2/3))/(x+y),
    #                       "sol3": lambda x,y: 1000*(- 0.714*y*(y-1)/(abs(x+y)**(1/3)))/(x+y),
    #                       "sol4": lambda x,y: 1000*(- 23.2*(y-x)**2/(x+y))/(x+y)
    #                       },
    # ^ Define operator for SymPy as well
    complexity_of_constants=4,
    # ^ Punish constants more than variables
    weight_randomize=0.8,
    # ^ Randomize the tree much more frequently
    loss="loss(x, y) = (x - y)^2",
    # ^ Custom loss function (julia syntax)
)

model.fit(X, y)
#print(model)
