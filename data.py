import numpy as np
import pandas as pd
import urllib.request
import torch
from sklearn.model_selection import train_test_split

def get_data():
  np.random.seed(1)
  def lc_read_csv(url):
      req = urllib.request.Request("https://nds.iaea.org/relnsd/v0/data?" + url)
      req.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0')
      return pd.read_csv(urllib.request.urlopen(req))

  df = lc_read_csv("fields=ground_states&nuclides=all")
  #selection
  df = df[df.binding != ' ']
  df = df[df.binding != 0]
  vocab_size = (df.z.nunique(), df.n.nunique())
  X = torch.tensor(df[["z", "n"]].values).int()
  y = torch.tensor(df.binding.astype(float).values).view(-1, 1).float()
  y = (y - y.mean()) / y.std()

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
  return X_train , X_test, y_train, y_test, vocab_size

print(get_data()[1].shape)


