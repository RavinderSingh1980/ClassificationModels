# pip install ucimlrepo
from ucimlrepo import fetch_ucirepo
bean = fetch_ucirepo(id=602)
X = bean.data.features
y = bean.data.targets.squeeze()
print(X.shape, y.value_counts())
