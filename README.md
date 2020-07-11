# Extragradient with player sampling for faster Nash equilibrium finding

Training GANs and convex games with gradient extrapolation and alternation of player updates.

Paper https://arxiv.org/abs/1905.12363

```
Jelassi, S., Enrich, C. D., Scieur, D., Mensch, A., & Bruna, J. (2020). Extragradient with player sampling for faster Nash equilibrium finding. Proceedings of the International Conference on Machine Learning
```


## Installation

```bash
python setup.py develop
```

## Experiments

```bash
cd scripts
```

Find equilibrium in a matrix game
```
python matrix_games.py
```

Train a CIFAR GAN
```
python gan_train.py
```

Find equilibrium in a simple 2-player one parameter game
```
python toy.py
```

