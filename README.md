# Extra-gradient with player sampling

Training GANs and convex games with gradient extrapolation and alternation of player updates.

Paper https://arxiv.org/abs/1905.12363

```
Jelassi, S., Enrich, C. D., Scieur, D., Mensch, A., & Bruna, J. (2019). Extra-gradient with player sampling for provable fast convergence in n-player games. Proceedings of the International Conference on Machine Learning
```


## How to install

```bash
python setup.py develop
```

## Run

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

