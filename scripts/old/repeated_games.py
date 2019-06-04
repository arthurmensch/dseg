import matplotlib.pyplot as plt
import torch
from gamesrl.extrapolation import ExtrapolatedObjective, simultaneous_grad
from gamesrl.games import RockPaperScissor, \
    MatchingPennies, RandomMatrixGame, RandomQuadraticGame, \
    IteratedPrisonerDilemna, IteratedMatchingPennies
from gamesrl.strategies import BypassedSoftmax
from torch.nn import Parameter

torch.manual_seed(200)

step_size = .01
extra_step_size = .01
extrapolate = 'lola'
n_iter = 1000
game = 'ipd'
param = 'flat`'

if game == 'ipd':
    game = IteratedPrisonerDilemna()
elif game == 'mp':
    game = IteratedMatchingPennies()
else:
    raise ValueError()

n_players = game.n_players
dim_strategy = game.dim_strategy


if param == 'flat':
    strategies = [Parameter(torch.empty(dim_strategy))
                  for _ in range(n_players)]
    for s in strategies:
        s.data.uniform_(0., 1.)
elif param == 'sigmoid':
    strategies = [Parameter(torch.empty(dim_strategy, 2))
                  for _ in range(n_players)]
    for s in strategies:
        s.data.normal_(0., 1.)

strat_rec = [[] for _ in range(n_players)]
vs_rec = [[] for _ in range(n_players)]


def objective_fn(strategies):
    if param == 'sigmoid':
        strategies = [BypassedSoftmax.apply(s, 1)[:, 0] for s in strategies]
    elif param == 'flat':
        strategies = [torch.clamp(s, min=0., max=1.) for s in strategies]
    return game(strategies)


for i in range(n_iter):
    values = objective_fn(strategies)
    if extrapolate != 'None':
        gradients = simultaneous_grad(values, strategies,
                                      create_graph=self.lola)
        values = objective_fn([p + step_size * g for p, g
                                  in zip(parameters, gradients)])
    for l, v, l_rec, v_rec in zip(strategies, values, strat_rec, vs_rec):
        if param == 'flat':
            l_rec.append(torch.clamp(l.detach(), min=0., max=1)[None, :])
        else:
            l_rec.append(torch.softmax(l.detach(), 1)[:, 0][None, :])
        v_rec.append(v.item())
    gradients = simultaneous_grad(values, strategies)
    for l, g in zip(strategies, gradients):
        l.data += step_size * g

strat_rec = [(torch.cat(rec, dim=0)) for rec in strat_rec]

fig, axes = plt.subplots(1, 3, figsize=(3 * (n_players + 1), 3))
strategy_names = ['p(C|0)', 'p(C|CC)', 'p(C|CD)', 'p(C|DC)', 'p(C|DD)']

for i, (ax, rec) in enumerate(zip(axes, strat_rec)):
    for a in range(dim_strategy):
        ax.plot(range(n_iter), rec[:, a].numpy(), label=strategy_names[a])
    ax.set_title(f'Player #{i}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Colaboration probability')
    ax.set_ylim([-0.1, 1.1])
    ax.set_xlim([0, n_iter - 1])
axes[n_players - 1].legend(ncol=2)
for i, v_rec in enumerate(vs_rec):
    axes[n_players].plot(range(0, n_iter), v_rec, label=f'P{i}')
axes[n_players].set_title('Reward')
axes[n_players].set_xlabel('Iteration')
axes[n_players].legend()
plt.show()
