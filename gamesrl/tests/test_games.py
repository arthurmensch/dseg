import torch
from gamesrl.games import IteratedPrisonerDilemna
from numpy.testing import assert_almost_equal


def test_ipd_tit_for_tat():
    game = IteratedPrisonerDilemna()
    strategies = [torch.tensor([1, 1, 0., 1, 0]),
                  torch.tensor([1, 1, 1, 0., 0])]
    values = game(strategies)
    assert_almost_equal(values[0].item(), -1, decimal=6)
    assert_almost_equal(values[1].item(), -1, decimal=6)


def test_ipd_always_defect():
    game = IteratedPrisonerDilemna()
    strategies = [torch.tensor([0, 0, 0., 0, 0]),
                  torch.tensor([0, 0, 0, 0., 0])]
    values = game(strategies)
    assert_almost_equal(values[0].item(), -2, decimal=6)
    assert_almost_equal(values[1].item(), -2, decimal=6)


def test_ipd_always_colab():
    game = IteratedPrisonerDilemna()
    strategies = [torch.tensor([1., 1., 1., 1, 1]),
                  torch.tensor([1., 1., 1., 1, 1])]
    values = game(strategies)
    assert_almost_equal(values[0].item(), 0, decimal=6)
    assert_almost_equal(values[1].item(), 0, decimal=6)
