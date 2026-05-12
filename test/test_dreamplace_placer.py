import math

import torch

from macro_place.benchmark import Benchmark


def test_macro_clearance_loss_penalizes_pairs_closer_than_min_gap():
    from submissions.dreamplace_placer.placer import macro_clearance_loss

    pos = torch.tensor(
        [
            [0.0, 0.0],
            [15.0, 0.0],
            [40.0, 0.0],
        ],
        requires_grad=True,
    )
    sizes = torch.tensor(
        [
            [4.0, 4.0],
            [4.0, 4.0],
            [4.0, 4.0],
        ]
    )

    loss = macro_clearance_loss(pos, sizes, num_hard=3, min_gap=12.0)

    assert torch.isclose(loss, torch.tensor(1.0))
    loss.backward()
    assert pos.grad is not None
    assert pos.grad[0, 0] > 0
    assert pos.grad[1, 0] < 0


def test_macro_clearance_loss_zero_when_all_pairs_have_min_gap():
    from submissions.dreamplace_placer.placer import macro_clearance_loss

    pos = torch.tensor([[0.0, 0.0], [16.0, 0.0]])
    sizes = torch.tensor([[4.0, 4.0], [4.0, 4.0]])

    assert macro_clearance_loss(pos, sizes, num_hard=2, min_gap=12.0).item() == 0.0


def test_weighted_average_wirelength_is_differentiable_and_matches_two_pin_hpwl():
    from submissions.dreamplace_placer.placer import weighted_average_wirelength_loss

    pin_xy = torch.tensor([[0.0, 0.0], [10.0, 5.0]], requires_grad=True)
    data = {
        "pin_net_idx": torch.tensor([0, 0]),
        "num_nets": 1,
    }
    net_weights = torch.tensor([1.0])

    wl = weighted_average_wirelength_loss(
        pin_xy,
        data,
        net_weights=net_weights,
        norm=100.0,
        gamma=0.25,
    )

    assert abs(wl.item() - 0.15) < 1e-3
    wl.backward()
    assert pin_xy.grad is not None
    assert torch.isfinite(pin_xy.grad).all()


def test_weighted_average_wirelength_is_translation_invariant():
    from submissions.dreamplace_placer.placer import weighted_average_wirelength_loss

    data = {
        "pin_net_idx": torch.tensor([0, 0]),
        "num_nets": 1,
    }
    net_weights = torch.tensor([1.0])
    origin = torch.tensor([[0.0, 0.0], [10.0, 5.0]])
    shifted = torch.tensor([[100.0, 100.0], [110.0, 105.0]])

    wl_origin = weighted_average_wirelength_loss(origin, data, net_weights, norm=100.0, gamma=0.25)
    wl_shifted = weighted_average_wirelength_loss(shifted, data, net_weights, norm=100.0, gamma=0.25)

    assert torch.allclose(wl_shifted, wl_origin, atol=1e-4)


def test_nesterov_step_uses_plan_formula():
    from submissions.dreamplace_placer.placer import nesterov_step

    x = torch.tensor([1.0])
    x_prev = torch.tensor([0.0])
    grad = torch.tensor([2.0])

    x_next, t_cur = nesterov_step(x, x_prev, grad, step_size=0.1, t_prev=1.0)

    expected_t = (1.0 + math.sqrt(5.0)) / 2.0
    expected_x = 1.0 + (1.0 / expected_t) * (1.0 - 0.0) - 0.1 * 2.0
    assert abs(t_cur - expected_t) < 1e-8
    assert torch.allclose(x_next, torch.tensor([expected_x]))


def test_tetris_legalize_reserves_fixed_macros_before_movable_macros():
    from submissions.dreamplace_placer.placer import _tetris_legalize

    b = Benchmark(
        name="synthetic_fixed",
        canvas_width=100.0,
        canvas_height=100.0,
        num_macros=2,
        num_hard_macros=2,
        num_soft_macros=0,
        macro_positions=torch.tensor([[50.0, 50.0], [50.0, 50.0]]),
        macro_sizes=torch.tensor([[20.0, 20.0], [5.0, 5.0]]),
        macro_fixed=torch.tensor([False, True]),
        macro_names=["movable", "fixed"],
        num_nets=0,
        net_nodes=[],
        net_weights=torch.zeros(0),
        grid_rows=10,
        grid_cols=10,
        hard_macro_indices=[0, 1],
        soft_macro_indices=[],
    )

    pos = _tetris_legalize(b.macro_positions, b, gap=0.02)
    dx = abs(pos[0, 0].item() - pos[1, 0].item())
    dy = abs(pos[0, 1].item() - pos[1, 1].item())

    assert dx >= (20.0 + 5.0) / 2 + 0.02 or dy >= (20.0 + 5.0) / 2 + 0.02


def test_placer_handles_processed_benchmark_without_pin_connectivity():
    from submissions.dreamplace_placer.placer import DreamPlaceMacroPlacer

    b = Benchmark(
        name="synthetic_no_pins",
        canvas_width=100.0,
        canvas_height=100.0,
        num_macros=3,
        num_hard_macros=2,
        num_soft_macros=1,
        macro_positions=torch.tensor([[20.0, 20.0], [40.0, 20.0], [30.0, 35.0]]),
        macro_sizes=torch.tensor([[5.0, 5.0], [5.0, 5.0], [2.0, 2.0]]),
        macro_fixed=torch.tensor([False, False, False]),
        macro_names=["h0", "h1", "s0"],
        num_nets=3,
        net_nodes=[],
        net_weights=torch.ones(3),
        grid_rows=10,
        grid_cols=10,
        hard_macro_indices=[0, 1],
        soft_macro_indices=[2],
    )

    pos = DreamPlaceMacroPlacer().place(b)

    assert pos.shape == (3, 2)
    assert torch.isfinite(pos).all()
