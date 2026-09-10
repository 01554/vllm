# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The expert pool through the real Marlin consumer (CUDA).

A layer with a small ``moe_expert_pool_rows`` resident count: decode steps
(one token) must match the uncached layer while experts are promoted and
evicted through the shared bank (bank rows > expert count, so the
logical-align + physical-remap path is exercised), and a wider batch must
match through the bank + host-view partition path. The pool tables must
stay consistent throughout."""

import subprocess
import sys

import pytest
import torch

from tests.kernels.expert_pool.marlin_fixture import (
    TOP_K,
    E,
    K,
    dist_env,  # noqa: F401
    make_layer,
    quantized_weights,
    routing,
    vllm_config,
)
from vllm.model_executor.layers.fused_moe.expert_pool.install import (
    install_expert_pool,
)
from vllm.model_executor.layers.fused_moe.expert_pool.pool import verify_bank_rows
from vllm.model_executor.layers.fused_moe.expert_pool.tables import (
    check_global_tables,
    resident_per_layer,
    set_gate,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    is_fp4_marlin_supported,
)
from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required"),
    pytest.mark.skipif(
        current_platform.is_cuda() and not is_fp4_marlin_supported(),
        reason="FP4 Marlin not supported on this GPU",
    ),
]

SLOTS = 4  # of E=8 experts resident per layer at start; top_k=2 staging rows


@pytest.mark.parametrize("lanes", [1, 10, 64])
@pytest.mark.parametrize("block", [8, 16, 32, 48, 64])
def test_small_alignment_preserves_route_groups_on_graph_replay(lanes, block):
    """Grouping, duplicate lanes and padding survive changed graph inputs."""
    from vllm.model_executor.layers.fused_moe.expert_pool.layer import (
        mask_routes,
        physical_block_experts_device,
    )
    from vllm.model_executor.layers.fused_moe.expert_pool.small_align import small_align
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    experts = 512
    mapping = torch.arange(experts, device="cuda", dtype=torch.int32) + experts
    mapping[3] = -1
    # Match serving topk_ids [tokens, top_k]; the CUDA reference reads size(1).
    ids = torch.zeros((1, lanes), device="cuda", dtype=torch.int64)
    small_align(ids, mapping, block, 2 * experts)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = small_align(ids, mapping, block, 2 * experts)

    def groups(result):
        sorted_ids, blocks, count = [t.cpu() for t in result]
        n = count.item()
        assert n % block == 0 and 0 <= n <= lanes * block
        groups_by_row: dict[int, list[int]] = {}
        for offset in range(0, n, block):
            row = blocks[offset // block].item()
            group = sorted_ids[offset : offset + block].tolist()
            assert all(0 <= i <= lanes for i in group)
            groups_by_row.setdefault(row, []).extend(i for i in group if i < lanes)
        return n, {k: sorted(v) for k, v in groups_by_row.items()}

    for values in (
        [-1] * lanes,
        [2] * lanes,
        [i % 7 for i in range(lanes)],
        [-1 if i % 3 == 0 else i for i in range(lanes)],
    ):
        ids.copy_(torch.tensor([values], device="cuda"))
        graph.replay()
        routed = mask_routes(ids, mapping)
        sorted_ids, logical, count = moe_align_block_size(
            routed, block, experts, None, ignore_invalid_experts=True
        )
        reference = (
            sorted_ids,
            physical_block_experts_device(logical, count, block, mapping, experts),
            count,
        )
        assert groups(actual) == groups(reference)
        n = actual[2].item()
        assert torch.all(actual[0][n:] == lanes)
        assert torch.all(actual[1][n // block :] == -1)


def _decode(order, device):
    logits = torch.full((1, E), -10.0, device=device)
    logits[0, order[0]] = 3.0
    logits[0, order[1]] = 2.0
    return logits


def test_two_layer_pool_decode_prefill_decode_matches_the_uncached_layers(
    dist_env,  # noqa: F811
):
    """Two layers share one bank (2 * SLOTS + staging = 10 rows > E = 8), so
    every bank call takes the logical-align + physical-remap path and a miss
    on one layer can evict the other layer's row. Decode on both layers,
    then a wide batch (bank + host-view partitions), then decode again on
    the same pool; every output must match the uncached layer."""
    device = torch.accelerator.current_accelerator()
    # One config per layer and side: a layer registers in the static forward
    # context of the config it was built under, which set_forward_context
    # must see again at run time (the legacy lookup also resolves layers by
    # call order within a config, so two layers never share one).
    ref_cfgs, pool_cfgs, refs, layers = [], [], [], []
    for seed_offset in (0, 1):
        params = quantized_weights(device, seed_offset=seed_offset)
        ref_cfgs.append(vllm_config(0))
        pool_cfgs.append(vllm_config(SLOTS))
        refs.append(make_layer(ref_cfgs[-1], params))
        layers.append(make_layer(pool_cfgs[-1], params, host_source=True))
    # As after the real loader: the small per-expert globals stay on the
    # device (never allocated in host memory), the big tensors are pinned.
    for layer in layers:
        for name in ("w13_weight_scale_2", "w2_weight_scale_2"):
            p = getattr(layer.routed_experts, name)
            p.data = p.data.to(device)
        # A non-contiguous (strided) host source must be densified into the
        # pool's own pinned copy, values preserved, not stride-preserved.
        p = layer.routed_experts.w2_weight_scale
        wide = torch.zeros((p.shape[0], 2, *p.shape[1:]), dtype=p.dtype).pin_memory()
        wide[:, 0].copy_(p.data)
        strided_values = p.data.clone()
        p.data = wide[:, 0]
        assert not p.data.is_contiguous()
    model = torch.nn.ModuleDict({"a": layers[0], "b": layers[1]})
    pool = install_expert_pool(model, device, max_decode_tokens=1)
    assert pool is not None
    # The pool owns pinned host copies of every source; the initial bank rows
    # match them byte for byte.
    for pl in (layer.routed_experts.expert_pool_layer for layer in layers):
        assert all(
            t.device.type == "cpu" and t.is_pinned() and t.is_contiguous()
            for t in pl.sources.values()
        )
    src = layers[-1].routed_experts.expert_pool_layer.sources["w2_weight_scale"]
    assert src.shape == strided_values.shape and src.dtype == strided_values.dtype
    assert torch.equal(src, strided_values)
    report = verify_bank_rows(pool, model.expert_pool_sources, sample=SLOTS)
    assert report == {"rows_checked": 2 * SLOTS, "rows_resident": 2 * SLOTS}
    assert pool.rows == 2 * SLOTS + TOP_K and pool.rows > E
    assert resident_per_layer(pool.tables) == [SLOTS, SLOTS]
    pls = [layer.routed_experts.expert_pool_layer for layer in layers]
    assert all(pl is not None and pl.bank_rows == pool.rows for pl in pls)
    check_global_tables(pool.tables)
    from vllm.forward_context import set_forward_context

    def run(i, x, logits, n):
        with set_forward_context(None, ref_cfgs[i], num_tokens=n):
            want = refs[i](x, logits)
        with set_forward_context(None, pool_cfgs[i], num_tokens=n):
            got = layers[i](x, logits)
        torch.accelerator.synchronize(device)
        torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)
        check_global_tables(pool.tables)
        assert int(pool.tables.error[0]) == 0

    x = torch.randn(1, K, dtype=torch.bfloat16, device=device)
    tables = pool.tables
    # Gate closed: a miss is staged into a shared staging row (physical row
    # >= E) and the step map must point there; the output still matches.
    set_gate(tables, False)
    run(0, x, _decode([6, 7], device), 1)  # experts 6, 7 are not resident
    step_map = pls[0].buffers.step_map.cpu().tolist()
    assert step_map[6] >= E and step_map[7] >= E, step_map
    assert resident_per_layer(tables) == [SLOTS, SLOTS]  # placement untouched
    set_gate(tables, True)
    # Decode steps whose routes walk every expert of both layers: misses
    # promote (evicting the least recently used row of either layer) or
    # stage into the shared staging rows.
    for order in ([0, 1], [4, 5], [6, 7], [2, 3], [0, 6], [7, 1]):
        run(0, x, _decode(order, device), 1)
    hot0_before = tables.layer_slice(tables.hot_phys, 0).cpu().clone()
    row_key_before = tables.row_key.cpu().clone()
    for order in ([4, 5], [6, 7], [2, 6]):
        run(1, x, _decode(order, device), 1)
    # Cross-layer eviction: layer 1's misses took rows from layer 0.
    hot0_after = tables.layer_slice(tables.hot_phys, 0).cpu()
    assert not torch.equal(hot0_before, hot0_after)
    row_key_after = tables.row_key.cpu()
    changed = (row_key_before != row_key_after).nonzero().flatten().tolist()
    assert changed and all(
        int(row_key_before[r]) // E == 0 and int(row_key_after[r]) // E == 1
        for r in changed
    ), (row_key_before.tolist(), row_key_after.tolist())
    assert sum(resident_per_layer(tables)) == 2 * SLOTS
    # Wide batch on layer 0: resident rows from the bank, the rest through
    # the host view; every route covered exactly once.
    xb = torch.randn(8, K, dtype=torch.bfloat16, device=device)
    run(0, xb, routing(list(range(E)), device), 8)
    assert pls[0].partition_steps == 1
    # Decode again on the same pool after the wide batch.
    for order in ([3, 4], [7, 0]):
        run(0, x, _decode(order, device), 1)
    run(1, x, _decode([0, 1], device), 1)
    assert pls[0].decode_steps == 9 and pls[1].decode_steps == 4  # 1 gate-closed


def _three_layer_pool(device):
    """Three pool layers sharing one bank, as install_expert_pool builds them."""
    from vllm.model_executor.layers.fused_moe.expert_pool.tables import set_gate

    pool_cfgs, layers = [], []
    for seed_offset in (0, 1, 2):
        params = quantized_weights(device, seed_offset=seed_offset)
        pool_cfgs.append(vllm_config(SLOTS))
        layers.append(make_layer(pool_cfgs[-1], params, host_source=True))
    for layer in layers:
        for name in ("w13_weight_scale_2", "w2_weight_scale_2"):
            p = getattr(layer.routed_experts, name)
            p.data = p.data.to(device)
    model = torch.nn.ModuleDict({"a": layers[0], "b": layers[1], "c": layers[2]})
    pool = install_expert_pool(model, device, max_decode_tokens=1)
    assert pool is not None
    set_gate(pool.tables, True)
    pls = [layer.routed_experts.expert_pool_layer for layer in layers]
    return pool, pls, pool_cfgs


def test_invalid_lanes_are_padding_and_the_error_is_sticky_without_firing(
    dist_env,  # noqa: F811
):
    """Device planner contract, checked with the step alone (no consumer, so
    no device assertion fires and the CUDA context stays usable): an
    out-of-range id, a negative non-sentinel id, and a non-finite router
    weight each set the sticky error and the `ok` flag, leave the tables
    identical to the same step with the lane as padding, route the other
    lanes, and hide the lane in safe_ids; the error persists across later
    clean steps until clear_error."""
    from vllm.model_executor.layers.fused_moe.expert_pool.tables import (
        check_global_tables,
        clear_error,
        step,
    )

    device = torch.accelerator.current_accelerator()
    pool, pls, _ = _three_layer_pool(device)
    ref_pool, ref_pls, _ = _three_layer_pool(device)
    tables, ref_tables = pool.tables, ref_pool.tables

    def run(p, layer, ids, weights):
        step(
            p.pool.tables,
            layer,
            torch.tensor([ids], dtype=torch.int32, device=device),
            p.buffers,
            torch.tensor([weights], dtype=torch.float32, device=device),
        )
        torch.accelerator.synchronize(device)

    for layer, ids, weights, ref_ids in (
        (0, [E + 3, 2], [0.5, 0.5], [-1, 2]),
        (1, [1, -9], [0.5, 0.5], [1, -1]),
        (2, [1, 2], [float("nan"), 0.5], [-1, 2]),
    ):
        run(pls[layer], layer, ids, weights)
        run(ref_pls[layer], layer, ref_ids, [0.5, 0.5])
        for name in ("hot_phys", "cold_phys", "row_key"):
            assert torch.equal(getattr(tables, name), getattr(ref_tables, name)), name
        assert torch.equal(pls[layer].buffers.routes, ref_pls[layer].buffers.routes)
        assert pls[layer].buffers.safe_ids[:2].tolist() == ref_ids
        assert int(tables.error[0]) == 1 and not bool(tables.ok[0])
        assert int(ref_tables.error[0]) == 0 and bool(ref_tables.ok[0])
        # Sticky across clean steps on other layers.
        for other in range(3):
            run(pls[other], other, [1, 2], [0.5, 0.5])
            run(ref_pls[other], other, [1, 2], [0.5, 0.5])
        assert int(tables.error[0]) == 1 and not bool(tables.ok[0])
        with pytest.raises(RuntimeError):
            check_global_tables(tables)
        clear_error(tables)
        check_global_tables(tables)
        check_global_tables(ref_tables)


def test_decode_graph_capture_and_replay_match_eager(dist_env):  # noqa: F811
    """The decode path (step, copy, sanitized routes, one-launch assertion,
    Marlin) is captured once and replayed with different valid inputs; every
    replay matches the eager result on identically placed twin pools."""
    from vllm.forward_context import set_forward_context

    device = torch.accelerator.current_accelerator()
    pool, pls, cfgs = _three_layer_pool(device)
    twin, tpls, tcfgs = _three_layer_pool(device)
    x_static = torch.zeros(1, K, dtype=torch.bfloat16, device=device)
    ids_static = torch.zeros(1, TOP_K, dtype=torch.int32, device=device)
    w_static = torch.full((1, TOP_K), 0.5, dtype=torch.float32, device=device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with (
        torch.cuda.stream(stream),
        set_forward_context(None, cfgs[0], num_tokens=1),
    ):
        for _ in range(2):  # warm up the kernels before capture
            pls[0].apply(x_static, w_static, ids_static)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with (
        set_forward_context(None, cfgs[0], num_tokens=1),
        torch.cuda.graph(graph, stream=stream),
    ):
        out_static = pls[0].apply(x_static, w_static, ids_static)
    # Mirror only the two warm-up steps on the twin: capture records the
    # kernels without executing them, so the placements agree here.
    with set_forward_context(None, tcfgs[0], num_tokens=1):
        for _ in range(2):
            tpls[0].apply(x_static, w_static, ids_static)
    torch.accelerator.synchronize(device)
    for name in ("hot_phys", "row_key"):
        assert torch.equal(getattr(pool.tables, name), getattr(twin.tables, name))
    for order in ([3, 4], [7, 0], [0, 0], [6, 1]):
        x = torch.randn(1, K, dtype=torch.bfloat16, device=device)
        ids = torch.tensor([order], dtype=torch.int32, device=device)
        x_static.copy_(x)
        ids_static.copy_(ids)
        graph.replay()
        with set_forward_context(None, tcfgs[0], num_tokens=1):
            want = tpls[0].apply(x, w_static, ids)
        torch.accelerator.synchronize(device)
        torch.testing.assert_close(out_static, want, rtol=2e-2, atol=2e-2)
        assert bool(pool.tables.ok[0]) and bool(twin.tables.ok[0])
        # The replayed step moved the placement exactly as the eager step.
        for name in ("hot_phys", "row_key"):
            assert torch.equal(getattr(pool.tables, name), getattr(twin.tables, name))
        assert torch.equal(pls[0].buffers.step_map, tpls[0].buffers.step_map)
        assert torch.equal(pls[0].buffers.safe_ids, tpls[0].buffers.safe_ids)


def test_step_graph_keeps_the_error_sticky_across_replays(dist_env):  # noqa: F811
    """The planner step alone (no consumer, so no assertion fires) captured
    in a CUDA graph: a replay with an invalid id sets the sticky error, a
    later clean replay keeps it, clear_error resets it, and the placement
    matches a twin stepped with padding throughout."""
    from vllm.model_executor.layers.fused_moe.expert_pool.tables import (
        check_global_tables,
        clear_error,
        step,
    )

    device = torch.accelerator.current_accelerator()
    pool, pls, _ = _three_layer_pool(device)
    twin, tpls, _ = _three_layer_pool(device)
    ids_static = torch.tensor([[1, 2]], dtype=torch.int32, device=device)
    w_static = torch.full((1, TOP_K), 0.5, dtype=torch.float32, device=device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step(pool.tables, 0, ids_static, pls[0].buffers, w_static)
    torch.cuda.current_stream().wait_stream(stream)
    step(twin.tables, 0, ids_static, tpls[0].buffers, w_static)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        step(pool.tables, 0, ids_static, pls[0].buffers, w_static)
    torch.accelerator.synchronize(device)

    def replay(ids, twin_ids):
        ids_static.copy_(torch.tensor([ids], dtype=torch.int32, device=device))
        graph.replay()
        step(
            twin.tables,
            0,
            torch.tensor([twin_ids], dtype=torch.int32, device=device),
            tpls[0].buffers,
            w_static,
        )
        torch.accelerator.synchronize(device)
        for name in ("hot_phys", "row_key"):
            assert torch.equal(getattr(pool.tables, name), getattr(twin.tables, name))
        assert torch.equal(pls[0].buffers.safe_ids, tpls[0].buffers.safe_ids)

    replay([3, 4], [3, 4])
    assert bool(pool.tables.ok[0])
    replay([E + 5, 4], [-1, 4])  # invalid lane: padding for the twin
    assert int(pool.tables.error[0]) == 1 and not bool(pool.tables.ok[0])
    replay([5, 6], [5, 6])  # clean replay keeps the sticky error
    assert int(pool.tables.error[0]) == 1 and not bool(pool.tables.ok[0])
    with pytest.raises(RuntimeError):
        check_global_tables(pool.tables)
    clear_error(pool.tables)
    replay([7, 0], [7, 0])
    assert bool(pool.tables.ok[0])
    check_global_tables(pool.tables)


def test_consumer_with_clamp_activation_only_sees_sanitized_routes(
    dist_env,  # noqa: F811
):
    """The activation path with a clamp limit indexes the step map by
    expert id (masked only by >= 0). The consumer receives the kernel's
    safe_ids, so a step with an out-of-range or negative id produces the
    same output as the same step with that lane as padding (the oracle);
    checked by calling the consumer directly, without the assertion."""
    import dataclasses

    from vllm.model_executor.layers.fused_moe.expert_pool.tables import (
        clear_error,
        step,
    )

    device = torch.accelerator.current_accelerator()
    pool, pls, _ = _three_layer_pool(device)
    twin, tpls, _ = _three_layer_pool(device)
    for p in (pls[0], tpls[0]):
        cfg = p.experts.activation_config
        p.experts.activation_config = dataclasses.replace(cfg, clamp_limit=7.0)
    x = torch.randn(1, K, dtype=torch.bfloat16, device=device)
    w = torch.full((1, TOP_K), 0.5, dtype=torch.float32, device=device)
    for bad, oracle in (
        ([E + 3, 2], [-1, 2]),
        ([1, -9], [1, -1]),
        ([E + 1, E + 2], [-1, -1]),
    ):
        ids = torch.tensor([bad], dtype=torch.int32, device=device)
        step(pool.tables, 0, ids, pls[0].buffers, w)
        step(
            twin.tables,
            0,
            torch.tensor([oracle], dtype=torch.int32, device=device),
            tpls[0].buffers,
            w,
        )
        safe = pls[0].buffers.safe_ids[: ids.numel()].view(ids.shape)
        got = pls[0]._run_marlin(
            x,
            w,
            safe,
            ((pls[0].bank, pls[0].buffers.step_map, pls[0].bank_rows),),
            decode=True,
        )
        want = tpls[0]._run_marlin(
            x,
            w,
            torch.tensor([oracle], dtype=torch.int32, device=device),
            ((tpls[0].bank, tpls[0].buffers.step_map, tpls[0].bank_rows),),
            decode=True,
        )
        torch.accelerator.synchronize(device)
        assert safe.tolist() == [oracle]
        torch.testing.assert_close(got, want, rtol=0, atol=0)
        assert int(pool.tables.error[0]) == 1
        clear_error(pool.tables)


ASSERT_CASES = {
    "first_layer_oob_id": (0, [E + 3, 2], [0.5, 0.5], "eager"),
    "middle_layer_nan_weight": (1, [1, 2], [float("nan"), 0.5], "eager"),
    "last_layer_negative_id": (2, [1, -9], [0.5, 0.5], "eager"),
    "single_layer_partial_oob_id": (1, [E + 3, 2], [0.5, 0.5], "single"),
    "graph_replay_oob_id": (0, [E + 3, 2], [0.5, 0.5], "graph"),
}


def _run_assert_case(name):
    """Subprocess body: one invalid decode forward must fail at the caller's
    synchronization (device assertion). Exits 0 only when it did."""
    from tests.kernels.moe.modular_kernel_tools.parallel_utils import _set_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        is_workspace_manager_initialized,
    )

    cfg = vllm_config(0)
    _set_vllm_config(cfg, 1, rank=0, local_rank=0)
    device = torch.accelerator.current_accelerator()
    if not is_workspace_manager_initialized():
        init_workspace_manager(device)
    pool, pls, cfgs = _three_layer_pool(device)
    bad_layer, bad_ids, bad_w, mode = ASSERT_CASES[name]
    x = torch.randn(1, K, dtype=torch.bfloat16, device=device)
    good_ids = torch.tensor([[1, 2]], dtype=torch.int32, device=device)
    good_w = torch.full((1, TOP_K), 0.5, dtype=torch.float32, device=device)
    # A clean full forward first.
    for i in range(3):
        with set_forward_context(None, cfgs[i], num_tokens=1):
            pls[i].apply(x, good_w, good_ids)
    torch.accelerator.synchronize(device)
    ids = torch.tensor([bad_ids], dtype=torch.int32, device=device)
    w = torch.tensor([bad_w], dtype=torch.float32, device=device)
    # Clean setup, capture and clean replay run outside the guarded region:
    # any failure there is a real failure of this test.
    graph = None
    ids_static = None
    if mode == "graph":
        ids_static = good_ids.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with (
            torch.cuda.stream(stream),
            set_forward_context(None, cfgs[0], num_tokens=1),
        ):
            pls[0].apply(x, good_w, ids_static)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with (
            set_forward_context(None, cfgs[0], num_tokens=1),
            torch.cuda.graph(graph, stream=stream),
        ):
            pls[0].apply(x, good_w, ids_static)
        graph.replay()
        torch.accelerator.synchronize(device)  # clean replay passes
    # Only the invalid input and its synchronization may raise, and only
    # with the device-side assertion; anything else is a failure.
    try:
        if mode == "eager":
            for i in range(3):
                with set_forward_context(None, cfgs[i], num_tokens=1):
                    pls[i].apply(
                        x,
                        w if i == bad_layer else good_w,
                        ids if i == bad_layer else good_ids,
                    )
            torch.accelerator.synchronize(device)
        elif mode == "single":
            # Partial execution: one non-final layer, then synchronize.
            with set_forward_context(None, cfgs[bad_layer], num_tokens=1):
                pls[bad_layer].apply(x, w, ids)
            torch.accelerator.synchronize(device)
        else:
            assert graph is not None and ids_static is not None
            ids_static.copy_(ids)  # invalid input into the captured buffer
            graph.replay()
            torch.accelerator.synchronize(device)
    except RuntimeError as exc:
        # Leave from inside the handler: once the device assertion fired the
        # CUDA context is poisoned, and unwinding out of this frame frees
        # pinned/device tensors whose release re-raises and aborts the
        # process (SIGABRT). The verdict is printed and flushed first.
        text = str(exc)
        if "device-side assert" in text or "Expert pool: invalid routing" in text:
            print(f"expected device assertion: {text[:160]}")
            _exit_now(0)
        print(f"unexpected error: {text[:300]}")
        _exit_now(4)
    print("no failure raised")
    _exit_now(3)


def _exit_now(code):
    """Exit without running destructors (see _run_assert_case)."""
    import os

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


@pytest.mark.parametrize("name", sorted(ASSERT_CASES))
def test_invalid_routing_fails_at_synchronization_in_a_subprocess(
    dist_env,  # noqa: F811
    name,
):
    """The one-launch device assertion fires for an invalid lane at the
    first, a middle, or the last layer, and inside a captured graph on
    replay; each case runs in its own process because a device assertion
    poisons the CUDA context."""
    proc = subprocess.run(
        [sys.executable, __file__, "--assert-case", name],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (name, proc.stdout[-3000:], proc.stderr[-6000:])
    assert "expected device assertion" in proc.stdout


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--assert-case", required=True, choices=sorted(ASSERT_CASES))
    _run_assert_case(ap.parse_args().assert_case)  # exits from inside
