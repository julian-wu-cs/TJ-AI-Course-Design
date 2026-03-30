"""Eight-Queens benchmark script used in the report appendix.

Implemented methods:
1) Imperative permutation generation + logic filtering
2) Pure logic programming via Kanren permuteq
3) Pure logic programming with custom neq relations
4) DFS over partial placements
5) BFS over partial placements
6) Bit-mask backtracking
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time
from collections import deque
from pathlib import Path
from typing import Callable

from kanren import eq, lall, membero, permuteq, run, var, vars
from unification import isvar, reify


Placement = tuple[int, ...]  # col -> row


def can_place(prefix: Placement, next_row: int) -> bool:
    """Return True if next_row is safe for the next column."""
    next_col = len(prefix)
    for old_col, old_row in enumerate(prefix):
        same_row = old_row == next_row
        same_diag = abs(old_row - next_row) == abs(old_col - next_col)
        if same_row or same_diag:
            return False
    return True


def diagonal_clear_goal(board_term):
    """Kanren relation: all queens in board_term avoid diagonal conflicts."""

    def goal(subst):
        board_value = reify(board_term, subst)
        if isvar(board_value):
            return (subst,)
        board_len = len(board_value)
        for left_col in range(board_len):
            for right_col in range(left_col + 1, board_len):
                if abs(board_value[left_col] - board_value[right_col]) == abs(left_col - right_col):
                    return ()
        return (subst,)

    return goal


def neq_rel(left, right):
    """Custom non-equality relation for Kanren 0.2.x."""

    def goal(subst):
        left_value = reify(left, subst)
        right_value = reify(right, subst)
        if isvar(left_value) or isvar(right_value):
            return (subst,)
        if left_value == right_value:
            return ()
        return (subst,)

    return goal


def diagonal_neq_rel(left, right, distance: int):
    """Custom relation: abs(left-right) must not equal distance."""

    def goal(subst):
        left_value = reify(left, subst)
        right_value = reify(right, subst)
        if isvar(left_value) or isvar(right_value):
            return (subst,)
        if abs(left_value - right_value) == distance:
            return ()
        return (subst,)

    return goal


def solve_imperative_then_logic(n: int) -> tuple[list[Placement], int]:
    domain = tuple(range(n))
    all_permutations = list(itertools.permutations(domain))
    symbol = var()
    solutions = run(0, symbol, membero(symbol, all_permutations), diagonal_clear_goal(symbol))
    return list(solutions), len(all_permutations)


def solve_pure_logic_permuteq(n: int) -> tuple[list[Placement], int]:
    domain = tuple(range(n))
    queens = tuple(vars(n))
    constraints = [permuteq(domain, queens), diagonal_clear_goal(queens)]
    solutions = run(0, queens, lall(*constraints))

    # Permutation space size: n!
    expanded_states = 1
    for value in range(2, n + 1):
        expanded_states *= value
    return list(solutions), expanded_states


def counted_member_rel(symbol, domain, counter):
    """Custom member relation with external counter for node statistics."""

    def goal(subst):
        for candidate in domain:
            counter[0] += 1
            yield from eq(symbol, candidate)(subst)

    return goal


def solve_pure_logic_custom_neq(n: int) -> tuple[list[Placement], int]:
    domain = tuple(range(n))
    queens = tuple(vars(n))
    counter = [0]
    constraints = []

    for col_idx in range(n):
        constraints.append(counted_member_rel(queens[col_idx], domain, counter))
        for prev_col in range(col_idx):
            constraints.append(neq_rel(queens[col_idx], queens[prev_col]))
            constraints.append(diagonal_neq_rel(queens[col_idx], queens[prev_col], col_idx - prev_col))

    solutions = run(0, queens, lall(*constraints))
    return list(solutions), counter[0]


def solve_dfs(n: int) -> tuple[list[Placement], int]:
    frontier: list[Placement] = [tuple()]
    solutions: list[Placement] = []
    expanded_states = 0

    while frontier:
        prefix = frontier.pop()
        expanded_states += 1
        if len(prefix) == n:
            solutions.append(prefix)
            continue
        # Reverse append to keep lexicographic order stable after stack pop.
        for row in range(n - 1, -1, -1):
            if can_place(prefix, row):
                frontier.append(prefix + (row,))

    return solutions, expanded_states


def solve_bfs(n: int) -> tuple[list[Placement], int]:
    frontier: deque[Placement] = deque([tuple()])
    solutions: list[Placement] = []
    expanded_states = 0

    while frontier:
        prefix = frontier.popleft()
        expanded_states += 1
        if len(prefix) == n:
            solutions.append(prefix)
            continue
        for row in range(n):
            if can_place(prefix, row):
                frontier.append(prefix + (row,))

    return solutions, expanded_states


def solve_bitmask_backtracking(n: int) -> tuple[list[Placement], int]:
    full_mask = (1 << n) - 1
    row_to_col = [0] * n
    solutions: list[Placement] = []
    expanded_states = 0

    def backtrack(row: int, col_mask: int, main_diag_mask: int, anti_diag_mask: int) -> None:
        nonlocal expanded_states
        expanded_states += 1

        if row == n:
            col_to_row = [0] * n
            for r, c in enumerate(row_to_col):
                col_to_row[c] = r
            solutions.append(tuple(col_to_row))
            return

        available_positions = full_mask & ~(col_mask | main_diag_mask | anti_diag_mask)
        while available_positions:
            lowest = available_positions & -available_positions
            available_positions ^= lowest

            chosen_col = lowest.bit_length() - 1
            row_to_col[row] = chosen_col
            backtrack(
                row + 1,
                col_mask | lowest,
                ((main_diag_mask | lowest) << 1) & full_mask,
                (anti_diag_mask | lowest) >> 1,
            )

    backtrack(0, 0, 0, 0)
    return solutions, expanded_states


def canonicalize(solutions: list[Placement]) -> list[Placement]:
    return sorted(tuple(sol) for sol in solutions)


def benchmark_method(
    solver: Callable[[int], tuple[list[Placement], int]],
    n: int,
    repeat: int,
) -> dict[str, float | int]:
    elapsed_samples: list[float] = []
    reference_solutions = None
    reference_count = 0
    stable_expanded_states = None

    for _ in range(repeat):
        started_at = time.perf_counter()
        solutions, expanded_states = solver(n)
        elapsed_samples.append(time.perf_counter() - started_at)

        normalized = canonicalize(solutions)
        if reference_solutions is None:
            reference_solutions = normalized
            reference_count = len(normalized)
            stable_expanded_states = expanded_states
        else:
            if normalized != reference_solutions:
                raise RuntimeError("The solver returned inconsistent solution sets across repeats.")
            if expanded_states != stable_expanded_states:
                raise RuntimeError("Expanded-state count changed across repeats.")

    return {
        "solutions": reference_count,
        "expanded_states": int(stable_expanded_states or 0),
        "time_mean_ms": round(statistics.mean(elapsed_samples) * 1000, 3),
        "time_std_ms": round(statistics.pstdev(elapsed_samples) * 1000, 3),
        "time_min_ms": round(min(elapsed_samples) * 1000, 3),
        "time_max_ms": round(max(elapsed_samples) * 1000, 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eight-Queens benchmark (6 methods).")
    parser.add_argument("--n", type=int, default=8, help="Board size, default 8.")
    parser.add_argument("--repeat", type=int, default=3, help="Benchmark repeats per method.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/eight_queens_benchmark.json"),
        help="Path to benchmark JSON output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    methods: list[tuple[str, Callable[[int], tuple[list[Placement], int]]]] = [
        ("imperative_plus_logic", solve_imperative_then_logic),
        ("logic_permuteq", solve_pure_logic_permuteq),
        ("logic_custom_neq", solve_pure_logic_custom_neq),
        ("dfs", solve_dfs),
        ("bfs", solve_bfs),
        ("bitwise_backtracking", solve_bitmask_backtracking),
    ]

    result = {"n": args.n, "repeat": args.repeat, "methods": {}}
    for method_name, method_impl in methods:
        result["methods"][method_name] = benchmark_method(method_impl, args.n, args.repeat)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
