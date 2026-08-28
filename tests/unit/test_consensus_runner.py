import numpy as np

from lspe.models.base import ForwardResult
from lspe.networks.consensus_runner import (
    ConsensusMappingProtocol,
    _category_rows,
    _select_candidate,
)
from lspe.networks.mapping_runner import _fixed_continuations, _stable_nonisolated_nodes


class _Prompt:
    def __init__(self, prompt_id: str, category: str) -> None:
        self.prompt_id = prompt_id
        self.category = category
        self.prompt = "test prompt"
        self.pair_kind = None
        self.pair_id = None
        self.pair_member = None


class _Adapter:
    def format_prompt(self, messages):
        return [10]

    def forward(self, tokens):
        logits = np.zeros((1, len(tokens), 32), dtype=np.float32)
        logits[0, -1, 20 + len(tokens)] = 5.0
        return ForwardResult(logits=logits, hidden_summaries={}, cache=None)

    def decode(self, token_ids):
        return str(token_ids[0])


def test_category_rows_preserve_four_positions_and_stratified_folds() -> None:
    prompts = [_Prompt(f"p{index}", "a") for index in range(40)]
    continuations = [
        {"prompt_id": prompt.prompt_id, "row_index": row}
        for row, prompt in enumerate(prompt for prompt in prompts for _ in range(4))
    ]
    rows, folds = _category_rows(prompts, continuations)
    assert np.array_equal(rows["a"], np.arange(160))
    assert all(len(fold) == 40 for fold in folds["a"])


def test_category_folds_keep_paraphrases_together() -> None:
    prompts = [_Prompt(f"p{index}", "a") for index in range(40)]
    prompts[0].pair_kind = prompts[1].pair_kind = "paraphrase"
    prompts[0].pair_id = prompts[1].pair_id = "pair-1"
    continuations = [
        {"prompt_id": prompt.prompt_id, "row_index": row}
        for row, prompt in enumerate(prompt for prompt in prompts for _ in range(4))
    ]
    _, folds = _category_rows(prompts, continuations)
    first_rows = set(range(0, 4))
    second_rows = set(range(4, 8))
    containing = [set(fold) for fold in folds["a"]]
    assert any(first_rows <= fold and second_rows <= fold for fold in containing)


def test_consensus_selection_never_uses_heldout_ari() -> None:
    selected = _select_candidate(
        [
            {
                "density": 0.1,
                "community_count": 4,
                "eligible_nodes": 50,
                "tuning_ari": 0.8,
                "heldout_ari": 0.1,
            },
            {
                "density": 0.2,
                "community_count": 3,
                "eligible_nodes": 50,
                "tuning_ari": 0.7,
                "heldout_ari": 0.99,
            },
        ]
    )
    assert selected["density"] == 0.1


def test_v2_continuation_plan_observes_two_positions_per_mode(tmp_path) -> None:
    rows = _fixed_continuations(
        _Adapter(),
        [_Prompt("p0", "a")],
        ConsensusMappingProtocol(),
        tmp_path,
    )
    assert [(row["mode"], row["generated_position"]) for row in rows] == [
        ("greedy", 0),
        ("greedy", 1),
        ("sampled", 0),
        ("sampled", 1),
    ]
    assert [len(row["observation_token_ids"]) for row in rows] == [2, 3, 2, 3]


def test_final_fit_prunes_isolates_from_tuning_eligible_graph() -> None:
    graph = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    assert np.array_equal(_stable_nonisolated_nodes([graph]), [0, 1])
