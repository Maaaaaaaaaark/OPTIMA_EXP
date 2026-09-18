import json
from types import SimpleNamespace

from train import generate


class SequentialLoader:
    data_type = "qa"
    dataset_name = "fake"

    def __init__(self):
        self.index = 0

    def sample_once(self, rng=None):
        index = self.index
        self.index += 1
        return f"question-{index}", f"answer-{index}", f"a-{index}", f"b-{index}"


def test_resume_advances_loader_past_completed_tasks(tmp_path, monkeypatch):
    raw_path = tmp_path / "iteration_0.jsonl"
    raw_path.write_text(
        json.dumps({"task_id": 0, "results": []}) + "\n", encoding="utf-8"
    )
    cfg = SimpleNamespace(
        sample_count=2,
        thread_count=1,
        seed=42,
        raw_path=lambda iteration: str(raw_path),
    )

    generated = []

    def fake_generate_one_task(cfg, task, iteration, output_path):
        generated.append(task)
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"task_id": task.task_id, "results": []}) + "\n")

    monkeypatch.setattr(generate, "generate_one_task", fake_generate_one_task)
    loader = SequentialLoader()

    generate.generate_all(cfg, loader, iteration=0)

    assert loader.index == 2
    assert len(generated) == 1
    assert generated[0].task_id == 1
    assert generated[0].question == "question-1"
