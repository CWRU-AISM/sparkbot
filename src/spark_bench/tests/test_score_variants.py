"""The curated name wins over a higher-scoring variant when it detects anything."""
import numpy as np
import torch

from spark_bench.libero_pro.perception import prompts as perception


class _Sam:
    def __init__(self, table):
        self.table = table

    def load_models(self, load_da3=False):
        pass

    class _Inner:
        def __init__(self, table):
            self.table = table

        def set_image(self, img):
            return None

        def set_text_prompt(self, prompt, state):
            n, conf = self.table.get(prompt, (0, 0.0))
            if n == 0:
                return {}
            masks = torch.zeros((n, 4, 4), dtype=torch.bool)
            masks[:, 1, 1] = True
            return {'masks': masks, 'scores': torch.full((n,), conf)}

    @property
    def _sam3(self):
        return self._Inner(self.table)


def test_name_wins_when_it_fires(monkeypatch):
    table = {'alphabet soup': (12, 0.14), 'Campbell soup can': (1, 0.95)}
    monkeypatch.setattr(perception, 'get_sam3', lambda: _Sam(table))
    chosen = perception._score_variants(np.zeros((4, 4, 3), np.uint8),
                                        {'alphabet soup': ['alphabet soup', 'Campbell soup can']}, 3)
    assert chosen[0][1] == 'alphabet soup'


def test_variant_used_when_name_silent(monkeypatch):
    table = {'black bowl': (0, 0.0), 'dark ceramic bowl': (1, 0.9)}
    monkeypatch.setattr(perception, 'get_sam3', lambda: _Sam(table))
    chosen = perception._score_variants(np.zeros((4, 4, 3), np.uint8),
                                        {'black bowl': ['dark ceramic bowl']}, 3)
    assert chosen[0][1] == 'dark ceramic bowl'


def test_dedupe_keeps_task_name_on_shared_object():
    chosen = [(0.05, 'bbq sauce', (100.0, 100.0)), (0.89, 'dark brown sauce bottle', (105.0, 102.0)),
              (0.9, 'basket', (400.0, 200.0))]
    out = perception._dedupe_chosen(chosen, instruction='pick up the bbq sauce and place it in the basket')
    assert out == ['bbq sauce', 'basket']
    assert perception._dedupe_chosen(chosen) == ['dark brown sauce bottle', 'basket']
