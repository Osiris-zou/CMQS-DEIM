import ast
import unittest
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]

class RuntimeWiringTest(unittest.TestCase):
    def test_engine_passes_epoch_in_both_training_branches(self):
        tree = ast.parse((ROOT / 'engine/solver/det_engine.py').read_text(encoding='utf-8'))
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name)
                 and node.func.id == 'model'
                 and any(kw.arg == 'epoch' for kw in node.keywords)]
        self.assertGreaterEqual(len(calls), 2)

    def test_model_forwards_epoch_to_decoder(self):
        text = (ROOT / 'engine/deim/deim.py').read_text(encoding='utf-8')
        self.assertIn('self.decoder(x, targets, epoch=epoch)', text)

    def test_main_exit_epochs(self):
        for rel, expected in [
            ('configs/deim_dfine/deim-l-cmqs.yml', 10),
            ('configs/deim_dfine/deim-s-cmqs.yml', 24),
        ]:
            cfg = yaml.safe_load((ROOT / rel).read_text(encoding='utf-8'))
            self.assertEqual(cfg['DFINETransformer']['query_select_gt_stop_epoch'], expected)
            self.assertEqual(cfg['DFINETransformer']['query_select_cost_mode'], 'sum')

if __name__ == '__main__':
    unittest.main()
