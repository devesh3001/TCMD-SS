import tempfile
import unittest
from pathlib import Path
from scripts.prepare_revision import disjoint, preserve_write
from src.generative.freeze import freeze, claim_final


class ProtocolTests(unittest.TestCase):
    def test_linked_groups_cannot_cross(self):
        a = dict(source_id="a", base_id="a", group_id="a", split_group_id="linked")
        b = dict(source_id="b", base_id="b", group_id="b", split_group_id="linked")
        with self.assertRaises(ValueError):
            disjoint({"train": [a], "test": [b]})

    def test_preserves_experiments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"result.json"
            preserve_write(path, "old")
            preserve_write(path, "old")
            with self.assertRaises(FileExistsError):
                preserve_write(path, "new")
            self.assertEqual(path.read_text(), "old")

    def test_final_claim_is_once_and_content_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root/"model.txt"
            model.write_text("weights")
            decisions = dict(architecture="test", dino_layers=[11], top_fraction=.005,
                             cluster_count=0, fusion="max", sampling_steps=15, seeds=[42,1042],
                             threshold_procedure="clean B q95", development_complete=True)
            freeze(root, [model], decisions)
            model.write_text("changed")
            with self.assertRaises(ValueError):
                claim_final(root)
            model.write_text("weights")
            claim_final(root)
            with self.assertRaises(FileExistsError):
                claim_final(root)


if __name__ == "__main__":
    unittest.main()
