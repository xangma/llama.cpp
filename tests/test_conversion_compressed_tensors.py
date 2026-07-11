from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from conversion.base import ModelBase


class TestMixedCompressedTensors(unittest.TestCase):
    def test_selects_fp8_layer_override(self) -> None:
        # This deliberately reverses group insertion order: target priority,
        # rather than dictionary order, must select the layer-specific FP8 group.
        groups = {
            "group_1": {
                "format": "nvfp4-pack-quantized",
                "targets": [
                    r"re:.*mlp\.experts\.\d+\.(gate|up|down)_proj$",
                ],
            },
            "group_0": {
                "format": "float-quantized",
                "targets": [
                    r"re:.*layers\.(32|33|34|35|36|37|38|39)\.mlp\.experts\.\d+\.(gate|up|down)_proj$",
                ],
            },
        }

        fp8 = ModelBase._get_compressed_tensors_group(
            groups, "model.layers.32.mlp.experts.0.down_proj.weight"
        )
        nvfp4 = ModelBase._get_compressed_tensors_group(
            groups, "model.layers.31.mlp.experts.0.down_proj.weight"
        )

        self.assertIs(fp8, groups["group_0"])
        self.assertEqual(fp8["format"], "float-quantized")
        self.assertIs(nvfp4, groups["group_1"])
        self.assertEqual(nvfp4["format"], "nvfp4-pack-quantized")


if __name__ == "__main__":
    unittest.main()
