import unittest

import torch

from lcapt.analysis import make_feature_grid


class TestAnalysis(unittest.TestCase):
    def test_make_feature_grid_produces_correct_shape_3D_1channel(self):
        inputs = torch.randn(5, 1, 100)
        grid = make_feature_grid(inputs)
        self.assertEqual(grid.numpy().shape, (5, 100))

    def test_make_feature_grid_produces_correct_shape_3D_multi_channel(self):
        inputs = torch.randn(5, 3, 100)
        grid = make_feature_grid(inputs)
        self.assertEqual(grid.numpy().shape, (17, 206))

    def test_make_feature_grid_produces_correct_shape_4D(self):
        inputs = torch.randn(5, 3, 10, 10)
        grid = make_feature_grid(inputs)
        self.assertEqual(grid.numpy().shape, (38, 26, 3))

    def test_make_feature_grid_produces_correct_shape_5D_one_timestep(self):
        inputs = torch.randn(5, 3, 1, 10, 10)
        grid = make_feature_grid(inputs)
        self.assertEqual(grid.numpy().shape, (38, 26, 3))

    def test_make_feature_grid_produces_correct_shape_5D_multi_timestep(self):
        inputs = torch.randn(5, 3, 4, 10, 10)
        grid = make_feature_grid(inputs)
        self.assertEqual(grid.numpy().shape, (4, 38, 26, 3))


if __name__ == "__main__":
    unittest.main()
