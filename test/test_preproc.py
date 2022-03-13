import unittest

import torch
from torch.testing import assert_close

from lcapt.preproc import standardize_inputs


class TestPreprocessing(unittest.TestCase):
    def test_standardize_inputs_raises_NotImplementedError(self):
        with self.assertRaises(NotImplementedError):
            standardize_inputs(torch.zeros(1, 2, 3, 4, 5, 6))

    def test_standardize_inputs_3D_tensor(self):
        inputs = torch.rand(1, 3, 100) * 100
        outputs = standardize_inputs(inputs)
        self.assertEqual(outputs.shape, inputs.shape)
        for chann in range(3):
            self.assertLess(outputs[:, chann].mean().abs().item(), 5e-7)
            assert_close(outputs[:, chann].std().item(), 1.0)

    def test_standardize_inputs_4D_tensor(self):
        inputs = torch.rand(1, 3, 100, 100) * 100
        outputs = standardize_inputs(inputs)
        self.assertEqual(outputs.shape, inputs.shape)
        for chann in range(3):
            self.assertLess(outputs[:, chann].mean().abs().item(), 5e-7)
            assert_close(outputs[:, chann].std().item(), 1.0)

    def test_standardize_inputs_5D_tensor(self):
        inputs = torch.rand(1, 3, 4, 100, 100) * 100
        outputs = standardize_inputs(inputs)
        self.assertEqual(outputs.shape, inputs.shape)
        for chann in range(3):
            for depth in range(4):
                self.assertLess(outputs[:, chann, depth].mean().abs().item(), 5e-7)
                assert_close(outputs[:, chann, depth].std().item(), 1.0)

    def test_standardize_inputs_already_standardized(self):
        inputs = standardize_inputs(torch.rand(1, 3, 10000))
        outputs = standardize_inputs(inputs)
        assert_close(inputs, outputs)

    def test_standardize_inputs_returns_torch_tensor(self):
        inputs = torch.rand(1, 3, 4, 100, 100)
        outputs = standardize_inputs(inputs)
        self.assertEqual(type(outputs), torch.Tensor)


if __name__ == "__main__":
    unittest.main()
