import unittest
import torch
from quantized_operations import QuantizedAvgPool

class TestQuantAvgPool(unittest.TestCase):
    def test_fwd(self):
        layer = QuantizedAvgPool()
        input = torch.tensor([[[[2.,4.],[6.,8.]],[[2.,4.],[6.,8.]]],[[[2.,4.],[6.,8.]],[[2.,4.],[6.,8.]]]])
        expected = torch.tensor([[5.,5.],[5.,5.]])
        out = layer.forward(input)
        torch.equal(out, expected)

if __name__ == "main":
    unittest.main()