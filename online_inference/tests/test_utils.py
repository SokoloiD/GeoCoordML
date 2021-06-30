import pytest
import numpy as np
from online_inference.src.model.utils.transformers import GpsToPixelTransformer


def test_gps_to_pix_transformer():
    transformer = GpsToPixelTransformer()
    transformer.fit([[0, 0], [1, 0], [0, 1]],
                    [[0, 0], [1, 0], [0, 1]])
    assert np.allclose(transformer.rotate_matrix, np.eye(3), rtol=1e-05, atol=1e-08)
    transformer.fit([[55.85032, 48.89209], [55.80093, 49.31117], [55.63561, 49.10654]],
                    [[4070, 8147], [24224, 12310], [14768, 26056]])

    assert np.allclose(transformer.transform([55.80093, 49.31117]),
                       np.array([24224, 12310]),
                       rtol=1e-05, atol=1e-08)
