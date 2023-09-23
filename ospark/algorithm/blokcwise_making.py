from typing import List, Tuple
import tensorflow as tf
import numpy as np
import random
import math


class BlockwiseMasking:
    """
    Blockwise masking algorithm.
    """

    def __init__(self, patch_matrix_shape: List[int], masking_ratio: float):
        """

        Args:
            patch_matrix_shape: List[int]
            masking_ratio:  float
        """

        self._patch_matrix_shape     = patch_matrix_shape

        self._patch_number   = patch_matrix_shape[0] * patch_matrix_shape[1]
        self._masking_ratio  = masking_ratio

    @property
    def patch_matrix_shape(self) -> List[int]:
        return self._patch_matrix_shape

    @property
    def patch_number(self) -> int:
        return self._patch_number

    @property
    def masking_ratio(self) -> float:
        return self._masking_ratio

    def pipeline(self, input_data: tf.Tensor) -> Tuple[tf.Tensor, List[List[int]]]:
        """
        Create image mask matrix.

        Args:
            input_data: tf.Tensor

        Returns:
            mask_matrices: tf.Tensor
            masking_indices: List[List[int]]
        """

        batch_size      = input_data.shape[0]
        masking_indices = []
        for i in range(batch_size):
            masking_index = set()
            while (len(masking_index) / self.patch_number) < self.masking_ratio:
                s = random.uniform(16, 0.4 * self.patch_number - len(masking_index))
                r = random.uniform(0.3, 1 / 0.3)
                a = math.floor(math.sqrt(s * r))
                b = math.floor(math.sqrt(s / r))
                t = random.randint(0, max(self.patch_matrix_shape[0] - a, 1))
                l = random.randint(0, max(self.patch_matrix_shape[1] - b, 1))

                start_index    = self.patch_matrix_shape[1] * t + l
                masking_index |= {start_index + i + j * self.patch_matrix_shape[1] for i in range(a) for j in range(b)}
                masking_index  = set(filter(lambda value: value < self._patch_number - 1, masking_index))
            masking_indices.append(list(masking_index))
        mask_matrices = tf.convert_to_tensor(self.create_mask_matrix(masking_indices=masking_indices))
        return mask_matrices, masking_indices

    def create_mask_matrix(self, masking_indices: List[set]) -> np.ndarray:
        """
        Create mask matrix.

        Args:
            masking_indices:

        Returns:
            mask_matrices: np.ndarray
        """

        mask_matrices = np.ones(shape=[len(masking_indices), self._patch_number]).astype(np.float32)
        for i, masking_index in enumerate(masking_indices):
            mask_matrices[i, masking_index] = 0
        return mask_matrices