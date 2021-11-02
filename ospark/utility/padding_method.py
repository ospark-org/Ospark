from enum import Enum
from typing import Optional
import numpy as np


class PaddingMethod(Enum):

    align_image_center = lambda image, pad_height, pad_width:\
                        np.pad(image, np.array([[np.ceil(pad_height / 2), np.floor(pad_height / 2)],
                                                [np.ceil(pad_width / 2), np.floor(pad_width / 2)],
                                                [0, 0]]).astype(np.int32))
    align_upper_left   = lambda image, pad_height, pad_width:\
                        np.pad(image, np.array([[0, pad_height],
                                                [0, pad_width],
                                                [0, 0]]).astype(np.int32))
    align_upper_right  = lambda image, pad_height, pad_width:\
                        np.pad(image, np.array([[0, pad_height],
                                                [pad_width, 0],
                                                [0, 0]]).astype(np.int32))
    align_lower_left   = lambda image, pad_height, pad_width:\
                        np.pad(image, np.array([[pad_height, 0],
                                                [0, pad_width],
                                                [0, 0]]).astype(np.int32))
    align_lower_right  = lambda image, pad_height, pad_width:\
                        np.pad(image, np.array([[pad_height, 0],
                                                [pad_width, 0],
                                                [0, 0]]).astype(np.int32))


class PaddingManager:

    def __init__(self, padding_method: Optional[str]=None):
        """
        Padding method

        Args:
             padding_method: Optional[str]
                Method has center, upper_left, upper_right, lower_left, lower_right. Default use "center".
        """
        if padding_method == "center" or padding_method is None:
            self._padding = lambda height, width: np.array([[np.ceil(height / 2), np.floor(height / 2)],
                                                            [np.ceil(width / 2), np.floor(width / 2)],
                                                            [0, 0]]).astype(np.int32)
        else:
            y_align, x_align = padding_method.split("_")
            if y_align == "upper":
                y_axis_pad = lambda height: [0, height]
            elif y_align == "lower":
                y_axis_pad = lambda height: [height, 0]
            else:
                raise NameError(f"Check if input {y_align} is wrong")

            if x_align == "left":
                x_axis_pad = lambda width: [0, width]
            elif x_align == "right":
                x_axis_pad = lambda width: [width, 0]
            else:
                raise NameError(f"Check if input {x_align} is wrong")

            self._padding = lambda padding_scale: np.array([y_axis_pad(padding_scale[1]),
                                                            x_axis_pad(padding_scale[0]),
                                                            [0, 0]]).astype(np.int32)

    @property
    def padding(self) -> str:
        return self._padding

    def __call__(self, image: np.ndarray, padding_scale: np.ndarray) -> np.ndarray:
        return np.pad(array=image, pad_width=self.padding(padding_scale))