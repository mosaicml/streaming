# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""ADE20K Semantic segmentation and scene parsing dataset.

Please refer to the `ADE20K dataset <https://groups.csail.mit.edu/vision/datasets/ADE20K/>`_ for
more details about this dataset.
"""

from typing import Any, Callable, Dict, Optional, Tuple

from streaming import StreamingDataset

__all__ = ['StreamingADE20K']


class StreamingADE20K(StreamingDataset):
    """Implementation of the ADE20K dataset using StreamingDataset.

    Args:
        joint_transform (Callable, optional): A function/transforms that takes in an image and a
            target  and returns the transformed versions of both. Defaults to ``None``.
        transform (Callable, optional): A function/transform that takes in an image and returns a
            transformed version. Defaults to ``None``.
        target_transform (Callable, optional): A function/transform that takes in the target and
            transforms it. Defaults to ``None``.
        **kwargs (Dict[str, Any]): Keyword arguments.
    """

    def __init__(self,
                 *,
                 joint_transform: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def get_item(self, idx: int) -> Tuple[Any, Any]:
        """Get sample by global index, blocking to load its shard if missing.

        Args:
            idx (int): Sample index.

        Returns:
            Tuple[Any, Any]: Sample data and label.
        """
        obj = super().get_item(idx)
        x = obj['x']
        y = obj['y']
        if self.joint_transform:
            x, y = self.joint_transform((x, y))
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y
