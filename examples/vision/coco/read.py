# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""COCO (Common Objects in Context) dataset.

COCO is a large-scale object detection, segmentation, and captioning dataset. Please refer to the
`COCO dataset <https://cocodataset.org>`_ for more details.
"""

from typing import Any, Callable, Dict, Optional

from streaming import StreamingDataset

__all__ = ['StreamingCOCO']


class StreamingCOCO(StreamingDataset):
    """Implementation of the COCO dataset using StreamingDataset.

    Args:
        transform (callable, optional): A function/transform that takes in an image and bboxes and
            returns a transformed version. Defaults to ``None``.
        **kwargs (Dict[str, Any]): Keyword arguments.
    """

    def __init__(self, *, transform: Optional[Callable] = None, **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)
        self.transform = transform

    def get_item(self, idx: int) -> Any:
        """Get sample by global index, blocking to load its shard if missing.

        Args:
            idx (int): Sample index.

        Returns:
            Any: Sample data.
        """
        x = super().get_item(idx)
        img = x['img'].convert('RGB')
        img_id = x['img_id']
        htot = x['htot']
        wtot = x['wtot']
        bbox_sizes = x['bbox_sizes']
        bbox_labels = x['bbox_labels']
        if self.transform:
            img, (htot,
                  wtot), bbox_sizes, bbox_labels = self.transform(img, (htot, wtot), bbox_sizes,
                                                                  bbox_labels)
        return img, img_id, (htot, wtot), bbox_sizes, bbox_labels
