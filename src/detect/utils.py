import numpy as np


def xyxy2cxcywh(bboxes: np.ndarray):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] / 2
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] / 2
    return bboxes


def cxcywh2xyxy(bboxes: np.ndarray):
    bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2
    bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2
    bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1]
    return bboxes


def xyxy2xywh(bboxes: np.ndarray):
    res = bboxes.copy()
    res[:, 2] = res[:, 2] - res[:, 0]
    res[:, 3] = res[:, 3] - res[:, 1]
    return res


class Tiler:
    """TIle image with overlapped tiles.

    Args:
        img_size (tuple[int, int]): Input size of the image (W, H)
        tile_size (tuple[int, int]): Size of the crops (W, H)
        overlap (float, optional): Minimal desired overlap as a fraction of the
            crop side length. Defaults to 0.0.
    """

    def __init__(self, img_size,
                 tile_size,
                 overlap: float = 0.0):
        self.img_size = img_size
        self.tile_size = tile_size
        self.overlap = overlap
        self.grid = (1, 1)
        self.tiles = self._get_tiles()

    def __len__(self) -> int:
        return len(self.tiles)

    @staticmethod
    def _tile_1D(img_size: int, tile_size: int, overlap: float = 0.0):
        n_tiles = 1
        overlap_px = 0.0
        if img_size > tile_size:
            n_tiles = int(np.ceil(img_size*(1.0 + overlap) / tile_size))
            overlap_px = (n_tiles * tile_size-img_size) / (n_tiles - 1)
        step = int(np.floor(tile_size - overlap_px))
        return step, n_tiles

    def _get_tiles(self):
        w, h = self.img_size
        w_tile, h_tile = self.tile_size
        w_step, n_col = self._tile_1D(w, w_tile, self.overlap)
        h_step, n_row = self._tile_1D(h, h_tile, self.overlap)
        tile_coords = [(j * w_step, i * h_step)
                       for i in range(n_row) for j in range(n_col)]
        self.grid = (n_col, n_row)
        return tile_coords

    def __getitem__(self, idx: int):
        assert idx < len(self)
        return self.tiles[idx]
