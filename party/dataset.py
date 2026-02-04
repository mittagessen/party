#
# Copyright 2015 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
Utility functions for data loading and training of VGSL networks.
"""
import io
import gc
import torch
import ctypes
import torch.nn.functional as F
import numpy as np

import tempfile
import pyarrow as pa

from pathlib import Path
from itertools import islice

from typing import (TYPE_CHECKING, Any, Callable, Literal, Optional, Tuple,
                    Union, Sequence)

from PIL import Image
from functools import partial
from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms import v2
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from torch.distributed import get_rank, get_world_size, is_initialized


from scipy.special import comb

from party.tokenizer import OctetTokenizer


if TYPE_CHECKING:
    from os import PathLike
    from kraken.containers import Segmentation

__all__ = ['BinnedBaselineDataset', 'ValidationBaselineDataset', 'get_default_transforms', 'collate_null']

import logging

logger = logging.getLogger(__name__)

try:
    import pillow_jxl # NOQA
except ImportError:
    logger.info('No JPEG-XL plugin found')

Image.MAX_IMAGE_PIXELS = 20000 ** 2


def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch


def get_default_transforms(image_size: tuple[int, int] = (2560, 1920), dtype=torch.float32):
    return v2.Compose([v2.Resize(image_size),
                       v2.ToImage(),
                       v2.ToDtype(dtype, scale=True),
                       v2.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])])


def _to_curve(baseline, im_size, min_points: int = 8):
    """
    Converts poly(base)lines to Bezier curves.
    """
    from shapely.geometry import LineString

    baseline = np.array(baseline)
    if len(baseline) < min_points:
        ls = LineString(baseline)
        baseline = np.stack([np.array(ls.interpolate(x, normalized=True).coords)[0] for x in np.linspace(0, 1, 8)])
    # control points
    curve = np.concatenate(([baseline[0]], bezier_fit(baseline), [baseline[-1]]))/im_size
    curve = curve.flatten()
    return pa.scalar(curve, type=pa.list_(pa.float32()))


def _to_bbox(boundary, im_size):
    """
    Converts a bounding polygon to a bbox in xyxyc_xc_yhw format.
    """
    flat_box = [point for pol in boundary for point in pol]
    xmin, xmax = min(flat_box[::2]), max(flat_box[::2])
    ymin, ymax = min(flat_box[1::2]), max(flat_box[1::2])
    w = xmax - xmin
    h = ymax - ymin
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    bbox = np.array([[xmin, ymin], [xmax, ymax], [cx, cy], [w, h]]) / im_size
    bbox = bbox.flatten()
    return pa.scalar(bbox, type=pa.list_(pa.float32()))


def compile(files: Optional[list[Union[str, 'PathLike']]] = None,
            output_file: Union[str, 'PathLike'] = None,
            normalize_whitespace: bool = True,
            normalization: Optional[Literal['NFD', 'NFC', 'NFKD', 'NFKC']] = None,
            max_line_tokens: int = 384,
            resize: Optional[Tuple[int, int]] = None,
            allow_textless: bool = False,
            callback: Callable[[int, int], None] = lambda chunk, lines: None) -> None:
    """
    Compiles a collection of XML facsimile files into a binary arrow dataset.

    Args:
        files: list of XML files
        output_file: destination to write arrow file to
        normalize_whitespace: whether to normalize all whitespace to ' '
        normalization: Unicode normalization to apply to data.
        max_line_tokens: maximum number of tokens per line
        resize: optional (height, width) tuple to resize images to
        allow_textless: if True, include lines without text
        callback: progress callback
    """
    from kraken.lib import functional_im_transforms as F_t
    from kraken.lib.xml import XMLPage

    text_transforms: list[Callable[[str], str]] = []

    # pyarrow structs
    line_struct = pa.struct([('text', pa.string()),
                             ('lang', pa.list_(pa.string())),
                             ('curve', pa.list_(pa.float32())),
                             ('bbox', pa.list_(pa.float32()))])
    page_struct = pa.struct([('im', pa.binary()),
                             ('lines', pa.list_(line_struct))])

    tokenizer = OctetTokenizer()

    if normalization:
        text_transforms.append(partial(F_t.text_normalize, normalization=normalization))
    if normalize_whitespace:
        text_transforms.append(F_t.text_whitespace_normalize)

    num_lines = 0
    # helper variables to enable padding to longest sequence without iterating
    # over set during training.
    max_lines_in_page = 0
    max_octets_in_line = 0
    schema = pa.schema([('pages', page_struct)])

    callback(0, len(files))

    with tempfile.NamedTemporaryFile() as tmpfile:
        with pa.OSFile(tmpfile.name, 'wb') as sink:
            with pa.ipc.new_file(sink, schema) as writer:
                for file in files:
                    try:
                        page = XMLPage(file).to_container()
                        # pick image format with smallest size
                        image_candidates = list(set(page.imagename.with_suffix(x) for x in ['.jxl', '.png']).union([page.imagename]))
                        cand_idxs = np.argsort([t.stat().st_size if t.exists() else np.inf for t in image_candidates])
                        im_path = None
                        for idx in cand_idxs:
                            try:
                                with Image.open(image_candidates[idx]) as im:
                                    if resize:
                                        im = im.convert('RGB')
                                        im = im.resize((resize[1], resize[0]), Image.LANCZOS)
                                        im_size = im.size
                                        im_buf = io.BytesIO()
                                        im.save(im_buf, format='JPEG', quality=95)
                                        resized_im_bytes = im_buf.getvalue()
                                    else:
                                        im_size = im.size
                                im_path = image_candidates[idx]
                                break
                            except Exception:
                                continue
                    except Exception:
                        continue
                    if im_path is None:
                        continue
                    page_data = []
                    prev_max_octets_in_line = max_octets_in_line
                    for line in page.lines:
                        try:
                            text = line.text
                            for func in text_transforms:
                                text = func(text)
                            if not text and not allow_textless:
                                logger.info(f'Text line "{line.text}" is empty after transformations')
                                continue
                            if not text:
                                text = ''
                            if not line.baseline:
                                logger.info('No baseline given for line')
                                continue
                            line_langs = line.language if line.language else ['und']
                            max_octets_in_line = max(len(tokenizer.encode(text, add_bos=False, add_eos=False)), max_octets_in_line)
                            page_data.append(pa.scalar({'text': pa.scalar(text),
                                                        'lang': line_langs,
                                                        'curve': _to_curve(line.baseline, im_size),
                                                        'bbox': _to_bbox(line.boundary, im_size)},
                                                       line_struct))
                            num_lines += 1
                        except Exception:
                            continue
                    # skip pages with lines longer than max_line_tokens
                    if max_octets_in_line > max_line_tokens:
                        max_octets_in_line = prev_max_octets_in_line
                        continue
                    if len(page_data) > 1:
                        if resize:
                            im = resized_im_bytes
                        else:
                            with open(im_path, 'rb') as fp:
                                im = fp.read()
                        ar = pa.array([pa.scalar({'im': im,
                                                  'lines': page_data}, page_struct)], page_struct)
                        writer.write(pa.RecordBatch.from_arrays([ar], schema=schema))
                        max_lines_in_page = max(len(page_data), max_lines_in_page)
                    callback(1, len(files))
        with pa.memory_map(tmpfile.name, 'rb') as source:
            metadata = {'num_lines': num_lines.to_bytes(4, 'little'),
                        'max_lines_in_page': max_lines_in_page.to_bytes(4, 'little'),
                        'max_octets_in_line': max_octets_in_line.to_bytes(4, 'little')}
            schema = schema.with_metadata(metadata)
            ds_table = pa.ipc.open_file(source).read_all()
            new_table = ds_table.replace_schema_metadata(metadata)
            with pa.OSFile(output_file, 'wb') as sink:
                with pa.ipc.new_file(sink, schema=schema) as writer:
                    for batch in new_table.to_batches():
                        writer.write(batch)


def _validation_worker_init_fn(worker_id):
    """ Fix random seeds so that augmentation always produces the same
        results when validating. Temporarily increase the logging level
        for lightning because otherwise it will display a message
        at info level about the seed being changed. """
    from lightning.pytorch import seed_everything
    seed_everything(42)


def collate_null(batch):
    return batch[0]


def collate_sequences(im, page_data, max_seq_len: int, index: int):
    """
    Sorts and pads image data.
    """
    if isinstance(page_data[0][0], str):
        labels = [x for x, _, _ in page_data]
    else:
        labels = torch.stack([F.pad(x, pad=(0, max_seq_len-len(x)), value=-100) for x, _, _ in page_data]).long()
    curves = None
    boxes = None
    if page_data[0][1] is not None:
        curves = torch.stack([x for _, x, _ in page_data])
    if page_data[0][2] is not None:
        boxes = torch.stack([x for _, _, x in page_data])
    gc.collect()
    libc = ctypes.CDLL("libc.so.6")
    libc.malloc_trim(0)
    gc.collect()
    return {'image': im,
            'tokens': labels,
            'curves': curves,
            'boxes': boxes,
            'index': index}


class BinnedBaselineDataset(Dataset):
    """
    Dataset for training a line recognition model from baseline data.

    Images are binned, i.e. a sample return `batch_size` lines sampled with
    replacement from a single page at index `n` of the dataset. As lines are
    sampled with replacement it is not useful to set batch sizes above the
    average number of lines contained on a page.

    The length of the dataset is the number of pages contained in the source
    files. A property `num_batches` contains the number of random samples
    required to roughly sample each line once over an epoch (under the
    assumption that each page contains the same number of lines).

    Args:
        prompt_mode: Select line prompt sampling mode: `boxes` for bbox-only,
                     `curves` for curves-only, and `both` for randomly
                     switching between the two.
        augmentation: Enables augmentation.
        batch_size: Maximum size of a batch. All samples from a batch will
                    come from a single page.
    """
    def __init__(self,
                 files: Sequence[Union[str, 'PathLike']],
                 im_transforms: Callable[[Any], torch.Tensor] = None,
                 augmentation: Callable[[Any], torch.Tensor] = None,
                 prompt_mode: Literal['boxes', 'curves', 'both'] = 'both',
                 batch_size: int = 32) -> None:
        super().__init__()
        self.files = files
        self.prompt_mode = prompt_mode
        self.transforms = im_transforms
        self.aug = augmentation
        self.batch_size = batch_size
        self.max_seq_len = 0
        self._len = 0

        self.tokenizer = OctetTokenizer()

        self.arrow_table = None
        self.pages_per_file = []

        for file in files:
            with pa.memory_map(file, 'rb') as source:
                ds_table = pa.ipc.open_file(source).read_all()
                raw_metadata = ds_table.schema.metadata
                if not raw_metadata or b'num_lines' not in raw_metadata:
                    raise ValueError(f'{file} does not contain a valid metadata record.')
                if not self.arrow_table:
                    self.arrow_table = ds_table
                else:
                    self.arrow_table = pa.concat_tables([self.arrow_table, ds_table])
                self.pages_per_file.append(len(ds_table))
                self._len += int.from_bytes(raw_metadata[b'num_lines'], 'little')
                self.max_seq_len = max(int.from_bytes(raw_metadata[b'max_octets_in_line'], 'little'), self.max_seq_len)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.arrow_table.column('pages')[index].as_py()
        logger.debug(f'Attempting to load {item["im"]}')
        im, page_data = item['im'], item['lines']
        try:
            im = decode_image(torch.frombuffer(bytearray(im), dtype=torch.uint8), mode=ImageReadMode.RGB)
        except Exception:
            return self[0]

        if self.prompt_mode == 'both':
            rng = np.random.default_rng()
            return_boxes = rng.choice([False, True], 1)
        elif self.prompt_mode == 'boxes':
            return_boxes = True
        else:
            return_boxes = False

        lines = []
        for line in page_data:
            tokens = torch.tensor(self.tokenizer.encode(line['text'], langs=line['lang'], add_bos=True, add_eos=True), dtype=torch.int32)
            curve = torch.tensor(line['curve']).view(4, 2) if not return_boxes else None
            bbox = torch.tensor(line['bbox']).view(4, 2) if return_boxes else None
            lines.append((tokens, curve, bbox))

        if self.aug is not None:
            im, lines = self.aug(im, lines)
        if not lines:
            return self[np.random.randint(len(self))]
        im = self.transforms(im)
        indices = np.random.choice(len(lines), self.batch_size, replace=True)
        sample = [lines[i] for i in indices]

        return collate_sequences(im.unsqueeze(0), sample, self.max_seq_len, index)

    def __len__(self) -> int:
        return len(self.arrow_table)

    @property
    def num_batches(self):
        """
        Number of batches in the dataset.
        """
        return self._len // self.batch_size


class ValidationBaselineDataset(IterableDataset):
    """
    Dataset for validation.

    Args:
        im_transforms: Function taking an PIL.Image and returning a tensor
                       suitable for forward passes.
        prompt_mode: Select line prompt sampling mode: `boxes` for bbox-only,
                     `curves` for curves-only, and `both` for randomly
                     switching between the two.
    """
    def __init__(self,
                 files: Sequence[Union[str, 'PathLike']],
                 im_transforms: Callable[[Any], torch.Tensor] = None,
                 prompt_mode: Literal['boxes', 'curves', 'both'] = 'both',
                 batch_size: int = 32) -> None:
        super().__init__()
        self.files = files
        self.transforms = im_transforms
        self.prompt_mode = prompt_mode
        self.batch_size = batch_size
        self.max_seq_len = 0

        self.tokenizer = OctetTokenizer()

        self.arrow_table = None

        for file in files:
            with pa.memory_map(file, 'rb') as source:
                ds_table = pa.ipc.open_file(source).read_all()
                raw_metadata = ds_table.schema.metadata
                if not raw_metadata or b'num_lines' not in raw_metadata:
                    raise ValueError(f'{file} does not contain a valid metadata record.')
                if not self.arrow_table:
                    self.arrow_table = ds_table
                else:
                    self.arrow_table = pa.concat_tables([self.arrow_table, ds_table])
                self.max_seq_len = max(int.from_bytes(raw_metadata[b'max_octets_in_line'], 'little'), self.max_seq_len)

    def __iter__(self):
        device_rank, world_size = (get_rank(), get_world_size()) if is_initialized() else (0, 1)

        # workers split
        worker_info = get_worker_info()
        worker_rank, num_workers = (worker_info.id, worker_info.num_workers) if worker_info else (0, 1)

        num_replicas = num_workers * world_size
        replica_rank = worker_rank * world_size + device_rank

        len_ds = self.arrow_table.column('pages').length()

        for idx in range(replica_rank, len_ds, num_replicas):
            item = self.arrow_table.column('pages')[idx].as_py()
            logger.debug(f'Attempting to load {item["im"]}')
            im, page_data = item['im'], item['lines']
            im = decode_image(torch.frombuffer(bytearray(im), dtype=torch.uint8), mode=ImageReadMode.RGB)
            im = self.transforms(im)

            im = im.unsqueeze(0)

            if self.prompt_mode == 'both':
                return_boxes = True
                return_curves = True
            elif self.prompt_mode == 'boxes':
                return_boxes = True
                return_curves = False
            else:
                return_boxes = False
                return_curves = True

            for lines in batched(page_data, self.batch_size):
                curves = []
                boxes = []
                tokens = []
                for line in lines:
                    if return_curves:
                        curves.append(torch.tensor(line['curve']).view(4, 2))
                    if return_boxes:
                        boxes.append(torch.tensor(line['bbox']).view(4, 2))
                    tokens.append(torch.tensor(self.tokenizer.encode(line['text'], langs=line['lang'], add_bos=True, add_eos=True), dtype=torch.int32))
                tokens = torch.stack([F.pad(x, pad=(0, self.max_seq_len-len(x)), value=-100) for x in tokens]).long()
                boxes = torch.stack(boxes) if len(boxes) else None
                curves = torch.stack(curves) if len(curves) else None
                yield {'image': im,
                       'boxes': boxes,
                       'curves': curves,
                       'tokens': tokens}


class TestBaselineDataset(Dataset):
    """
    Dataset for validation.

    Args:
        im_transforms: Function taking an PIL.Image and returning a tensor
                       suitable for forward passes.
        prompt_mode: Select line prompt sampling mode: `boxes` for bbox-only,
                     `curves` for curves-only, and `both` for randomly
                     switching between the two.
    """
    def __init__(self,
                 files: Sequence[Union[str, 'PathLike']],
                 prompt_mode: Literal['boxes', 'curves'] = 'curves') -> None:
        super().__init__()
        self.files = files
        self.prompt_mode = prompt_mode
        self.max_seq_len = 0

        self.arrow_table = None

        for file in files:
            with pa.memory_map(file, 'rb') as source:
                ds_table = pa.ipc.open_file(source).read_all()
                raw_metadata = ds_table.schema.metadata
                if not raw_metadata or b'num_lines' not in raw_metadata:
                    raise ValueError(f'{file} does not contain a valid metadata record.')
                if not self.arrow_table:
                    self.arrow_table = ds_table
                else:
                    self.arrow_table = pa.concat_tables([self.arrow_table, ds_table])
                self.max_seq_len = max(int.from_bytes(raw_metadata[b'max_octets_in_line'], 'little'), self.max_seq_len)

    def __len__(self):
        return len(self.arrow_table)

    def __getitem__(self, index: int) -> tuple[Image.Image, 'Segmentation']:
        from kraken.containers import Segmentation, BaselineLine, BBoxLine

        item = self.arrow_table.column('pages')[index].as_py()
        im, page_data = item['im'], item['lines']
        im = Image.open(io.BytesIO(im)).convert('RGB')

        all_langs = list({lang for line in page_data for lang in line['lang']})
        if self.prompt_mode == 'curves':
            lines = [BaselineLine(id='_foo',
                                  baseline=line['curve'],
                                  boundary=[],
                                  text=line['text']) for line in page_data]
        elif self.prompt_mode == 'boxes':
            lines = [BBoxLine(id='_foo', bbox=line['bbox'], text=line['text']) for line in page_data]
        return im, Segmentation(type=lines[0].type,
                                imagename='default.jpg',
                                script_detection=False,
                                text_direction='horizontal-lr',
                                lines=lines,
                                language=all_langs)


# magic lsq cubic bezier fit function from the internet.
def Mtk(n, t, k):
    return t**k * (1-t)**(n-k) * comb(n, k)


def BezierCoeff(ts):
    return [[Mtk(3, t, k) for k in range(4)] for t in ts]


def bezier_fit(bl):
    x = bl[:, 0]
    y = bl[:, 1]
    dy = y[1:] - y[:-1]
    dx = x[1:] - x[:-1]
    dt = (dx ** 2 + dy ** 2)**0.5
    t = dt/dt.sum()
    t = np.hstack(([0], t))
    t = t.cumsum()

    Pseudoinverse = np.linalg.pinv(BezierCoeff(t))  # (9,4) -> (4,9)

    control_points = Pseudoinverse.dot(bl)  # (4,9)*(9,2) -> (4,2)
    medi_ctp = control_points[1:-1, :]
    return medi_ctp
