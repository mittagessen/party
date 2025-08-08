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
import torch
import torch.nn.functional as F
import numpy as np
import lightning.pytorch as L

import tempfile
import pyarrow as pa

from pathlib import Path
from itertools import islice

from torch.distributed import get_rank, get_world_size, is_initialized

from typing import (TYPE_CHECKING, Any, Callable, List, Literal, Optional,
                    Tuple, Union, Sequence)

from functools import partial
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, IterableDataset

from PIL import Image

from scipy.special import comb

from party.tokenizer import OctetTokenizer, LANG_TO_ISO
from party.sampler import LossAwareSampler


if TYPE_CHECKING:
    from os import PathLike
    from kraken.containers import Segmentation

__all__ = ['TextLineDataModule']

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


def get_default_transforms(dtype=torch.float32):
    return v2.Compose([v2.Resize((2560, 1920)),
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


def compile(files: Optional[List[Union[str, 'PathLike']]] = None,
            output_file: Union[str, 'PathLike'] = None,
            normalize_whitespace: bool = True,
            normalization: Optional[Literal['NFD', 'NFC', 'NFKD', 'NFKC']] = None,
            max_line_tokens: int = 384,
            callback: Callable[[int, int], None] = lambda chunk, lines: None) -> None:
    """
    Compiles a collection of XML facsimile files into a binary arrow dataset.

    Args:
        files: List of XML files
        output_file: destination to write arrow file to
        normalize_whitespace: whether to normalize all whitespace to ' '
        normalization: Unicode normalization to apply to data.
        max_line_tokens: maximum number of tokens per line
        callback: progress callback
    """
    from kraken.lib import functional_im_transforms as F_t
    from kraken.lib.xml import XMLPage

    text_transforms: List[Callable[[str], str]] = []

    # pyarrow structs
    line_struct = pa.struct([('text', pa.string()),
                             ('curve', pa.list_(pa.float32())),
                             ('bbox', pa.list_(pa.float32()))])
    page_struct = pa.struct([('im', pa.binary()),
                             ('lang', pa.string()),
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
                                    im_size = im.size
                                im_path = image_candidates[idx]
                                break
                            except Exception:
                                continue
                        # parse language by traversing path component
                        for part in Path(file).parts[::-1]:
                            if (lang := LANG_TO_ISO.get(part, 'und')) != 'und':
                                break
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
                            if not text:
                                logger.info(f'Text line "{line.text}" is empty after transformations')
                                continue
                            if not line.baseline:
                                logger.info('No baseline given for line')
                                continue
                            max_octets_in_line = max(len(tokenizer.encode(text, add_bos=False, add_eos=False)), max_octets_in_line)
                            page_data.append(pa.scalar({'text': pa.scalar(text),
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
                        with open(im_path, 'rb') as fp:
                            im = fp.read()
                        ar = pa.array([pa.scalar({'im': im,
                                                  'lang': lang,
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
    return {'image': im,
            'tokens': labels,
            'curves': curves,
            'boxes': boxes,
            'index': index}


class TextLineDataModule(L.LightningDataModule):
    def __init__(self,
                 training_data: List[Union[str, 'PathLike']],
                 evaluation_data: List[Union[str, 'PathLike']],
                 prompt_mode: Literal['boxes', 'curves', 'both'] = 'both',
                 augmentation: bool = False,
                 batch_size: int = 16,
                 num_workers: int = 8):
        super().__init__()

        self.save_hyperparameters()

        self.im_transforms = get_default_transforms()

        # tokenizer is stateless so we can just initiate it here
        tokenizer = OctetTokenizer()

        self.pad_id = tokenizer.pad_id
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id

        self.num_classes = tokenizer.max_label + 1

    def setup(self, stage: str):
        """
        Actually builds the datasets.
        """
        self.train_set = BinnedBaselineDataset(self.hparams.training_data,
                                               im_transforms=self.im_transforms,
                                               augmentation=self.hparams.augmentation,
                                               batch_size=self.hparams.batch_size)
        self.val_set = ValidationBaselineDataset(self.hparams.evaluation_data,
                                                 im_transforms=self.im_transforms,
                                                 augmentation=self.hparams.augmentation)
        self.train_set.max_seq_len = max(self.train_set.max_seq_len, self.val_set.max_seq_len)
        self.val_set.max_seq_len = self.train_set.max_seq_len

    def train_dataloader(self):
        # shared tensor for perfect sample labeling 
        perfect_samples = torch.zeros(self.train_set.num_pages, dtype=torch.bool).share_memory_()
        perfect_sample_it = torch.zeros(1).share_memory_()
        return DataLoader(self.train_set,
                          num_workers=self.hparams.num_workers,
                          batch_size=1,
                          sampler=LossAwareSampler(self.train_set,
                                                   perfect_samples=perfect_samples,
                                                   perfect_sample_it=perfect_sample_it),
                          prefetch_factor=1,
                          pin_memory=True,
                          shuffle=False,
                          collate_fn=collate_null)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          shuffle=False,
                          batch_size=1,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          collate_fn=collate_null,
                          worker_init_fn=_validation_worker_init_fn)


class BinnedBaselineDataset(Dataset):
    """
    Dataset for training a line recognition model from baseline data.

    Images are binned, so the batch_size parameter of the data loader is an
    upper limit of the number of samples returned.

    Args:
        im_transforms: Function taking an PIL.Image and returning a tensor
                       suitable for forward passes.
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
                 prompt_mode: Literal['boxes', 'curves', 'both'] = 'both',
                 augmentation: bool = False,
                 batch_size: int = 32) -> None:
        super().__init__()
        self.files = files
        self.transforms = im_transforms
        self.prompt_mode = prompt_mode
        self.aug = None
        self.batch_size = batch_size
        self.max_seq_len = 0
        self._len = 0

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
                self._len += int.from_bytes(raw_metadata[b'num_lines'], 'little')
                self.max_seq_len = max(int.from_bytes(raw_metadata[b'max_octets_in_line'], 'little'), self.max_seq_len)

        if augmentation:
            from party.augmentation import DefaultAugmenter
            self.aug = DefaultAugmenter()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.arrow_table.column('pages')[index].as_py()
        logger.debug(f'Attempting to load {item["im"]}')
        im, lang, page_data = item['im'], item['lang'], item['lines']
        try:
            im = Image.open(io.BytesIO(im)).convert('RGB')
        except Exception:
            return self[0]

        im = self.transforms(im)
        if self.aug:
            im = im.permute((1, 2, 0)).numpy()
            o = self.aug(image=im)
            im = torch.tensor(o['image'].transpose(2, 0, 1))

        # sample randomly between baselines
        sample = []
        if self.prompt_mode == 'both':
            rng = np.random.default_rng()
            return_boxes = rng.choice([False, True], 1)
        elif self.prompt_mode == 'boxes':
            return_boxes = True
        else:
            return_boxes = False
        for x in rng.choice(len(page_data), self.batch_size, replace=True, shuffle=False):
            line = page_data[x]
            tokens = torch.tensor(self.tokenizer.encode(line['text'], langs=[lang], add_bos=True, add_eos=True), dtype=torch.int32)
            curve = torch.tensor(line['curve']).view(4, 2) if not return_boxes else None
            bbox = torch.tensor(line['bbox']).view(4, 2) if return_boxes else None
            sample.append((tokens, curve, bbox))
        return collate_sequences(im.unsqueeze(0), sample, self.max_seq_len, index)

    def __len__(self) -> int:
        return len(self.arrow_table)

    @property
    def num_pages(self):
        """
        Number of pages in the dataset.
        """
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
        augmentation: Enables augmentation.
    """
    def __init__(self,
                 files: Sequence[Union[str, 'PathLike']],
                 im_transforms: Callable[[Any], torch.Tensor] = None,
                 prompt_mode: Literal['boxes', 'curves', 'both'] = 'both',
                 augmentation: bool = False,
                 batch_size: int = 32) -> None:
        super().__init__()
        self.files = files
        self.transforms = im_transforms
        self.prompt_mode = prompt_mode
        self.aug = None
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

        if augmentation:
            from party.augmentation import DefaultAugmenter
            self.aug = DefaultAugmenter()

    def __iter__(self):
        device_rank, world_size = (get_rank(), get_world_size()) if is_initialized() else (0, 1)

        # workers split
        worker_info = get_worker_info()
        worker_rank, num_workers = (worker_info.id, worker_info.num_workers) if worker_info else (0, 1)

        num_replicas = num_workers * world_size
        replica_rank = worker_rank * world_size + device_rank

        for item in self.arrow_table.column('pages')[replica_rank::num_replicas]:
            item = item.as_py()
            logger.debug(f'Attempting to load {item["im"]}')
            im, lang, page_data = item['im'], item['lang'], item['lines']
            im = Image.open(io.BytesIO(im)).convert('RGB')
            im = self.transforms(im)
            if self.aug:
                im = im.permute((1, 2, 0)).numpy()
                o = self.aug(image=im)
                im = torch.tensor(o['image'].transpose(2, 0, 1))

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
                    tokens.append(torch.tensor(self.tokenizer.encode(line['text'], langs=[lang], add_bos=True, add_eos=True), dtype=torch.int32))

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

    def __getitem__(self, index: int) -> Tuple[Image.Image, 'Segmentation']:
        from kraken.containers import Segmentation, BaselineLine, BBoxLine

        item = self.arrow_table.column('pages')[index].as_py()
        im, lang, page_data = item['im'], item['lang'], item['lines']
        im = Image.open(io.BytesIO(im)).convert('RGB')

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
                                language=[lang])


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
