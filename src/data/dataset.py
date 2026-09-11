from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, root: str | Path, transform: Callable | None = None):
        self.root = Path(root).resolve()
        self.records = manifest.loc[manifest['status'].eq('ok')].to_dict('records')
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        path = (self.root / record['path']).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Manifest path escapes dataset root')
        with Image.open(path) as source:
            # RGB provides a consistent tensor shape; the audit retains original modes.
            image = source.convert('RGB')
        return {'image': self.transform(image) if self.transform else image,
                'label': record['label'], 'group_id': record['group_id'], 'path': record['path']}
