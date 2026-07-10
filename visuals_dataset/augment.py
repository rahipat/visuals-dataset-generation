"""
Image augmenter that wraps HiKER corruptions.
Provides decode/encode utilities and ImageAugmenter class.
"""

import random
import numpy as np
from io import BytesIO
from typing import Dict, Optional
from pathlib import Path
import importlib.util
import random

from PIL import Image as PILImage


def decode_image(data: bytes) -> PILImage.Image:
    """Decode raw image bytes into a PIL Image (RGB)."""
    return PILImage.open(BytesIO(data)).convert("RGB")


def encode_image(image: PILImage.Image, format: str = "JPEG", quality: int = 90) -> bytes:
    """Encode a PIL Image back to bytes."""
    buf = BytesIO()
    image.save(buf, format=format, quality=quality)
    return buf.getvalue()


class ImageAugmenter:
    """
    Image augmenter that delegates to HiKER-SGG_Alterations/corruptions.py.
    Applies: snow, frost, fog, rain, sunglare, brightness,
    wildfire_smoke, dust.
    (waterdrop is skipped by default — it requires OpenGL/arcade; see below.)
    """
    
    FIXED_SEED = 42

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed if seed is not None else self.FIXED_SEED
        self.rng = random.Random(self.seed)
        np.random.seed(self.seed)
        self.corruptions_mod = self._load_corruptions_module()
    
    def _load_corruptions_module(self):
        """Load HiKER corruptions module by path."""
        try:
            # Get the visuals_dataset folder (where augment.py lives)
            workspace = Path(__file__).resolve().parent
            # Go up to the repo root (parent of visuals_dataset)
            repo_root = workspace.parent
            hiker_path = repo_root / 'HiKER-SGG_Alterations' / 'corruptions.py'
            
            if not hiker_path.exists():
                raise FileNotFoundError(f'HiKER corruptions not found at {hiker_path}')
            
            spec = importlib.util.spec_from_file_location('hiker_corruptions', str(hiker_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        except Exception as e:
            raise RuntimeError(f'Failed to load HiKER corruptions: {e}')
    
    def apply_all(self, image: PILImage.Image) -> Dict[str, PILImage.Image]:
        """
        Apply all configured corruptions from HiKER.
        Returns dict mapping corruption name -> PIL Image.
        Skips corruptions that fail (e.g., waterdrop on Windows).
        """
        result = {}
        
        # List of corruptions to apply with their function names and severity
        corruptions_to_apply = [
            ('snow', 3),
            ('frost', 3),
            ('fog', 3),
            ('rain', 3),
            ('sunglare', 3),
            ('brightness', 3),
            ('wildfire_smoke', 3),
            ('dust', 3),
            # VISUALS-MOD: waterdrop skipped by default — it needs OpenGL/arcade
            # (GPU + display) and fails on headless CPU nodes. corruptions.py imports
            # arcade lazily, so leaving this out keeps the module OpenGL-free. To use
            # it, install `arcade`/`pyvirtualdisplay`, run on a GPU partition under
            # `xvfb-run`, and uncomment the line below.
            # ('waterdrop', 3),
        ]
        
        for corruption_name, severity in corruptions_to_apply:
            try:
                fn = getattr(self.corruptions_mod, corruption_name, None)
                if fn is None:
                    print(f"[warn] Corruption '{corruption_name}' not found in HiKER module")
                    continue
                intensity = self.rng.uniform(0.2, 0.8)
                
                # Call the corruption function with randomized intensity
                intensity = self.rng.uniform(0.55, 0.9)
                try:
                    corrupted = fn(image, severity=severity, intensity=intensity)
                except TypeError:
                    corrupted = fn(image, severity=severity)
                
                # Ensure result is a PIL Image
                if not isinstance(corrupted, PILImage.Image):
                    # Convert numpy array to PIL Image
                    import numpy as np
                    arr = np.asarray(corrupted)
                    if arr.dtype != np.uint8:
                        arr = np.clip(arr, 0, 255).astype(np.uint8)
                    corrupted = PILImage.fromarray(arr)
                
                # Ensure RGB
                if corrupted.mode != 'RGB':
                    corrupted = corrupted.convert('RGB')
                
                result[corruption_name] = corrupted
            
            except Exception as e:
                print(f"[warn] Failed to apply '{corruption_name}': {type(e).__name__}: {e}")
                continue
        
        if not result:
            raise RuntimeError('No corruptions were successfully applied')
        
        return result
