#!/usr/bin/env python3
"""Baixa e prepara UM exame do grupo Pneumothorax do ChestPathCT5-S100.
Fonte: https://zenodo.org/records/18256797 (licença CC0).
Não cria anotações de pneumotórax; somente publica a TC real.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile
import zipfile

import nibabel as nib
import numpy as np
import requests

ARCHIVE_URL = "https://zenodo.org/records/18256797/files/Med3DVlM_Pneumothorax.zip?download=1"
DATA_DIR = Path("atlas-torax/dados")
OUTPUT = DATA_DIR / "pneumotorax.nii.gz"
META = DATA_DIR / "pneumotorax_fonte.json"
MAX_MB = 28

def download(dest):
    with requests.get(ARCHIVE_URL, stream=True, timeout=(45, 180), allow_redirects=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    if not zipfile.is_zipfile(dest):
        raise RuntimeError("Download não retornou um arquivo ZIP válido.")

def case_member(z):
    names = sorted((m for m in z.infolist()
                    if not m.is_dir() and m.filename.lower().endswith((".nii.gz", ".nii"))),
                   key=lambda x: x.filename)
    if not names:
        raise RuntimeError("Nenhum volume NIfTI encontrado no ZIP da coleção.")
    print("Exames disponíveis:", [n.filename for n in names])
    return names[0]

def prepare(path, label):
    img = nib.load(str(path))
    if len(img.shape) != 3:
        raise RuntimeError("O caso selecionado não tem três dimensões.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    best = None
    for stride in ((2,2,2), (3,3,2), (3,3,3), (4,4,3)):
        # O exame é convertido para HU inteiros e subamostrado para caber no iPad.
        # O novo affine preserva o posicionamento espacial dos voxels restantes.
        arr = np.asanyarray(img.dataobj)[::stride[0], ::stride[1], ::stride[2]]
        arr = np.clip(np.rint(arr), -32768, 32767).astype(np.int16)
        affine = img.affine.copy()
        affine[:3, :3] = affine[:3, :3] @ np.diag(stride)
        header = img.header.copy()
        header.set_data_dtype(np.int16)
        header.set_slope_inter(1, 0)
        reduced = nib.Nifti1Image(arr, affine, header=header)
        nib.save(reduced, str(OUTPUT))
        mb = OUTPUT.stat().st_size / 1048576
        print("Subamostragem:", stride, "dimensões:", arr.shape, "MB:", round(mb, 2))
        del arr, reduced
        if mb <= MAX_MB:
            best = stride
            break
    if best is None:
        raise RuntimeError("Arquivo ainda muito grande; revisão manual necessária.")
    META.write_text(json.dumps({
        "fonte": "ChestPathCT5-S100 / Zenodo 18256797",
        "licenca": "CC0 1.0",
        "categoria": "Pneumothorax",
        "arquivo_original": label,
        "stride": best,
        "arquivo_publicado": str(OUTPUT),
        "observacao": "Exame real do grupo pneumotórax; sem contornos segmentados.",
    }, ensure_ascii=False, indent=2) + "\n")

def main():
    with tempfile.TemporaryDirectory() as td:
        zp = Path(td) / "pneumothorax.zip"
        print("Baixando grupo Pneumothorax (~2,6 GB).")
        download(zp)
        with zipfile.ZipFile(zp) as z:
            item = case_member(z)
            if item.file_size > 2_000_000_000:
                raise RuntimeError("Caso individual excessivamente grande.")
            ext = ".nii.gz" if item.filename.lower().endswith(".nii.gz") else ".nii"
            raw = Path(td) / ("case" + ext)
            with z.open(item) as src, raw.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            prepare(raw, item.filename)
    print("Arquivo público preparado:", OUTPUT)

if __name__ == "__main__":
    main()
