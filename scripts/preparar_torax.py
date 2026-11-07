"""Preparacao educacional de uma TC toracica publica para o GitHub Pages.
A TC original e do exemplo SlicerLungCTAnalyzer; nao ha mascaras nesta etapa.
"""
from pathlib import Path
import SimpleITK as sitk

source = Path('/tmp/medatlas-torax/torax.nrrd')
out = Path('atlas-torax/tc-real/dados/torax.nii.gz')
if not source.exists() or source.stat().st_size < 100_000:
    raise RuntimeError('A TC publica nao foi baixada corretamente.')
image = sitk.ReadImage(str(source))
if image.GetDimension() != 3:
    raise RuntimeError('A amostra nao e um volume 3D.')
if image.GetPixelID() != sitk.sitkInt16:
    image = sitk.Cast(image, sitk.sitkInt16)
out.parent.mkdir(parents=True, exist_ok=True)
# Para reduzir uso de memoria do Safari, faz subamostragem isotropica em indices;
# SimpleITK preserva a origem, orientacao e spacing fisico correspondente.
for factor in (2, 3, 4):
    reduced = sitk.Shrink(image, [factor]*3)
    sitk.WriteImage(reduced, str(out), useCompression=True)
    mb = out.stat().st_size / (1024*1024)
    print('Fator', factor, 'dimensoes', reduced.GetSize(), 'arquivo MB', round(mb, 2))
    if mb < 24: break
else:
    raise RuntimeError('Volume ainda maior que 24 MB; revisar compactacao.')
if out.stat().st_size < 100_000:
    raise RuntimeError('A conversao nao produziu uma TC valida.')
