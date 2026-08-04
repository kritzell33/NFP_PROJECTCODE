# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the NFP Assessor.
# Build ON WINDOWS, from the repo root, inside the activated .venv:
#     pyinstaller packaging/assessor.spec
# Output: dist/NFP_Assessor/NFP_Assessor.exe  (one-dir mode: faster startup,
# fewer antivirus false positives than one-file).

from PyInstaller.utils.hooks import collect_data_files

# Bundle the package's Jinja2 templates plus sample data & scenarios so the
# frozen app works out of the box.
datas = collect_data_files("nfp")
datas += [
    ("../data", "data"),
    ("../scenarios", "scenarios"),
]

a = Analysis(
    ["launch_gui.py"],
    pathex=[".."],
    datas=datas,
    hiddenimports=[],
    excludes=["pymc", "arviz", "jupyterlab"],  # calibration stack never ships
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="NFP_Assessor",
    console=False,      # windowed app; set True temporarily when debugging
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="NFP_Assessor",
)
