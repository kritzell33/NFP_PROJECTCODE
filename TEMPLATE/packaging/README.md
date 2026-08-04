# Packaging the Assessor to a Windows .exe

Prereqs: Windows machine, the main `.venv` active, `pip install -e ".[gui,dev]"`
already done (that pulls in PyInstaller).

    packaging\build_exe.bat

or manually from the repo root:

    pyinstaller packaging\assessor.spec

Output lands in `dist\NFP_Assessor\` - the whole folder is the app; the exe
inside won't run if separated from it. Zip the folder to share it, or wrap it
with Inno Setup later for a real installer.

Notes
- One-dir mode is deliberate: faster startup and far fewer antivirus false
  positives than `--onefile`.
- The calibration stack (PyMC/ArviZ) is explicitly excluded; the exe consumes
  posterior files, it never fits models.
- If the frozen app can't find templates or data, run the exe from a console
  (set `console=True` in the spec temporarily) to see the traceback.
- This spec is written for Windows and hasn't been exercised on this repo's
  CI yet - expect one debugging pass on first build; that's normal with
  PyInstaller.
