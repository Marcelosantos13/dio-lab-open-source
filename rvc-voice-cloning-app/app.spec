# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

# Configurações do aplicativo
app_name = 'RVC_Voice_Cloning_App'
main_script = 'main.py'

# Dados adicionais para incluir
added_files = [
    ('examples', 'examples'),
    ('README.md', '.'),
    ('requirements.txt', '.'),
]

# Imports ocultos necessários
hidden_imports = [
    'gradio',
    'torch',
    'torchaudio',
    'librosa',
    'soundfile',
    'numpy',
    'scipy',
    'matplotlib',
    'seaborn',
    'sklearn',
    'tensorboard',
    'praat',
    'resampy',
    'ffmpeg',
    'pydub',
    'noisereduce',
    'numba',
    'llvmlite',
    'cffi',
    'pycparser',
]

# Análise do script principal
a = Analysis(
    [main_script],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# Remover duplicatas
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# Executável
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=app_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

# Coletar arquivos
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=app_name,
)