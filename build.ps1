$ErrorActionPreference='Stop'
$env:TEMP=Join-Path $PSScriptRoot '.build-temp'
$env:TMP=$env:TEMP
$env:PYINSTALLER_CONFIG_DIR=Join-Path $PSScriptRoot '.pyinstaller-cache'
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
Set-Location $PSScriptRoot
python -m PyInstaller --noconfirm --windowed --onedir --name WhisperMeeting --collect-all faster_whisper --collect-all pyaudiowpatch --collect-all ctranslate2 --collect-all tokenizers --exclude-module torch --exclude-module transformers realtime.py
if($LASTEXITCODE -ne 0){throw 'Build failed'}
foreach($folder in @('model-turbo','cuda')){if(Test-Path $folder){Copy-Item -Recurse -Force $folder 'dist/WhisperMeeting/'}}
