param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11+ ile sanal ortam olusturulamadi. -Python parametresini kullanin.' }
}
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Bagimlilik kurulumu basarisiz.' }
& .\.venv\Scripts\python.exe -c 'import numpy, scipy, matplotlib; print("Python dependencies OK")'
if ($LASTEXITCODE -ne 0) { throw 'Python DLL sorunu. Bagimsiz python.org yorumlayicisi kullanin; README kurulum bolumune bakin.' }
& .\.venv\Scripts\python.exe lora_master.py --prepare-only
if ($LASTEXITCODE -ne 0) { throw 'Dunya hazirlanamadi.' }
Write-Host 'Hazir. Start-LORA.cmd dosyasini acin.'
