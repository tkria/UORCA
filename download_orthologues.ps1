# Download the mammalian orthologue table used by the explorer's
# "Check for orthologues" option, and verify it against a fixed SHA-256.
#
# Usage (from the UORCA repository root):
#   powershell -ExecutionPolicy Bypass -File download_orthologues.ps1
#
# Source: Allen Institute GeneOrthology, commit dd9e22d (2023-11-13), file
# csv/mammalian_orthologs_20231113.csv. Licence: BSD-2-Clause,
# Copyright (c) 2018 Allen Institute. See data/README.md.
# macOS/Linux: use download_orthologues.sh instead.

$ErrorActionPreference = 'Stop'
# The progress bar makes Invoke-WebRequest very slow on Windows PowerShell 5.1
$ProgressPreference = 'SilentlyContinue'

$Url = 'https://raw.githubusercontent.com/AllenInstitute/GeneOrthology/dd9e22dd03ef60cf418d1fff04690ab8fbd99f2c/csv/mammalian_orthologs_20231113.csv'
$ExpectedSha256 = '9989602d1815885c462f029d362bcf518af3390ba167f2a8b1ceb707a67281cf'
$DataDir = Join-Path $PSScriptRoot 'data'
$Target = Join-Path $DataDir 'mammalian_orthologues.csv'
$TmpFile = Join-Path $DataDir ('.mammalian_orthologues.' + [System.Guid]::NewGuid().ToString('N'))

New-Item -ItemType Directory -Force -Path $DataDir | Out-Null

try {
    Write-Host "Downloading $Url"
    # TLS 1.2 is required by GitHub; Windows PowerShell 5.1 may not enable it by default.
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $Url -OutFile $TmpFile -UseBasicParsing

    $ActualSha256 = (Get-FileHash -Algorithm SHA256 -Path $TmpFile).Hash.ToLowerInvariant()
    Write-Host "SHA-256 expected: $ExpectedSha256"
    Write-Host "SHA-256 actual:   $ActualSha256"
    if ($ActualSha256 -ne $ExpectedSha256) {
        throw "SHA-256 mismatch. $Target was not written."
    }

    Move-Item -Force -Path $TmpFile -Destination $Target
    Write-Host "Wrote $Target"
}
catch {
    # Write-Error would itself throw under ErrorActionPreference Stop and skip
    # the exit code; write to stderr directly, then exit 1.
    [Console]::Error.WriteLine("Error: orthologue download failed: $_")
    exit 1
}
finally {
    if (Test-Path $TmpFile) { Remove-Item -Force $TmpFile }
}
