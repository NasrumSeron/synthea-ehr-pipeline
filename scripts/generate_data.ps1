# Download Synthea v4.0.0 and generate the synthetic cohort into data/raw/csv.
# Needs Java 17+. Same seeds + same end date -> same patients (row order may differ;
# the pipeline sorts by patient Id).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$Version = "v4.0.0"
$Jar = "tools/synthea-with-dependencies.jar"
$Sha256 = "ed43c20ad40ba5c3bc724503a5af032715fe3c491620b766148e7c2361e6ecc1"
$Url = "https://github.com/synthetichealth/synthea/releases/download/$Version/synthea-with-dependencies.jar"

function Get-Sha256($Path) { (Get-FileHash $Path -Algorithm SHA256).Hash.ToLower() }

New-Item -ItemType Directory -Force tools | Out-Null
if (-not (Test-Path $Jar) -or (Get-Sha256 $Jar) -ne $Sha256) {
    Write-Host "Downloading Synthea $Version (~200 MB)..."
    $ProgressPreference = "SilentlyContinue"
    Invoke-WebRequest -Uri $Url -OutFile "$Jar.part"
    $actual = Get-Sha256 "$Jar.part"
    if ($actual -ne $Sha256) {
        Remove-Item -Force "$Jar.part"
        Write-Error "checksum mismatch for downloaded jar (got $actual)"
    }
    Move-Item -Force "$Jar.part" $Jar
}

# Generate into a temp dir; replace data/raw only if Synthea succeeds.
if (Test-Path data/raw.tmp) { Remove-Item -Recurse -Force data/raw.tmp }
& java -jar $Jar -p 1000 -s 42 -cs 42 -r 20261101 -e 20261101 `
    --exporter.baseDirectory data/raw.tmp `
    --exporter.csv.export true `
    --exporter.fhir.export false `
    --exporter.hospital.fhir.export false `
    --exporter.practitioner.fhir.export false
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if (Test-Path data/raw) { Remove-Item -Recurse -Force data/raw }
Move-Item data/raw.tmp data/raw
Write-Host "Done: data/raw/csv"
