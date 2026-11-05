#!/usr/bin/env bash
# Download Synthea v4.0.0 and generate the synthetic cohort into data/raw/csv.
# Needs Java 17+ and curl. Same seeds + same end date -> same patients (row order may differ;
# the pipeline sorts by patient Id).
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION="v4.0.0"
JAR="tools/synthea-with-dependencies.jar"
SHA256="ed43c20ad40ba5c3bc724503a5af032715fe3c491620b766148e7c2361e6ecc1"
URL="https://github.com/synthetichealth/synthea/releases/download/${VERSION}/synthea-with-dependencies.jar"

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

mkdir -p tools
if [ ! -f "$JAR" ] || [ "$(sha256 "$JAR")" != "$SHA256" ]; then
  echo "Downloading Synthea ${VERSION} (~200 MB)..."
  curl -fSL -o "$JAR.part" "$URL"
  actual="$(sha256 "$JAR.part")"
  if [ "$actual" != "$SHA256" ]; then
    rm -f "$JAR.part"
    echo "error: checksum mismatch for downloaded jar (got $actual)" >&2
    exit 1
  fi
  mv "$JAR.part" "$JAR"
fi

# Generate into a temp dir; replace data/raw only if Synthea succeeds.
rm -rf data/raw.tmp
java -jar "$JAR" -p 1000 -s 42 -cs 42 -r 20261101 -e 20261101 \
  --exporter.baseDirectory data/raw.tmp \
  --exporter.csv.export true \
  --exporter.fhir.export false \
  --exporter.hospital.fhir.export false \
  --exporter.practitioner.fhir.export false
rm -rf data/raw
mv data/raw.tmp data/raw
echo "Done: data/raw/csv"
