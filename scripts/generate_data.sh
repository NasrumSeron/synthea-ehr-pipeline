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

mkdir -p tools
if [ ! -f "$JAR" ]; then
  echo "Downloading Synthea ${VERSION} (~200 MB)..."
  curl -fSL -o "$JAR" "$URL"
fi
actual="$(sha256sum "$JAR" | cut -d' ' -f1)"
if [ "$actual" != "$SHA256" ]; then
  echo "error: checksum mismatch for $JAR (got $actual)" >&2
  exit 1
fi

rm -rf data/raw
java -jar "$JAR" -p 1000 -s 42 -cs 42 -r 20261101 -e 20261101 \
  --exporter.baseDirectory data/raw \
  --exporter.csv.export true \
  --exporter.fhir.export false \
  --exporter.hospital.fhir.export false \
  --exporter.practitioner.fhir.export false
echo "Done: data/raw/csv"
