#!/bin/bash

# Download the mammalian orthologue table used by the explorer's
# "Check for orthologues" option, and verify it against a fixed SHA-256.
#
# Usage: ./download_orthologues.sh      (run from the UORCA repository root)
#
# Source: Allen Institute GeneOrthology, commit dd9e22d (2023-11-13), file
# csv/mammalian_orthologs_20231113.csv. Licence: BSD-2-Clause,
# Copyright (c) 2018 Allen Institute. See data/README.md.
# Windows: use download_orthologues.ps1 instead.

set -euo pipefail

URL="https://raw.githubusercontent.com/AllenInstitute/GeneOrthology/dd9e22dd03ef60cf418d1fff04690ab8fbd99f2c/csv/mammalian_orthologs_20231113.csv"
EXPECTED_SHA256="9989602d1815885c462f029d362bcf518af3390ba167f2a8b1ceb707a67281cf"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
TARGET="${DATA_DIR}/mammalian_orthologues.csv"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

sha256_of() {
    if command -v sha256sum &> /dev/null; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum &> /dev/null; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        echo -e "${RED}Error: neither sha256sum nor shasum is installed${NC}" >&2
        exit 1
    fi
}

mkdir -p "${DATA_DIR}"
TMP_FILE="$(mktemp "${DATA_DIR}/.mammalian_orthologues.XXXXXX")"
trap 'rm -f "${TMP_FILE}"' EXIT

echo -e "${YELLOW}Downloading ${URL}${NC}"
if command -v curl &> /dev/null; then
    curl -fL -# -o "${TMP_FILE}" "${URL}"
elif command -v wget &> /dev/null; then
    wget -q --show-progress -O "${TMP_FILE}" "${URL}"
else
    echo -e "${RED}Error: neither curl nor wget is installed${NC}" >&2
    exit 1
fi

ACTUAL_SHA256="$(sha256_of "${TMP_FILE}")"
echo "SHA-256 expected: ${EXPECTED_SHA256}"
echo "SHA-256 actual:   ${ACTUAL_SHA256}"
if [ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]; then
    echo -e "${RED}Error: SHA-256 mismatch. ${TARGET} was not written.${NC}" >&2
    exit 1
fi

chmod 644 "${TMP_FILE}"
mv "${TMP_FILE}" "${TARGET}"
trap - EXIT
echo -e "${GREEN}Wrote ${TARGET}${NC}"
