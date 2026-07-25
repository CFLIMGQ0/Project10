#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${MREPATH_PYTHON:-/home/administrator/miniconda3/envs/mrepath-train/bin/python}"
THIRD_PARTY_ROOT="${MREPATH_THIRD_PARTY_ROOT:-/home/administrator/.cache/mrepath/third_party}"
PATCH_ROOT="${PROJECT_DIR}/patches/multimodal_baselines"

mkdir -p "${THIRD_PARTY_ROOT}"

clone_pinned() {
    local name="$1"
    local url="$2"
    local commit="$3"
    local repo="${THIRD_PARTY_ROOT}/${name}"

    if [[ ! -d "${repo}/.git" ]]; then
        git clone "${url}" "${repo}"
        git -C "${repo}" checkout --detach "${commit}"
    fi

    local actual
    actual="$(git -C "${repo}" rev-parse HEAD)"
    if [[ "${actual}" != "${commit}" ]]; then
        echo "${name}: expected ${commit}, found ${actual}" >&2
        echo "Use a clean checkout at the pinned commit and run this script again." >&2
        return 1
    fi
}

apply_once() {
    local repo="$1"
    local patch="$2"
    if git -C "${repo}" apply --reverse --check "${patch}" >/dev/null 2>&1; then
        echo "$(basename "${repo}"): compatibility patch already applied"
    else
        git -C "${repo}" apply --check "${patch}"
        git -C "${repo}" apply "${patch}"
        echo "$(basename "${repo}"): applied $(basename "${patch}")"
    fi
}

clone_pinned \
    SurvPath \
    https://github.com/mahmoodlab/SurvPath.git \
    3f73ddd6705ec67d643020c5bb04fb13f9f382cc
clone_pinned \
    MOTCat \
    https://github.com/Innse/MOTCat.git \
    0da379f73d92c096122df139a9410d08b096e6c1
clone_pinned \
    CMTA \
    https://github.com/FT-ZHOU-ZZZ/CMTA.git \
    31340f6c74575668de35ee6c5d761467f11089e2
clone_pinned \
    PORPOISE \
    https://github.com/mahmoodlab/PORPOISE.git \
    3390dc40e15995ba852d7a561faf0e996cf20501
clone_pinned \
    PIBD \
    https://github.com/mahmoodlab/PIBD.git \
    bd5bd94e6f8d48e7679c6e68209a3a65c9e56a78

apply_once "${THIRD_PARTY_ROOT}/SurvPath" \
    "${PATCH_ROOT}/survpath-coadread.patch"
apply_once "${THIRD_PARTY_ROOT}/PORPOISE" \
    "${PATCH_ROOT}/porpoise-coadread.patch"
apply_once "${THIRD_PARTY_ROOT}/CMTA" \
    "${PATCH_ROOT}/cmta-coadread.patch"

"${PYTHON_BIN}" -m pip install \
    nystrom-attention \
    POT \
    x-transformers \
    tensorboardX

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/prepare_coadread_legacy_baselines.py"

echo "The pinned multimodal baseline repositories and COREAD adapters are ready."
