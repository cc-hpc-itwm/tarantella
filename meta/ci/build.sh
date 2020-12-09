#!/usr/bin/env bash

set -euo pipefail

SRC_DIR="${1:?}"
INSTALL_DIR="${2:?}"
BUILD_TYPE="${3:?}"

CI_DIR="${SRC_DIR}/meta/ci"

# Build in build subfolder of source
cd "${SRC_DIR}"
if [ -d "build" ]; then
    rm -r build
fi
if [ -d "${INSTALL_DIR}" ]; then
    echo "Install folder '${INSTALL_DIR}' must not exist"
    exit 1
fi

mkdir build && cd build

cmake "${SRC_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DLINK_IB=OFF   \
    -DENABLE_TESTING=ON \
    -DTIMEOUT_GASPI_TERMINATE_BARRIER=1000

make -j$(nproc)

