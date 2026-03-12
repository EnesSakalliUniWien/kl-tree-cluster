#!/bin/sh
set -eu

cd "$(dirname "$0")"

OS="$(uname -s)"
ARCH="$(uname -m)"

case "${OS}-${ARCH}" in
  Darwin-arm64)
    TARGET="aarch64-apple-darwin"
    ;;
  Darwin-x86_64)
    TARGET="x86_64-apple-darwin"
    ;;
  Linux-x86_64)
    TARGET="x86_64-unknown-linux-gnu"
    ;;
  *)
    echo "Unsupported platform: ${OS}-${ARCH}" >&2
    exit 1
    ;;
esac

VERSION="0.15.0"
ARCHIVE="tectonic-${VERSION}-${TARGET}.tar.gz"
URL="https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40${VERSION}/${ARCHIVE}"

mkdir -p tools

if [ ! -f "tools/${ARCHIVE}" ]; then
  echo "Downloading ${ARCHIVE}..."
  curl -L "${URL}" -o "tools/${ARCHIVE}"
fi

tar -xzf "tools/${ARCHIVE}" -C tools
chmod +x tools/tectonic
echo "Installed tools/tectonic"
