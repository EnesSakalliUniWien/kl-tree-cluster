#!/bin/sh
set -eu

cd "$(dirname "$0")"

OS="$(uname -s)"
ARCH="$(uname -m)"

case "${OS}-${ARCH}" in
  Darwin-arm64)
    ARCHIVE="vale_3.14.0_macOS_arm64.tar.gz"
    ;;
  Darwin-x86_64)
    ARCHIVE="vale_3.14.0_macOS_64-bit.tar.gz"
    ;;
  Linux-x86_64)
    ARCHIVE="vale_3.14.0_Linux_64-bit.tar.gz"
    ;;
  Linux-aarch64|Linux-arm64)
    ARCHIVE="vale_3.14.0_Linux_arm64.tar.gz"
    ;;
  *)
    echo "Unsupported platform: ${OS}-${ARCH}" >&2
    exit 1
    ;;
esac

URL="https://github.com/errata-ai/vale/releases/download/v3.14.0/${ARCHIVE}"

mkdir -p tools

if [ ! -f "tools/${ARCHIVE}" ]; then
  echo "Downloading ${ARCHIVE}..."
  curl -L "${URL}" -o "tools/${ARCHIVE}"
fi

tar -xzf "tools/${ARCHIVE}" -C tools
chmod +x tools/vale
echo "Installed tools/vale"
