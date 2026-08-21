#!/usr/bin/env bash
# Boot the rolnix PMM harness under QEMU and stream its serial output.
set -euo pipefail

cd "$(dirname "$0")/.."

profile="${1:-debug}"
case "$profile" in
    debug|release) ;;
    *) echo "usage: $0 [debug|release]" >&2; exit 2 ;;
esac

# cargo's built-in dev profile is invoked as `--profile dev`, not `debug`;
# its on-disk artifact directory is `debug`.
cargo_profile="dev"
artifact_dir="debug"
[ "$profile" = "release" ] && { cargo_profile="release"; artifact_dir="release"; }

cargo build -p rolnix-boot --target x86_64-unknown-none --profile "$cargo_profile"

kernel="target/x86_64-unknown-none/$artifact_dir/rolnix-boot"

# QEMU's multiboot loader only accepts ELF images marked i386 (EM_386), even
# though the harness self-transitions to long mode. Patch e_machine (bytes
# 18-19 of the ELF header) from EM_X86_64 (62) to EM_386 (3); idempotent.
if [ "$(od -An -tu1 -j 18 -N 1 "$kernel")" != "3" ]; then
    printf '\x03' | dd of="$kernel" bs=1 seek=18 conv=notrunc status=none
fi

exec qemu-system-x86_64 \
    -kernel "$kernel" \
    -serial stdio \
    -m 512 \
    -display none \
    -no-reboot \
    -no-shutdown
