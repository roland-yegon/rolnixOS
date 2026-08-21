use std::env;
use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let linker = manifest.join("link.ld");
    println!("cargo:rerun-if-changed={}", linker.display());
    // Link the raw ELF with our own script: kernel loaded at 1 MiB, no PIE,
    // no startup objects (our entry.s provides `_start`), no libc.
    println!("cargo:rustc-link-arg=-T{}", linker.display());
    println!("cargo:rustc-link-arg=-no-pie");
    println!("cargo:rustc-link-arg=-static");
    println!("cargo:rustc-link-arg=-nostartfiles");
}
