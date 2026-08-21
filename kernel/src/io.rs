//! x86 port-mapped I/O: single-byte `in`/`out` helpers.
//!
//! Port access is inherently a ring-0, x86-only operation. Like `arch.rs`,
//! the host/test build gets no-op stubs so the crate still compiles for
//! `cargo test`.

/// Write one byte to a device port.
///
/// # Safety
///
/// `port` must be a port the caller is allowed to touch; the caller must also
/// ensure no other code accesses the same port concurrently.
#[cfg(all(target_arch = "x86_64", not(test)))]
pub unsafe fn outb(port: u16, value: u8) {
    // Safety: the caller upholds the port-access contract; `out` is always
    // available in ring 0 and does not touch memory.
    unsafe {
        core::arch::asm!(
            "out dx, al",
            in("dx") port,
            in("al") value,
            options(nostack, preserves_flags),
        );
    }
}

/// Read one byte from a device port.
///
/// # Safety
///
/// `port` must be a port the caller is allowed to touch; the caller must also
/// ensure no other code accesses the same port concurrently.
#[cfg(all(target_arch = "x86_64", not(test)))]
pub unsafe fn inb(port: u16) -> u8 {
    let value: u8;
    // Safety: the caller upholds the port-access contract; `in` is always
    // available in ring 0 and does not touch memory.
    unsafe {
        core::arch::asm!(
            "in al, dx",
            out("al") value,
            in("dx") port,
            options(nostack, preserves_flags),
        );
    }
    value
}

/// Host/build-stub for [`outb`].
///
/// # Safety
///
/// No-op stub; the caller follows the [`outb`] contract.
#[cfg(not(all(target_arch = "x86_64", not(test))))]
pub unsafe fn outb(_port: u16, _value: u8) {}

/// Host/build-stub for [`inb`].
///
/// # Safety
///
/// No-op stub; the caller follows the [`inb`] contract.
#[cfg(not(all(target_arch = "x86_64", not(test))))]
pub unsafe fn inb(_port: u16) -> u8 {
    0
}
