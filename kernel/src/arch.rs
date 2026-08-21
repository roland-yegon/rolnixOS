use core::sync::atomic::{AtomicUsize, Ordering};

use crate::frame::FrameIndex;

/// Size of a physical frame.
pub const FRAME_SIZE: usize = 4096;

/// Number of per-CPU free-list shards.
pub const SHARD_COUNT: usize = 64;

/// Maximum number of frames moved from the global free list into a shard on
/// a refill.
pub const SHARD_REFILL_BATCH: u32 = 64;

/// Base virtual address of the physmap, set once during early boot.
static PHYS_MAP_BASE: AtomicUsize = AtomicUsize::new(0);

/// Set the physmap base. Must be called exactly once, before any allocation.
pub fn set_physmap_base(base: usize) {
    let prev = PHYS_MAP_BASE.swap(base, Ordering::Relaxed);
    assert!(prev == 0 || prev == base, "physmap base set twice");
}

/// The configured physmap base.
pub fn physmap_base() -> usize {
    let b = PHYS_MAP_BASE.load(Ordering::Relaxed);
    assert!(b != 0, "physmap not initialized");
    b
}

/// Translate a physical address to its physmap virtual address.
///
/// The caller is responsible for `phys` being covered by the physmap (i.e. a
/// RAM address below the end of the frame table). The conversion itself is a
/// plain integer cast.
pub fn phys_to_virt(phys: usize) -> *mut u8 {
    (physmap_base() + phys) as *mut u8
}

/// Reverse of [`phys_to_virt`]: the physical address backing a physmap VA.
pub fn virt_to_phys(virt: usize) -> usize {
    virt - physmap_base()
}

/// Physical address of the start of a frame.
pub fn frame_phys(frame: FrameIndex) -> usize {
    (frame as usize) * FRAME_SIZE
}

/// Identifier of the CPU currently executing.
///
/// Stub that always returns 0: per-CPU areas do not exist yet. Once a
/// GSBASE-based per-CPU area lands, this should read the CPU id from it.
/// Until then every CPU routes through shard 0, which is correct but does not
/// yet exploit sharding.
pub fn current_cpu() -> usize {
    0
}

/// Disable local interrupts and return the previous RFLAGS.
///
/// # Safety
///
/// The returned value must be passed to [`irq_restore`] exactly once, on the
/// same CPU, before the interrupt state is otherwise used.
#[cfg(all(target_arch = "x86_64", not(test)))]
pub unsafe fn irq_save() -> u64 {
    let flags: u64;
    // Safety: pushfq/pop are always available; cli is the paired operation
    // with irq_restore's popfq.
    unsafe {
        core::arch::asm!(
            "pushfq",
            "pop {flags}",
            "cli",
            flags = out(reg) flags,
            options(nomem, nostack),
        );
    }
    flags
}

/// Restore RFLAGS previously saved by [`irq_save`] on this CPU.
///
/// # Safety
///
/// `flags` must come from a matching [`irq_save`] on the same CPU.
#[cfg(all(target_arch = "x86_64", not(test)))]
pub unsafe fn irq_restore(flags: u64) {
    // Safety: popfq restores the exact state captured by irq_save.
    unsafe {
        core::arch::asm!(
            "push {flags}",
            "popfq",
            flags = in(reg) flags,
            options(nomem, nostack),
        );
    }
}

/// Enable maskable interrupts (set RFLAGS.IF).
///
/// # Safety
///
/// The IDT must be installed with gates for every unmasked IRQ before this is
/// called; otherwise the first IRQ turns into a #GP.
#[cfg(all(target_arch = "x86_64", not(test)))]
pub unsafe fn enable_interrupts() {
    // Safety: caller guarantees the IDT/IRQ state is consistent.
    unsafe {
        core::arch::asm!("sti", options(nostack));
    }
}

/// Disable maskable interrupts on the local CPU (clear RFLAGS.IF).
///
/// # Safety
///
/// The caller must pair this with a matching [`enable_interrupts`] (or
/// [`irq_restore`]) on the same CPU before relying on interrupt state.
#[cfg(all(target_arch = "x86_64", not(test)))]
pub unsafe fn disable_interrupts() {
    // Safety: cli is always available in ring 0 and has no other effects.
    unsafe {
        core::arch::asm!("cli", options(nostack));
    }
}

/// Invalidate the TLB entry for a single virtual page.
///
/// Must be called after unmapping (or otherwise changing) a page's mapping on
/// the current CPU.
///
/// # Safety
///
/// `va` must be the address of a page that is (or was just) mapped; the
/// invalidation targets the page-table entry the caller has modified.
#[cfg(all(target_arch = "x86_64", not(test)))]
pub unsafe fn invlpg(va: usize) {
    // Safety: `va` must be the address of a mapped page being torn down; the
    // invalidation is a plain instruction with no other preconditions.
    unsafe {
        core::arch::asm!("invlpg [{0}]", in(reg) va, options(nostack, preserves_flags));
    }
}

/// Load CR3, switching to a new page table root (full TLB flush).
///
/// # Safety
///
/// `phys` must be the physical address of a valid, fully populated PML4 whose
/// mappings cover all memory the running code will touch.
#[cfg(all(target_arch = "x86_64", not(test)))]
pub unsafe fn cr3_write(phys: usize) {
    // Safety: `phys` must be the physical address of a valid, populated PML4.
    unsafe {
        core::arch::asm!("mov cr3, {0}", in(reg) phys, options(nostack));
    }
}

/// Current CR3 (physical address of the active PML4).
#[cfg(all(target_arch = "x86_64", not(test)))]
pub fn cr3_read() -> usize {
    let value: usize;
    // Safety: reading CR3 has no side effects.
    unsafe {
        core::arch::asm!("mov {0}, cr3", out(reg) value, options(nostack));
    }
    value
}

/// Host/build-stub for [`irq_save`]: nothing to disable.
///
/// # Safety
///
/// No-op stub; the caller follows the [`irq_save`] contract.
#[cfg(not(all(target_arch = "x86_64", not(test))))]
pub unsafe fn irq_save() -> u64 {
    0
}

/// Host/build-stub for [`irq_restore`]: nothing to restore.
///
/// # Safety
///
/// No-op stub; the caller follows the [`irq_restore`] contract.
#[cfg(not(all(target_arch = "x86_64", not(test))))]
pub unsafe fn irq_restore(_flags: u64) {}

/// Host/build-stub for [`invlpg`]: nothing to invalidate.
///
/// # Safety
///
/// No-op stub; the caller follows the [`invlpg`] contract.
#[cfg(not(all(target_arch = "x86_64", not(test))))]
pub unsafe fn invlpg(_va: usize) {}

/// Host/build-stub for [`cr3_write`]: no paging on the host.
///
/// # Safety
///
/// No-op stub; the caller follows the [`cr3_write`] contract.
#[cfg(not(all(target_arch = "x86_64", not(test))))]
pub unsafe fn cr3_write(_phys: usize) {}

/// Host/build-stub for [`cr3_read`].
#[cfg(not(all(target_arch = "x86_64", not(test))))]
pub fn cr3_read() -> usize {
    0
}

/// Host/build-stub for [`enable_interrupts`].
///
/// # Safety
///
/// No-op stub; the caller follows the [`enable_interrupts`] contract.
#[cfg(not(all(target_arch = "x86_64", not(test))))]
pub unsafe fn enable_interrupts() {}

/// Host/build-stub for [`disable_interrupts`].
///
/// # Safety
///
/// No-op stub; the caller follows the [`disable_interrupts`] contract.
#[cfg(not(all(target_arch = "x86_64", not(test))))]
pub unsafe fn disable_interrupts() {}
