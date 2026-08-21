//! x86-64 TSS + GDT: a Rust-built descriptor table with a TSS providing a
//! kernel stack (`rsp0`) and a dedicated double-fault stack (`ist[0]`).
//!
//! The early GDT from `entry.s` is replaced here. Selectors 0x08/0x10/0x18
//! keep their meaning (32-bit code, data, 64-bit code), so the segments
//! loaded before `lgdt` stay valid; a 16-byte TSS descriptor is appended at
//! selector 0x20 and ring-3 user code/data segments at 0x30/0x38. `init`
//! loads the GDT and executes `ltr`. The IDT's #DF
//! gate then references `DF_GATE_IST`, so a double fault switches to the
//! IST stack instead of pushing onto whatever stack the faulting frame was
//! using (which is what escalates most #DFs in the first place).
//!
//! Host/build-stub note: same convention as `arch.rs`/`idt.rs`; on a non-x86
//! host build `init` is a no-op so the crate still builds for `cargo test`.
#![cfg_attr(not(all(target_arch = "x86_64", not(test))), allow(dead_code))]

/// Selector of the code64 segment; matches the IDT gate selector.
pub const GDT_CODE64_SELECTOR: u16 = 0x18;
/// Selector of the TSS descriptor (index 4 -> 0x20).
pub const TSS_SELECTOR: u16 = 0x20;
/// Ring-3 code64 segment (index 6, RPL 3 -> 0x33), used for user programs.
pub const USER_CODE64_SELECTOR: u16 = 0x33;
/// Ring-3 data segment (index 7, RPL 3 -> 0x3B), used as the user stack/data
/// segment. The RPL is baked in because `iretq` demands RPL == new CPL.
pub const USER_DATA_SELECTOR: u16 = 0x3B;
/// IST index (1-based) the #DF gate uses; refers to `TSS.ist[0]`.
pub const DF_GATE_IST: u8 = 1;
/// Exception vector of #DF, which the IDT gives an IST-bearing gate.
pub const DF_VECTOR: usize = 8;

/// AMD64 TSS, 104 bytes (limit 103). The hardware layout has `rsp0` at byte
/// offset 0x04 and `ist[0]` at 0x24 — no alignment padding — so this must be
/// `packed`; `repr(C)` alone would insert 4 bytes after `reserved1` and shift
/// every field. Access through `addr_of_mut!` + `write_unaligned`.
#[repr(C, packed)]
struct TaskStateSegment {
    reserved1: u32,
    rsp0: u64,
    rsp1: u64,
    rsp2: u64,
    reserved2: u64,
    ist: [u64; 7],
    reserved3: u64,
    reserved4: u16,
    iomap_base: u16,
}

/// 10-byte pointer loaded via `lgdt`.
#[repr(C, packed)]
struct GdtPtr {
    limit: u16,
    base: u64,
}

const DF_STACK_SIZE: usize = 16384;
const KERNEL_STACK_SIZE: usize = 16384;

/// The kernel-owned descriptor table; statics live in the image's bss, so
/// they are identity-mapped and reserved by `pmm_init`.
static mut GDT: [u64; 8] = [0; 8];
static mut TSS: TaskStateSegment = TaskStateSegment {
    reserved1: 0,
    rsp0: 0,
    rsp1: 0,
    rsp2: 0,
    reserved2: 0,
    ist: [0; 7],
    reserved3: 0,
    reserved4: 0,
    iomap_base: 0,
};

/// Dedicated stack used by the #DF gate via `TSS.ist[0]`.
static mut DF_STACK: [u8; DF_STACK_SIZE] = [0; DF_STACK_SIZE];
/// Kernel stack for privilege transitions into ring 0 (`TSS.rsp0`).
static mut KERNEL_STACK: [u8; KERNEL_STACK_SIZE] = [0; KERNEL_STACK_SIZE];

/// Point `TSS.rsp0` (the stack ring-3->0 transitions use) at `top`. Called by
/// the scheduler on every process switch; the boot value from [`init`] covers
/// the pre-scheduler period. A no-op on host builds.
pub fn set_rsp0(top: usize) {
    #[cfg(all(target_arch = "x86_64", not(test)))]
    {
        // Safety: single CPU; rsp0 is only read by the hardware on a ring-3->0
        // transition, which cannot occur while we are inside the scheduler
        // (IF=0) on this CPU.
        unsafe {
            let tss = core::ptr::addr_of_mut!(TSS);
            core::ptr::addr_of_mut!((*tss).rsp0).write_unaligned(top as u64);
        }
    }
    #[cfg(not(all(target_arch = "x86_64", not(test))))]
    {
        let _ = top;
    }
}

/// Build a 64-bit code/data segment descriptor.
///
/// `access` is the type/access byte (P/DPL/S/type), `flags` the G/D/B/L/AVL
/// nibble. Matches the descriptor bytes in `entry.s` for the shared selectors.
fn code_data_descriptor(base: u32, limit: u32, access: u8, flags: u8) -> u64 {
    (limit as u64 & 0xFFFF)                              // limit[15:0]
        | ((base as u64 & 0xFFFF) << 16)                 // base[15:0]
        | ((base as u64 & 0x00FF_0000) << 16)            // base[23:16]
        | ((access as u64) << 40)                        // type/access
        | ((((limit >> 16) as u64 & 0xF) | flags as u64) << 48) // limit[19:16] | flags
        | ((base as u64 & 0xFF00_0000) << 32)            // base[31:24]
}

/// Build the 16-byte system descriptor for a TSS. The first 8 bytes share the
/// code/data descriptor layout (base[15:0] at bits 16-31, access at bits
/// 40-47); the upper 8 bytes hold base[63:32].
fn system_descriptor(base: u64, limit: u16, access: u8) -> [u64; 2] {
    let lo = code_data_descriptor(base as u32, limit as u32, access, 0x0);
    let hi = ((base >> 32) & 0xFF)
        | (((base >> 40) & 0xFF) << 8)
        | (((base >> 48) & 0xFF) << 16)
        | (((base >> 56) & 0xFF) << 24);
    [lo, hi]
}

/// Install the GDT + TSS. Boot-only, single-threaded, called exactly once
/// before any #DF can be raised. A no-op on host builds.
pub fn init() {
    #[cfg(all(target_arch = "x86_64", not(test)))]
    {
        // `addr_of_mut!` on statics is safe; the dereferences below are the
        // unsafe part, done under the single-threaded boot-path comment.
        let gdt = core::ptr::addr_of_mut!(GDT);
        let tss = core::ptr::addr_of_mut!(TSS);

        // Safety: unique access during the single-threaded boot path. The
        // struct is packed, so fields are written unaligned via addr_of_mut.
        unsafe {
            core::ptr::addr_of_mut!((*tss).rsp0).write_unaligned(
                core::ptr::addr_of!(KERNEL_STACK) as usize as u64 + KERNEL_STACK_SIZE as u64,
            );
            core::ptr::addr_of_mut!((*tss).ist[0]).write_unaligned(
                core::ptr::addr_of!(DF_STACK) as usize as u64 + DF_STACK_SIZE as u64,
            );
        }

        let base = core::ptr::addr_of!(TSS) as usize as u64;
        let [tss_lo, tss_hi] = system_descriptor(base, 103, 0x89);

        // Safety: unique access during the single-threaded boot path.
        unsafe {
            (*gdt)[0] = 0;
            (*gdt)[1] = code_data_descriptor(0, 0xFFFFF, 0x9A, 0xCF); // 0x08 code32
            (*gdt)[2] = code_data_descriptor(0, 0xFFFFF, 0x92, 0xCF); // 0x10 data
            (*gdt)[3] = code_data_descriptor(0, 0xFFFFF, 0x9A, 0xAF); // 0x18 code64
            (*gdt)[4] = tss_lo; // 0x20 TSS
            (*gdt)[5] = tss_hi;
            (*gdt)[6] = code_data_descriptor(0, 0xFFFFF, 0xFA, 0xAF); // 0x30 code64 ring3
            (*gdt)[7] = code_data_descriptor(0, 0xFFFFF, 0xF2, 0xCF); // 0x38 data ring3
        }

        // Safety: GDT is a module static whose address is stable; GdtPtr is a
        // stack copy read only for the instruction duration.
        let gdt_ptr = GdtPtr {
            limit: (core::mem::size_of::<[u64; 8]>() - 1) as u16,
            base: core::ptr::addr_of!(GDT) as usize as u64,
        };
        let sel = TSS_SELECTOR;
        unsafe {
            core::arch::asm!("lgdt [{0}]", in(reg) &gdt_ptr, options(nostack, preserves_flags));
            core::arch::asm!("ltr [{0}]", in(reg) &sel, options(nostack, preserves_flags));
        }
    }
}
