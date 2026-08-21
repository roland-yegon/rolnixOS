#![no_std]
#![no_main]
#![deny(unsafe_op_in_unsafe_fn)]

//! Multiboot v1 -> long mode -> PMM + kernel heap + kernel address space
//! self-test harness.
//!
//! `entry.s` calls `rust_entry(magic, info)` with long mode and paging on,
//! physical RAM identity-mapped (low 4 GiB) and mirrored at
//! `PHYS_MAP_BASE`. This crate builds a `BootInfo` from the multiboot memory
//! map, runs `pmm_init`, installs the kernel heap and a Rust-built kernel
//! address space, then exercises the allocator / untyped / deferral / VMM /
//! heap paths and prints verdicts over COM1.

core::arch::global_asm!(
    include_str!("entry.s"),
    include_str!("user_prog.s"),
    options(att_syntax)
);

extern crate alloc;

mod multiboot;
mod serial;
mod shell;

use core::fmt::Write;
use core::panic::PanicInfo;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use rolnix_kernel::arch::frame_phys;
use rolnix_kernel::boot::{BootInfo, MemoryRegion, PmInitOut, RegionKind, UntypedSpec};
use rolnix_kernel::frame::{add_mapping, frame_info, pin_frame, remove_mapping, unpin_frame, OwnerTag};
use rolnix_kernel::spinlock::SpinLock;
use rolnix_kernel::untyped::{retype_frame, UntypedCap};
use rolnix_kernel::vmm::{Flags, PageTable};
use rolnix_kernel::{alloc_frame, heap_init, phys_to_virt, pmm_init, release_frame};

const PHYS_MAP_BASE: u64 = 0xFFFF_8000_0000_0000;
const FRAME_SIZE: usize = 4096;
const MAX_REGIONS: usize = 64;
const UNTYPED_LEN: u64 = 64 * 1024 * 1024;
/// Kernel heap carved out of RAM by `pmm_init` (boot-time, never moved).
const KERNEL_HEAP_SIZE: usize = 8 * 1024 * 1024;
/// A VA high in the canonical upper half, outside the identity/physmap
/// windows, used by the VMM self-test.
const TEST_VA: usize = 0xFFFF_9000_0000_0000;
const TEST_VA2: usize = 0xFFFF_9000_0020_0000;

/// PIT frequency used by the timer demo.
const TIMER_HZ: u32 = 100;
/// Number of IRQ0 ticks the timer demo waits for before declaring victory.
const TIMER_DEMO_TICKS: usize = 100;

/// IRQ0 (PIT) tick count, updated by `timer_handler`.
static TICKS: AtomicUsize = AtomicUsize::new(0);

/// The kernel `PageTable`, kept in a static so it survives the boot stack's
/// reuse: processes exit by re-entering the shell on the boot stack, whose new
/// frames overwrite `rust_entry`'s locals. Set once by `rust_entry` after
/// `activate`; used by `spawn_process` to map USER pages.
static mut KERNEL_PT: Option<PageTable> = None;

// ---------------------------------------------------------------------------
// Serial output
// ---------------------------------------------------------------------------

static WRITER: SpinLock<serial::SerialPort> = SpinLock::new(serial::SerialPort::const_new());
static INIT: AtomicBool = AtomicBool::new(false);

fn with_serial<F, R>(f: F) -> R
where
    F: FnOnce(&mut serial::SerialPort) -> R,
{
    // lock_irqsave disables interrupts while the lock is held. Without that, a
    // preempting tick could switch to a process that prints and spin forever
    // on the lock; it also guards handler re-entry mid-byte.
    let mut guard = WRITER.lock_irqsave();
    if !INIT.swap(true, Ordering::Relaxed) {
        guard.init();
    }
    f(&mut guard)
}

macro_rules! kprintln {
    ($($arg:tt)*) => {
        with_serial(|p| { let _ = writeln!(p, $($arg)*); })
    };
}

/// Print a string without a trailing newline (used for prompts and echo).
fn kprint(s: &str) {
    with_serial(|p| {
        let _ = p.write_str(s);
    });
}

/// Print a string followed by a newline (function form, for the shell).
fn kprint_line(s: &str) {
    with_serial(|p| {
        let _ = writeln!(p, "{s}");
    });
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    let mut buf = [0u8; 1024];
    let mut w = RawSink { buf: &mut buf, len: 0 };
    let _ = core::fmt::write(&mut w, format_args!("PANIC: {info}\r\n"));
    let len = w.len;
    // Raw (lock-free) so a panic inside a print cannot deadlock re-entering
    // the serial lock.
    serial::raw_write(&buf[..len]);
    halt()
}

/// A `fmt::Write` over a fixed stack buffer, for panic/fault reporting that
/// must not touch the serial lock.
struct RawSink<'a> {
    buf: &'a mut [u8],
    len: usize,
}

impl Write for RawSink<'_> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        for &b in s.as_bytes() {
            if self.len < self.buf.len() {
                self.buf[self.len] = b;
                self.len += 1;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Freestanding runtime builtins
// ---------------------------------------------------------------------------
//
// The kernel links with `-nostartfiles` and no libc, so the toolchain's
// prebuilt `alloc` rlib and compiler codegen expect these symbols to come
// from somewhere. They used to arrive via the C runtime objects; now they are
// provided here. With `panic = "abort"` the unwinding ones are never called.

#[no_mangle]
extern "C" fn memcpy(dst: *mut u8, src: *const u8, n: usize) -> *mut u8 {
    let mut i = 0;
    while i < n {
        // Safety: the caller promises dst/src valid for n bytes.
        unsafe {
            *dst.add(i) = *src.add(i);
        }
        i += 1;
    }
    dst
}

#[no_mangle]
extern "C" fn memset(s: *mut u8, c: i32, n: usize) -> *mut u8 {
    let mut i = 0;
    while i < n {
        // Safety: the caller promises s valid for n bytes.
        unsafe {
            *s.add(i) = c as u8;
        }
        i += 1;
    }
    s
}

#[no_mangle]
extern "C" fn memcmp(s1: *const u8, s2: *const u8, n: usize) -> i32 {
    let mut i = 0;
    while i < n {
        // Safety: the caller promises s1/s2 valid for n bytes.
        let a = unsafe { *s1.add(i) };
        let b = unsafe { *s2.add(i) };
        if a != b {
            return a as i32 - b as i32;
        }
        i += 1;
    }
    0
}

/// Resolved for `alloc`'s `DW.ref.rust_eh_personality`; unreachable with
/// `panic = "abort"`.
#[no_mangle]
extern "C" fn rust_eh_personality() {}

/// Unwinding resume point; never reached with `panic = "abort"`.
#[no_mangle]
extern "C" fn _Unwind_Resume() -> ! {
    loop {
        core::hint::spin_loop();
    }
}

/// Receives formatted exception reports from the kernel's IDT handler. May be
/// called while a normal print holds the serial lock (exception during
/// printing), so it must bypass the lock entirely.
fn fault_sink(bytes: &[u8]) {
    let s = core::str::from_utf8(bytes).unwrap_or("<non-utf8>\r\n");
    serial::raw_write(s.as_bytes());
}

fn halt() -> ! {
    loop {
        unsafe { core::arch::asm!("hlt") }
    }
}

// ---------------------------------------------------------------------------
// Memory map
// ---------------------------------------------------------------------------

fn kernel_image() -> (u64, u64) {
    // Safety: symbols come from the linker script (boot/link.ld); the array
    // is never moved, so the addresses are the actual load addresses.
    let start = core::ptr::addr_of!(_skernel) as usize as u64;
    let end = core::ptr::addr_of!(_ekernel) as usize as u64;
    (start, end - start)
}

extern "C" {
    static _skernel: u8;
    static _ekernel: u8;
    static boot_stack_top: u8;
}

// Boundaries of the ring-3 demo program (see `user_prog.s`). The bytes
// between them are copied into USER-mapped frames.
extern "C" {
    static _user_prog_start: u8;
    static _user_prog_end: u8;
}

/// Copy the multiboot map into `out`, frame-aligning usable entries and
/// sorting by start address. Returns the number of regions written.
fn fill_regions(out: &mut [MemoryRegion; MAX_REGIONS], info: usize) -> usize {
    let mut raw = [multiboot::MemEntry { start: 0, len: 0, kind: 0 }; MAX_REGIONS];
    let n = match multiboot::parse(info, &mut raw) {
        Some(n) => n,
        None => {
            kprintln!("PANIC: bootloader provided no usable memory map");
            halt();
        }
    };

    let mut count = 0usize;
    for e in &raw[..n] {
        let kind =
            if e.kind == multiboot::MT_USABLE { RegionKind::Usable } else { RegionKind::Reserved };
        let mut start = e.start;
        let mut len = e.len;
        if kind == RegionKind::Usable {
            // Keep only whole frames inside the reported range.
            start = (start + FRAME_SIZE as u64 - 1) & !(FRAME_SIZE as u64 - 1);
            let end = e.start.checked_add(e.len).unwrap_or(e.start) & !(FRAME_SIZE as u64 - 1);
            if end <= start {
                continue;
            }
            len = end - start;
        }
        out[count] = MemoryRegion { start, len, kind };
        count += 1;
    }

    // Insertion sort by start address (multiboot maps are normally ordered;
    // this makes the harness robust to those that are not).
    for i in 1..count {
        let key = out[i];
        let mut j = i;
        while j > 0 && out[j - 1].start > key.start {
            out[j] = out[j - 1];
            j -= 1;
        }
        out[j] = key;
    }
    count
}

/// Reserve `UNTYPED_LEN` off the top of the highest usable region, so the
/// kernel pool keeps all the low memory it can see.
fn pick_untyped_spec(regions: &[MemoryRegion]) -> UntypedSpec {
    let (mut best_start, mut best_end) = (0u64, 0u64);
    for r in regions {
        if r.kind == RegionKind::Usable && r.end() > best_end {
            best_start = r.start;
            best_end = r.end();
        }
    }
    if best_end == 0 {
        kprintln!("PANIC: no usable region for the untyped cap");
        halt();
    }
    let len = UNTYPED_LEN.min(best_end - best_start);
    UntypedSpec { base_phys: best_end - len, len }
}

// ---------------------------------------------------------------------------
// Self-test
// ---------------------------------------------------------------------------

fn self_test(caps: &[UntypedCap], out: PmInitOut, pt: &mut PageTable) -> ! {
    kprintln!(
        "pmm_init OK: total_frames={} free_frames={} frame_table_frames={} untyped={} heap={:#x}+{:#x} ram_end={:#x}",
        out.total_frames,
        out.free_frames,
        out.frame_table_frames,
        out.untyped_created,
        out.heap_phys,
        out.heap_len,
        out.ram_end,
    );
    kprintln!(
        "kernel address space active: pml4={:#x} translate(0x1000)={:?} translate(physmap)={:?}",
        pt.root_phys(),
        pt.translate(0x1000),
        pt.translate(rolnix_kernel::physmap_base()),
    );

    // --- alloc / release round trip through the physmap --------------------
    let a = alloc_frame().expect("alloc");
    let ptr = phys_to_virt(frame_phys(a));
    unsafe {
        ptr.write(0xAB);
        assert_eq!(ptr.read(), 0xAB);
    }
    assert_eq!(frame_info(a).owner_tag(), OwnerTag::Kernel);

    let b = alloc_frame().expect("alloc");
    assert_ne!(a, b);
    release_frame(a);
    let a2 = alloc_frame().expect("alloc");
    assert_eq!(a2, a, "released frame must be recycled");
    release_frame(b);
    release_frame(a2);

    // --- retype from the untyped region ------------------------------------
    let f1 = retype_frame(&caps[0], 7).expect("retype");
    assert_eq!(frame_info(f1).owner_tag(), OwnerTag::Domain);

    let mut count = 1u32;
    while retype_frame(&caps[0], 7).is_ok() {
        count += 1;
    }
    kprintln!("retyped {count} frames from the untyped region");

    // Once-only grant: a released frame returns to the pool, not the region.
    release_frame(f1);
    assert_eq!(frame_info(f1).owner_tag(), OwnerTag::Free);
    assert!(retype_frame(&caps[0], 7).is_err(), "region must stay exhausted");

    // --- deferral: pinned frame is Pending until drained --------------------
    let d = alloc_frame().expect("alloc");
    pin_frame(d);
    release_frame(d);
    assert_eq!(frame_info(d).owner_tag(), OwnerTag::Pending);
    let e = alloc_frame().expect("alloc");
    assert_ne!(e, d);
    unpin_frame(d);
    assert_eq!(frame_info(d).owner_tag(), OwnerTag::Free);
    assert_eq!(alloc_frame(), Some(d));
    release_frame(e);

    // --- mapping helpers ----------------------------------------------------
    let m = alloc_frame().expect("alloc");
    assert_eq!(add_mapping(m), 1);
    assert_eq!(remove_mapping(m), 0);
    release_frame(m);

    vmm_self_test(pt);
    heap_self_test();

    kprintln!("PMM SELF-TEST OK");
    timer_demo();
    run_shell()
}

/// IRQ0 handler: count timer ticks and drive the round-robin scheduler. Runs
/// in interrupt-gate context (IF=0), so it must be short. `frame` is the
/// preempted context (in ring 3 that is the user process's registers; in ring
/// 0 it is the current process on its kernel stack).
unsafe extern "C" fn timer_handler(_irq: u8, frame: &rolnix_kernel::InterruptFrame) {
    TICKS.fetch_add(1, Ordering::Relaxed);
    rolnix_kernel::sched::on_tick(frame);
}

/// Prove the whole interrupt stack works: remap the PIC, program the PIT at
/// 100 Hz, unmask IRQ0, enable interrupts and wait for 100 ticks.
fn timer_demo() {
    kprintln!("timer demo: PIC + PIT @ {TIMER_HZ} Hz, unmasking IRQ0, sti...");
    rolnix_kernel::pic::remap();
    rolnix_kernel::pit::set_frequency(TIMER_HZ);
    rolnix_kernel::set_irq_handler(0, Some(timer_handler));
    rolnix_kernel::pic::unmask(0);
    // Safety: the IDT has interrupt gates for 0x20..0x2F and IRQ0 is now
    // unmasked, so the PIT's first IRQ0 will be handled, not #GP'd.
    unsafe { rolnix_kernel::enable_interrupts(); }
    while TICKS.load(Ordering::Relaxed) < TIMER_DEMO_TICKS {
        core::hint::spin_loop();
    }
    // Safety: pairs with enable_interrupts above; stops IRQ delivery here.
    unsafe { rolnix_kernel::disable_interrupts(); }
    rolnix_kernel::pic::mask_all();
    let ticks = TICKS.load(Ordering::Relaxed);
    kprintln!("timer demo OK: {ticks} ticks at {TIMER_HZ} Hz");
}

/// Bring up the PS/2 keyboard, register the user-mode console, seed the
/// scheduler with the shell as the idle process, then hand over to it forever.
fn run_shell() -> ! {
    if !rolnix_kernel::keyboard::init() {
        kprintln!("PS/2 init failed; keyboard input unavailable");
    }
    rolnix_kernel::set_irq_handler(1, Some(rolnix_kernel::keyboard::irq_handler));
    rolnix_kernel::pic::unmask(1);
    // Keep the PIT counting so the shell's `ticks` command stays live.
    rolnix_kernel::pic::unmask(0);
    // Every ring-3 SYS_PUTC goes to COM1.
    rolnix_kernel::user::set_console(user_putc);
    // The shell is process 0; its kernel stack is the boot stack, which
    // survives because the shell never returns out of it.
    let boot_stack = core::ptr::addr_of!(boot_stack_top) as usize;
    rolnix_kernel::sched::init_idle(boot_stack);
    // Safety: the IDT has gates for 0x20..0x2F and IRQ0/IRQ1 handlers are
    // registered, so the first timer/keyboard IRQ is handled, not #GP'd.
    unsafe { rolnix_kernel::enable_interrupts(); }
    kprintln!("rolnix shell ready: type 'help' (PS/2 keyboard + COM1)");
    shell::run()
}

/// Deliberately fault (shell command `fault`) to demonstrate the IDT: the
/// #PF handler prints a report (vector, error code, CR2, RIP) and halts
/// instead of triple-faulting.
fn exception_demo() {
    kprintln!("exception demo: writing to unmapped VA 0xFFFF_9000_1000_0000");
    let p = 0xFFFF_9000_1000_0000 as *mut u64;
    // Safety: deliberate; the #PF handler is expected to report and halt.
    unsafe {
        p.write(0x1234_5678);
    }
    // Unreachable on a working IDT.
    kprintln!("exception demo: NO EXCEPTION (IDT broken!)");
}

/// Deliberately double-fault (shell command `double`) to prove the TSS + IST:
/// clobber RSP to an unmapped canonical VA and execute `ud2`. The #UD handler
/// pushes onto the unmapped stack -> #PF, the #PF handler pushes onto the
/// same stack -> #PF again -> #DF, which switches to `TSS.ist[0]` and is
/// reported instead of triple-faulting.
fn double_fault_demo() {
    kprintln!("double fault demo: clobbering RSP to an unmapped VA, then ud2");
    // Safety: deliberate; the #DF gate + IST are expected to report and halt.
    // `nostack` is deliberately absent: the asm replaces RSP.
    unsafe {
        core::arch::asm!(
            "mov rsp, {0}",
            "ud2",
            in(reg) 0xFFFF_9000_1000_0000usize,
            options(nomem, preserves_flags),
        );
    }
    // Unreachable on a working #DF gate.
    kprintln!("double fault demo: NO DOUBLE FAULT (TSS/IST broken!)");
}

/// Spawn a ring-3 process running the `_user_prog` blob, mapped into its
/// private 2 MiB region (`user_code_va`/`user_stack_top`). Returns the pid, or
/// `None` if the table is full or a frame cannot be allocated. The table
/// write, the mapping and the frame build all happen with IRQs off so a tick
/// can never observe a half-registered process.
fn spawn_process(name: &str) -> Option<u32> {
    // Safety: single CPU; paired irq_restore below.
    let saved = unsafe { rolnix_kernel::arch::irq_save() };
    let result = spawn_process_locked(name);
    // Safety: pairs with the irq_save above.
    unsafe { rolnix_kernel::arch::irq_restore(saved) };
    result
}

fn spawn_process_locked(name: &str) -> Option<u32> {
    // Safety: set once by `rust_entry` before the shell; the shell is
    // single-threaded, so the exclusive borrow is sound.
    let pt = unsafe { core::ptr::addr_of_mut!(KERNEL_PT).as_mut() }
        .and_then(|o| o.as_mut())
        .expect("kernel page table not initialized");
    let pid = rolnix_kernel::sched::free_slot()?;
    let code_va = rolnix_kernel::sched::user_code_va(pid);
    let stack_va = rolnix_kernel::sched::user_stack_top(pid) - FRAME_SIZE;
    let start = core::ptr::addr_of!(_user_prog_start) as usize;
    let end = core::ptr::addr_of!(_user_prog_end) as usize;
    let len = end - start;

    if pt.translate(code_va).is_some() {
        // The region already has a mapping (killed process reused the slot
        // without freeing frames): nothing to do, the image is still there.
    } else {
        let flags = Flags::PRESENT.with(Flags::WRITABLE).with(Flags::USER);
        let code_frames = (len + FRAME_SIZE - 1) / FRAME_SIZE;
        for i in 0..code_frames {
            let f = alloc_frame()?;
            let va = code_va + i * FRAME_SIZE;
            pt.map(va, frame_phys(f), flags).expect("map user code");
            let copy_len = core::cmp::min(FRAME_SIZE, len - i * FRAME_SIZE);
            // Safety: `va` is now mapped writable; the source is the kernel
            // image (identity-mapped, readable in ring 0).
            unsafe {
                core::ptr::copy_nonoverlapping(
                    (start + i * FRAME_SIZE) as *const u8,
                    va as *mut u8,
                    copy_len,
                );
            }
        }
        let sf = alloc_frame()?;
        pt.map(stack_va, frame_phys(sf), flags).expect("map user stack");
    }

    rolnix_kernel::sched::start_process(pid, code_va, name);
    kprintln!("spawned pid {pid} ('{name}') at {code_va:#x}, stack {stack_va:#x}");
    Some(pid)
}

/// `SYS_PUTC` sink: one byte to COM1.
unsafe extern "C" fn user_putc(c: u8) {
    with_serial(|p| p.write_byte(c));
}

/// Exercise the Rust-built kernel address space: map a pool frame at a high
/// VA, read/write through it (paging is on, so this is a real translation),
/// then unmap and verify the frame returns to the pool.
fn vmm_self_test(pt: &mut PageTable) {
    assert_eq!(pt.translate(0x1000), Some(0x1000), "identity window");
    assert_eq!(pt.translate(rolnix_kernel::physmap_base()), Some(0), "physmap window");
    assert_eq!(pt.translate(TEST_VA), None, "test VA must start unmapped");

    let flags = Flags::PRESENT.with(Flags::WRITABLE);
    let f = alloc_frame().expect("vmm frame");
    let pa = frame_phys(f);
    pt.map(TEST_VA, pa, flags).expect("map");
    assert_eq!(pt.translate(TEST_VA), Some(pa), "translate after map");
    // Paging is on, so this is a real translation: the write must land in the
    // mapped frame and read back.
    let page_ptr = TEST_VA as *mut u64;
    unsafe {
        page_ptr.write(0xDEAD_BEEF);
        assert_eq!(page_ptr.read(), 0xDEAD_BEEF);
    }
    pt.unmap(TEST_VA).expect("unmap");
    assert_eq!(pt.translate(TEST_VA), None, "translate after unmap");
    assert_eq!(frame_info(f).mappings(), 0);
    release_frame(f);
    assert_eq!(frame_info(f).owner_tag(), OwnerTag::Free, "frame back in pool");

    // A second map through a different PD entry must also work (grows a
    // second PT).
    let f2 = alloc_frame().expect("vmm frame 2");
    pt.map(TEST_VA2, frame_phys(f2), flags).expect("map 2");
    assert_eq!(pt.translate(TEST_VA2), Some(frame_phys(f2)));
    pt.unmap(TEST_VA2).expect("unmap 2");
    release_frame(f2);

    kprintln!("VMM SELF-TEST OK");
}

/// Exercise the kernel heap through the `alloc` crate.
fn heap_self_test() {
    let before = rolnix_kernel::heap_allocated();
    let mut v: Vec<u64> = Vec::new();
    for i in 0..1000u64 {
        v.push(i);
    }
    let mut s = String::new();
    let _ = write!(s, "heap string {:#x}", 0x1234u64);
    let b = Box::new(42u32);
    let sum: u64 = v.iter().sum();
    assert_eq!(v.len(), 1000);
    assert_eq!(sum, 1000 * 999 / 2);
    assert!(s.starts_with("heap string 0x1234"));
    assert_eq!(*b, 42);
    let peak = rolnix_kernel::heap_allocated();
    assert!(peak > before, "heap must have been used");
    drop(v);
    drop(s);
    drop(b);
    assert_eq!(rolnix_kernel::heap_allocated(), before, "heap must be fully reclaimed");
    kprintln!("KERNEL HEAP OK (allocated {} bytes at peak, total {})", peak, rolnix_kernel::heap_total());
}

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------

/// Called by `entry.s` from 64-bit mode. `magic` is the multiboot magic,
/// `info` the multiboot info pointer (both passed through from the bootloader
/// via EBX/EAX).
#[no_mangle]
pub extern "C" fn rust_entry(magic: u64, info: usize) {
    // Install the IDT before anything can fault, so every exception is
    // reported over COM1 instead of triple-faulting silently.
    rolnix_kernel::idt::set_fault_sink(fault_sink);
    rolnix_kernel::idt::init();
    // Load the GDT + TSS (kernel stack, IST double-fault stack) so the #DF
    // gate installed above can escalate onto a dedicated stack.
    rolnix_kernel::tss::init();
    if magic != multiboot::MAGIC as u64 {
        kprintln!("PANIC: bad multiboot magic {magic:#x}");
        halt();
    }
    let mut region_buf = [MemoryRegion { start: 0, len: 0, kind: RegionKind::Reserved };
        MAX_REGIONS];
    let n = fill_regions(&mut region_buf, info);
    let regions = &region_buf[..n];
    let ki = kernel_image();

    kprintln!("rolnix boot harness: {n} regions, kernel image {:#x}..{:#x}", ki.0, ki.0 + ki.1);
    for r in regions {
        kprintln!("  region {:#x} len {:#x} {:?}", r.start, r.len, r.kind);
    }

    let untyped = pick_untyped_spec(regions);
    kprintln!("untyped spec: {:#x} len {:#x}", untyped.base_phys, untyped.len);

    // Placeholder caps; `pmm_init` overwrites them via `out_untyped`.
    let mut caps = [unsafe { core::mem::zeroed::<UntypedCap>() }];
    let boot = BootInfo {
        physmap_base: PHYS_MAP_BASE as usize,
        regions,
        kernel_image: ki,
    };
    match pmm_init(&boot, core::slice::from_ref(&untyped), &mut caps, KERNEL_HEAP_SIZE) {
        Ok(out) => {
            // Install the kernel heap over the region pmm_init carved out.
            // Safety: pmm_init reserved and zeroed [heap_phys, +heap_len).
            heap_init(phys_to_virt(out.heap_phys as usize) as usize, out.heap_len as usize);

            // Build the kernel address space (identity + physmap windows) and
            // switch to it. The image, stack, frame table and heap all live in
            // the identity-mapped window, so execution continues seamlessly.
            let pt = match PageTable::new_kernel(out.ram_end as usize) {
                Some(pt) => pt,
                None => {
                    kprintln!("PANIC: cannot allocate kernel page tables");
                    halt();
                }
            };
            pt.activate();
            // Safety: single-threaded boot; store the table once, before the
            // shell runs, so it outlives any reuse of the boot stack.
            unsafe {
                *core::ptr::addr_of_mut!(KERNEL_PT) = Some(pt);
            }
            // Safety: stored above; the self-test runs before the stack is
            // ever reused, so this reference is the only one.
            let pt =
                unsafe { core::ptr::addr_of_mut!(KERNEL_PT).as_mut() }.expect("kernel pt is set");
            self_test(&caps, out, pt.as_mut().expect("kernel pt is set"))
        }
        Err(e) => {
            kprintln!("pmm_init FAILED: {e}");
            halt();
        }
    }
}
