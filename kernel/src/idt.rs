//! x86-64 IDT: exception gates, per-vector stubs and a fault reporter.
//!
//! The IDT is fully static: vectors 0..32 get interrupt gates whose stubs
//! push `[vector, error_code]` (a dummy `0` where the CPU pushes none) and
//! tail-jump into `isr_common`, which saves the GPRs and calls
//! `handle_exception`. The handler reports the exception through the
//! registered [`FaultSink`] (the boot harness wires it to COM1) and halts.
//! The #DF gate (vector 8) additionally gets an IST index, so it runs on the
//! [`crate::tss`]'s dedicated double-fault stack.
//!
//! IRQ gates live at 0x20..0x2F (the [`crate::pic`] remap vectors). Their
//! stubs push `[vector, 0]` and jump into `irq_common`, which saves the same
//! frame, calls `handle_irq` (acknowledges the PIC, dispatches to a
//! registered [`IrqHandler`]) and iretq's back.
//!
//! Unhandled vectors (0x30..0xFF) stay all-zero (not present), so any stray
//! interrupt gate fault is reported as a #GP.
//!
//! Host/build-stub note: this file follows the `arch.rs` convention. On a
//! real x86-64 (non-test) build the asm stubs, CR2 access and halt loop are
//! compiled in; otherwise everything degrades to no-ops so the crate still
//! builds for `cargo test` on the host. Those host builds legitimately use
//! nothing below, so silence the dead-code noise there.
#![cfg_attr(not(all(target_arch = "x86_64", not(test))), allow(dead_code))]

use core::fmt;
use core::sync::atomic::{AtomicUsize, Ordering};

/// Code segment selector for 64-bit mode; matches `entry.s`'s GDT (0x18).
const GDT_CODE64: u16 = 0x18;

/// Interrupt-gate attribute byte: present, ring 0, 64-bit interrupt gate.
const GATE_ATTRS: u8 = 0x8E;
/// Same gate with DPL 3, so ring-3 code can enter via `int 0x80`.
const GATE_ATTRS_USER: u8 = 0xEE;

/// First vector of the remapped IRQ range; must match [`crate::pic::MASTER_BASE`].
const IRQ_BASE: u8 = crate::pic::MASTER_BASE;

// ---------------------------------------------------------------------------
// IDT layout
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
#[repr(C, packed)]
struct IdtEntry {
    offset_low: u16,
    selector: u16,
    ist: u8,
    attrs: u8,
    offset_mid: u16,
    offset_high: u32,
    reserved: u32,
}

impl IdtEntry {
    /// A not-present entry: triggering it raises #GP.
    const fn missing() -> IdtEntry {
        IdtEntry {
            offset_low: 0,
            selector: 0,
            ist: 0,
            attrs: 0,
            offset_mid: 0,
            offset_high: 0,
            reserved: 0,
        }
    }

    const fn gate(offset: u64, selector: u16, ist: u8, attrs: u8) -> IdtEntry {
        IdtEntry {
            offset_low: offset as u16,
            selector,
            ist,
            attrs,
            offset_mid: (offset >> 16) as u16,
            offset_high: (offset >> 32) as u32,
            reserved: 0,
        }
    }
}

#[repr(C, align(16))]
struct Idt {
    entries: [IdtEntry; 256],
}

/// 10-byte descriptor-table pointer loaded via `lidt`.
#[repr(C, packed)]
struct Idtr {
    limit: u16,
    base: u64,
}

// ---------------------------------------------------------------------------
// Per-vector stubs and the common save/restore frame
// ---------------------------------------------------------------------------

/// What the CPU pushed (plus our vector/error-code words) looks like to
/// `handle_exception`. Field order matches the stack: lowest address first.
/// `isr_common` pushes the interrupted `rsp`, then `r15..rax`; the stub
/// pushed `[vector]` and `[error_code]`; the CPU pushed `[rip][cs][rflags]`
/// for a ring-0 interrupt, or `[rip][cs][rflags][user_rsp][user_ss]` for a
/// ring-3 interrupt (the extra two words are the CPU-pushed RSP/SS). The
/// `user_*` fields therefore only carry meaningful values for ring-3->0
/// transitions; for ring-0 interrupts they read whatever is below the frame.
#[repr(C)]
pub struct InterruptFrame {
    pub rsp: u64,
    pub r15: u64,
    pub r14: u64,
    pub r13: u64,
    pub r12: u64,
    pub r11: u64,
    pub r10: u64,
    pub r9: u64,
    pub r8: u64,
    pub rbp: u64,
    pub rdi: u64,
    pub rsi: u64,
    pub rdx: u64,
    pub rcx: u64,
    pub rbx: u64,
    pub rax: u64,
    pub vector: u64,
    pub error_code: u64,
    pub rip: u64,
    pub cs: u64,
    pub rflags: u64,
    pub user_rsp: u64,
    pub user_ss: u64,
}

// `ec` = 1 for vectors that carry a hardware error code; 0 otherwise.
// Exception stubs push `[vector, error_code]` (a dummy 0 where the CPU pushes
// none) and tail-jump to `isr_common`. IRQ stubs push `[vector, 0]` and jump
// to `irq_common`. Both go through the same `push_frame`/`pop_frame` save
// sequence; `irq_common` must preserve the frame pointer across the handler
// call in a callee-saved register (r12), unlike `isr_common` whose handler
// never returns.
#[cfg(all(target_arch = "x86_64", not(test)))]
core::arch::global_asm!(
    "
.section .text
.macro push_frame
    push rax
    push rbx
    push rcx
    push rdx
    push rsi
    push rdi
    push rbp
    push r8
    push r9
    push r10
    push r11
    push r12
    push r13
    push r14
    push r15
    # original RSP = current RSP + 15 GPRs (120) + vector/error_code (16) + CPU rip/cs/rflags (24) = 160
    lea rax, [rsp + 160]
    push rax
.endm
.macro pop_frame
    pop rax
    pop r15
    pop r14
    pop r13
    pop r12
    pop r11
    pop r10
    pop r9
    pop r8
    pop rbp
    pop rdi
    pop rsi
    pop rdx
    pop rcx
    pop rbx
    pop rax
.endm
.macro isr_stub num, ec
.global isr_\\num
isr_\\num:
    .if \\ec
    push \\num
    .else
    push 0
    push \\num
    .endif
    jmp isr_common
.endm
.macro irq_stub num, vec
.global irq_\\num
irq_\\num:
    push 0
    push \\vec
    jmp irq_common
.endm
isr_stub 0, 0
isr_stub 1, 0
isr_stub 2, 0
isr_stub 3, 0
isr_stub 4, 0
isr_stub 5, 0
isr_stub 6, 0
isr_stub 7, 0
isr_stub 8, 1
isr_stub 9, 0
isr_stub 10, 1
isr_stub 11, 1
isr_stub 12, 1
isr_stub 13, 1
isr_stub 14, 1
isr_stub 15, 0
isr_stub 16, 0
isr_stub 17, 1
isr_stub 18, 0
isr_stub 19, 0
isr_stub 20, 0
isr_stub 21, 0
isr_stub 22, 0
isr_stub 23, 0
isr_stub 24, 0
isr_stub 25, 0
isr_stub 26, 0
isr_stub 27, 0
isr_stub 28, 0
isr_stub 29, 0
isr_stub 30, 0
isr_stub 31, 0
isr_common:
    push_frame
    mov rdi, rsp
    and rsp, -16
    call {exc_handler}
    mov rsp, rdi
    pop_frame
    add rsp, 16
    iretq
irq_stub 0, 0x20
irq_stub 1, 0x21
irq_stub 2, 0x22
irq_stub 3, 0x23
irq_stub 4, 0x24
irq_stub 5, 0x25
irq_stub 6, 0x26
irq_stub 7, 0x27
irq_stub 8, 0x28
irq_stub 9, 0x29
irq_stub 10, 0x2a
irq_stub 11, 0x2b
irq_stub 12, 0x2c
irq_stub 13, 0x2d
irq_stub 14, 0x2e
irq_stub 15, 0x2f
irq_common:
    push_frame
    mov rdi, rsp
    mov r12, rsp
    and rsp, -16
    call {irq_handler}
    mov rsp, r12
    pop_frame
    add rsp, 16
    iretq
.macro syscall_stub num
.global isr_\\num
isr_\\num:
    push 0
    push \\num
    jmp syscall_common
.endm
syscall_stub 0x80
syscall_common:
    push_frame
    mov rdi, rsp
    mov r12, rsp
    and rsp, -16
    call {syscall_handler}
    mov rsp, r12
    pop_frame
    add rsp, 16
    iretq
",
    exc_handler = sym handle_exception,
    irq_handler = sym handle_irq,
    syscall_handler = sym crate::user::handle_syscall,
);

// ---------------------------------------------------------------------------
// Static IDT + IDTR
// ---------------------------------------------------------------------------

extern "C" {
    fn isr_0();
    fn isr_1();
    fn isr_2();
    fn isr_3();
    fn isr_4();
    fn isr_5();
    fn isr_6();
    fn isr_7();
    fn isr_8();
    fn isr_9();
    fn isr_10();
    fn isr_11();
    fn isr_12();
    fn isr_13();
    fn isr_14();
    fn isr_15();
    fn isr_16();
    fn isr_17();
    fn isr_18();
    fn isr_19();
    fn isr_20();
    fn isr_21();
    fn isr_22();
    fn isr_23();
    fn isr_24();
    fn isr_25();
    fn isr_26();
    fn isr_27();
    fn isr_28();
    fn isr_29();
    fn isr_30();
    fn isr_31();
}

const STUBS: [unsafe extern "C" fn(); 32] = [
    isr_0, isr_1, isr_2, isr_3, isr_4, isr_5, isr_6, isr_7, isr_8, isr_9, isr_10,
    isr_11, isr_12, isr_13, isr_14, isr_15, isr_16, isr_17, isr_18, isr_19, isr_20,
    isr_21, isr_22, isr_23, isr_24, isr_25, isr_26, isr_27, isr_28, isr_29, isr_30,
    isr_31,
];

extern "C" {
    fn irq_0();
    fn irq_1();
    fn irq_2();
    fn irq_3();
    fn irq_4();
    fn irq_5();
    fn irq_6();
    fn irq_7();
    fn irq_8();
    fn irq_9();
    fn irq_10();
    fn irq_11();
    fn irq_12();
    fn irq_13();
    fn irq_14();
    fn irq_15();
}

// Ring-3 syscall gate (`int $0x80`) stub.
extern "C" {
    fn isr_0x80();
}

/// Stubs for the remapped IRQ vectors 0x20..0x2F (IRQ0..IRQ15).
const IRQ_STUBS: [unsafe extern "C" fn(); 16] = [
    irq_0, irq_1, irq_2, irq_3, irq_4, irq_5, irq_6, irq_7, irq_8, irq_9, irq_10,
    irq_11, irq_12, irq_13, irq_14, irq_15,
];

/// Built once at boot: every exception vector gets an interrupt gate; the
/// rest stay not-present so stray interrupts surface as a #GP.
static mut IDT: Idt = Idt { entries: [IdtEntry::missing(); 256] };
static mut IDTR: Idtr = Idtr {
    limit: (core::mem::size_of::<Idt>() - 1) as u16,
    base: 0,
};

/// Load the static IDT into the processor.
///
/// Boot-only, single-threaded: called exactly once, before any exception can
/// be raised, and never mutated afterwards.
pub fn init() {
    #[cfg(all(target_arch = "x86_64", not(test)))]
    {
        // Safety: single-threaded boot path, called exactly once, before any
        // exception can be raised.
        let idt = core::ptr::addr_of_mut!(IDT);
        let idtr = core::ptr::addr_of_mut!(IDTR);
        for (i, stub) in STUBS.iter().enumerate() {
            // #DF gets an IST so its handler runs on the TSS's dedicated
            // stack (the current stack is usually what caused the #DF).
            let ist = if i == crate::tss::DF_VECTOR { crate::tss::DF_GATE_IST } else { 0 };
            // Safety: unique access during the single-threaded boot path.
            unsafe {
                (*idt).entries[i] =
                    IdtEntry::gate(*stub as usize as u64, GDT_CODE64, ist, GATE_ATTRS);
            }
        }
        for (i, stub) in IRQ_STUBS.iter().enumerate() {
            // Safety: unique access during the single-threaded boot path.
            unsafe {
                (*idt).entries[IRQ_BASE as usize + i] =
                    IdtEntry::gate(*stub as usize as u64, GDT_CODE64, 0, GATE_ATTRS);
            }
        }
        // Vector 0x80: ring-3 syscall gate (`int $0x80`). DPL 3 lets user code
        // in; the CPU switches to `TSS.rsp0` and the handler iretq's back.
        // Safety: unique access during the single-threaded boot path.
        unsafe {
            (*idt).entries[0x80] =
                IdtEntry::gate(isr_0x80 as *const () as usize as u64, GDT_CODE64, 0, GATE_ATTRS_USER);
        }
        // Safety: IDT is a module static whose address is stable.
        unsafe {
            (*idtr).base = core::ptr::addr_of!(IDT) as usize as u64;
        }
        // Safety: IDTR now points at the fully populated IDT.
        unsafe {
            core::arch::asm!("lidt [{0}]", in(reg) idtr, options(nostack, preserves_flags));
        }
    }
}

// ---------------------------------------------------------------------------
// Fault reporting
// ---------------------------------------------------------------------------

/// A sink receiving the formatted exception report, byte by byte.
pub type FaultSink = fn(&[u8]);

static FAULT_SINK: AtomicUsize = AtomicUsize::new(0);

/// Register the sink that receives exception reports. The boot harness wires
/// this to its serial port; until set, reports go nowhere and the kernel
/// halts.
pub fn set_fault_sink(sink: FaultSink) {
    FAULT_SINK.store(sink as usize, Ordering::Relaxed);
}

fn emit(bytes: &[u8]) {
    let raw = FAULT_SINK.load(Ordering::Relaxed);
    if raw != 0 {
        // Safety: only `set_fault_sink` stores values here, always a valid
        // `FaultSink` fn pointer.
        let sink: FaultSink = unsafe { core::mem::transmute(raw) };
        sink(bytes);
    }
}

struct SinkWriter;

impl fmt::Write for SinkWriter {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        emit(s.as_bytes());
        Ok(())
    }
}

const EXCEPTION_NAMES: [&str; 32] = [
    "divide error",
    "debug",
    "nmi",
    "breakpoint",
    "overflow",
    "bound range",
    "invalid opcode",
    "device not available",
    "double fault",
    "coprocessor overrun",
    "invalid tss",
    "segment not present",
    "stack fault",
    "general protection",
    "page fault",
    "reserved",
    "x87 exception",
    "alignment check",
    "machine check",
    "simd exception",
    "virtualization",
    "reserved",
    "reserved",
    "reserved",
    "reserved",
    "reserved",
    "reserved",
    "reserved",
    "reserved",
    "reserved",
    "reserved",
    "reserved",
];

fn page_fault_reason(code: u64) -> &'static str {
    match code & 0x1F {
        0b00000 => "not present, read",
        0b00001 => "protection violation, read",
        0b00010 => "not present, write",
        0b00011 => "protection violation, write",
        0b00100 => "not present, instruction fetch",
        0b00101 => "protection violation, instruction fetch",
        0b00110 => "not present, write, reserved-bit",
        0b01010 => "not present, write, user-mode",
        0b01110 => "not present, write, user-mode, reserved-bit",
        0b10010 => "not present, write, instruction fetch",
        _ => "unspecified",
    }
}

/// Report the exception described by `frame`, then halt. Never returns.
#[cfg(all(target_arch = "x86_64", not(test)))]
#[no_mangle]
extern "C" fn handle_exception(frame: &InterruptFrame) -> ! {
    let mut w = SinkWriter;
    let name = EXCEPTION_NAMES
        .get(frame.vector as usize)
        .copied()
        .unwrap_or("reserved");
    let _ = fmt::Write::write_fmt(
        &mut w,
        format_args!("\r\nEXCEPTION #{:02} {}", frame.vector, name),
    );
    if frame.vector == 14 {
        let _ = fmt::Write::write_fmt(
            &mut w,
            format_args!(
                " ({}): cr2={:#x}",
                page_fault_reason(frame.error_code),
                read_cr2()
            ),
        );
    } else if frame.error_code != 0 {
        let _ = fmt::Write::write_fmt(
            &mut w,
            format_args!(": error_code={:#x}", frame.error_code),
        );
    }
    let _ = fmt::Write::write_fmt(
        &mut w,
        format_args!(
            "\r\n  rip={:#x} cs={:#x} rflags={:#x} rsp={:#x}\r\n  rax={:#x} rbx={:#x} rbp={:#x} rdi={:#x}\r\n",
            frame.rip,
            frame.cs,
            frame.rflags,
            frame.rsp,
            frame.rax,
            frame.rbx,
            frame.rbp,
            frame.rdi,
        ),
    );
    halt_loop()
}

#[cfg(all(target_arch = "x86_64", not(test)))]
fn read_cr2() -> u64 {
    let value: u64;
    // Safety: reading CR2 has no side effects.
    unsafe {
        core::arch::asm!("mov {0}, cr2", out(reg) value, options(nostack));
    }
    value
}

#[cfg(all(target_arch = "x86_64", not(test)))]
fn halt_loop() -> ! {
    loop {
        // Safety: hlt is always available in ring 0; the CPU waits for the
        // next interrupt (none are enabled), so this parks the machine.
        unsafe {
            core::arch::asm!("hlt", options(nostack));
        }
    }
}

// ---------------------------------------------------------------------------
// IRQ dispatch
// ---------------------------------------------------------------------------

/// A handler registered for one of the 16 remapped IRQs.
///
/// Runs in interrupt-gate context (maskable interrupts disabled): it must be
/// short, must not block, and must not `halt` (the CPU will not be woken).
/// `frame` is the full saved [`InterruptFrame`] on the interrupted process's
/// kernel stack; a scheduler may stash its address to resume the process.
pub type IrqHandler = unsafe extern "C" fn(irq: u8, frame: &InterruptFrame);

/// One slot per IRQ (0..15). Populated at boot via [`set_irq_handler`], before
/// interrupts are enabled; read by `handle_irq` with interrupts disabled.
static mut IRQ_HANDLERS: [Option<IrqHandler>; 16] = [None; 16];

/// Register `handler` for `irq` (0..15), replacing any previous one. Pass
/// `None` to unregister.
///
/// Boot-only: call before enabling interrupts on the single boot CPU.
pub fn set_irq_handler(irq: u8, handler: Option<IrqHandler>) {
    #[cfg(all(target_arch = "x86_64", not(test)))]
    {
        // Safety: only the single-threaded boot path writes this, before
        // interrupts are enabled, and only with valid fn pointers.
        unsafe {
            *core::ptr::addr_of_mut!(IRQ_HANDLERS).cast::<Option<IrqHandler>>().add(irq as usize) =
                handler;
        }
    }
    #[cfg(not(all(target_arch = "x86_64", not(test))))]
    {
        let _ = (irq, handler);
    }
}

/// Acknowledge the IRQ and run its registered handler, if any.
#[cfg(all(target_arch = "x86_64", not(test)))]
#[no_mangle]
extern "C" fn handle_irq(frame: &InterruptFrame) {
    let irq = (frame.vector - IRQ_BASE as u64) as u8;
    // EOI first so a slow handler does not hold the line; with interrupt
    // gates IF stays clear until iretq, so no nested IRQ can re-enter us.
    crate::pic::send_eoi(irq);
    // Safety: the registry only ever holds valid `IrqHandler` pointers or
    // None, written once at boot; reads happen with interrupts disabled.
    let handler = unsafe {
        core::ptr::addr_of!(IRQ_HANDLERS)
            .cast::<Option<IrqHandler>>()
            .add(irq as usize)
            .read()
    };
    if let Some(h) = handler {
        // Safety: `h` came from the registry, which only stores handlers
        // meant to run in this exact interrupt context.
        unsafe { h(irq, frame) };
    }
}
