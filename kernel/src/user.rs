//! Ring-3 user mode: `int 0x80` syscalls for the scheduler's processes.
//!
//! The scheduler iretq's a hand-built [`InterruptFrame`] into each process
//! (see [`crate::sched::start_process`]); from there the program issues
//! syscalls via `int 0x80` (gate DPL 3, vector 0x80):
//!   * [`SYS_PUTC`] (rax=0, rdi=byte): write a byte to the registered console.
//!   * [`SYS_GETC`] (rax=1): non-blocking keyboard poll; `u64::MAX` = none.
//!   * [`SYS_EXIT`] (rax=2): mark the process dead ([`crate::sched`]).
//!   * [`SYS_GETPID`] (rax=3): return the process id.
//!   * [`SYS_YIELD`] (rax=4): give up the rest of the timeslice.
//!
//! The three control syscalls rewrite nothing: the shared `syscall_common`
//! iretq returns the program to ring 3 unless the scheduler switched away,
//! in which case that same frame is iretq'd by `sched_switch` later. IRQs
//! from ring 3 work unchanged: the CPU switches to `TSS.rsp0`, `irq_common`
//! iretq's with the 5 CPU-pushed words, and the
//! [`InterruptFrame::user_rsp`]/[`user_ss`] fields expose the user stack.
//!
//! Host/build-stub note: same convention as `idt.rs`; on a non-x86 host build
//! everything degrades to no-ops so the crate still builds for `cargo test`.
#![cfg_attr(not(all(target_arch = "x86_64", not(test))), allow(dead_code))]

use core::sync::atomic::{AtomicUsize, Ordering};

#[cfg(all(target_arch = "x86_64", not(test)))]
use crate::idt::InterruptFrame;

/// Syscall numbers for the `int 0x80` gate.
pub const SYS_PUTC: u64 = 0;
pub const SYS_GETC: u64 = 1;
pub const SYS_EXIT: u64 = 2;
pub const SYS_GETPID: u64 = 3;
pub const SYS_YIELD: u64 = 4;

/// A ring-0 callback that emits one byte to the console on the user's behalf.
pub type ConsoleSink = unsafe extern "C" fn(u8);

static CONSOLE: AtomicUsize = AtomicUsize::new(0);

/// Register the byte sink `SYS_PUTC` writes to (e.g. the boot COM1 port).
/// Boot-only, once, before any user process can run.
pub fn set_console(sink: ConsoleSink) {
    CONSOLE.store(sink as usize, Ordering::Relaxed);
}

fn emit(c: u8) {
    let raw = CONSOLE.load(Ordering::Relaxed);
    if raw != 0 {
        // Safety: only `set_console` stores values here, always a valid
        // `ConsoleSink` fn pointer.
        let sink: ConsoleSink = unsafe { core::mem::transmute(raw) };
        unsafe { sink(c) };
    }
}

/// Dispatches `int 0x80` from ring 3. Runs in interrupt-gate context (IF=0)
/// on the `TSS.rsp0` stack; `iretq` in `syscall_common` returns to ring 3
/// unless `SYS_YIELD`/`SYS_EXIT` made the scheduler switch away first.
#[cfg(all(target_arch = "x86_64", not(test)))]
#[no_mangle]
pub extern "C" fn handle_syscall(frame: &mut InterruptFrame) {
    match frame.rax {
        SYS_PUTC => emit(frame.rdi as u8),
        SYS_GETC => {
            frame.rax = match crate::keyboard::poll_char() {
                Some(c) => c as u64,
                None => u64::MAX,
            };
        }
        SYS_EXIT => crate::sched::exit_current(frame),
        SYS_GETPID => frame.rax = crate::sched::current_pid() as u64,
        SYS_YIELD => crate::sched::yield_now(frame),
        _ => frame.rax = u64::MAX,
    }
}
