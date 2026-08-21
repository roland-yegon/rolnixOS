//! Minimal interactive shell over COM1 (serial) and the PS/2 keyboard.
//!
//! [`run`] never returns: it prompts on COM1, feeds each incoming character
//! (keyboard ring via IRQ1, serial polled) through a line editor, and
//! dispatches completed lines to a small command set. The fault demos are
//! commands, so the machine stays interactive until one of them (or `halt`)
//! stops it.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use super::{TICKS, halt, kprint, kprint_line, with_serial};

const MAX_LINE: usize = 127;

/// Run the shell forever.
pub fn run() -> ! {
    let mut line = String::new();
    loop {
        kprint("rolnix> ");
        line.clear();
        loop {
            // PS/2 keyboard: IRQ-driven ring buffer.
            if let Some(c) = rolnix_kernel::keyboard::poll_char() {
                if edit_line(&mut line, c) {
                    break;
                }
                continue;
            }
            // COM1: polled, outside the serial lock so echo can re-enter it.
            if let Some(b) = with_serial(|p| p.try_read()) {
                if edit_line(&mut line, b as char) {
                    break;
                }
                continue;
            }
            core::hint::spin_loop();
        }
        execute(&line);
    }
}

/// Feed one character into the line buffer. Returns `true` when the line is
/// complete (Enter), `false` otherwise.
fn edit_line(line: &mut String, c: char) -> bool {
    match c {
        '\n' | '\r' => {
            kprint("\r\n");
            true
        }
        '\u{8}' | '\u{7f}' => {
            if line.pop().is_some() {
                kprint("\u{8} \u{8}");
            }
            false
        }
        c if c.is_ascii_graphic() || c == ' ' => {
            if line.len() < MAX_LINE {
                line.push(c);
                let mut buf = [0u8; 4];
                kprint(c.encode_utf8(&mut buf));
            }
            false
        }
        _ => false,
    }
}

/// Execute one completed command line.
fn execute(line: &str) {
    let mut parts = line.split_whitespace();
    let Some(cmd) = parts.next() else {
        return;
    };
    let rest = parts.collect::<Vec<_>>().join(" ");
    match cmd {
        "help" => help(),
        "echo" => kprint_line(&rest),
        "info" => info(),
        "ticks" => {
            let t = TICKS.load(core::sync::atomic::Ordering::Relaxed);
            kprint_line(&format!("ticks = {t}"));
        }
        "spawn" => {
            let name = if rest.is_empty() { "proc" } else { &rest };
            match super::spawn_process(name) {
                Some(pid) => kprint_line(&format!("started pid {pid} ('{name}')")),
                None => kprint_line("spawn failed: no free slot or out of memory"),
            }
        }
        "ps" => ps(),
        "kill" => {
            if rest.is_empty() {
                kprint_line("usage: kill <pid>");
            } else if let Ok(pid) = rest.trim().parse::<u32>() {
                if rolnix_kernel::sched::kill(pid) {
                    kprint_line(&format!("pid {pid} killed"));
                } else {
                    kprint_line(&format!("kill {pid}: not a live process"));
                }
            } else {
                kprint_line("kill: pid must be a number");
            }
        }
        "fault" => super::exception_demo(),
        "double" => super::double_fault_demo(),
        "halt" | "exit" | "shutdown" => halt(),
        _ => kprint_line(&format!("unknown command '{cmd}' (type 'help')")),
    }
}

/// Print one line per live process (shell command `ps`).
fn ps() {
    let mut buf = [rolnix_kernel::sched::ProcDesc { pid: 0, running: false, name: [0; 16] };
        rolnix_kernel::sched::MAX_PROCESSES];
    let n = rolnix_kernel::sched::snapshot(&mut buf);
    kprint_line(&format!("{n} process(es):"));
    for d in buf.iter().take(n) {
        let name = core::str::from_utf8(&d.name)
            .unwrap_or("<bad>")
            .trim_end_matches('\0');
        let state = if d.running { "running" } else { "ready" };
        kprint_line(&format!("  pid {}  {state:<8} {name}", d.pid));
    }
}

fn help() {
    kprint_line("commands: help, echo <text>, info, ticks, spawn [name], ps, kill <pid>, fault, double, halt");
}

fn info() {
    let free = rolnix_kernel::free_frame_count();
    let allocated = rolnix_kernel::heap_allocated();
    let total = rolnix_kernel::heap_total();
    kprint_line(&format!("frames free: {free}   heap: {allocated}/{total}"));
}
