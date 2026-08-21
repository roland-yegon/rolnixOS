//! PS/2 keyboard driver: 8042 controller init + scancode-set-1 translation.
//!
//! [`init`] resets the 8042 and the keyboard and enables keyboard IRQs; from
//! then on every IRQ1 pushes the translated character into a small ring
//! buffer that consumers drain with [`poll_char`]. Shift and caps-lock state
//! is tracked inside the IRQ handler; extended (0xE0-prefixed) keys are
//! consumed but not yet mapped.
//!
//! Host/build-stub note: like `arch.rs`, the port-touching pieces are real
//! only on x86-64 non-test builds; the pure translation table always
//! compiles, and the host build's remaining dead code is silenced.
#![cfg_attr(not(all(target_arch = "x86_64", not(test))), allow(dead_code))]

use crate::arch::{irq_restore, irq_save};
#[cfg(all(target_arch = "x86_64", not(test)))]
use crate::io;

const PS2_DATA: u16 = 0x60;
const PS2_STATUS: u16 = 0x64;
const PS2_CMD: u16 = 0x64;

// 8042 controller commands (written to port 0x64).
const CMD_DISABLE_KBD: u8 = 0xAD;
const CMD_ENABLE_KBD: u8 = 0xAE;
const CMD_CONTROLLER_SELF_TEST: u8 = 0xAA;
const CMD_READ_CONFIG: u8 = 0x20;
const CMD_WRITE_CONFIG: u8 = 0x60;

/// Config bit 0: enable the keyboard's IRQ line.
const CONFIG_KBD_INT: u8 = 0x01;
/// Config bit 6: 8042 translates the keyboard's set-2 codes to set 1.
/// Not relied on (QEMU ignores it): scancode set 1 is selected directly.
const CONFIG_TRANSLATE: u8 = 0x40;

// Keyboard commands (written to port 0x60).
const KC_RESET: u8 = 0xFF;
const KC_ENABLE_SCANNING: u8 = 0xF4;
const KC_SET_SCANCODE_SET: u8 = 0xF0;
/// Parameter for [`KC_SET_SCANCODE_SET`]: send the XT (set 1) codes.
const SCANCODE_SET_1: u8 = 0x01;

// 8042 status register bits.
const STATUS_OUTPUT_FULL: u8 = 0x01;
const STATUS_INPUT_FULL: u8 = 0x02;

// Scancode-set-1 special codes.
const SC_E0: u8 = 0xE0;
const SC_LSHIFT: u8 = 0x2A;
const SC_RSHIFT: u8 = 0x36;
const SC_CAPS: u8 = 0x3A;
const BREAK_BIT: u8 = 0x80;

// ---------------------------------------------------------------------------
// Character ring buffer (IRQ1 producer, boot-harness consumer)
// ---------------------------------------------------------------------------

const RING_SIZE: usize = 64;
static mut RING: [u8; RING_SIZE] = [0; RING_SIZE];
static mut RING_HEAD: usize = 0;
static mut RING_TAIL: usize = 0;

// IRQ state maintained by the handler. Reading/writing these from the
// consumer is safe because `irq_save` disables interrupts around its access.
static mut EXTENDED: bool = false;
static mut SHIFT: bool = false;
static mut CAPS: bool = false;

/// Take the next typed character, if any. Interrupt-safe: disables
/// interrupts for the duration of the pop.
pub fn poll_char() -> Option<char> {
    // Safety: irq_save serializes us against the IRQ1 producer.
    let flags = unsafe { irq_save() };
    let c = unsafe {
        if RING_HEAD != RING_TAIL {
            let c = RING[RING_TAIL];
            RING_TAIL = (RING_TAIL + 1) % RING_SIZE;
            Some(c)
        } else {
            None
        }
    };
    // Safety: restores the exact flags captured above.
    unsafe { irq_restore(flags) };
    c.map(|c| c as char)
}

fn push_char(c: u8) {
    // Safety: irq_save serializes us against the poller; we are already in
    // interrupt context but the save makes the ordering explicit.
    let flags = unsafe { irq_save() };
    unsafe {
        let head = RING_HEAD;
        let next = (head + 1) % RING_SIZE;
        if next != RING_TAIL {
            RING[head] = c;
            RING_HEAD = next;
        }
    }
    // Safety: restores the exact flags captured above.
    unsafe { irq_restore(flags) };
}

// ---------------------------------------------------------------------------
// IRQ1 handler
// ---------------------------------------------------------------------------

/// IRQ1 entry: read one scancode, update key state and queue any character.
///
/// Runs in interrupt-gate context with maskable interrupts disabled; keep it
/// short and do not block.
///
/// # Safety
///
/// Only callable from the IDT IRQ1 gate (ring 0, IF=0). It assumes the 8042
/// has a scancode pending in its output buffer and that the keyboard state is
/// not concurrently touched by another CPU.
#[cfg(all(target_arch = "x86_64", not(test)))]
#[no_mangle]
pub unsafe extern "C" fn irq_handler(_irq: u8, _frame: &crate::idt::InterruptFrame) {
    // Safety: IRQ1 means the 8042 has a scancode in its output buffer. The
    // state below is only touched here (IF=0) and by poll_char under
    // irq_save, so no two CPUs or re-entered handlers race on it.
    unsafe {
        let code = io::inb(PS2_DATA);

        // Consume one extended (0xE0) make/break without mapping it yet.
        if EXTENDED {
            EXTENDED = false;
            return;
        }
        if code == SC_E0 {
            EXTENDED = true;
            return;
        }
        if code & BREAK_BIT != 0 {
            // Key release: only shift release resets state.
            if code & !BREAK_BIT == SC_LSHIFT || code & !BREAK_BIT == SC_RSHIFT {
                SHIFT = false;
            }
            return;
        }
        match code {
            SC_LSHIFT | SC_RSHIFT => SHIFT = true,
            SC_CAPS => CAPS = !CAPS,
            _ => {
                if let Some(c) = translate(code, SHIFT, CAPS) {
                    push_char(c as u8);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Scancode -> character translation (pure, always compiled)
// ---------------------------------------------------------------------------

/// Scancode-set-1 make codes for the printable keys, lowercase / unshifted.
const BASE_CHARS: [(u8, char); 52] = [
    (0x02, '1'),
    (0x03, '2'),
    (0x04, '3'),
    (0x05, '4'),
    (0x06, '5'),
    (0x07, '6'),
    (0x08, '7'),
    (0x09, '8'),
    (0x0A, '9'),
    (0x0B, '0'),
    (0x1E, 'a'),
    (0x30, 'b'),
    (0x2E, 'c'),
    (0x20, 'd'),
    (0x12, 'e'),
    (0x21, 'f'),
    (0x22, 'g'),
    (0x23, 'h'),
    (0x17, 'i'),
    (0x24, 'j'),
    (0x25, 'k'),
    (0x26, 'l'),
    (0x32, 'm'),
    (0x31, 'n'),
    (0x18, 'o'),
    (0x19, 'p'),
    (0x10, 'q'),
    (0x13, 'r'),
    (0x1F, 's'),
    (0x14, 't'),
    (0x16, 'u'),
    (0x2F, 'v'),
    (0x11, 'w'),
    (0x2D, 'x'),
    (0x15, 'y'),
    (0x2C, 'z'),
    (0x0C, '-'),
    (0x0D, '='),
    (0x1A, '['),
    (0x1B, ']'),
    (0x2B, '\\'),
    (0x27, ';'),
    (0x28, '\''),
    (0x29, '`'),
    (0x33, ','),
    (0x34, '.'),
    (0x35, '/'),
    (0x0E, '\u{8}'), // backspace
    (0x0F, '\t'),    // tab
    (0x1C, '\n'),    // enter
    (0x39, ' '),     // space
    (0x01, '\u{1b}'), // escape
];

/// Map a make code to its character, applying shift and caps-lock.
fn translate(code: u8, shift: bool, caps: bool) -> Option<char> {
    let ch = BASE_CHARS.iter().find(|(c, _)| *c == code)?.1;
    let lower = ch.is_ascii_lowercase();
    if lower {
        let upper = shift ^ caps;
        Some(if upper { (ch as u8 - 32) as char } else { ch })
    } else if shift {
        Some(shifted_symbol(ch).unwrap_or(ch))
    } else {
        Some(ch)
    }
}

/// Shifted US-layout symbols for the digit and punctuation rows.
fn shifted_symbol(ch: char) -> Option<char> {
    Some(match ch {
        '1' => '!',
        '2' => '@',
        '3' => '#',
        '4' => '$',
        '5' => '%',
        '6' => '^',
        '7' => '&',
        '8' => '*',
        '9' => '(',
        '0' => ')',
        '-' => '_',
        '=' => '+',
        '[' => '{',
        ']' => '}',
        '\\' => '|',
        ';' => ':',
        '\'' => '"',
        '`' => '~',
        ',' => '<',
        '.' => '>',
        '/' => '?',
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// 8042 + keyboard init
// ---------------------------------------------------------------------------

/// Bring the PS/2 keyboard up: 8042 self-test, IRQ enable, keyboard reset,
/// scanning on. Returns false if the controller or device misbehaved.
pub fn init() -> bool {
    #[cfg(all(target_arch = "x86_64", not(test)))]
    {
        // Safety: the 8042 legacy ports are always present on x86 and the
        // boot path is single-threaded; nothing else touches them yet.
        unsafe {
            if !wait_input_clear() {
                return false;
            }
            io::outb(PS2_CMD, CMD_DISABLE_KBD);
            if !wait_input_clear() {
                return false;
            }
            // Flush anything the bootloader left in the output buffer.
            while status() & STATUS_OUTPUT_FULL != 0 {
                let _ = io::inb(PS2_DATA);
            }
            // Controller self-test: must answer 0x55.
            io::outb(PS2_CMD, CMD_CONTROLLER_SELF_TEST);
            if read_byte() != Some(0x55) {
                return false;
            }
            // Read config, enable the keyboard IRQ bit, turn the 8042
            // translation OFF (we select the scancode set directly), write
            // it back.
            if !wait_input_clear() {
                return false;
            }
            io::outb(PS2_CMD, CMD_READ_CONFIG);
            let cfg = match read_byte() {
                Some(c) => c,
                None => return false,
            };
            if !wait_input_clear() {
                return false;
            }
            io::outb(PS2_CMD, CMD_WRITE_CONFIG);
            if !wait_input_clear() {
                return false;
            }
            io::outb(PS2_DATA, (cfg | CONFIG_KBD_INT) & !CONFIG_TRANSLATE);
            if !wait_input_clear() {
                return false;
            }
            io::outb(PS2_CMD, CMD_ENABLE_KBD);
            // Keyboard reset: ACK then BAT (0xAA); tolerate a missing BAT.
            if !wait_input_clear() {
                return false;
            }
            io::outb(PS2_DATA, KC_RESET);
            if read_byte() != Some(0xFA) {
                return false;
            }
            let _bat = read_byte();
            // Pin the keyboard to scancode set 1 (the XT codes the
            // translation table below understands), independent of the 8042.
            if !wait_input_clear() {
                return false;
            }
            io::outb(PS2_DATA, KC_SET_SCANCODE_SET);
            if read_byte() != Some(0xFA) {
                return false;
            }
            if !wait_input_clear() {
                return false;
            }
            io::outb(PS2_DATA, SCANCODE_SET_1);
            if read_byte() != Some(0xFA) {
                return false;
            }
            // Enable scanning: further keystrokes now produce scancodes.
            if !wait_input_clear() {
                return false;
            }
            io::outb(PS2_DATA, KC_ENABLE_SCANNING);
            read_byte() == Some(0xFA)
        }
    }
    #[cfg(not(all(target_arch = "x86_64", not(test))))]
    {
        true
    }
}

#[cfg(all(target_arch = "x86_64", not(test)))]
fn status() -> u8 {
    // Safety: reading the 8042 status register has no side effects.
    unsafe { io::inb(PS2_STATUS) }
}

#[cfg(all(target_arch = "x86_64", not(test)))]
fn wait_input_clear() -> bool {
    // Safety: the loop only reads the status register via `status`.
    for _ in 0..100_000 {
        if status() & STATUS_INPUT_FULL == 0 {
            return true;
        }
        core::hint::spin_loop();
    }
    false
}

#[cfg(all(target_arch = "x86_64", not(test)))]
fn read_byte() -> Option<u8> {
    // Safety: only reads the data port once the controller signals output.
    unsafe {
        for _ in 0..100_000 {
            if status() & STATUS_OUTPUT_FULL != 0 {
                return Some(io::inb(PS2_DATA));
            }
            core::hint::spin_loop();
        }
    }
    None
}
