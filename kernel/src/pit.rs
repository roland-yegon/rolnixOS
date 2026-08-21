//! Intel 8254 PIT: drive IRQ0 as a periodic interrupt source.
//!
//! Channel 0 of the PIT is wired to PIC input 0 (IRQ0). [`set_frequency`]
//! programs it in mode 3 (square wave), which makes the interrupt line pulse
//! `hz` times per second.

use crate::io;

const PIT_CMD: u16 = 0x43;
const PIT_CH0: u16 = 0x40;
/// 1.19318 MHz reference clock, per the PIT datasheet.
const PIT_REFERENCE_HZ: u64 = 1_193_182;

/// Program channel 0 to fire `hz` times per second.
///
/// The count divisor is `reference / hz`, clamped to the PIT's 16-bit range
/// (so `hz` is 18..1193182); the channel then counts down, so the effective
/// rate is `reference / divisor`.
pub fn set_frequency(hz: u32) {
    let divisor = (PIT_REFERENCE_HZ / hz as u64).clamp(2, 65535) as u16;
    // Safety: the PIT legacy ports are always present on x86; nothing else
    // touches them during this single-threaded boot-time sequence.
    unsafe {
        // Channel 0, read/load lobyte then hibyte, mode 3, binary count.
        io::outb(PIT_CMD, 0x36);
        io::outb(PIT_CH0, (divisor & 0xFF) as u8);
        io::outb(PIT_CH0, (divisor >> 8) as u8);
    }
}
