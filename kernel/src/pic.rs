//! 8259 PIC: remap the interrupt vectors and gate individual IRQs.
//!
//! The BIOS leaves the PIC mapping IRQs onto CPU exception vectors 0x00..0x0F.
//! That must be fixed before interrupts can be enabled: [`remap`] moves the
//! master to 0x20..0x27 and the slave to 0x28..0x2F, then masks everything.
//! [`idt::set_irq_handler`] / [`unmask`] selectively turn individual IRQs on.

use crate::io;

const PIC1_CMD: u16 = 0x20;
const PIC1_DATA: u16 = 0x21;
const PIC2_CMD: u16 = 0xA0;
const PIC2_DATA: u16 = 0xA1;

/// ICW1: initialize, cascade mode, edge triggered, ICW4 required.
const ICW1: u8 = 0x11;
/// ICW4: 8086 mode (fully nested, not buffered, normal EOI).
const ICW4_8086: u8 = 0x01;
/// Standard end-of-interrupt command.
const OCW2_EOI: u8 = 0x20;

/// First vector of the remapped master PIC (IRQ0..IRQ7 -> 0x20..0x27).
pub const MASTER_BASE: u8 = 0x20;
/// First vector of the remapped slave PIC (IRQ8..IRQ15 -> 0x28..0x2F).
pub const SLAVE_BASE: u8 = 0x28;

/// Remap the PIC onto the CPU-exception-free vectors 0x20..0x2F.
///
/// Must run exactly once, before interrupts are enabled. Leaves every IRQ
/// masked; the caller unmask()s what it wants. The vectors must agree with
/// the IRQ gates [`crate::idt::init`] installs.
pub fn remap() {
    // Safety: the PIC legacy ports are always present on x86; nothing else
    // touches them during this single-threaded boot-time sequence.
    unsafe {
        io::outb(PIC1_CMD, ICW1);
        io::outb(PIC2_CMD, ICW1);
        io::outb(PIC1_DATA, MASTER_BASE);
        io::outb(PIC2_DATA, SLAVE_BASE);
        io::outb(PIC1_DATA, 0x04); // slave cascade on master IRQ2
        io::outb(PIC2_DATA, 0x02); // cascade identity of the slave
        io::outb(PIC1_DATA, ICW4_8086);
        io::outb(PIC2_DATA, ICW4_8086);
    }
    mask_all();
}

/// Mask every IRQ on both controllers.
pub fn mask_all() {
    // Safety: as in [`remap`]; this only touches the two mask registers.
    unsafe {
        io::outb(PIC1_DATA, 0xFF);
        io::outb(PIC2_DATA, 0xFF);
    }
}

/// Enable delivery of `irq`.
pub fn unmask(irq: u8) {
    let (data, bit) = pic_mask_port(irq);
    // Safety: as in [`remap`]; read-modify-write of a single mask bit.
    unsafe {
        let cur = io::inb(data);
        io::outb(data, cur & !(1 << bit));
    }
}

/// Disable delivery of `irq`.
pub fn mask(irq: u8) {
    let (data, bit) = pic_mask_port(irq);
    // Safety: as in [`remap`]; read-modify-write of a single mask bit.
    unsafe {
        let cur = io::inb(data);
        io::outb(data, cur | (1 << bit));
    }
}

/// Acknowledge an interrupt from `irq` (EOI). Slave IRQs need EOI to both
/// controllers.
pub fn send_eoi(irq: u8) {
    // Safety: EOI is a plain write to the legacy PIC command ports.
    unsafe {
        if irq >= 8 {
            io::outb(PIC2_CMD, OCW2_EOI);
        }
        io::outb(PIC1_CMD, OCW2_EOI);
    }
}

/// Which mask register and bit govern `irq`.
fn pic_mask_port(irq: u8) -> (u16, u8) {
    if irq < 8 {
        (PIC1_DATA, irq)
    } else {
        (PIC2_DATA, irq - 8)
    }
}
