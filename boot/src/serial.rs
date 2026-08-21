use core::fmt;

const COM1: u16 = 0x3F8;

const UART_RBR: u16 = 0; // RX buffer (read)
const UART_THR: u16 = 0; // TX holding (write)
const UART_IER: u16 = 1; // interrupt enable
const UART_FCR: u16 = 2; // FIFO control
const UART_LCR: u16 = 3; // line control
const UART_LSR: u16 = 5; // line status

unsafe fn outb(port: u16, value: u8) {
    unsafe {
        core::arch::asm!("out dx, al", in("dx") port, in("al") value, options(nomem, nostack, preserves_flags));
    }
}

unsafe fn inb(port: u16) -> u8 {
    unsafe {
        let value: u8;
        core::arch::asm!("in al, dx", out("al") value, in("dx") port, options(nomem, nostack, preserves_flags));
        value
    }
}

/// Minimal 16550 UART driver for QEMU's `-serial stdio` on COM1.
///
/// A unit struct so it can live in a `static`; `init` runs the port setup
/// exactly once at first use.
pub struct SerialPort;

impl SerialPort {
    /// Const placeholder for the static; the port is configured by [`init`].
    pub const fn const_new() -> Self {
        SerialPort
    }

    /// Configure the UART (unsafe: touches MMIO; call once, on the BSP).
    pub fn init(&mut self) {
        // Safety: COM1 at 0x3F8 is standard ISA legacy I/O, writable from ring
        // 0; only reached from the single-threaded boot path.
        unsafe {
            outb(COM1 + UART_IER, 0x00); // no interrupts
            outb(COM1 + UART_LCR, 0x80); // DLAB on
            outb(COM1 + UART_THR, 0x01); // divisor low: 115200
            outb(COM1 + UART_IER, 0x00); // divisor high
            outb(COM1 + UART_LCR, 0x03); // 8N1, DLAB off
            outb(COM1 + UART_FCR, 0xC7); // FIFO enable, clear, 14-byte trigger
        }
    }

    pub fn write_byte(&mut self, byte: u8) {
        // Wait for the transmit holding register to drain.
        unsafe {
            while inb(COM1 + UART_LSR) & 0x20 == 0 {}
            outb(COM1 + UART_THR, byte);
        }
    }

    /// Poll for a received byte (returns immediately if the RX FIFO is empty).
    pub fn try_read(&mut self) -> Option<u8> {
        // Safety: LSR/RBR polling has no side effects beyond the read itself.
        unsafe {
            if inb(COM1 + UART_LSR) & 0x01 == 0 {
                None
            } else {
                Some(inb(COM1 + UART_RBR))
            }
        }
    }
}

/// Write `bytes` straight to the UART, bypassing the serial lock. Only for
/// panic/fault reports that may fire while a normal print already holds the
/// lock (a re-entrant acquire would spin forever). Interleaving with an
/// in-flight print is acceptable for diagnostics; the caller is expected to
/// run with interrupts disabled, so no tick can preempt mid-write.
pub fn raw_write(bytes: &[u8]) {
    // Safety: the port is expected to be initialized (normally by the first
    // `with_serial` call); writing COM1 THR is ring-0-legal.
    unsafe {
        for &b in bytes {
            while inb(COM1 + UART_LSR) & 0x20 == 0 {}
            outb(COM1 + UART_THR, if b == b'\n' { b'\r' } else { b });
        }
    }
}

impl fmt::Write for SerialPort {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for &b in s.as_bytes() {
            if b == b'\n' {
                self.write_byte(b'\r');
            }
            self.write_byte(b);
        }
        Ok(())
    }
}
