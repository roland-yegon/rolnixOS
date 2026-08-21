//! Multiboot v1 info parsing. The bootloader passes a pointer in EBX; we copy
//! the memory map out into our own static array (the bootloader's info block
//! may live in memory we hand to the PMM).

pub const MAGIC: u32 = 0x2BADB002;

pub const MT_USABLE: u32 = 1;

/// Parsed from the bootloader: one e820-style entry.
#[derive(Clone, Copy)]
pub struct MemEntry {
    pub start: u64,
    pub len: u64,
    pub kind: u32,
}

/// Parse `info_ptr` into a fixed-size array of memory-map entries.
///
/// Returns the number of entries written, or `None` on a malformed map. The
/// multiboot memory map is a series of `[size:u32][base:u64][len:u64][type:u32]`
/// records; each record advances by `4 + size` (size excludes the size field
/// itself).
pub fn parse(info_ptr: usize, out: &mut [MemEntry]) -> Option<usize> {
    let mbi = info_ptr as *const u32;
    let flags = unsafe { mbi.read() };
    let mut count = 0usize;

    // FLAG_MEM (bit 0): mem_upper at word 2. Fallback if no mmap present.
    let mem_upper = if flags & 0x1 != 0 { unsafe { mbi.add(2).read() } } else { 0 };

    // FLAG_MMAP (bit 6): mmap_length at word 11, mmap_addr at word 12
    // (offsets 44 and 48 of multiboot_info_t).
    if flags & (1 << 6) != 0 {
        let mmap_addr = unsafe { mbi.add(12).read() } as usize;
        let mmap_len = unsafe { mbi.add(11).read() } as usize;
        let mut p = mmap_addr as *const u32;
        let end = mmap_addr + mmap_len;
        while (p as usize) < end && count < out.len() {
            let size = unsafe { p.read() };
            // Packed layout after the size field: addr:u64, len:u64, type:u32.
            // x86 tolerates the 4-byte alignment of addr.
            let base = unsafe { (p.add(1) as *const u64).read() };
            let len = unsafe { (p.add(1).add(2) as *const u64).read() };
            let ty = unsafe { p.add(1).add(4).read() };
            if size < 20 {
                break; // malformed; stop, do not loop forever
            }
            out[count] = MemEntry { start: base, len, kind: ty };
            count += 1;
            p = (p as usize + 4 + size as usize) as *const u32;
        }
    }

    if count == 0 && mem_upper != 0 {
        out[count] = MemEntry {
            start: 0x100000,
            len: mem_upper as u64 * 1024,
            kind: MT_USABLE,
        };
        count += 1;
    }

    if count == 0 {
        None
    } else {
        Some(count)
    }
}
