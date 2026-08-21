//! Virtual memory: kernel page tables and address-space management.
//!
//! The kernel runs on its own 4-level page tables built in Rust over PMM
//! frames. At boot, [`PageTable::new_kernel`] reproduces the harness's
//! low-level windows with 2 MiB huge pages: an identity map of all of RAM and
//! a mirror at the physmap base. Individual pages can then be mapped and
//! unmapped at 4 KiB granularity, which participates in frame reference
//! counting so the PMM stays coherent.
//!
//! Lifecycle: the root frame is deliberately never freed — the kernel address
//! space lives for the kernel's lifetime. Tearing a space down (walking and
//! freeing every level) is future work.

use core::ptr;

use crate::allocator::alloc_frame;
use crate::arch::{cr3_write, frame_phys, phys_to_virt, physmap_base, FRAME_SIZE};
use crate::frame::{add_mapping, remove_mapping, total_frames, FrameIndex};

/// Entries per page-table level.
pub const TABLE_ENTRIES: usize = 512;
/// A 2 MiB huge page.
pub const LARGE_PAGE: usize = 2 * 1024 * 1024;

const PTE_PRESENT: u64 = 1 << 0;
const PTE_WRITABLE: u64 = 1 << 1;
const PTE_USER: u64 = 1 << 2;
const PTE_HUGE: u64 = 1 << 7;

/// Physical-address field masks (bits 12..51, offset by the page size).
const ADDR_MASK_4K: u64 = 0x000f_ffff_f000;
const ADDR_MASK_2M: u64 = 0x0000_0fff_ffe0_0000;
const ADDR_MASK_1G: u64 = 0x000f_ffff_c000_0000;

/// Page-table flags. Use [`Flags::with`] to combine.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Flags(u64);

impl Flags {
    pub const PRESENT: Flags = Flags(PTE_PRESENT);
    pub const WRITABLE: Flags = Flags(PTE_WRITABLE);
    pub const USER: Flags = Flags(1 << 2);
    pub const HUGE: Flags = Flags(PTE_HUGE);
    pub const NO_EXEC: Flags = Flags(1 << 63);

    pub const fn bits(self) -> u64 {
        self.0
    }

    pub const fn with(self, other: Flags) -> Flags {
        Flags(self.0 | other.0)
    }
}

/// Why a page-table operation failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapError {
    /// Address or length not aligned as required.
    NotAligned,
    /// Virtual address is not canonical.
    NonCanonical,
    /// Physical address is not backed by RAM.
    OutOfRange,
    /// Ran out of frames while growing the tree.
    AllocFailed,
    /// An incompatible mapping (e.g. huge vs. 4 KiB) is already present.
    Conflict,
    /// No mapping exists at the address.
    NotMapped,
}

/// A 4-level page table rooted at a PMM frame.
pub struct PageTable {
    root: FrameIndex,
}

impl PageTable {
    /// Allocate and zero a fresh, empty root. Nothing is mapped yet.
    pub fn new() -> Option<PageTable> {
        let root = alloc_frame()?;
        let pt = PageTable { root };
        pt.zero_table(root);
        Some(pt)
    }

    /// Build the kernel address space: identity-map all of RAM and mirror it
    /// at the physmap base, both as 2 MiB huge pages (uncounted window
    /// mappings — these cover the whole of physical memory, not individual
    /// frames).
    pub fn new_kernel(ram_end: usize) -> Option<PageTable> {
        let root = alloc_frame()?;
        let pt = PageTable { root };
        pt.zero_table(root);
        let window = align_up(ram_end, LARGE_PAGE);
        let kern = Flags::PRESENT.with(Flags::WRITABLE);
        pt.map_window(0, 0, window, kern).ok()?;
        pt.map_window(physmap_base(), 0, window, kern).ok()?;
        Some(pt)
    }

    /// Physical address of the active root (for CR3).
    pub fn root_phys(&self) -> usize {
        frame_phys(self.root)
    }

    /// Load this page table on the current CPU (full TLB flush).
    ///
    /// # Safety
    ///
    /// The table must map everything the running code needs (at minimum the
    /// kernel image, stack and heap) before this is called.
    pub fn activate(&self) {
        // Safety: the caller ensures the table is populated and complete.
        unsafe { cr3_write(self.root_phys()) };
    }

    /// Map one 4 KiB page at `va` -> `pa`. `pa` must be RAM-backed and the
    /// physical frame's reference count is incremented.
    pub fn map(&mut self, va: usize, pa: usize, flags: Flags) -> Result<(), MapError> {
        if !is_canonical(va) {
            return Err(MapError::NonCanonical);
        }
        if va % FRAME_SIZE != 0 || pa % FRAME_SIZE != 0 {
            return Err(MapError::NotAligned);
        }
        if pa >= total_frames() * FRAME_SIZE {
            return Err(MapError::OutOfRange);
        }
        // The PT level index (0) is resolved last inside the loop.
        let user = flags.bits() & PTE_USER;
        let mut frame = self.root;
        for level in (1..=3).rev() {
            let entry = unsafe { &mut *table_va(frame).add(index(va, level)) };
            if *entry & PTE_PRESENT == 0 {
                let new = alloc_frame().ok_or(MapError::AllocFailed)?;
                unsafe { ptr::write_bytes(table_va(new) as *mut u8, 0, FRAME_SIZE) };
                *entry = (frame_phys(new) as u64) | PTE_PRESENT | PTE_WRITABLE | user;
                frame = new;
            } else {
                if *entry & PTE_HUGE != 0 {
                    return Err(MapError::Conflict);
                }
                // In the monolithic kernel address space user code must be
                // able to *walk* the upper levels to reach a user leaf, so
                // promote a supervisor-only intermediate to USER. This is
                // harmless: the leaf's U/S bit still gates the actual memory.
                if user != 0 {
                    *entry |= user;
                }
                frame = phys_of(*entry);
            }
        }
        let pt = table_va(frame);
        let pt_entry = unsafe { &mut *pt.add(index(va, 0)) };
        if *pt_entry & PTE_PRESENT != 0 {
            return Err(MapError::Conflict);
        }
        *pt_entry = (pa as u64) | flags.bits() | PTE_PRESENT;
        add_mapping((pa / FRAME_SIZE) as FrameIndex);
        Ok(())
    }

    /// Remove the 4 KiB mapping at `va`, dropping the physical frame's
    /// reference count and invalidating the TLB entry.
    pub fn unmap(&mut self, va: usize) -> Result<(), MapError> {
        if !is_canonical(va) {
            return Err(MapError::NonCanonical);
        }
        if va % FRAME_SIZE != 0 {
            return Err(MapError::NotAligned);
        }
        let mut frame = self.root;
        for level in (1..=3).rev() {
            let entry = unsafe { &mut *table_va(frame).add(index(va, level)) };
            if *entry & PTE_PRESENT == 0 {
                return Err(MapError::NotMapped);
            }
            if *entry & PTE_HUGE != 0 {
                return Err(MapError::Conflict);
            }
            frame = phys_of(*entry);
        }
        let pt = table_va(frame);
        let pt_entry = unsafe { &mut *pt.add(index(va, 0)) };
        if *pt_entry & PTE_PRESENT == 0 {
            return Err(MapError::NotMapped);
        }
        let pa = (*pt_entry & ADDR_MASK_4K) as usize;
        *pt_entry = 0;
        remove_mapping((pa / FRAME_SIZE) as FrameIndex);
        // Safety: the page is no longer mapped; invalidating its TLB entry is
        // required before any reuse of the VA.
        unsafe { invlpg_checked(va) };
        Ok(())
    }

    /// Software walk: physical address backing `va`, or `None`.
    pub fn translate(&self, va: usize) -> Option<usize> {
        if !is_canonical(va) {
            return None;
        }
        let mut table = self.root;
        let e0 = unsafe { *table_va(table).add(index(va, 3)) };
        if e0 & PTE_PRESENT == 0 {
            return None;
        }
        table = phys_of(e0);

        let e1 = unsafe { *table_va(table).add(index(va, 2)) };
        if e1 & PTE_PRESENT == 0 {
            return None;
        }
        if e1 & PTE_HUGE != 0 {
            return Some(((e1 & ADDR_MASK_1G) as usize) + (va & 0x3fff_ffff));
        }
        table = phys_of(e1);

        let e2 = unsafe { *table_va(table).add(index(va, 1)) };
        if e2 & PTE_PRESENT == 0 {
            return None;
        }
        if e2 & PTE_HUGE != 0 {
            return Some(((e2 & ADDR_MASK_2M) as usize) + (va & 0x1f_ffff));
        }
        table = phys_of(e2);

        let e3 = unsafe { *table_va(table).add(index(va, 0)) };
        if e3 & PTE_PRESENT == 0 {
            return None;
        }
        Some(((e3 & ADDR_MASK_4K) as usize) + (va & 0xfff))
    }

    /// Map a run of 2 MiB huge pages from `[base_pa, base_pa + bytes)` at
    /// `[base_va, base_va + bytes)`. `bytes` must be a multiple of
    /// [`LARGE_PAGE`]. These are uncounted window mappings.
    fn map_window(
        &self,
        base_va: usize,
        base_pa: usize,
        bytes: usize,
        flags: Flags,
    ) -> Result<(), MapError> {
        if bytes == 0 {
            return Ok(());
        }
        if base_va % LARGE_PAGE != 0 || base_pa % LARGE_PAGE != 0 || bytes % LARGE_PAGE != 0 {
            return Err(MapError::NotAligned);
        }
        let leaf = flags.with(Flags::PRESENT).with(Flags::HUGE);
        let mut va = base_va;
        let mut pa = base_pa;
        let end = base_va + bytes;
        while va < end {
            let mut frame = self.root;
            // `map_window` walks two table levels (PML4 and PDP); the level
            // index used in the walk must be one above the table being
            // created, matching `translate`/`map` (level 3 = PML4, 2 = PDP).
            for level in (1..=2).rev() {
                let entry = unsafe { &mut *table_va(frame).add(index(va, level + 1)) };
            if *entry & PTE_PRESENT == 0 {
                let new = alloc_frame().ok_or(MapError::AllocFailed)?;
                unsafe { ptr::write_bytes(table_va(new) as *mut u8, 0, FRAME_SIZE) };
                *entry = (frame_phys(new) as u64) | PTE_PRESENT | PTE_WRITABLE;
                frame = new;
            } else {
                    if *entry & PTE_HUGE != 0 {
                        return Err(MapError::Conflict);
                    }
                    frame = phys_of(*entry);
                }
            }
            let pd = table_va(frame);
            let pd_entry = unsafe { &mut *pd.add(index(va, 1)) };
            if *pd_entry & PTE_PRESENT != 0 {
                return Err(MapError::Conflict);
            }
            *pd_entry = (pa as u64) | leaf.bits();
            va += LARGE_PAGE;
            pa += LARGE_PAGE;
        }
        Ok(())
    }

    fn zero_table(&self, frame: FrameIndex) {
        // Safety: `frame` is a fresh PMM frame mapped through the physmap.
        unsafe {
            ptr::write_bytes(table_va(frame) as *mut u8, 0, FRAME_SIZE);
        }
    }
}

fn align_up(value: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

/// Physmap virtual address of a table frame.
fn table_va(frame: FrameIndex) -> *mut u64 {
    phys_to_virt(frame_phys(frame)) as *mut u64
}

/// Index within a level's table for `va`. `level` 0 = PT (offset 12),
/// 3 = PML4 (offset 39).
#[inline]
fn index(va: usize, level: usize) -> usize {
    (va >> (12 + 9 * level)) & 0x1ff
}

/// The frame a present (non-huge) PTE points to.
#[inline]
fn phys_of(entry: u64) -> FrameIndex {
    ((entry >> 12) & 0x000f_ffff_ffff) as FrameIndex
}

/// True if `va` is a canonical 48-bit address.
fn is_canonical(va: usize) -> bool {
    let sign = (va >> 47) & 1;
    let top = (va >> 48) as u16;
    if sign == 0 {
        top == 0
    } else {
        top == 0xffff
    }
}

#[cfg(not(all(target_arch = "x86_64", not(test))))]
unsafe fn invlpg_checked(_va: usize) {}
#[cfg(all(target_arch = "x86_64", not(test)))]
unsafe fn invlpg_checked(va: usize) {
    // Safety: the caller has removed the mapping for `va`.
    unsafe { crate::arch::invlpg(va) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boot::testutil::*;
    use crate::boot::UntypedSpec;
    use crate::frame::{frame_info, OwnerTag};
    use crate::owned::OwnedFrame;
    use crate::untyped::UntypedCap;

    const TEST_VA: usize = 0xFFFF_9000_0000_0000;

    #[test]
    fn map_translate_unmap_round_trip() {
        let _g = test_lock();
        let specs: [UntypedSpec; 0] = [];
        let mut caps = [UntypedCap::new(0, 0, 0)];
        let _out = fresh_pmm(&specs, &mut caps);
        let mut pt = PageTable::new().expect("root");
        assert_eq!(pt.translate(TEST_VA), None);

        let f = OwnedFrame::alloc().expect("frame");
        let pa = f.phys();
        pt.map(TEST_VA, pa, Flags::PRESENT.with(Flags::WRITABLE)).expect("map");
        assert_eq!(pt.translate(TEST_VA), Some(pa));
        assert_eq!(frame_info(f.frame()).mappings(), 1);
        // Write through the mapping (host physmap: table writes land in the
        // test buffer; the page frame is the same memory).
        let page_ptr = phys_to_virt(pa) as *mut u64;
        unsafe {
            page_ptr.write(0xCAFE);
            assert_eq!(page_ptr.read(), 0xCAFE);
        }

        pt.unmap(TEST_VA).expect("unmap");
        assert_eq!(pt.translate(TEST_VA), None);
        let idx = f.frame();
        assert_eq!(frame_info(idx).mappings(), 0);
        drop(f);
        assert_eq!(frame_info(idx).owner_tag(), OwnerTag::Free);
    }

    #[test]
    fn kernel_windows_translate() {
        let _g = test_lock();
        let specs: [UntypedSpec; 0] = [];
        let mut caps = [UntypedCap::new(0, 0, 0)];
        let out = fresh_pmm(&specs, &mut caps);
        let pt = PageTable::new_kernel(out.ram_end as usize).expect("kernel space");

        // Identity and physmap windows via huge pages.
        assert_eq!(pt.translate(0x1000), Some(0x1000));
        let pm = crate::arch::physmap_base();
        assert_eq!(pt.translate(pm), Some(0));
        assert_eq!(pt.translate(pm + 0x1000), Some(0x1000));
        assert_eq!(pt.translate(pm + out.ram_end as usize - 1), Some(out.ram_end as usize - 1));
        // Nothing beyond the windows.
        assert_eq!(pt.translate(TEST_VA), None);
        assert_eq!(pt.translate(0x1_0000_0000), None);

        // A 4 KiB map inside an existing huge window must conflict.
        let mut pt = pt;
        assert_eq!(pt.map(pm, 0, Flags::PRESENT), Err(MapError::Conflict));
    }

    #[test]
    fn map_rejects_bad_inputs() {
        let _g = test_lock();
        let specs: [UntypedSpec; 0] = [];
        let mut caps = [UntypedCap::new(0, 0, 0)];
        let _out = fresh_pmm(&specs, &mut caps);
        let mut pt = PageTable::new().expect("root");
        let f = OwnedFrame::alloc().expect("frame");

        // Unaligned VA/PA.
        assert_eq!(pt.map(TEST_VA + 1, f.phys(), Flags::PRESENT), Err(MapError::NotAligned));
        assert_eq!(pt.map(TEST_VA, f.phys() + 1, Flags::PRESENT), Err(MapError::NotAligned));
        // Non-canonical (bit 47 set but bits 48..63 zero).
        assert_eq!(
            pt.map(0x8000_0000_0000, f.phys(), Flags::PRESENT),
            Err(MapError::NonCanonical)
        );
        // PA beyond RAM.
        assert_eq!(
            pt.map(TEST_VA, 1 << 40, Flags::PRESENT),
            Err(MapError::OutOfRange)
        );
        drop(f);
    }
}
