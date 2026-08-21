use crate::arch::{phys_to_virt, set_physmap_base, FRAME_SIZE};
use crate::frame::{init_table_zeroed, try_set_owner, FrameInfo, FrameIndex, FrameOwner, UntypedId};
use crate::untyped::UntypedCap;

/// Classification of a physical memory region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RegionKind {
    /// Available for general use by the kernel.
    Usable = 0,
    /// Not usable: MMIO, ACPI tables, holes, firmware.
    Reserved = 1,
}

/// A physical memory region, as reported by the boot handoff (e820-style).
#[derive(Clone, Copy, Debug)]
pub struct MemoryRegion {
    pub start: u64,
    pub len: u64,
    pub kind: RegionKind,
}

impl MemoryRegion {
    pub fn end(&self) -> u64 {
        self.start + self.len
    }
}

/// Everything the long-mode entry trampoline hands to the PMM.
pub struct BootInfo<'a> {
    /// Base virtual address of the physmap (all of RAM mapped at +`phys`).
    pub physmap_base: usize,
    /// Physical memory map, in address order.
    pub regions: &'a [MemoryRegion],
    /// Physical `[start, start+len)` of the kernel image; `(0, 0)` if none.
    pub kernel_image: (u64, u64),
}

/// Bump allocator used before the frame table exists.
///
/// Allocates from the usable regions in order. Not thread-safe and not
/// reentrant: intended for the single BSP boot path. The PMM proper takes
/// over once [`pmm_init`] returns; after that, use [`crate::alloc_frame`] /
/// [`crate::OwnedFrame`].
pub struct BootAllocator<'a> {
    regions: &'a [MemoryRegion],
    next_region: usize,
    cursor: u64,
}

impl<'a> BootAllocator<'a> {
    pub const fn new(regions: &'a [MemoryRegion]) -> BootAllocator<'a> {
        BootAllocator {
            regions,
            next_region: 0,
            cursor: 0,
        }
    }

    /// Bump-allocate `size` bytes from the first usable region that can
    /// satisfy the request, aligned to `align`. Returns the physical address.
    pub fn alloc(&mut self, size: usize, align: usize) -> Option<u64> {
        debug_assert!(size != 0);
        debug_assert!(align.is_power_of_two());
        while self.next_region < self.regions.len() {
            let region = self.regions[self.next_region];
            if region.kind != RegionKind::Usable {
                self.next_region += 1;
                self.cursor = 0;
                continue;
            }
            let start = align_up(self.cursor.max(region.start), align as u64);
            if let Some(end) = start.checked_add(size as u64) {
                if end <= region.end() {
                    self.cursor = end;
                    if self.cursor == region.end() {
                        self.next_region += 1;
                        self.cursor = 0;
                    }
                    return Some(start);
                }
            }
            self.next_region += 1;
            self.cursor = 0;
        }
        None
    }

    /// Convenience: allocate `n` whole frames, returning the base frame
    /// index.
    pub fn alloc_frames(&mut self, n: u32) -> Option<FrameIndex> {
        let phys = self.alloc(n as usize * FRAME_SIZE, FRAME_SIZE)?;
        Some((phys / FRAME_SIZE as u64) as FrameIndex)
    }
}

fn align_up(value: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

fn round_down(value: u64, align: u64) -> u64 {
    value & !(align - 1)
}

fn round_up(value: u64, align: u64) -> u64 {
    align_up(value, align)
}

fn ceil_div(a: u64, b: u64) -> u64 {
    (a + b - 1) / b
}

/// Cap on the number of effective regions after the kernel image is split
/// out. Plenty for any bootloader-provided map.
const MAX_EFFECTIVE_REGIONS: usize = 128;

/// Physical end of the legacy low-memory area, kept out of the kernel pool:
/// the real-mode IVT/BDA, the multiboot info structure and any real-mode
/// trampolines live there, so page-table and frame allocations must not
/// clobber them. The kernel image (and everything above it) is unaffected.
const LEGACY_END: u64 = 1024 * 1024;

/// Build the effective region list into `out`, returning its length.
///
/// Reserved regions pass through unchanged. Usable regions keep their
/// frame-aligned ranges, except the kernel image is split out (with
/// frame-aligned boundaries) so it never reaches the frame table or the free
/// list. Regions remain in address order.
fn build_effective<'a>(
    boot: &BootInfo<'a>,
    out: &mut [MemoryRegion; MAX_EFFECTIVE_REGIONS],
) -> Result<usize, &'static str> {
    let (ki_start, ki_len) = boot.kernel_image;
    let ki_ks = if ki_len != 0 { round_down(ki_start, FRAME_SIZE as u64) } else { 0 };
    let ki_ke = if ki_len != 0 { round_up(ki_start + ki_len, FRAME_SIZE as u64) } else { 0 };

    let mut n = 0usize;
    for region in boot.regions {
        if region.kind != RegionKind::Usable {
            out[n] = *region;
            n += 1;
        } else {
            let start = region.start;
            let end = region.end();
            if ki_len != 0 && ki_ks < end && start < ki_ke {
                if start < ki_ks {
                    out[n] = MemoryRegion { start, len: ki_ks - start, kind: RegionKind::Usable };
                    n += 1;
                }
                if ki_ke < end {
                    out[n] = MemoryRegion { start: ki_ke, len: end - ki_ke, kind: RegionKind::Usable };
                    n += 1;
                }
            } else {
                out[n] = *region;
                n += 1;
            }
        }
        if n >= MAX_EFFECTIVE_REGIONS {
            return Err("too many memory regions");
        }
    }
    Ok(n)
}

/// A request to turn a physical range into an untyped region at boot.
#[derive(Clone, Copy, Debug)]
pub struct UntypedSpec {
    pub base_phys: u64,
    pub len: u64,
}

/// Result of [`pmm_init`].
pub struct PmInitOut {
    pub total_frames: usize,
    /// Frames in the kernel pool (the global free list).
    pub free_frames: u64,
    pub frame_table_frames: usize,
    /// Number of untyped caps created.
    pub untyped_created: usize,
    /// Physical start of the kernel heap region; 0 if `heap_len` was 0.
    pub heap_phys: u64,
    /// Length of the kernel heap region (always a multiple of FRAME_SIZE).
    pub heap_len: u64,
    /// Highest usable physical end (i.e. `total_frames * FRAME_SIZE`).
    pub ram_end: u64,
}

/// Initialize the physical memory manager.
///
/// Boot-only, single-threaded, idempotent-guarded. Steps:
///   1. establish the physmap base and the frame table (via [`BootAllocator`]),
///   2. carve `heap_len` bytes right after the frame table for the kernel
///      heap (see [`crate::heap::heap_init`]); the region is zeroed and kept
///      out of the frame pool,
///   3. mark every frame Reserved, then every usable frame Free (skipping the
///      kernel image, frame table and heap),
///   4. reserve the given `specs` as untyped regions (`Free -> Untyped`),
///   5. build the kernel-pool free list from the remaining Free frames.
///
/// Each spec becomes an `UntypedCap` written into `out_untyped[i]`. After
/// this returns, [`crate::alloc_frame`] / [`crate::OwnedFrame`] take over.
pub fn pmm_init<'a>(
    boot: &BootInfo<'a>,
    specs: &[UntypedSpec],
    out_untyped: &mut [UntypedCap],
    heap_len: usize,
) -> Result<PmInitOut, &'static str> {
    if specs.len() > out_untyped.len() {
        return Err("untyped output buffer too small");
    }
    assert!(heap_len % FRAME_SIZE == 0, "heap length must be a multiple of FRAME_SIZE");
    set_physmap_base(boot.physmap_base);

    for region in boot.regions {
        if region.kind == RegionKind::Usable
            && (region.start % FRAME_SIZE as u64 != 0 || region.len % FRAME_SIZE as u64 != 0)
        {
            return Err("usable region not frame-aligned");
        }
    }

    // Effective region list: usable regions with the kernel image split out
    // (frame-aligned) and dropped, so it never ends up in the frame table or
    // the free list.
    let mut effective =
        [MemoryRegion { start: 0, len: 0, kind: RegionKind::Reserved }; MAX_EFFECTIVE_REGIONS];
    let effective_len = build_effective(boot, &mut effective)?;
    let regions = &effective[..effective_len];

    // Total frame count: up to the highest usable end.
    let mut max_end = 0u64;
    for region in regions {
        if region.kind == RegionKind::Usable {
            max_end = max_end.max(region.end());
        }
    }
    if max_end == 0 {
        return Err("no usable memory");
    }
    let total = ceil_div(max_end, FRAME_SIZE as u64) as usize;
    let ft_bytes = total * core::mem::size_of::<FrameInfo>();

    // Carve the frame table out of usable memory with the boot allocator.
    let mut boot_alloc = BootAllocator::new(regions);
    let ft_phys = boot_alloc.alloc(ft_bytes, FRAME_SIZE).ok_or("cannot allocate frame table")?;
    let ft_va = phys_to_virt(ft_phys as usize) as usize;
    init_table_zeroed(ft_va, total, ft_bytes)?; // zeroed => owner == Reserved
    let ft_end = ft_phys + ft_bytes as u64;

    // Carve the kernel heap right after the frame table and zero it.
    let heap_end = if heap_len != 0 {
        let hp = boot_alloc.alloc(heap_len, FRAME_SIZE).ok_or("cannot allocate kernel heap")?;
        let hv = phys_to_virt(hp as usize) as usize;
        // Safety: the boot allocator reserved `heap_len` bytes for us.
        unsafe {
            core::ptr::write_bytes(hv as *mut u8, 0, heap_len);
        }
        hp + heap_len as u64
    } else {
        ft_end
    };

    // Mark usable frames Free, skipping the frame table and the heap.
    for region in regions {
        if region.kind != RegionKind::Usable {
            continue;
        }
        let first = ceil_div(region.start, FRAME_SIZE as u64);
        let last = ceil_div(region.end(), FRAME_SIZE as u64);
        for idx in first..last {
            let phys = idx * FRAME_SIZE as u64;
            if phys < LEGACY_END {
                continue; // stays Reserved (low memory: IVT/BDA, multiboot info)
            }
            if phys >= ft_phys && phys < heap_end {
                continue; // stays Reserved
            }
            if !try_set_owner(idx as FrameIndex, FrameOwner::RESERVED, FrameOwner::FREE) {
                return Err("overlapping memory map entries");
            }
        }
    }

    // Reserve the untyped regions (Free -> Untyped).
    for (i, spec) in specs.iter().enumerate() {
        carve_untyped_region(spec, i as UntypedId, &mut out_untyped[i])?;
    }

    // Build the kernel pool from whatever is still Free.
    let free = crate::allocator::build_global_freelist();

    Ok(PmInitOut {
        total_frames: total,
        free_frames: free,
        frame_table_frames: ft_bytes / FRAME_SIZE,
        untyped_created: specs.len(),
        heap_phys: if heap_len != 0 { heap_end - heap_len as u64 } else { 0 },
        heap_len: heap_len as u64,
        ram_end: max_end,
    })
}

fn carve_untyped_region(
    spec: &UntypedSpec,
    id: UntypedId,
    out: &mut UntypedCap,
) -> Result<(), &'static str> {
    if spec.base_phys % FRAME_SIZE as u64 != 0 || spec.len % FRAME_SIZE as u64 != 0 {
        return Err("untyped spec not frame-aligned");
    }
    let base_frame = spec.base_phys / FRAME_SIZE as u64;
    let num = spec.len / FRAME_SIZE as u64;
    if num == 0 || base_frame + num >= u32::MAX as u64 {
        return Err("untyped spec has invalid range");
    }
    for i in 0..num {
        let frame = (base_frame + i) as FrameIndex;
        if !try_set_owner(frame, FrameOwner::FREE, FrameOwner::untyped(id)) {
            return Err("untyped spec overlaps a reserved range or another region");
        }
    }
    *out = UntypedCap::new(id, base_frame as FrameIndex, num as u32);
    Ok(())
}

#[cfg(test)]
pub(crate) mod testutil {
    use super::*;
    use crate::allocator::_test_reset as _alloc_reset;
    use crate::frame::_test_reset as _frame_reset;
    use std::sync::{Mutex, MutexGuard, OnceLock};

    pub const RAM_SIZE: usize = 16 * 1024 * 1024;
    /// Kernel heap carved by the boot path in tests.
    pub const HEAP_SIZE: usize = 64 * 1024;

    pub static REGIONS: [MemoryRegion; 1] = [MemoryRegion {
        start: 0,
        len: RAM_SIZE as u64,
        kind: RegionKind::Usable,
    }];

    /// Serializes all tests that touch the global frame table / allocator,
    /// which cargo runs on multiple threads by default. Shared across test
    /// modules (boot, vmm, heap) so they never interleave. Poison is ignored:
    /// the `double_release_panics` test deliberately panics while holding it.
    pub fn test_lock() -> MutexGuard<'static, ()> {
        static TEST_LOCK: Mutex<()> = Mutex::new(());
        TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A host-heap buffer standing in for RAM; `pmm_init` maps the physmap
    /// onto it via `physmap_base`.
    pub fn aligned_ram_buffer() -> &'static [u8] {
        static RAM: OnceLock<&'static [u8]> = OnceLock::new();
        // 2 MiB alignment so the test physmap base is huge-page aligned, as
        // the real kernel's is.
        const ALIGN: usize = 2 * 1024 * 1024;
        RAM.get_or_init(|| {
            let v = vec![0u8; RAM_SIZE + ALIGN];
            let aligned = (v.as_ptr() as usize + ALIGN - 1) & !(ALIGN - 1);
            let shift = aligned - v.as_ptr() as usize;
            // `Vec::drain` shifts data in place but does NOT move the buffer
            // pointer, so to get an aligned slice we leak a boxed slice and
            // take the aligned sub-range.
            let boxed: Box<[u8]> = v.into_boxed_slice();
            let raw = Box::leak(boxed);
            &raw[shift..shift + RAM_SIZE]
        })
    }

    /// Fresh `BootInfo` pointing the physmap at a host-heap "RAM" buffer, and
    /// a re-armed frame table + empty free lists so `pmm_init` can run again.
    pub fn fresh_pmm(specs: &[UntypedSpec], out: &mut [UntypedCap]) -> PmInitOut {
        _frame_reset();
        _alloc_reset();
        let buf = aligned_ram_buffer();
        let boot = BootInfo {
            physmap_base: buf.as_ptr() as usize,
            regions: &REGIONS,
            kernel_image: (0, 0),
        };
        pmm_init(&boot, specs, out, HEAP_SIZE).expect("pmm_init failed")
    }
}

#[cfg(test)]
mod tests {
    use super::testutil::*;
    use super::*;
    use crate::allocator::debug_validate_free_lists;
    use crate::frame::{add_mapping, frame_gen, frame_info, pin_frame, remove_mapping, unpin_frame, OwnerTag};
    use crate::owned::OwnedFrame;
    use crate::untyped::RetypeError;
    use crate::{alloc_frame, free_frame_count, release_frame, retype_frame};

    #[test]
    fn boot_allocator_bumps() {
        static R: [MemoryRegion; 3] = [
            MemoryRegion { start: 0, len: 0x1000, kind: RegionKind::Reserved },
            MemoryRegion { start: 0x1000, len: 0x2000, kind: RegionKind::Usable },
            MemoryRegion { start: 0x10000, len: 0x1000, kind: RegionKind::Usable },
        ];
        let mut ba = BootAllocator::new(&R);
        assert_eq!(ba.alloc(0x1000, 0x1000), Some(0x1000));
        assert_eq!(ba.alloc(0x1000, 0x1000), Some(0x2000));
        // The gap [0x3000, 0x10000) is skipped by advancing regions.
        assert_eq!(ba.alloc(0x1000, 0x1000), Some(0x10000));
        assert_eq!(ba.alloc(0x100, 0x100), None);
    }

    #[test]
    fn kernel_image_split_keeps_image_reserved() {
        let _g = test_lock();
        let specs: [UntypedSpec; 0] = [];
        let mut caps = [UntypedCap::new(0, 0, 0)];
        crate::frame::_test_reset();
        crate::allocator::_test_reset();
        let buf = aligned_ram_buffer();
        // Kernel image occupies 1 MiB..3 MiB (frame-aligned).
        let boot = BootInfo {
            physmap_base: buf.as_ptr() as usize,
            regions: &REGIONS,
            kernel_image: (0x100000, 0x200000),
        };
        let out = pmm_init(&boot, &specs, &mut caps, 0).expect("pmm_init failed");

        // Drain the whole pool: no frame may land inside the kernel image.
        let ki_first = 0x100000 / 4096;
        let ki_last = 0x300000 / 4096;
        for _ in 0..out.free_frames {
            let f = alloc_frame().expect("alloc");
            assert!(
                !(ki_first..ki_last).contains(&(f as u64)),
                "allocator returned kernel-image frame {f:#x}"
            );
        }
        assert_eq!(alloc_frame(), None, "pool must be exhausted");
    }

    #[test]
    fn heap_carve_stays_out_of_pool() {
        let _g = test_lock();
        let specs: [UntypedSpec; 0] = [];
        let mut caps = [UntypedCap::new(0, 0, 0)];
        let out = fresh_pmm(&specs, &mut caps);
        assert!(out.heap_len == HEAP_SIZE as u64);
        assert!(out.heap_phys % FRAME_SIZE as u64 == 0);
        let h_first = out.heap_phys / FRAME_SIZE as u64;
        let h_last = (out.heap_phys + out.heap_len) / FRAME_SIZE as u64;

        // Drain the pool: no frame may land inside the heap region.
        for _ in 0..out.free_frames {
            let f = alloc_frame().expect("alloc");
            assert!(
                !(h_first..h_last).contains(&(f as u64)),
                "allocator returned heap frame {f:#x}"
            );
        }
        assert_eq!(alloc_frame(), None, "pool must be exhausted");
    }

    #[test]
    fn pmm_integration() {
        let _g = test_lock();
        // Reserve the 1 MiB..2 MiB range as an untyped region; everything
        // else becomes the kernel pool.
        let specs = [UntypedSpec { base_phys: 0x100000, len: 0x100000 }];
        let mut caps = [UntypedCap::new(0, 0, 0)];
        let out = fresh_pmm(&specs, &mut caps);
        assert_eq!(out.total_frames, RAM_SIZE / 4096);
        assert_eq!(out.frame_table_frames, out.total_frames * 24 / 4096);
        assert_eq!(out.ram_end, RAM_SIZE as u64);
        assert_eq!(debug_validate_free_lists(), free_frame_count());
        assert!(out.free_frames > 0);

        // --- alloc / release round trip through OwnedFrame -----------------
        let a = OwnedFrame::alloc().expect("alloc");
        let idx_a = a.frame();
        let ptr = a.as_ptr();
        unsafe { ptr.write(0xAB) };
        assert_eq!(unsafe { ptr.read() }, 0xAB);
        assert_eq!(frame_info(idx_a).owner_tag(), OwnerTag::Kernel);

        let b = OwnedFrame::alloc().expect("alloc");
        assert_ne!(a.frame(), b.frame());
        drop(a);
        // Released to shard 0; the next allocation should reuse the frame.
        let a2 = OwnedFrame::alloc().expect("alloc");
        assert_eq!(a2.frame(), idx_a);
        drop(b);
        drop(a2);

        // --- retype from the untyped region --------------------------------
        let f1 = retype_frame(&caps[0], 7).expect("retype");
        assert!((256..512).contains(&f1), "frame {f1} outside 1MiB..2MiB");
        assert_eq!(frame_info(f1).owner_tag(), OwnerTag::Domain);
        assert_eq!(frame_info(f1).owner().id(), 7);

        let mut count = 1u32;
        while retype_frame(&caps[0], 7).is_ok() {
            count += 1;
        }
        assert_eq!(count, 256, "region has 256 frames");

        // Once-only grant semantics: a released frame returns to the global
        // pool, not to the region, so the region stays exhausted.
        release_frame(f1);
        assert_eq!(frame_info(f1).owner_tag(), OwnerTag::Free);
        assert_eq!(retype_frame(&caps[0], 7), Err(RetypeError::Exhausted));

        // --- deferral: pinned frame is not released until drained -----------
        let d = alloc_frame().expect("alloc");
        pin_frame(d);
        release_frame(d);
        assert_eq!(frame_info(d).owner_tag(), OwnerTag::Pending);
        let e = OwnedFrame::alloc().expect("alloc");
        assert_ne!(e.frame(), d);
        unpin_frame(d);
        assert_eq!(frame_info(d).owner_tag(), OwnerTag::Free);
        assert_eq!(alloc_frame(), Some(d));
        drop(e);

        assert_eq!(debug_validate_free_lists(), free_frame_count());
    }

    #[test]
    #[should_panic]
    fn double_release_panics() {
        let _g = test_lock();
        let specs: [UntypedSpec; 0] = [];
        let mut caps = [UntypedCap::new(0, 0, 0)];
        fresh_pmm(&specs, &mut caps);

        let f = alloc_frame().expect("alloc");
        release_frame(f);
        release_frame(f); // second release must panic
    }

    #[test]
    fn split_untyped_is_pure_bookkeeping() {
        let _g = test_lock();
        let specs = [UntypedSpec { base_phys: 0x100000, len: 0x100000 }];
        let mut caps = [UntypedCap::new(0, 0, 0)];
        fresh_pmm(&specs, &mut caps);

        let (a, b) = caps[0].split(0x40000).expect("split");
        assert_eq!(a.base_frame(), 256);
        assert_eq!(a.num_frames(), 64);
        assert_eq!(b.base_frame(), 320);
        assert_eq!(b.num_frames(), 192);

        // Retyping from either half works; the owner id is unchanged.
        let f = retype_frame(&a, 1).expect("retype a");
        assert!((256..320).contains(&f));
        let g = retype_frame(&b, 1).expect("retype b");
        assert!((320..512).contains(&g));
        let _ = f;
        let _ = g;
    }

    #[test]
    fn mapping_helpers_compile() {
        let _g = test_lock();
        let specs: [UntypedSpec; 0] = [];
        let mut caps = [UntypedCap::new(0, 0, 0)];
        fresh_pmm(&specs, &mut caps);

        let f = alloc_frame().expect("alloc");
        assert_eq!(add_mapping(f), 1);
        assert_eq!(frame_gen(f), 0);
        assert_eq!(remove_mapping(f), 0);
        release_frame(f);
        assert!(frame_gen(f) >= 1);
    }
}
