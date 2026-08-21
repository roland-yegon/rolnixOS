//! The kernel heap: a first-fit free-list allocator with coalescing.
//!
//! Established once at boot by [`heap_init`] over a contiguous region the
//! boot allocator carved out of RAM (see [`crate::boot::pmm_init`]). Backs
//! the `alloc` crate (`Vec`, `String`, `Box`, ...) via the `#[global_allocator]`
//! static below.
//!
//! Layout: every block starts with a 16-byte header holding its total size
//! and, when free, a next pointer. Free blocks are kept in a singly-linked
//! list sorted by address so neighbours can be coalesced on free. The
//! returned payload is aligned to the request; its base address is stashed in
//! the 8 bytes just before it so `free` can recover the block header without
//! knowing the original alignment.

use core::alloc::{GlobalAlloc, Layout};
use core::ptr;

use crate::spinlock::SpinLock;

const HEADER: usize = 16;
const MIN_FREE: usize = 32;

/// A block header at the start of a heap block.
#[repr(C)]
struct Block {
    size: usize,
    next: *mut Block,
}

struct FreeList {
    /// Address of the first free block; 0 when empty.
    head: usize,
    /// Bytes of heap handed out (for diagnostics only).
    allocated: usize,
    /// Total heap size.
    total: usize,
}

impl FreeList {
    const fn empty() -> FreeList {
        FreeList { head: 0, allocated: 0, total: 0 }
    }
}

static FREELIST: SpinLock<FreeList> = SpinLock::new(FreeList::empty());

fn block_mut(addr: usize) -> &'static mut Block {
    // Safety: `addr` is the base of a live heap block (header region).
    unsafe { &mut *(addr as *mut Block) }
}

fn align_up(value: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

/// Initialize the heap over `[start, start + size)`. Boot-only, single-shot.
///
/// `start` and `size` must be 16-byte aligned. The region must be reserved by
/// the caller (pmm_init carves it out of RAM and keeps it out of the pool).
pub fn heap_init(start: usize, size: usize) {
    assert!(start % HEADER == 0 && size % HEADER == 0, "heap misaligned");
    assert!(size >= MIN_FREE, "heap too small");
    let mut list = FREELIST.lock_irqsave();
    assert!(list.head == 0, "heap already initialized");
    // Safety: `start` is a reserved, writable region of `size` bytes; the
    // header occupies the first 16 bytes, the rest is payload space.
    let block = block_mut(start);
    block.size = size;
    block.next = ptr::null_mut();
    list.head = start;
    list.allocated = 0;
    list.total = size;
}

/// Bytes of heap currently allocated.
pub fn heap_allocated() -> usize {
    FREELIST.lock_irqsave().allocated
}

/// Total heap size.
pub fn heap_total() -> usize {
    FREELIST.lock_irqsave().total
}

fn heap_alloc(layout: Layout) -> *mut u8 {
    let align = layout.align().max(HEADER);
    let size = layout.size();
    let mut list = FREELIST.lock_irqsave();
    let mut prev = 0usize;
    let mut cur = list.head;
    while cur != 0 {
        let base = cur;
        let block_size = block_mut(cur).size;
        let payload = align_up(base + HEADER, align);
        // The payload must fit inside the block.
        if payload.checked_add(size).is_some_and(|end| end <= base + block_size) {
            // Split boundary rounded up to the header alignment so the free
            // tail starts on a properly aligned address. Block sizes are
            // always multiples of HEADER, so tail_start <= base + block_size.
            let tail_start = align_up(payload + size, HEADER);
            let remainder = base + block_size - tail_start;
            let next = block_mut(cur).next as usize;
            // Unlink `cur`: its range is now (partly) allocated.
            if remainder >= MIN_FREE {
                // Split off the free tail and link it where `cur` was, so the
                // list stays sorted.
                let tail = block_mut(tail_start);
                tail.size = remainder;
                tail.next = next as *mut Block;
                if prev == 0 {
                    list.head = tail_start;
                } else {
                    block_mut(prev).next = tail_start as *mut Block;
                }
                block_mut(cur).size = tail_start - base;
            } else {
                if prev == 0 {
                    list.head = next;
                } else {
                    block_mut(prev).next = next as *mut Block;
                }
            }
            let consumed = if remainder >= MIN_FREE { tail_start - base } else { block_size };
            // Safety: `payload - 8` is inside the block's region (>= base, and
            // below `tail_start`), so writing the base there is within the
            // block.
            unsafe { *((payload - 8) as *mut usize) = base };
            list.allocated += consumed;
            return payload as *mut u8;
        }
        prev = cur;
        cur = block_mut(cur).next as usize;
    }
    ptr::null_mut()
}

fn heap_free(ptr: *mut u8) {
    let payload = ptr as usize;
    // Safety: `ptr` must be a pointer previously returned by heap_alloc, so
    // the 8 bytes before it hold the block base.
    let base = unsafe { *((payload - 8) as *const usize) };
    let mut list = FREELIST.lock_irqsave();
    // Walk to the insertion point (first free block at or above `base`).
    let mut prev = 0usize;
    let mut cur = list.head;
    while cur != 0 && cur < base {
        prev = cur;
        cur = block_mut(cur).next as usize;
    }
    let size = block_mut(base).size;
    let block = block_mut(base);
    // Safety: `base` was allocated (recovered from the payload), so its header
    // is valid and the range [base, base+size) is free to manage.
    block.size = size;
    block.next = cur as *mut Block;
    if prev == 0 {
        list.head = base;
    } else {
        block_mut(prev).next = base as *mut Block;
    }
    // Coalesce with the next neighbour.
    if cur != 0 && base + size == cur {
        let next = block_mut(cur).next as usize;
        block.size += block_mut(cur).size;
        block.next = next as *mut Block;
    }
    // Coalesce with the previous neighbour.
    if prev != 0 && prev + block_mut(prev).size == base {
        block_mut(prev).size += block.size;
        block_mut(prev).next = block.next;
    }
    list.allocated -= size;
}

/// The kernel's global allocator.
#[derive(Default)]
pub struct KernelHeap;

impl KernelHeap {
    pub const fn new() -> KernelHeap {
        KernelHeap
    }
}

// Safety: all access to the free list goes through the FREELIST spinlock;
// the unit struct has no fields to alias.
unsafe impl GlobalAlloc for KernelHeap {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        heap_alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        heap_free(ptr)
    }
}

/// The one true global allocator for any binary linking this crate. Gated out
/// under `cfg(test)` so host tests use the standard system allocator.
#[cfg(not(test))]
#[global_allocator]
static KERNEL_HEAP: KernelHeap = KernelHeap::new();

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_heap(size: usize) {
        static BUF: std::sync::Mutex<Option<Vec<u8>>> = std::sync::Mutex::new(None);
        let mut guard = BUF.lock().unwrap();
        if guard.as_ref().map_or(true, |v| v.len() < size + 64) {
            *guard = Some(vec![0u8; size + 64]);
        }
        let v = guard.as_mut().unwrap();
        let start = (v.as_ptr() as usize + 63) & !63;
        let base = v.as_mut_ptr().wrapping_add(start - v.as_ptr() as usize) as usize;
        FREELIST.lock_irqsave().head = 0;
        heap_init(base, size);
    }

    #[test]
    fn alloc_free_round_trip() {
        fresh_heap(16 * 1024);
        let a = heap_alloc(Layout::from_size_align(32, 8).unwrap());
        assert!(!a.is_null());
        unsafe { a.write_bytes(0xAA, 32) };
        let b = heap_alloc(Layout::from_size_align(64, 8).unwrap());
        assert!(!b.is_null());
        assert_ne!(a, b);
        heap_free(a);
        heap_free(b);
        // After both frees, the whole heap must be one coalesced free block.
        let list = FREELIST.lock_irqsave();
        let head = list.head;
        assert_ne!(head, 0);
        assert_eq!(block_mut(head).size, 16 * 1024);
        assert!(block_mut(head).next.is_null());
        assert_eq!(list.allocated, 0);
    }

    #[test]
    fn large_alignment() {
        fresh_heap(64 * 1024);
        let a = heap_alloc(Layout::from_size_align(100, 4096).unwrap());
        assert!(!a.is_null());
        assert_eq!(a as usize % 4096, 0);
        heap_free(a);
    }

    #[test]
    fn reuses_freed_space() {
        fresh_heap(16 * 1024);
        let a = heap_alloc(Layout::from_size_align(128, 8).unwrap());
        heap_free(a);
        let b = heap_alloc(Layout::from_size_align(128, 8).unwrap());
        assert_eq!(a, b);
        heap_free(b);
    }
}
