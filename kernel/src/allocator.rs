use crate::arch::{current_cpu, frame_phys, phys_to_virt, SHARD_COUNT, SHARD_REFILL_BATCH};
use crate::frame::{frame_info, set_owner, FrameIndex, FrameOwner, OwnerTag};
use crate::spinlock::SpinLock;

/// Sentinel written into the first word of a free frame to mark list end.
const LIST_END: u32 = u32::MAX;

/// One per-CPU (or global) free list. The list lives *inside the free frames
/// themselves*: the first word of each free frame holds the index of the next
/// free frame, or `LIST_END`.
#[derive(Clone, Copy)]
struct Shard {
    head: u32,
    count: u32,
}

impl Shard {
    const fn empty() -> Shard {
        Shard { head: LIST_END, count: 0 }
    }

    fn is_empty(&self) -> bool {
        self.head == LIST_END
    }
}

/// The physical frame allocator. A static, never an owned value: reachable
/// from any context (IRQ handlers, drop paths, panic paths).
struct Allocator {
    shards: [SpinLock<Shard>; SHARD_COUNT],
    global: SpinLock<Shard>,
}

impl Allocator {
    const fn new() -> Allocator {
        Allocator {
            shards: [const { SpinLock::new(Shard::empty()) }; SHARD_COUNT],
            global: SpinLock::new(Shard::empty()),
        }
    }
}

static ALLOCATOR: Allocator = Allocator::new();

/// Read the embedded next pointer of a free frame.
fn list_next(frame: FrameIndex) -> u32 {
    // Safety: by the allocator's invariant `frame` is a member of some free
    // list, so its first word is owned by the allocator and holds a frame
    // index or LIST_END.
    unsafe { core::ptr::read_volatile(phys_to_virt(frame_phys(frame)) as *const u32) }
}

/// Write the embedded next pointer of a free frame.
fn set_list_next(frame: FrameIndex, next: u32) {
    // Safety: as in `list_next`; the first word of a free frame is free for
    // the allocator's use.
    unsafe { core::ptr::write_volatile(phys_to_virt(frame_phys(frame)) as *mut u32, next) }
}

/// Allocate a frame into kernel-internal ownership. Returns the frame index,
/// or `None` if the free lists are empty.
///
/// This is one of the small set of functions allowed to change a frame's
/// owner (Free -> Kernel).
pub fn alloc_frame() -> Option<FrameIndex> {
    let shard = current_cpu() % SHARD_COUNT;
    if let Some(frame) = pop_shard(shard) {
        return Some(frame);
    }
    refill_shard(shard);
    pop_shard(shard)
}

fn pop_shard(shard: usize) -> Option<FrameIndex> {
    let mut guard = ALLOCATOR.shards[shard].lock_irqsave();
    if guard.is_empty() {
        return None;
    }
    let frame = guard.head;
    debug_assert!(
        frame_info(frame).is_free(),
        "listed frame {frame} not free (owner={:?})",
        frame_info(frame).owner_tag()
    );
    guard.head = list_next(frame);
    guard.count -= 1;
    // The frame is no longer on any list: mark it kernel-owned.
    set_owner(frame, FrameOwner::FREE, FrameOwner::KERNEL);
    Some(frame)
}

/// Move up to a batch of frames from the global list into `shard`.
fn refill_shard(shard: usize) {
    let mut global = ALLOCATOR.global.lock_irqsave();
    let batch = global.count.min(SHARD_REFILL_BATCH);
    if batch == 0 {
        return;
    }

    // Peel `batch` frames off the head of the global list.
    let batch_head = global.head;
    let mut tail = batch_head;
    let mut cur = list_next(batch_head);
    for _ in 1..batch {
        tail = cur;
        cur = list_next(cur);
    }
    global.head = cur;
    global.count -= batch;
    drop(global);

    // Splice the peeled chain onto the front of our shard. Never hold both
    // locks at once.
    let mut guard = ALLOCATOR.shards[shard].lock_irqsave();
    set_list_next(tail, guard.head);
    guard.head = batch_head;
    guard.count += batch;
}

/// Release a frame back to the allocator.
///
/// This is one of the small set of functions allowed to change a frame's
/// owner. If the frame is still pinned or mapped it is not released: it
/// transitions to `Pending` and waits for the pin/mapping subsystems to
/// drain it.
///
/// The free decision and the list mutation happen under the current CPU's
/// shard lock, so the "in a free list iff Free" invariant is maintained
/// atomically. Lock order is strictly "other locks -> shard lock"; nothing
/// else is acquired while the shard lock is held.
pub fn release_frame(frame: FrameIndex) {
    let mut guard = ALLOCATOR.shards[current_cpu() % SHARD_COUNT].lock_irqsave();

    let info = frame_info(frame);
    let owner = info.owner();
    assert!(owner.tag() != OwnerTag::Reserved, "release of reserved frame {frame}");
    assert!(owner.tag() != OwnerTag::Free, "double release of frame {frame}");

    if info.pinned() != 0 || info.mappings() != 0 {
        if owner.tag() != OwnerTag::Pending {
            set_owner(frame, owner, FrameOwner::PENDING);
        }
        return;
    }

    set_owner(frame, owner, FrameOwner::FREE);
    info.bump_gen();

    set_list_next(frame, guard.head);
    guard.head = frame;
    guard.count += 1;
}

/// Release a draining frame whose pins and mappings have all been removed.
pub(crate) fn drain_if_ready(frame: FrameIndex) {
    let info = frame_info(frame);
    if info.owner_tag() == OwnerTag::Pending && info.pinned() == 0 && info.mappings() == 0 {
        release_frame(frame);
    }
}

/// Build the global free list from every frame currently marked Free.
///
/// Boot-only: must run before any allocation, and only after untyped regions
/// have been reserved. Returns the number of frames placed on the list.
pub(crate) fn build_global_freelist() -> u64 {
    let total = crate::frame::total_frames() as u32;
    let mut head = LIST_END;
    let mut prev = LIST_END;
    let mut count = 0u64;

    for idx in 0..total {
        if frame_info(idx).is_free() {
            if head == LIST_END {
                head = idx;
            } else {
                set_list_next(prev, idx);
            }
            prev = idx;
            count += 1;
        }
    }
    if prev != LIST_END {
        set_list_next(prev, LIST_END);
    }

    let mut guard = ALLOCATOR.global.lock_irqsave();
    assert!(guard.is_empty(), "global free list already built");
    guard.head = head;
    guard.count = count as u32;
    count
}

/// Total number of free frames across all lists.
pub fn free_frame_count() -> usize {
    let mut total = 0usize;
    for shard in &ALLOCATOR.shards {
        total += shard.lock_irqsave().count as usize;
    }
    total += ALLOCATOR.global.lock_irqsave().count as usize;
    total
}

/// Diagnostics: walk every free list and verify each listed frame is still
/// marked Free. Returns the number of frames walked.
pub fn debug_validate_free_lists() -> usize {
    let mut walked = 0usize;
    let mut check = |head: u32| {
        let mut cur = head;
        while cur != LIST_END {
            assert!(frame_info(cur).is_free(), "listed frame {cur} is not Free");
            walked += 1;
            cur = list_next(cur);
        }
    };
    for shard in &ALLOCATOR.shards {
        check(shard.lock_irqsave().head);
    }
    check(ALLOCATOR.global.lock_irqsave().head);
    walked
}

/// Test-only: empty every free list so a fresh [`crate::boot::pmm_init`] can
/// run. Never call outside `#[cfg(test)]`.
#[cfg(test)]
pub(crate) fn _test_reset() {
    for shard in &ALLOCATOR.shards {
        *shard.lock_irqsave() = Shard::empty();
    }
    *ALLOCATOR.global.lock_irqsave() = Shard::empty();
}
