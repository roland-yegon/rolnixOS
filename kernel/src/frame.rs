use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Index of a physical frame. `u32` bounds the addressable space at
/// 4 KiB * 2^32 = 16 TiB of RAM.
pub type FrameIndex = u32;

/// Identifier of a protection domain (a process).
pub type DomainId = u32;

/// Identifier of an untyped memory region.
pub type UntypedId = u32;

/// Tag identifying the class of owner of a frame.
///
/// `Reserved` is deliberately encoded as 0 so that zeroed frame-table memory
/// is the safe default.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum OwnerTag {
    /// Physically reserved: kernel image, frame table, MMIO, memory holes.
    Reserved = 0,
    /// On some free list; available for allocation.
    Free = 1,
    /// Held by kernel-internal code as an [`crate::owned::OwnedFrame`].
    Kernel = 2,
    /// Ownerless and draining: pins/mappings must reach zero before the
    /// frame returns to a free list.
    Pending = 3,
    /// Covered by an untyped region; not yet retyped into a typed object.
    Untyped = 4,
    /// Owned by a protection domain.
    Domain = 5,
}

/// The owner of a frame, encoded as a tagged `u64` (tag in the low byte,
/// region/domain id in the upper 32 bits).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct FrameOwner(u64);

impl FrameOwner {
    const fn encode(tag: OwnerTag, id: u32) -> u64 {
        (tag as u64) | ((id as u64) << 8)
    }

    fn decode(raw: u64) -> (OwnerTag, u32) {
        let tag = match raw & 0xff {
            0 => OwnerTag::Reserved,
            1 => OwnerTag::Free,
            2 => OwnerTag::Kernel,
            3 => OwnerTag::Pending,
            4 => OwnerTag::Untyped,
            5 => OwnerTag::Domain,
            _ => unreachable!("invalid owner tag"),
        };
        (tag, ((raw >> 8) & 0xffff_ffff) as u32)
    }

    pub const RESERVED: FrameOwner = FrameOwner(Self::encode(OwnerTag::Reserved, 0));
    pub const FREE: FrameOwner = FrameOwner(Self::encode(OwnerTag::Free, 0));
    pub const KERNEL: FrameOwner = FrameOwner(Self::encode(OwnerTag::Kernel, 0));
    pub const PENDING: FrameOwner = FrameOwner(Self::encode(OwnerTag::Pending, 0));

    pub const fn untyped(id: UntypedId) -> FrameOwner {
        FrameOwner(Self::encode(OwnerTag::Untyped, id))
    }

    pub const fn domain(id: DomainId) -> FrameOwner {
        FrameOwner(Self::encode(OwnerTag::Domain, id))
    }

    pub fn tag(self) -> OwnerTag {
        Self::decode(self.0).0
    }

    pub fn id(self) -> u32 {
        Self::decode(self.0).1
    }

    fn raw(self) -> u64 {
        self.0
    }

    fn from_raw(raw: u64) -> FrameOwner {
        FrameOwner(raw)
    }
}

/// Per-frame metadata. This is the single source of truth for physical
/// memory ownership; the free lists and capability table are derived views.
///
/// All fields are atomic because different CPUs may concurrently observe a
/// frame (e.g. one releasing it while another reads `pinned`). Mutations are
/// further serialized by the allocator shard lock or the pin/mapping
/// subsystems, as documented per field.
#[repr(C)]
pub struct FrameInfo {
    owner: AtomicU64,
    mappings: AtomicU32,
    pinned: AtomicU32,
    gen: AtomicU32,
}

impl FrameInfo {
    /// Current owner of the frame.
    pub fn owner(&self) -> FrameOwner {
        FrameOwner::from_raw(self.owner.load(Ordering::Acquire))
    }

    /// Tag of the current owner.
    pub fn owner_tag(&self) -> OwnerTag {
        self.owner().tag()
    }

    /// True if the frame is on a free list.
    pub fn is_free(&self) -> bool {
        self.owner_tag() == OwnerTag::Free
    }

    /// Number of address spaces currently mapping this frame.
    pub fn mappings(&self) -> u32 {
        self.mappings.load(Ordering::Acquire)
    }

    /// Number of in-flight pins (DMA, TLB shootdown, held syscall refs).
    pub fn pinned(&self) -> u32 {
        self.pinned.load(Ordering::Acquire)
    }

    /// Generation counter, bumped on every release. Stamped into
    /// capabilities; checked at capability-use boundaries.
    pub fn gen(&self) -> u32 {
        self.gen.load(Ordering::Relaxed)
    }

    /// Atomically move the frame to a new owner iff the current owner equals
    /// `expected`. Returns whether the transition happened.
    pub(crate) fn transition(&self, expected: FrameOwner, new: FrameOwner) -> bool {
        self.owner
            .compare_exchange(expected.raw(), new.raw(), Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    pub(crate) fn bump_gen(&self) -> u32 {
        self.gen.fetch_add(1, Ordering::Relaxed).wrapping_add(1)
    }

    pub(crate) fn pin(&self) -> u32 {
        self.pinned.fetch_add(1, Ordering::AcqRel).wrapping_add(1)
    }

    pub(crate) fn unpin(&self) -> u32 {
        debug_assert!(self.pinned.load(Ordering::Acquire) > 0);
        self.pinned.fetch_sub(1, Ordering::AcqRel).wrapping_sub(1)
    }

    pub(crate) fn add_mapping(&self) -> u32 {
        self.mappings.fetch_add(1, Ordering::AcqRel).wrapping_add(1)
    }

    pub(crate) fn remove_mapping(&self) -> u32 {
        debug_assert!(self.mappings.load(Ordering::Acquire) > 0);
        self.mappings.fetch_sub(1, Ordering::AcqRel).wrapping_sub(1)
    }
}

/// The frame table: a statically-mapped array of `FrameInfo`, one per frame,
/// indexed by frame number. Established once at boot and reserved forever.
struct FrameTable {
    /// Physmap virtual address of `FrameInfo[0]`; 0 means uninitialized.
    base: AtomicU64,
    num_frames: AtomicU64,
}

static FRAME_TABLE: FrameTable = FrameTable {
    base: AtomicU64::new(0),
    num_frames: AtomicU64::new(0),
};

/// Initialize the frame table over a zeroed region of `bytes` bytes.
pub(crate) fn init_table_zeroed(
    base_va: usize,
    num_frames: usize,
    bytes: usize,
) -> Result<(), &'static str> {
    assert!(base_va != 0, "frame table base is zero");
    let prev = FRAME_TABLE
        .base
        .compare_exchange(0, base_va as u64, Ordering::AcqRel, Ordering::Acquire);
    if prev.is_err() {
        return Err("frame table already initialized");
    }
    FRAME_TABLE.num_frames.store(num_frames as u64, Ordering::Release);
    // Safety: `base_va` is a physmap mapping of a region reserved by the boot
    // allocator for exactly `bytes` bytes of frame table.
    unsafe {
        core::ptr::write_bytes(base_va as *mut u8, 0, bytes);
    }
    Ok(())
}

/// Number of frames tracked by the frame table.
pub fn total_frames() -> usize {
    FRAME_TABLE.num_frames.load(Ordering::Acquire) as usize
}

/// Look up the metadata for a frame.
pub fn frame_info(idx: FrameIndex) -> &'static FrameInfo {
    let base = FRAME_TABLE.base.load(Ordering::Relaxed);
    assert!(base != 0, "frame table not initialized");
    let num = FRAME_TABLE.num_frames.load(Ordering::Relaxed);
    assert!(
        (idx as u64) < num,
        "frame index {idx} out of range (table has {num} frames)"
    );
    // Safety: `base` was established at init as the physmap address of the
    // reserved frame table region, which lives for the kernel's lifetime.
    // Each distinct index maps to a distinct, naturally-aligned FrameInfo.
    unsafe { &*((base as usize + (idx as usize) * core::mem::size_of::<FrameInfo>()) as *const FrameInfo) }
}

/// Panicking owner transition. Used where the expected owner is known
/// statically; a failure is an invariant violation.
pub(crate) fn set_owner(frame: FrameIndex, expected: FrameOwner, new: FrameOwner) {
    let ok = frame_info(frame).transition(expected, new);
    assert!(
        ok,
        "frame {frame}: owner transition {:?} -> {:?} failed",
        expected.tag(),
        new.tag()
    );
}

/// Non-panicking owner transition; returns whether it happened.
pub(crate) fn try_set_owner(frame: FrameIndex, expected: FrameOwner, new: FrameOwner) -> bool {
    frame_info(frame).transition(expected, new)
}

/// Increment the in-flight pin count. Returns the new count.
pub fn pin_frame(frame: FrameIndex) -> u32 {
    frame_info(frame).pin()
}

/// Decrement the in-flight pin count; if the frame is draining and no pins or
/// mappings remain, release it. Returns the new pin count.
///
/// Must not be called while holding the allocator shard lock.
pub fn unpin_frame(frame: FrameIndex) -> u32 {
    let n = frame_info(frame).unpin();
    crate::allocator::drain_if_ready(frame);
    n
}

/// Record a new mapping of this frame. Returns the new mapping count.
pub fn add_mapping(frame: FrameIndex) -> u32 {
    frame_info(frame).add_mapping()
}

/// Remove a mapping of this frame; if the frame is draining and no pins or
/// mappings remain, release it. Returns the new mapping count.
///
/// Must not be called while holding the allocator shard lock.
pub fn remove_mapping(frame: FrameIndex) -> u32 {
    let n = frame_info(frame).remove_mapping();
    crate::allocator::drain_if_ready(frame);
    n
}

/// Current generation of a frame.
pub fn frame_gen(frame: FrameIndex) -> u32 {
    frame_info(frame).gen()
}

/// Test-only: re-arm the frame table so a fresh [`crate::boot::pmm_init`] can
/// run. Never call outside `#[cfg(test)]`.
#[cfg(test)]
pub(crate) fn _test_reset() {
    FRAME_TABLE.base.store(0, Ordering::Relaxed);
    FRAME_TABLE.num_frames.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::{FrameOwner, OwnerTag};

    #[test]
    fn owner_encoding_round_trip() {
        for owner in [
            FrameOwner::RESERVED,
            FrameOwner::FREE,
            FrameOwner::KERNEL,
            FrameOwner::PENDING,
            FrameOwner::untyped(0x1234),
            FrameOwner::domain(0xdead_beef),
        ] {
            let tag = owner.tag();
            let id = owner.id();
            let back = match tag {
                OwnerTag::Untyped => FrameOwner::untyped(id),
                OwnerTag::Domain => FrameOwner::domain(id),
                _ => match tag {
                    OwnerTag::Reserved => FrameOwner::RESERVED,
                    OwnerTag::Free => FrameOwner::FREE,
                    OwnerTag::Kernel => FrameOwner::KERNEL,
                    OwnerTag::Pending => FrameOwner::PENDING,
                    _ => unreachable!(),
                },
            };
            assert_eq!(owner, back);
        }
        assert_eq!(FrameOwner::RESERVED.raw(), 0);
        assert_eq!(FrameOwner::untyped(7).tag(), OwnerTag::Untyped);
        assert_eq!(FrameOwner::domain(9).id(), 9);
    }
}
