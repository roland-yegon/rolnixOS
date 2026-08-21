use core::sync::atomic::{AtomicU32, Ordering};

use crate::arch::{frame_phys, FRAME_SIZE};
use crate::frame::{
    frame_info, try_set_owner, DomainId, FrameIndex, FrameOwner, OwnerTag, UntypedId,
};

/// A chunk of raw physical memory from which typed objects (frames, page
/// tables, endpoints) can be retyped.
///
/// The covered frames carry `owner == Untyped(id)`; retyping moves one from
/// `Untyped -> Domain`. All caps covering the same region share the same
/// `id`. Splitting a cap is pure bookkeeping and does not touch `FrameInfo`.
pub struct UntypedCap {
    /// Region id stamped into the covered frames' owner field.
    id: UntypedId,
    base_frame: FrameIndex,
    num_frames: u32,
    /// Hint for the next frame to try during retype. Only an optimization:
    /// correctness comes from the per-frame owner CAS.
    cursor: AtomicU32,
}

impl UntypedCap {
    pub(crate) const fn new(id: UntypedId, base_frame: FrameIndex, num_frames: u32) -> UntypedCap {
        UntypedCap {
            id,
            base_frame,
            num_frames,
            cursor: AtomicU32::new(base_frame),
        }
    }

    pub fn id(&self) -> UntypedId {
        self.id
    }

    pub fn base_frame(&self) -> FrameIndex {
        self.base_frame
    }

    pub fn num_frames(&self) -> u32 {
        self.num_frames
    }

    pub fn base_phys(&self) -> usize {
        frame_phys(self.base_frame)
    }

    pub fn end_phys(&self) -> usize {
        frame_phys(self.base_frame) + (self.num_frames as usize) * FRAME_SIZE
    }

    /// Split this cap into two covering the first `split_len` bytes and the
    /// remainder. Both result caps keep the same region id.
    pub fn split(&self, split_len: usize) -> Result<(UntypedCap, UntypedCap), &'static str> {
        if split_len == 0 || split_len % FRAME_SIZE != 0 {
            return Err("split length must be a nonzero multiple of the frame size");
        }
        let split_frames = split_len / FRAME_SIZE;
        if split_frames as u32 >= self.num_frames {
            return Err("split length must leave a nonempty remainder");
        }
        let a = UntypedCap::new(self.id, self.base_frame, split_frames as u32);
        let b = UntypedCap::new(
            self.id,
            self.base_frame + split_frames as u32,
            self.num_frames - split_frames as u32,
        );
        Ok((a, b))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetypeError {
    /// The region has no remaining untaken frames.
    Exhausted,
}

/// Retype the next free frame covered by `cap` into a frame owned by
/// `domain`.
///
/// The per-frame owner CAS makes this safe even if two threads retype from
/// caps covering the same region: only one CAS per frame succeeds.
pub fn retype_frame(cap: &UntypedCap, domain: DomainId) -> Result<FrameIndex, RetypeError> {
    let end = cap.base_frame + cap.num_frames;
    let mut idx = cap.cursor.load(Ordering::Relaxed);
    let mut tried = 0u32;

    while tried < cap.num_frames {
        if idx >= end {
            idx = cap.base_frame;
        }
        let info = frame_info(idx);
        if info.owner_tag() == OwnerTag::Untyped
            && info.owner().id() == cap.id
            && try_set_owner(idx, FrameOwner::untyped(cap.id), FrameOwner::domain(domain))
        {
            cap.cursor.store(idx.wrapping_add(1), Ordering::Relaxed);
            return Ok(idx);
        }
        idx = idx.wrapping_add(1);
        tried += 1;
    }
    Err(RetypeError::Exhausted)
}
