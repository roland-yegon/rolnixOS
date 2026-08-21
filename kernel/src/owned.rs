use crate::allocator::{alloc_frame, release_frame};
use crate::arch::{frame_phys, phys_to_virt};
use crate::frame::FrameIndex;

/// An owned physical frame, for kernel-internal allocations only.
///
/// Deliberately scoped away from the capability tree: kernel page tables and
/// transient kernel buffers are owned by kernel code paths and dropped by
/// kernel `Drop`, never by domain teardown. Dropping an `OwnedFrame` runs the
/// fast-path release (shard-lock CAS + free-list push, no tree walks), which
/// is safe in IRQ context.
///
/// Do not derive `Copy`/`Clone`: an `OwnedFrame` is a unique owner.
pub struct OwnedFrame {
    frame: FrameIndex,
}

impl OwnedFrame {
    /// Allocate a frame from the kernel pool.
    pub fn alloc() -> Option<OwnedFrame> {
        alloc_frame().map(|frame| OwnedFrame { frame })
    }

    /// The frame index backing this allocation.
    pub fn frame(&self) -> FrameIndex {
        self.frame
    }

    /// Physical address of the frame.
    pub fn phys(&self) -> usize {
        frame_phys(self.frame)
    }

    /// Physmap pointer to the frame's memory (kernel-internal use only).
    pub fn as_ptr(&self) -> *mut u8 {
        phys_to_virt(frame_phys(self.frame))
    }
}

impl Drop for OwnedFrame {
    fn drop(&mut self) {
        release_frame(self.frame);
    }
}
