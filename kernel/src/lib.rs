//! rolnix physical memory manager and kernel virtual memory.
//!
//! Entry contract: this crate assumes the caller has already
//!   1. switched to long mode with paging enabled,
//!   2. mapped all of physical RAM at a fixed high virtual address
//!      (the physmap), and
//!   3. placed a [`BootInfo`] describing the memory map somewhere readable.
//!
//! Before that handoff there is no frame table and no free list; the only
//! allocator is the [`BootAllocator`] (a bump allocator). After
//! [`boot::pmm_init`] returns, [`alloc_frame`] / [`OwnedFrame`] take over,
//! the kernel heap is live via [`heap::heap_init`], and the kernel address
//! space can be installed with [`vmm::PageTable::new_kernel`] /
//! [`vmm::PageTable::activate`].
//!
//! ## Ownership model
//!
//! Allocation state lives in exactly one place: the per-frame `FrameInfo`
//! array. The free list (per-CPU shards + global) and capability table are
//! *derived views* over that array, never independent records. The only
//! functions that may change a frame's owner are [`alloc_frame`],
//! [`release_frame`], [`untyped::retype_frame`] and the boot-time region
//! setup in [`boot::pmm_init`]; all route through a single CAS on the owner
//! field.
//!
//! Frames are uniquely owned. Kernel-internal allocations are
//! [`OwnedFrame`] values dropped by kernel code; user-visible memory is
//! owned by protection domains via retyped capabilities. A frame may only be
//! released when `pinned == 0 && mappings == 0`; otherwise it transitions to
//! `Pending` and is drained later by the pin/mapping subsystems.

#![cfg_attr(not(test), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]

extern crate alloc;

pub mod allocator;
pub mod arch;
pub mod boot;
pub mod frame;
pub mod heap;
pub mod idt;
pub mod io;
pub mod keyboard;
pub mod owned;
pub mod pic;
pub mod pit;
pub mod sched;
pub mod spinlock;
pub mod tss;
pub mod untyped;
pub mod user;
pub mod vmm;

pub use arch::{disable_interrupts, enable_interrupts, phys_to_virt, physmap_base, virt_to_phys};
pub use boot::{
    BootAllocator, BootInfo, MemoryRegion, PmInitOut, RegionKind, UntypedSpec, pmm_init,
};
pub use frame::{
    DomainId, FrameInfo, FrameIndex, FrameOwner, OwnerTag, UntypedId, add_mapping, frame_gen,
    frame_info, pin_frame, remove_mapping, unpin_frame,
};
pub use heap::{KernelHeap, heap_allocated, heap_init, heap_total};
pub use idt::{FaultSink, InterruptFrame, IrqHandler, init, set_fault_sink, set_irq_handler};
pub use owned::OwnedFrame;
pub use sched::{ProcDesc, MAX_PROCESSES, current_pid, free_slot, init_idle, kill, snapshot, start_process, user_code_va, user_stack_top};
pub use spinlock::SpinLock;
pub use untyped::{RetypeError, UntypedCap, retype_frame};
pub use allocator::{alloc_frame, free_frame_count, release_frame};
pub use vmm::{Flags, MapError, PageTable};
