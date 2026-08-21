use core::cell::UnsafeCell;
use core::ops::{Deref, DerefMut};
use core::sync::atomic::{AtomicBool, Ordering};

/// A test-and-set spinlock.
///
/// Unfair (no ticket), suitable for the short critical sections of the frame
/// allocator. Upgrade to a ticket/queued lock if contention shows up.
pub struct SpinLock<T> {
    locked: AtomicBool,
    value: UnsafeCell<T>,
}

// Safety: access to `value` is exclusively gated by the lock discipline.
// `T: Send` allows the guarded value to be shared across threads.
unsafe impl<T: Send> Sync for SpinLock<T> {}
unsafe impl<T: Send> Send for SpinLock<T> {}

impl<T> SpinLock<T> {
    pub const fn new(value: T) -> Self {
        SpinLock {
            locked: AtomicBool::new(false),
            value: UnsafeCell::new(value),
        }
    }

    /// Acquire without touching interrupt state.
    ///
    /// Only safe where the caller already guarantees this CPU cannot re-enter
    /// the critical section from an interrupt handler.
    pub fn lock(&self) -> SpinLockGuard<'_, T> {
        self.acquire();
        SpinLockGuard { lock: self, irq: None }
    }

    /// Acquire while disabling local interrupts; the guard restores the
    /// previous interrupt state on drop. IRQ-safe on the local CPU.
    pub fn lock_irqsave(&self) -> SpinLockGuard<'_, T> {
        // Disable IRQs before spinning so an interrupt on this CPU cannot
        // re-enter the same lock while we wait.
        let saved = unsafe { crate::arch::irq_save() };
        self.acquire();
        SpinLockGuard { lock: self, irq: Some(saved) }
    }

    fn acquire(&self) {
        while self.locked.swap(true, Ordering::Acquire) {
            while self.locked.load(Ordering::Relaxed) {
                core::hint::spin_loop();
            }
        }
    }

    fn release(&self) {
        self.locked.store(false, Ordering::Release);
    }
}

pub struct SpinLockGuard<'a, T> {
    lock: &'a SpinLock<T>,
    irq: Option<u64>,
}

impl<T> Deref for SpinLockGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        // Safety: the guard holds the lock for the lifetime of this reference.
        unsafe { &*self.lock.value.get() }
    }
}

impl<T> DerefMut for SpinLockGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // Safety: the guard holds the lock, granting exclusive access.
        unsafe { &mut *self.lock.value.get() }
    }
}

impl<T> Drop for SpinLockGuard<'_, T> {
    fn drop(&mut self) {
        self.lock.release();
        if let Some(saved) = self.irq {
            // Safety: `saved` came from irq_save in lock_irqsave; this is the
            // paired restore.
            unsafe { crate::arch::irq_restore(saved) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SpinLock;

    #[test]
    fn lock_round_trip() {
        static L: SpinLock<u32> = SpinLock::new(0);
        {
            let mut g = L.lock();
            *g += 1;
        }
        assert_eq!(*L.lock(), 1);
    }
}
