//! Preemptive round-robin multitasking for ring-3 processes.
//!
//! The PIT tick (IRQ0) preempts the current process and switches to the next
//! ready one; `int 0x80` adds cooperative [`yield_now`] and [`exit_current`].
//! The interactive shell is process 0 (idle), created by [`init_idle`];
//! [`start_process`] registers a ring-3 program (already mapped as USER by the
//! boot harness) and the scheduler starts running it on the next tick.
//!
//! A switch is a stack swap: on preemption the IDT stub saves a full
//! [`InterruptFrame`] on the process's private kernel stack (the `TSS.rsp0`
//! stack), and [`sched_switch`] merely moves RSP to the next process's frame
//! and iretq's it. A freshly created process carries a hand-built frame with
//! the iretq tail (RIP/CS/RFLAGS/RSP/SS), so its first switch is an entry
//! into ring 3.
//!
//! Concurrency: single CPU. The process table is touched only from interrupt
//! context (IF=0: tick/syscall) or from shell context under `irq_save`
//! (`start_process`/`kill`/`snapshot`), so no locks are needed.
//!
//! Host/build-stub note: same convention as `idt.rs`; the switching asm is
//! compiled out and the table operations become host-testable state.
#![cfg_attr(not(all(target_arch = "x86_64", not(test))), allow(dead_code))]

use crate::idt::InterruptFrame;

/// Maximum number of processes (slot 0 is always the idle shell).
pub const MAX_PROCESSES: usize = 16;
/// Per-process ring-0 stack size; the initial frame and every ring-3->0
/// interrupt entry land on it via `TSS.rsp0`.
const KSTACK_SIZE: usize = 16 * 1024;

/// Start of the per-process virtual region, above the identity window
/// (0..512 MiB) so it starts unmapped in the kernel page table.
pub const PROC_BASE: usize = 0x8000_0000;
/// Address-space stride between two processes' regions.
pub const PROC_STRIDE: usize = 2 * 1024 * 1024;
/// Offset of the user stack within a process's region (code is at the base).
const PROC_STACK_OFFSET: usize = 1024 * 1024;

/// The first (user-executable, USER-mapped) byte of process `pid`'s region.
pub fn user_code_va(pid: u32) -> usize {
    PROC_BASE + pid as usize * PROC_STRIDE
}

/// Top of process `pid`'s user stack (one page above [`user_code_va`]'s
/// region's stack slot).
pub fn user_stack_top(pid: u32) -> usize {
    PROC_BASE + pid as usize * PROC_STRIDE + PROC_STACK_OFFSET + 4096
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    Unused,
    Ready,
    Running,
    Zombie,
}

#[derive(Clone, Copy)]
struct Process {
    pid: u32,
    state: State,
    /// Top of the private ring-0 stack (`TSS.rsp0`).
    kernel_stack_top: usize,
    /// Address of the saved (or initial) [`InterruptFrame`] on that stack.
    ctx: usize,
    name: [u8; 16],
}

const UNUSED: Process = Process {
    pid: 0,
    state: State::Unused,
    kernel_stack_top: 0,
    ctx: 0,
    name: [0; 16],
};

static mut PROCESSES: [Process; MAX_PROCESSES] = [UNUSED; MAX_PROCESSES];
static mut KSTACKS: [u8; MAX_PROCESSES * KSTACK_SIZE] = [0; MAX_PROCESSES * KSTACK_SIZE];
static mut CURRENT: usize = 0;
static mut SCHEDULING: bool = false;

/// Ring-3 syscall gate selectors for the initial frame, so `sched_switch`
/// iretq's it into user mode like any preempted process's frame.
const USER_CODE64: u16 = crate::tss::USER_CODE64_SELECTOR;
const USER_DATA: u16 = crate::tss::USER_DATA_SELECTOR;

fn table_mut() -> *mut Process {
    core::ptr::addr_of_mut!(PROCESSES) as *mut Process
}

fn table_const() -> *const Process {
    core::ptr::addr_of!(PROCESSES) as *const Process
}

// The actual switch: drop RSP onto the next process's saved [`InterruptFrame`]
// and iretq it (the same tail `irq_common` uses, but with RSP taken from the
// argument). `sched_switch` is the one place that ends a timeslice; it never
// returns. `rdi` holds `ctx` (frame address): set RSP to it, then the 16 pops
// consume `rsp`+15 GPRs, `add $16` skips vector/error-code, iretq pops
// RIP/CS/RFLAGS/RSP/SS. The `call` return address is abandoned (never returns).
#[cfg(all(target_arch = "x86_64", not(test)))]
core::arch::global_asm!(
    "
.section .text
.globl sched_switch
sched_switch:
    mov %rdi, %rsp
    pop %rax
    pop %r15
    pop %r14
    pop %r13
    pop %r12
    pop %r11
    pop %r10
    pop %r9
    pop %r8
    pop %rbp
    pop %rdi
    pop %rsi
    pop %rdx
    pop %rcx
    pop %rbx
    pop %rax
    add $16, %rsp
    iretq
",
    options(att_syntax),
);

#[cfg(all(target_arch = "x86_64", not(test)))]
extern "C" {
    /// `ctx` is the address of an [`InterruptFrame`]; never returns.
    fn sched_switch(ctx: usize) -> !;
}

/// Register the boot flow as process 0 (idle) and enable the scheduler.
/// Boot-only, once, before the shell runs.
pub fn init_idle(kernel_stack_top: usize) {
    // Safety: single-threaded boot path, called exactly once, before the PIT
    // handler can run.
    unsafe {
        *core::ptr::addr_of_mut!(PROCESSES[0]) = Process {
            pid: 0,
            state: State::Ready,
            kernel_stack_top,
            ctx: 0,
            name: *b"idle\0\0\0\0\0\0\0\0\0\0\0\0",
        };
        CURRENT = 0;
        SCHEDULING = true;
    }
}

/// First slot that is not holding a live process (idle is slot 0, so `None`
/// means the table is full). The caller must keep interrupts disabled across
/// the matching [`start_process`] so no tick can allocate the slot in
/// between.
pub fn free_slot() -> Option<u32> {
    // Safety: called under irq_save by the boot harness; table writes happen
    // only here (under irq_save) and in interrupt context (IF=0).
    (1..MAX_PROCESSES as u32).find(|&i| {
        matches!(
            unsafe { table_const().add(i as usize).read().state },
            State::Unused | State::Zombie
        )
    })
}

/// Register a ring-3 process. `entry` must already be mapped USER-executable
/// (see [`user_code_va`]); the process's user stack and kernel stack are
/// assigned from the process number. Call with interrupts disabled.
pub fn start_process(pid: u32, entry: usize, name: &str) {
    let slot = pid as usize;
    let stack_base = core::ptr::addr_of!(KSTACKS) as usize + slot * KSTACK_SIZE;
    let stack_top = stack_base + KSTACK_SIZE;
    // The initial frame sits just below the top of the kernel stack; the
    // process is entered by iretq'ing it, then overwritten by the first real
    // ring-3->0 interrupt entry (also pushed from `stack_top`).
    let frame_addr = (stack_top - core::mem::size_of::<InterruptFrame>()) & !15;
    let frame = frame_addr as *mut InterruptFrame;
    // Safety: `frame` is inside this process's private bss kernel stack; the
    // slot is free (guarded by irq_save) and the single CPU can only reach it
    // through this write.
    unsafe {
        frame.write(InterruptFrame {
            rsp: 0,
            r15: 0,
            r14: 0,
            r13: 0,
            r12: 0,
            r11: 0,
            r10: 0,
            r9: 0,
            r8: 0,
            rbp: 0,
            rdi: pid as u64,
            rsi: 0,
            rdx: 0,
            rcx: 0,
            rbx: 0,
            rax: 0,
            vector: 0,
            error_code: 0,
            rip: entry as u64,
            cs: USER_CODE64 as u64,
            rflags: 0x202, // IF + reserved bit 1
            user_rsp: user_stack_top(pid) as u64,
            user_ss: USER_DATA as u64,
        });
    }
    let mut nb = [0u8; 16];
    for (dst, &b) in nb.iter_mut().zip(name.as_bytes().iter().take(15)) {
        *dst = b;
    }
    // Safety: same single-CPU, irq_save-guarded access as above.
    unsafe {
        *core::ptr::addr_of_mut!(PROCESSES[slot]) = Process {
            pid,
            state: State::Ready,
            kernel_stack_top: stack_top,
            ctx: frame_addr,
            name: nb,
        };
    }
}

/// Mark `pid` for termination. Returns `false` for pid 0, unknown pids and
/// already-dead ones. A zombie is skipped by the scheduler and its slot is
/// reusable by the next [`start_process`].
pub fn kill(pid: u32) -> bool {
    if pid == 0 || pid as usize >= MAX_PROCESSES {
        return false;
    }
    // Safety: single CPU; shell context (irq_save'd by the caller).
    let p = unsafe { &mut *table_mut().add(pid as usize) };
    if p.state == State::Zombie || p.state == State::Unused {
        return false;
    }
    p.state = State::Zombie;
    true
}

/// The pid of the process currently on the CPU.
pub fn current_pid() -> u32 {
    // Safety: single CPU; read from syscall context (IF=0).
    unsafe { table_const().add(CURRENT).read().pid }
}

/// Copy of a process for the shell's `ps` command.
#[derive(Clone, Copy)]
pub struct ProcDesc {
    pub pid: u32,
    pub running: bool,
    pub name: [u8; 16],
}

/// Fill `buf` with one entry per live process. Returns the count.
pub fn snapshot(buf: &mut [ProcDesc]) -> usize {
    // Safety: irq_save serializes us against the tick's table updates.
    let saved = unsafe { crate::arch::irq_save() };
    let mut n = 0;
    // Safety: single CPU, table stable under irq_save.
    unsafe {
        for i in 0..MAX_PROCESSES {
            let p = &*table_const().add(i);
            if p.state != State::Unused && p.state != State::Zombie && n < buf.len() {
                buf[n] = ProcDesc { pid: p.pid, running: p.state == State::Running, name: p.name };
                n += 1;
            }
        }
    }
    // Safety: `saved` came from the irq_save above on this CPU.
    unsafe { crate::arch::irq_restore(saved) };
    n
}

/// The next ready process after `CURRENT`, if any.
///
/// # Safety
///
/// Caller holds the only table access (IF=0 or irq_save).
unsafe fn pick_next_other() -> Option<usize> {
    // Safety: covered by the function's `# Safety` contract.
    unsafe {
        let cur = CURRENT;
        for i in 1..MAX_PROCESSES {
            let idx = (cur + i) % MAX_PROCESSES;
            if table_const().add(idx).read().state == State::Ready {
                return Some(idx);
            }
        }
    }
    None
}

/// Commit to running process `idx`: mark it Running, make it current, point
/// `TSS.rsp0` at its kernel stack, and iretq its frame. Never returns.
///
/// # Safety
///
/// `idx` must be Ready and its `ctx` a valid [`InterruptFrame`] on its own
/// kernel stack; the caller holds exclusive table access.
#[cfg(all(target_arch = "x86_64", not(test)))]
unsafe fn do_switch(idx: usize) -> ! {
    // Safety: covered by the function's `# Safety` contract.
    unsafe {
        let table = table_mut();
        (*table.add(idx)).state = State::Running;
        CURRENT = idx;
        let ctx = (*table.add(idx)).ctx;
        let ktop = (*table.add(idx)).kernel_stack_top;
        crate::tss::set_rsp0(ktop);
        sched_switch(ctx)
    }
}

/// PIT tick: preempt the current process unless nothing else is ready.
#[cfg(all(target_arch = "x86_64", not(test)))]
pub fn on_tick(frame: &InterruptFrame) {
    // Safety: the tick runs with IF=0 and owns the table.
    unsafe {
        let cur = CURRENT;
        let table = table_mut();
        (*table.add(cur)).state = State::Ready;
        (*table.add(cur)).ctx = frame as *const InterruptFrame as usize;
        if let Some(next) = pick_next_other() {
            do_switch(next)
        }
        // No other ready process: keep running (CURRENT is unchanged).
        (*table.add(cur)).state = State::Running;
    }
}

/// `SYS_YIELD`: give up the rest of the timeslice voluntarily.
#[cfg(all(target_arch = "x86_64", not(test)))]
pub fn yield_now(frame: &InterruptFrame) {
    // Safety: syscall context, IF=0.
    unsafe {
        let cur = CURRENT;
        let table = table_mut();
        (*table.add(cur)).state = State::Ready;
        (*table.add(cur)).ctx = frame as *const InterruptFrame as usize;
        if let Some(next) = pick_next_other() {
            do_switch(next)
        }
        (*table.add(cur)).state = State::Running;
    }
}

/// `SYS_EXIT`: mark the current process dead; it is never resumed.
#[cfg(all(target_arch = "x86_64", not(test)))]
pub fn exit_current(_frame: &InterruptFrame) -> ! {
    // Safety: syscall context, IF=0.
    unsafe {
        let cur = CURRENT;
        let table = table_mut();
        (*table.add(cur)).state = State::Zombie;
        let next = pick_next_other().expect("scheduler has no fallback process");
        do_switch(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cargo runs tests on many threads; the process table is a global static,
    /// so each test takes the shared kernel test lock (as `vmm` does).
    fn reset() {
        // Safety: test-only, single-threaded, no interrupts on the host.
        unsafe {
            let mut p = core::ptr::addr_of_mut!(PROCESSES) as *mut Process;
            let end = p.add(MAX_PROCESSES);
            while p < end {
                p.write(UNUSED);
                p = p.add(1);
            }
            CURRENT = 0;
            SCHEDULING = false;
        }
    }

    #[test]
    fn idle_is_slot_zero() {
        let _g = crate::boot::testutil::test_lock();
        reset();
        init_idle(0x1000);
        let mut buf = [ProcDesc { pid: 0, running: false, name: [0; 16] }; MAX_PROCESSES];
        let n = snapshot(&mut buf);
        assert_eq!(n, 1);
        assert_eq!(buf[0].pid, 0);
        assert_eq!(core::str::from_utf8(&buf[0].name).unwrap(), "idle\0\0\0\0\0\0\0\0\0\0\0\0");
    }

    #[test]
    fn start_process_assigns_stacks_and_frame() {
        let _g = crate::boot::testutil::test_lock();
        reset();
        let pid = free_slot().expect("free slot");
        assert_eq!(pid, 1);
        start_process(pid, 0x8000_0000, "demo");
        assert_eq!(user_code_va(1), 0x8020_0000);
        assert_eq!(user_code_va(2), 0x8040_0000);

        let p = unsafe { &*table_const().add(pid as usize) };
        assert_eq!(p.state, State::Ready);
        assert!(p.kernel_stack_top > 0);
        // The initial frame is a valid iretq tail into ring 3.
        assert_eq!(p.ctx % 16, 0, "frame must be 16-aligned");
        let f = unsafe { &*(p.ctx as *const InterruptFrame) };
        assert_eq!(f.rip, 0x8000_0000);
        assert_eq!(f.cs, USER_CODE64 as u64);
        assert_eq!(f.user_ss, USER_DATA as u64);
        assert_eq!(f.user_rsp, user_stack_top(pid) as u64);
        assert_eq!(f.rflags & (1 << 9), 1 << 9, "IF must be set");
        assert_eq!(f.rdi, pid as u64);
        assert_eq!(core::str::from_utf8(&p.name).unwrap(), "demo\0\0\0\0\0\0\0\0\0\0\0\0");
    }

    #[test]
    fn kill_turns_zombie_and_frees_slot() {
        let _g = crate::boot::testutil::test_lock();
        reset();
        let pid = free_slot().unwrap();
        start_process(pid, 0x8000_0000, "p");
        assert!(kill(pid));
        assert!(!kill(pid), "already zombie");
        assert!(!kill(0), "idle is not killable");
        assert_eq!(free_slot().unwrap(), pid, "zombie slot is reusable");
    }

    #[test]
    fn table_full_returns_none() {
        let _g = crate::boot::testutil::test_lock();
        reset();
        init_idle(0x1000);
        for _ in 1..MAX_PROCESSES {
            let pid = free_slot().expect("slot");
            start_process(pid, 0x8000_0000, "x");
        }
        assert_eq!(free_slot(), None);
    }

    #[test]
    fn pick_round_robins_after_current() {
        let _g = crate::boot::testutil::test_lock();
        reset();
        init_idle(0x1000);
        for pid in 1..=3u32 {
            start_process(pid, 0x8000_0000, "x");
        }
        let p = table_mut();
        // Safety: test-only direct table manipulation, no interrupts.
        unsafe {
            CURRENT = 0;
            assert_eq!(pick_next_other(), Some(1));
            CURRENT = 1;
            assert_eq!(pick_next_other(), Some(2));
            CURRENT = 2;
            assert_eq!(pick_next_other(), Some(3));
            CURRENT = 3;
            assert_eq!(pick_next_other(), Some(0), "wraps to idle");
            // A zombie is skipped.
            (*p.add(2)).state = State::Zombie;
            CURRENT = 1;
            assert_eq!(pick_next_other(), Some(3));
        }
    }
}
