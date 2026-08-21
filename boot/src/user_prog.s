/* Ring-3 process demo, copied verbatim into USER-mapped frames by
 * spawn_process. It is the same image for every pid; it prints a banner with
 * its pid, then five "alive" lines (yielding between them so the scheduler
 * interleaves several processes), then exits. Position-independent: only
 * rip-relative addressing and `int $0x80` syscalls. Syscall convention:
 * rax = number (0 = putc rdi:byte, 3 = getpid, 4 = yield, 2 = exit), args in
 * rdi/rsi/rdx. The kernel restores every register from the interrupt frame,
 * so values survive `int $0x80`.
 */
.section .text
.p2align 4
.globl _user_prog_start
_user_prog_start:
    /* banner "proc " */
    lea banner(%rip), %rbx
    mov $banner_len, %rcx
1:  movzbl (%rbx), %edi
    mov $0, %rax                 /* SYS_PUTC */
    int $0x80
    inc %rbx
    dec %rcx
    jnz 1b

    /* pid */
    mov $3, %rax                 /* SYS_GETPID */
    int $0x80
    mov %rax, %r12               /* pid lives in a callee-saved reg */
    mov %rax, %rdi
    call print_dec

    /* ": " */
    mov $0, %rax
    mov $58, %edi                /* ':' */
    int $0x80
    mov $0, %rax
    mov $32, %edi                /* ' ' */
    int $0x80

    mov $5, %r13                 /* five "alive" lines */
life:
    mov $0, %rax                 /* 'p' */
    mov $112, %edi
    int $0x80
    mov %r12, %rdi
    call print_dec
    lea alive(%rip), %rbx
    mov $alive_len, %rcx
2:  movzbl (%rbx), %edi
    mov $0, %rax
    int $0x80
    inc %rbx
    dec %rcx
    jnz 2b
    mov $6, %rax
    sub %r13, %rax               /* i = 6 - r13  => 1..5 */
    mov %rax, %rdi
    call print_dec
    mov $0, %rax
    mov $10, %edi                /* '\n' */
    int $0x80

    mov $4, %rax                 /* SYS_YIELD */
    int $0x80
    /* burn a bit of the timeslice so ticks also interleave processes */
    mov $2000000, %rbx
3:  dec %rbx
    jnz 3b
    dec %r13
    jnz life

    mov $2, %rax                 /* SYS_EXIT */
    int $0x80
    hlt                          /* unreachable if SYS_EXIT works */

/* print_dec: print u64 in rdi as decimal to SYS_PUTC. Clobbers rax, rcx, rdx,
 * rsi, rdi, and temporarily the ring-3 stack (restored before returning). */
print_dec:
    mov %rsp, %rsi               /* rsp before any digits pushed */
    mov %rdi, %rax
    mov $10, %rcx
1:  xor %rdx, %rdx
    div %rcx                     /* rdx:rax / 10 -> rax, remainder rdx */
    add $48, %rdx                /* '0' */
    push %rdx
    cmp $0, %rax
    jnz 1b
2:  pop %rdx                     /* most significant digit first */
    mov %rdx, %rdi
    mov $0, %rax
    int $0x80
    cmp %rsp, %rsi
    jne 2b
    ret

.section .rodata
banner:
    .ascii "proc "
banner_end:
    .equ banner_len, banner_end - banner
alive:
    .ascii ": alive "
alive_end:
    .equ alive_len, alive_end - alive
.globl _user_prog_end
_user_prog_end:
