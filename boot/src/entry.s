.set MULTIBOOT_MAGIC, 0x1BADB002
.set MULTIBOOT_FLAGS, 0x3          /* align modules, provide memory map */
.set MULTIBOOT_CHECKSUM, -(MULTIBOOT_MAGIC + MULTIBOOT_FLAGS)

/* Multiboot v1 header: must be within the first 8192 bytes of the file and
 * 4-byte aligned. The linker places .multiboot first, at 1 MiB. The "a"
 * flag is required: a non-alloc section would be orphaned by the linker. */
.section .multiboot, "a", @progbits
.p2align 4
    .long MULTIBOOT_MAGIC
    .long MULTIBOOT_FLAGS
    .long MULTIBOOT_CHECKSUM

/* ------------------------------------------------------------------ */
/* 32-bit entry: bootloader jumped here with EAX = magic, EBX = info.  */
.section .text
.code32
.globl _start
_start:
    cli
    /* EBP and EBX are untouched by the setup below: EBP carries the multiboot
     * magic, EBX already holds the multiboot info pointer. */
    movl %eax, %ebp

    movl $boot_stack_top, %esp

    /* CR4.PAE */
    movl %cr4, %eax
    orl $0x20, %eax
    movl %eax, %cr4

    /* EFER.LME (bit 8 of MSR 0xC0000080) */
    movl $0xC0000080, %ecx
    rdmsr
    orl $0x100, %eax
    wrmsr

    /* Page tables live in .bss (zeroed by the loader). Fill four page
     * directories with 2 MiB huge pages, identity-mapping physical 0..4 GiB.
     * pd0..pd3 are declared consecutively, so one loop of 2048 entries
     * walks all four. Entry: present | writable | huge (0x83). */
    movl $0x83, %eax
    movl $pd0, %edi
    movl $2048, %ecx
1:  movl %eax, 0(%edi)
    addl $0x200000, %eax
    addl $8, %edi
    decl %ecx
    jnz 1b

    /* PDPTs: low 4 GiB of the identity map, and the physmap window that
     * covers physical 0..1 GiB (all of it for -m 512). */
    movl $pd0, %eax
    orl $0x3, %eax
    movl %eax, pdpt_low + 0
    movl %eax, pdpt_physmap + 0
    movl $pd1, %eax
    orl $0x3, %eax
    movl %eax, pdpt_low + 8
    movl $pd2, %eax
    orl $0x3, %eax
    movl %eax, pdpt_low + 16
    movl $pd3, %eax
    orl $0x3, %eax
    movl %eax, pdpt_low + 24

    /* PML4[0] = identity map, PML4[256] = physmap (0xFFFF_8000_0000_0000). */
    movl $pdpt_low, %eax
    orl $0x3, %eax
    movl %eax, pml4 + 0
    movl $pdpt_physmap, %eax
    orl $0x3, %eax
    movl %eax, pml4 + 256 * 8

    /* CR3 = PML4, then enable paging + protected mode. */
    movl $pml4, %eax
    movl %eax, %cr3
    movl %cr0, %eax
    orl $0x80000001, %eax
    movl %eax, %cr0

    /* Load the GDT and far-jump into 64-bit code (selector 0x18). A
     * push / lret is used instead of `ljmp` for integrated-assembler
     * portability; the target is a 32-bit address (kernel at 1 MiB). */
    lgdt gdt64_ptr
    pushl $0x18
    pushl $long_mode
    lret

/* ------------------------------------------------------------------ */
.code64
long_mode:
    movw $0x10, %ax
    movw %ax, %ds
    movw %ax, %es
    movw %ax, %fs
    movw %ax, %gs
    movw %ax, %ss

    /* Enable SSE/FPU. Modern rustc codegen (e.g. the u64 `Display` decimal
     * formatter) uses SSE2, so leaving it disabled makes those instructions
     * #UD. 64-bit mode guarantees SSE2 exists: CR4.OSFXSR|OSXMMEXCPT, clear
     * CR0.EM, keep CR0.MP, clear CR0.TS, then fninit the x87 unit. */
    movq %cr4, %rax
    orq $0x600, %rax
    movq %rax, %cr4
    movq %cr0, %rax
    andq $0xFFFFFFFFFFFFFFF9, %rax /* ~(EM|TS) */
    orq $0x2, %rax               /* MP */
    movq %rax, %cr0
    fninit

    /* Rust entry: (magic: u64, info: usize). */
    movl %ebp, %edi
    movl %ebx, %esi
    call rust_entry

hlt_loop:
    hlt
    jmp hlt_loop

/* ------------------------------------------------------------------ */
/* Zeroed BSS: page tables (page-aligned) and the boot stack.          */
.section .boot_bss, "aw", @nobits
.p2align 12
pml4:         .skip 4096
pdpt_low:     .skip 4096
pdpt_physmap: .skip 4096
pd0:          .skip 4096
pd1:          .skip 4096
pd2:          .skip 4096
pd3:          .skip 4096
.p2align 4
boot_stack_bottom:
              .skip 32768
.globl boot_stack_top
boot_stack_top:

/* ------------------------------------------------------------------ */
.section .rodata
.p2align 3
gdt64:
    .quad 0x0000000000000000   /* null */
    .quad 0x00CF9A000000FFFF   /* 0x08: 32-bit code, base 0, limit 4 GiB */
    .quad 0x00CF92000000FFFF   /* 0x10: data */
    .quad 0x00AF9A000000FFFF   /* 0x18: 64-bit code */
gdt64_ptr:
    .word gdt64_ptr - gdt64 - 1
    .long gdt64

/* Silence the "missing .note.GNU-stack" warning: stack is non-executable. */
.section .note.GNU-stack, "", @progbits
