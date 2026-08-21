ENTRY(_start)

SECTIONS
{
    /* QEMU's -kernel loads the multiboot image; the kernel links at 1 MiB,
     * above the low-memory IVT/BDA region the boot protocol expects. */
    . = 1M;

    .multiboot : {
        /* KEEP: unreferenced, so --gc-sections would otherwise drop it. */
        KEEP(*(.multiboot))
    }

    .text : {
        *(.text .text.*)
    }

    .rodata : {
        *(.rodata .rodata.*)
    }

    .data : {
        *(.data .data.*)
    }

    .bss (NOLOAD) : ALIGN(4K) {
        _sbss = .;
        *(.bss .bss.*)
        /* Assembly BSS: page tables and the boot stack. */
        *(.boot_bss .boot_bss.*)
        _ebss = .;
    }

    /* Kernel image extent, frame-aligned, handed to pmm_init so it reserves
     * exactly the pages the kernel really occupies. */
    _skernel = ADDR(.multiboot);
    _ekernel = ALIGN(_ebss, 4K);

    /DISCARD/ : {
        *(.eh_frame*)
        *(.note.gnu.build-id)
        *(.comment)
    }
}
