#pragma once
/*
 * ddio.h - DDIO (Data Direct I/O) control for GPU->PM write path.
 *
 * Intel DDIO routes PCIe writes through the CPU L3 cache before reaching
 * memory (including PM). For GPU->PM writes, this is harmful:
 *   - Data stalls in CPU L3 cache
 *   - __threadfence_system() alone may not flush the CPU cache
 *   - Durability guarantee is lost until the cache line is evicted
 *
 * Solution (from GPM-ASPLOS22): disable DDIO before launching GPU kernels
 * that write to PM, re-enable after. With DDIO off, GPU PCIe writes go
 * directly to PM media, bypassing CPU cache entirely.
 *
 * Hardware requirement: Intel Skylake-X / Cascade Lake / Ice Lake server CPUs
 * Register: PCIe Root Port PERFCTRLSTS_0 (offset 0x180), Intel Xeon Scalable.
 *
 * Usage:
 *   1. Find your GPU's PCI bus: run `lspci | grep -i nvidia`
 *      Example output: "01:00.0 VGA ..." -> gpu_bus = 0x01
 *   2. Or call gpm_find_gpu_bus() to auto-detect.
 *   3. Call gpm_ddio_off(gpu_bus) before GPU PM kernels.
 *   4. Call gpm_ddio_on(gpu_bus)  after  GPU PM kernels.
 *
 * Adapted from GPM-ASPLOS22/LibGPM/change-ddio.h
 * Original: Copyright (c) 2020, Alireza Farshin, KTH Royal Institute of Technology
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pci/pci.h>
#include <sys/io.h>
#include <fcntl.h>
#include <unistd.h>

#ifndef DDIO_IMPL_H
#define DDIO_IMPL_H

/* Intel Xeon Scalable (Skylake-X/Cascade Lake) PCIe Root Port registers */
#define DDIO_PCI_VENDOR_INTEL           0x8086
#define DDIO_SKX_PERFCTRLSTS_0          0x180
#define DDIO_SKX_USE_ALLOC_FLOW_WR_BIT  0x80   /* bit 7: DDIO enabled when set */
#define DDIO_SKX_NOSNOOP_WR_BIT         0x08   /* bit 3: NoSnoopOpWrEn */

#define DDIO_NVIDIA_VENDOR_ID           0x10de

/* ------------------------------------------------------------------ */
/* Internal helpers                                                    */
/* ------------------------------------------------------------------ */

/*
 * Find the PCIe Root Port that bridges to the device on gpu_bus.
 * The root port's PCI_SUBORDINATE_BUS register equals the bus number
 * of the device hanging below it.
 */
static struct pci_dev *
_ddio_find_root_port(struct pci_access *pacc, uint8_t gpu_bus)
{
    struct pci_dev *dev;
    for (dev = pacc->devices; dev; dev = dev->next) {
        pci_fill_info(dev, PCI_FILL_IDENT | PCI_FILL_BASES);
        if (pci_read_byte(dev, PCI_SUBORDINATE_BUS) == gpu_bus)
            return dev;
    }
    fprintf(stderr, "ddio: cannot find PCIe root port for bus 0x%02x\n",
            gpu_bus);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

/*
 * gpm_find_gpu_bus: scan PCI bus and return the bus number of the
 * first NVIDIA GPU found.  Returns 0xFF on failure.
 *
 * Equivalent to running: lspci | grep -i nvidia
 * and extracting the bus part (XX in XX:YY.Z).
 */
static inline uint8_t gpm_find_gpu_bus(void)
{
    struct pci_access *pacc = pci_alloc();
    pci_init(pacc);
    pci_scan_bus(pacc);

    struct pci_dev *dev;
    uint8_t gpu_bus = 0xFF;
    for (dev = pacc->devices; dev; dev = dev->next) {
        pci_fill_info(dev, PCI_FILL_IDENT);
        if (dev->vendor_id == DDIO_NVIDIA_VENDOR_ID) {
            gpu_bus = dev->bus;
            char namebuf[256];
            char *name = pci_lookup_name(pacc, namebuf, sizeof(namebuf),
                                         PCI_LOOKUP_DEVICE,
                                         dev->vendor_id, dev->device_id);
            printf("ddio: found NVIDIA GPU at %04x:%02x:%02x.%d (%s)\n",
                   dev->domain, dev->bus, dev->dev, dev->func,
                   name ? name : "unknown");
            break;
        }
    }
    pci_cleanup(pacc);

    if (gpu_bus == 0xFF)
        fprintf(stderr, "ddio: no NVIDIA GPU found on PCI bus\n");

    return gpu_bus;
}

/*
 * gpm_ddio_status: returns 1 if DDIO is currently enabled, 0 if disabled.
 * Returns -1 on error.
 */
static inline int gpm_ddio_status(uint8_t gpu_bus)
{
    struct pci_access *pacc = pci_alloc();
    pci_init(pacc);
    pci_scan_bus(pacc);

    struct pci_dev *root = _ddio_find_root_port(pacc, gpu_bus);
    if (!root) {
        pci_cleanup(pacc);
        return -1;
    }

    uint32_t val = pci_read_long(root, DDIO_SKX_PERFCTRLSTS_0);
    pci_cleanup(pacc);

    return (val & DDIO_SKX_USE_ALLOC_FLOW_WR_BIT) ? 1 : 0;
}

/*
 * gpm_ddio_off: disable DDIO for the PCIe root port connected to gpu_bus.
 * Call this before launching GPU kernels that write to PM.
 * GPU->PM writes will bypass CPU L3, going directly to PM media.
 */
static inline void gpm_ddio_off(uint8_t gpu_bus)
{
    struct pci_access *pacc = pci_alloc();
    pci_init(pacc);
    pci_scan_bus(pacc);

    struct pci_dev *root = _ddio_find_root_port(pacc, gpu_bus);
    if (!root) {
        pci_cleanup(pacc);
        return;
    }

    uint32_t val = pci_read_long(root, DDIO_SKX_PERFCTRLSTS_0);
    if (val & DDIO_SKX_USE_ALLOC_FLOW_WR_BIT) {
        pci_write_long(root, DDIO_SKX_PERFCTRLSTS_0,
                       val & ~DDIO_SKX_USE_ALLOC_FLOW_WR_BIT);
        printf("ddio: DDIO disabled (GPU bus=0x%02x)\n", gpu_bus);
    } else {
        printf("ddio: DDIO was already disabled (GPU bus=0x%02x)\n", gpu_bus);
    }

    pci_cleanup(pacc);
}

/*
 * gpm_ddio_on: re-enable DDIO after GPU PM kernels complete.
 */
static inline void gpm_ddio_on(uint8_t gpu_bus)
{
    struct pci_access *pacc = pci_alloc();
    pci_init(pacc);
    pci_scan_bus(pacc);

    struct pci_dev *root = _ddio_find_root_port(pacc, gpu_bus);
    if (!root) {
        pci_cleanup(pacc);
        return;
    }

    uint32_t val = pci_read_long(root, DDIO_SKX_PERFCTRLSTS_0);
    if (!(val & DDIO_SKX_USE_ALLOC_FLOW_WR_BIT)) {
        pci_write_long(root, DDIO_SKX_PERFCTRLSTS_0,
                       val | DDIO_SKX_USE_ALLOC_FLOW_WR_BIT);
        printf("ddio: DDIO enabled (GPU bus=0x%02x)\n", gpu_bus);
    } else {
        printf("ddio: DDIO was already enabled (GPU bus=0x%02x)\n", gpu_bus);
    }

    pci_cleanup(pacc);
}

#endif /* DDIO_IMPL_H */
