/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

.text
.align 4

// ============================================================
// DistanceCodesN8x8S
//   Computes 8 distances in parallel for PQ codes (n8 layout).
//   AAPCS64:
//     x0 = M
//     x1 = sim_table
//     x2 = codes
//     x3 = result
//     s0 = dis0
// ============================================================
.global DistanceCodesN8x8S
.type   DistanceCodesN8x8S, %function
DistanceCodesN8x8S:
        // v3 = [dis0, dis0, dis0, dis0]
        dup  v3.4s, v0.s[0]
        stp  x29, x30, [sp, #-112]!
        mov  x29, sp
        stp  x25, x26, [sp, #64]
        add  x25, x2, x0          // codes + M
        stp  x21, x22, [sp, #32]
        add  x22, x25, x0         // codes + 2M
        stp  x19, x20, [sp, #16]
        add  x20, x22, x0         // codes + 3M
        stp  x23, x24, [sp, #48]
        add  x24, x20, x0         // codes + 4M
        add  x23, x24, x0         // codes + 5M
        add  x21, x23, x0         // codes + 6M
        add  x26, x21, x0         // codes + 7M
        cmp  x0, #0x3
        b.ls .Lsmall_M            // M <= 3

        sub  x7, x0, #0x4
        mov  x4, x1               // x4 = current table base
        mov  v1.16b, v3.16b       // v1 = base bias
        stp  x27, x28, [sp, #80]
        lsr  x6, x7, #2
        add  x8, x6, #0x1
        mov  x5, #0x0             // offset = 0
        str  x8, [sp, #104]
        lsl  x27, x8, #2
        nop

.Lloop_main:
        // load 4 bytes from each of 8 codes
        ldr  w17, [x2, x5]
        add  x9, x4, #0x400
        ldr  w16, [x25, x5]
        add  x8, x4, #0x800
        ldr  w13, [x24, x5]
        and  x18, x17, #0xff
        ldr  w12, [x23, x5]
        and  x10, x16, #0xff
        and  x14, x13, #0xff
        ldr  w15, [x22, x5]
        and  x6,  x12, #0xff

        // row0 gathers
        ldr  s2,  [x4, x18, lsl #2]
        ldr  s0,  [x4, x10, lsl #2]
        ubfx x19, x16, #8, #8
        ldr  s7,  [x4, x14, lsl #2]
        ubfx x10, x17, #8, #8
        ldr  s4,  [x4, x6,  lsl #2]
        ubfx x18, x13, #8, #8
        ldr  w14, [x20, x5]
        ubfx x6,  x12, #8, #8
        ldr  w11, [x21, x5]
        and  x30, x15, #0xff

        mov  v2.s[1], v0.s[0]     // pack row0
        ldr  s16, [x9, x18, lsl #2]
        ldr  s6,  [x9, x6,  lsl #2]
        and  x28, x11, #0xff
        ldr  s0,  [x9, x10, lsl #2]
        ubfx x18, x15, #8, #8
        ubfx x6,  x14, #8, #8
        ldr  s17, [x4, x30, lsl #2]
        mov  v7.s[1], v4.s[0]
        ldr  s4,  [x9, x19, lsl #2]
        and  x19, x14, #0xff
        ldr  s5,  [x4, x28, lsl #2]
        ldr  s26, [x9, x6,  lsl #2]
        ubfx x28, x17, #16, #8
        add  x6,  x4, #0xc00
        ldr  w10, [x26, x5]
        ldr  s22, [x4, x19, lsl #2]
        ubfx x30, x16, #16, #8
        ubfx x19, x15, #16, #8
        lsr  w17, w17, #24

        mov  v0.s[1], v4.s[0]
        ldr  s4,  [x9, x18, lsl #2]
        ubfx x18, x14, #16, #8
        ldr  s21, [x8, x30, lsl #2]
        ldr  s25, [x8, x19, lsl #2]
        lsr  w14, w14, #24
        mov  v2.s[2], v17.s[0]
        ldr  s17, [x8, x28, lsl #2]
        ldr  s29, [x8, x18, lsl #2]
        and  x28, x10, #0xff
        mov  v7.s[2], v5.s[0]

        // row3 base
        ldr  s5,  [x6, x17, lsl #2]
        ubfx x30, x11, #8, #8
        ubfx x19, x10, #8, #8
        ubfx x18, x13, #16, #8
        ubfx x17, x12, #16, #8
        lsr  w16, w16, #24
        lsr  w15, w15, #24

        mov  v16.s[1], v6.s[0]
        ldr  s20, [x9, x30, lsl #2]
        ldr  s27, [x6, x14, lsl #2]
        lsr  w13, w13, #24
        ldr  s23, [x9, x19, lsl #2]
        ubfx x14, x11, #16, #8
        ldr  s24, [x6, x16, lsl #2]
        lsr  w12, w12, #24
        ldr  s28, [x6, x15, lsl #2]
        ubfx x9,  x10, #16, #8

        // prefetch next tables
        prfm pldl1keep, [x4, #8192]

        lsr  w11, w11, #24
        ldr  s19, [x4, x28, lsl #2]
        lsr  w10, w10, #24
        ldr  s18, [x8, x17, lsl #2]
        add  x5,  x5, #0x4
        ldr  s6,  [x8, x18, lsl #2]
        add  x4,  x4, #0x1, lsl #12

        // pack and accumulate
        mov  v2.s[3], v22.s[0]
        ldr  s22, [x8, x14, lsl #2]
        mov  v0.s[2], v4.s[0]
        ldr  s4,  [x6, x13, lsl #2]
        mov  v17.s[1], v21.s[0]
        ldr  s21, [x6, x12, lsl #2]
        mov  v16.s[2], v20.s[0]
        ldr  s20, [x8, x9,  lsl #2]
        mov  v7.s[3], v19.s[0]
        ldr  s19, [x6, x11, lsl #2]
        mov  v6.s[1], v18.s[0]
        ldr  s18, [x6, x10, lsl #2]

        fadd v2.4s, v2.4s, v3.4s
        mov  v0.s[3], v26.s[0]
        mov  v17.s[2], v25.s[0]
        mov  v5.s[1], v24.s[0]
        mov  v16.s[3], v23.s[0]
        fadd v7.4s, v7.4s, v1.4s
        mov  v6.s[2], v22.s[0]
        mov  v4.s[1], v21.s[0]
        fadd v0.4s, v0.4s, v2.4s
        mov  v17.s[3], v29.s[0]
        mov  v5.s[2], v28.s[0]
        fadd v1.4s, v16.4s, v7.4s
        mov  v6.s[3], v20.s[0]
        mov  v4.s[2], v19.s[0]
        fadd v0.4s, v0.4s, v17.4s
        mov  v5.s[3], v27.s[0]
        fadd v1.4s, v1.4s, v6.4s
        mov  v4.s[3], v18.s[0]
        fadd v3.4s, v0.4s, v5.4s
        fadd v1.4s, v1.4s, v4.4s

        cmp  x5, x27
        b.ne .Lloop_main

        // advance pointers for tail handling
        ldr  x8, [sp, #104]
        and  x7, x7, #0xfffffffffffffffc
        ldp  x27, x28, [sp, #80]
        add  x2,  x2,  x5
        add  x25, x25, x5
        add  x22, x22, x5
        add  x20, x20, x5
        add  x24, x24, x5
        add  x23, x23, x5
        add  x21, x21, x5
        add  x26, x26, x5
        add  x1,  x1,  x8, lsl #12
        add  x5,  x7,  #0x4
        add  x7,  x7,  #0x5

.Ltail_ge4:
        cmp  x0, x7
        b.ls .Ltail_lt_full

        // 2-byte tail chunk (process 2 sub-quantizers)
        ldr  w7,  [x24], #2
        add  x4,  x1,  #0x400
        ldr  w10, [x2],  #2
        add  x5,  x5,  #0x2
        ldr  w8,  [x25], #2
        and  x12, x7,  #0xff
        ldr  w6,  [x23], #2
        and  x14, x10, #0xff
        and  x13, x8,  #0xff
        ldr  w9,  [x22], #2
        and  x11, x6,  #0xff
        ldr  s0,  [x1, x12, lsl #2]
        ubfx x12, x8,  #8, #8
        ldr  w8,  [x21], #2
        ldr  s5,  [x1, x13, lsl #2]
        and  x15, x9,  #0xff
        ldr  s4,  [x1, x11, lsl #2]
        ubfx x13, x10, #8, #8
        ldr  s2,  [x1, x14, lsl #2]
        ubfx x10, x6,  #8, #8
        ubfx x11, x7,  #8, #8
        and  x14, x8,  #0xff
        ldr  w7,  [x20], #2
        ubfx x9,  x9,  #8, #8
        mov  v2.s[1], v5.s[0]
        ldr  w6,  [x26], #2
        mov  v0.s[1], v4.s[0]
        ldr  s17, [x1, x15, lsl #2]
        ldr  s16, [x4, x12, lsl #2]
        ubfx x8,  x8,  #8, #8
        ldr  s7,  [x1, x14, lsl #2]
        and  x12, x7,  #0xff
        ldr  s6,  [x4, x10, lsl #2]
        and  x10, x6,  #0xff
        ldr  s5,  [x4, x13, lsl #2]
        ubfx x7,  x7,  #8, #8
        ldr  s4,  [x4, x11, lsl #2]
        ubfx x6,  x6,  #8, #8
        mov  v2.s[2], v17.s[0]
        ldr  s17, [x1, x12, lsl #2]
        mov  v5.s[1], v16.s[0]
        ldr  s16, [x4, x9,  lsl #2]
        mov  v0.s[2], v7.s[0]
        ldr  s7,  [x1, x10, lsl #2]
        mov  v4.s[1], v6.s[0]
        ldr  s6,  [x4, x8,  lsl #2]
        mov  v2.s[3], v17.s[0]
        prfm pldl1keep, [x1, #8192]
        mov  v5.s[2], v16.s[0]
        add  x1,  x1,  #0x800
        mov  v0.s[3], v7.s[0]
        ldr  s7,  [x4, x7,  lsl #2]
        mov  v4.s[2], v6.s[0]
        ldr  s6,  [x4, x6,  lsl #2]
        fadd v2.4s, v2.4s, v3.4s
        mov  v5.s[3], v7.s[0]
        fadd v0.4s, v0.4s, v1.4s
        mov  v4.s[3], v6.s[0]
        fadd v3.4s, v5.4s, v2.4s
        fadd v1.4s, v1.4s, v0.4s

.Ltail_lt_full:
        cmp  x5, x0
        b.cs .Lstore_out

        // final 1-byte tail
        ldrb w5, [x2]
        ldrb w4, [x25]
        ldrb w2, [x24]
        ldrb w0, [x23]
        ldr  s2, [x1, x5, lsl #2]
        ldr  s5, [x1, x4, lsl #2]
        ldr  s0, [x1, x2, lsl #2]
        ldr  s4, [x1, x0, lsl #2]
        ldrb w5, [x22]
        mov  v2.s[1], v5.s[0]
        ldrb w4, [x21]
        mov  v0.s[1], v4.s[0]
        ldrb w2, [x20]
        ldrb w0, [x26]
        ldr  s4, [x1, x5, lsl #2]
        ldr  s6, [x1, x4, lsl #2]
        ldr  s5, [x1, x2, lsl #2]
        mov  v2.s[2], v4.s[0]
        ldr  s4, [x1, x0, lsl #2]
        mov  v0.s[2], v6.s[0]
        mov  v2.s[3], v5.s[0]
        mov  v0.s[3], v4.s[0]
        fadd v3.4s, v3.4s, v2.4s
        fadd v1.4s, v1.4s, v0.4s

.Lstore_out:
        // store 8 results: [q3 q1]
        stp  q3, q1, [x3]
        ldp  x19, x20, [sp, #16]
        ldp  x21, x22, [sp, #32]
        ldp  x23, x24, [sp, #48]
        ldp  x25, x26, [sp, #64]
        ldp  x29, x30, [sp], #112
        ret

.Lsmall_M:
        // fall back to tail handler with small M
        mov  v1.16b, v3.16b
        mov  x7, #0x1
        mov  x5, #0x0
        b    .Ltail_ge4
.size DistanceCodesN8x8S, .-DistanceCodesN8x8S


// ============================================================
// DistanceCodesN8x4S
//   4 codes packed as [code0(M)][code1(M)][code2(M)][code3(M)]
//   AAPCS64: x0=M, x1=sim_table, x2=codes, x3=result, s0=dis
// ============================================================
.global DistanceCodesN8x4S
.type   DistanceCodesN8x4S, %function
DistanceCodesN8x4S:
    // acc = [dis, dis, dis, dis]
    dup     v16.4s, v0.s[0]

    // c0..c3
    mov     x10, x2
    add     x11, x2, x0
    add     x12, x11, x0
    add     x13, x12, x0

    mov     x6, x0
    cbz     x6, .L4_store

    cmp     x6, #4
    b.lt    .L4_tail_ge2

.L4_loop_main:
    prfm    pldl1keep, [x1, #8192]

    // read 4 bytes from each code stream
    ldr     w14, [x10], #4
    ldr     w15, [x11], #4
    ldr     w16, [x12], #4
    ldr     w17, [x13], #4

    // row bases
    mov     x8, x1
    add     x9, x1, #0x400
    add     x5, x1, #0x800
    add     x2, x1, #0xC00

    // row0 (byte0) -> v0
    and     w4, w14, #0xff
    ldr     s0, [x8, x4, lsl #2]
    and     w4, w15, #0xff
    ldr     s1, [x8, x4, lsl #2]
    and     w4, w16, #0xff
    ldr     s2, [x8, x4, lsl #2]
    and     w4, w17, #0xff
    ldr     s3, [x8, x4, lsl #2]
    mov     v0.s[1], v1.s[0]
    mov     v0.s[2], v2.s[0]
    mov     v0.s[3], v3.s[0]

    // row1 (byte1) -> v4
    ubfx    w4, w14, #8, #8
    ldr     s4, [x9, x4, lsl #2]
    ubfx    w4, w15, #8, #8
    ldr     s5, [x9, x4, lsl #2]
    ubfx    w4, w16, #8, #8
    ldr     s6, [x9, x4, lsl #2]
    ubfx    w4, w17, #8, #8
    ldr     s7, [x9, x4, lsl #2]
    mov     v4.s[1], v5.s[0]
    mov     v4.s[2], v6.s[0]
    mov     v4.s[3], v7.s[0]
    fadd    v0.4s, v0.4s, v4.4s

    // row2 (byte2) -> v4
    ubfx    w4, w14, #16, #8
    ldr     s4, [x5, x4, lsl #2]
    ubfx    w4, w15, #16, #8
    ldr     s5, [x5, x4, lsl #2]
    ubfx    w4, w16, #16, #8
    ldr     s6, [x5, x4, lsl #2]
    ubfx    w4, w17, #16, #8
    ldr     s7, [x5, x4, lsl #2]
    mov     v4.s[1], v5.s[0]
    mov     v4.s[2], v6.s[0]
    mov     v4.s[3], v7.s[0]

    // row3 (byte3) -> v5
    lsr     w14, w14, #24
    lsr     w15, w15, #24
    lsr     w16, w16, #24
    lsr     w17, w17, #24
    ldr     s5, [x2, x14, lsl #2]
    ldr     s1, [x2, x15, lsl #2]
    ldr     s2, [x2, x16, lsl #2]
    ldr     s3, [x2, x17, lsl #2]
    mov     v5.s[1], v1.s[0]
    mov     v5.s[2], v2.s[0]
    mov     v5.s[3], v3.s[0]

    // v0 += row2 + row3; acc += v0
    fadd    v4.4s, v4.4s, v5.4s
    fadd    v0.4s, v0.4s, v4.4s
    fadd    v16.4s, v16.4s, v0.4s

    // tab += 4 rows
    add     x1, x1, #0x1000
    subs    x6, x6, #4
    cmp     x6, #4
    b.ge    .L4_loop_main

.L4_tail_ge2:
    cmp     x6, #2
    b.lt    .L4_tail_ge1

    prfm    pldl1keep, [x1, #8192]

    // read 2 bytes per code
    ldrh    w14, [x10], #2
    ldrh    w15, [x11], #2
    ldrh    w16, [x12], #2
    ldrh    w17, [x13], #2

    mov     x8, x1
    add     x9, x1, #0x400

    // row0 (byte0)
    and     w4, w14, #0xff
    ldr     s0, [x8, x4, lsl #2]
    and     w4, w15, #0xff
    ldr     s1, [x8, x4, lsl #2]
    and     w4, w16, #0xff
    ldr     s2, [x8, x4, lsl #2]
    and     w4, w17, #0xff
    ldr     s3, [x8, x4, lsl #2]
    mov     v0.s[1], v1.s[0]
    mov     v0.s[2], v2.s[0]
    mov     v0.s[3], v3.s[0]

    // row1 (byte1)
    lsr     w14, w14, #8
    lsr     w15, w15, #8
    lsr     w16, w16, #8
    lsr     w17, w17, #8
    and     w4, w14, #0xff
    ldr     s4, [x9, x4, lsl #2]
    and     w4, w15, #0xff
    ldr     s5, [x9, x4, lsl #2]
    and     w4, w16, #0xff
    ldr     s6, [x9, x4, lsl #2]
    and     w4, w17, #0xff
    ldr     s7, [x9, x4, lsl #2]
    mov     v4.s[1], v5.s[0]
    mov     v4.s[2], v6.s[0]
    mov     v4.s[3], v7.s[0]

    fadd    v0.4s,  v0.4s,  v4.4s
    fadd    v16.4s, v16.4s, v0.4s

    add     x1, x1, #0x800
    subs    x6, x6, #2

.L4_tail_ge1:
    cbz     x6, .L4_store

    // 1 byte per code
    ldrb    w14, [x10], #1
    ldrb    w15, [x11], #1
    ldrb    w16, [x12], #1
    ldrb    w17, [x13], #1

    mov     x8, x1
    ldr     s0, [x8, x14, lsl #2]
    ldr     s1, [x8, x15, lsl #2]
    ldr     s2, [x8, x16, lsl #2]
    ldr     s3, [x8, x17, lsl #2]
    mov     v0.s[1], v1.s[0]
    mov     v0.s[2], v2.s[0]
    mov     v0.s[3], v3.s[0]
    fadd    v16.4s, v16.4s, v0.4s

.L4_store:
    str     q16, [x3]
    ret
.size DistanceCodesN8x4S, .-DistanceCodesN8x4S


// ============================================================
// DistanceCodesN8x2S
//   2 codes packed as [code0(M)][code1(M)]
//   AAPCS64: x0=M, x1=sim_table, x2=codes, x3=result, s0=dis
// ============================================================
.global DistanceCodesN8x2S
.type   DistanceCodesN8x2S, %function
DistanceCodesN8x2S:
    dup     v16.2s, v0.s[0]

    // c0, c1
    mov     x10, x2
    add     x11, x2, x0

    mov     x6, x0
    cbz     x6, .L2_store

    cmp     x6, #4
    b.lt    .L2_tail_ge2

.L2_loop_main:
    prfm    pldl1keep, [x1, #8192]

    ldr     w12, [x10], #4
    ldr     w13, [x11], #4

    mov     x8, x1
    add     x9, x1, #0x400
    add     x5, x1, #0x800
    add     x2, x1, #0xC00

    // row0 byte0
    and     w4, w12, #0xff
    ldr     s0, [x8, x4, lsl #2]
    and     w4, w13, #0xff
    ldr     s1, [x8, x4, lsl #2]
    mov     v0.s[1], v1.s[0]
    fadd    v16.2s, v16.2s, v0.2s

    // row1 byte1
    ubfx    w4, w12, #8, #8
    ldr     s0, [x9, x4, lsl #2]
    ubfx    w4, w13, #8, #8
    ldr     s1, [x9, x4, lsl #2]
    mov     v0.s[1], v1.s[0]
    fadd    v16.2s, v16.2s, v0.2s

    // row2 byte2
    ubfx    w4, w12, #16, #8
    ldr     s0, [x5, x4, lsl #2]
    ubfx    w4, w13, #16, #8
    ldr     s1, [x5, x4, lsl #2]
    mov     v0.s[1], v1.s[0]
    fadd    v16.2s, v16.2s, v0.2s

    // row3 byte3
    lsr     w12, w12, #24
    lsr     w13, w13, #24
    ldr     s0, [x2, x12, lsl #2]
    ldr     s1, [x2, x13, lsl #2]
    mov     v0.s[1], v1.s[0]
    fadd    v16.2s, v16.2s, v0.2s

    add     x1, x1, #0x1000
    subs    x6, x6, #4
    cmp     x6, #4
    b.ge    .L2_loop_main

.L2_tail_ge2:
    cmp     x6, #2
    b.lt    .L2_tail_ge1

    prfm    pldl1keep, [x1, #8192]

    ldrh    w12, [x10], #2
    ldrh    w13, [x11], #2

    mov     x8, x1
    add     x9, x1, #0x400

    // row0 byte0
    and     w4, w12, #0xff
    ldr     s0, [x8, x4, lsl #2]
    and     w4, w13, #0xff
    ldr     s1, [x8, x4, lsl #2]
    mov     v0.s[1], v1.s[0]
    fadd    v16.2s, v16.2s, v0.2s

    // row1 byte1
    lsr     w12, w12, #8
    lsr     w13, w13, #8
    and     w4, w12, #0xff
    ldr     s0, [x9, x4, lsl #2]
    and     w4, w13, #0xff
    ldr     s1, [x9, x4, lsl #2]
    mov     v0.s[1], v1.s[0]
    fadd    v16.2s, v16.2s, v0.2s

    add     x1, x1, #0x800
    subs    x6, x6, #2

.L2_tail_ge1:
    cbz     x6, .L2_store

    ldrb    w12, [x10]
    ldrb    w13, [x11]

    mov     x8, x1
    ldr     s0, [x8, x12, lsl #2]
    ldr     s1, [x8, x13, lsl #2]
    mov     v0.s[1], v1.s[0]
    fadd    v16.2s, v16.2s, v0.2s

.L2_store:
    str     d16, [x3]
    ret
.size DistanceCodesN8x2S, .-DistanceCodesN8x2S


// ============================================================
// DistanceCodesN8x1S
//   AAPCS64: x0=M, x1=sim_table, x2=code, s0=dis, return s0
// ============================================================
.global DistanceCodesN8x1S
.type   DistanceCodesN8x1S, %function
DistanceCodesN8x1S:
    // acc in s16
    fmov    s16, s0

    mov     x6, x0
    cbz     x6, .L1_ret

    cmp     x6, #4
    b.lt    .L1_tail_ge2

.L1_loop_main:
    prfm    pldl1keep, [x1, #8192]

    ldr     w12, [x2], #4

    mov     x8, x1
    add     x9, x1, #0x400
    add     x5, x1, #0x800
    add     x3, x1, #0xC00

    // row0 byte0
    and     w4, w12, #0xff
    ldr     s0, [x8, x4, lsl #2]
    fadd    s16, s16, s0

    // row1 byte1
    ubfx    w4, w12, #8, #8
    ldr     s0, [x9, x4, lsl #2]
    fadd    s16, s16, s0

    // row2 byte2
    ubfx    w4, w12, #16, #8
    ldr     s0, [x5, x4, lsl #2]
    fadd    s16, s16, s0

    // row3 byte3
    lsr     w12, w12, #24
    ldr     s0, [x3, x12, lsl #2]
    fadd    s16, s16, s0

    add     x1, x1, #0x1000
    subs    x6, x6, #4
    cmp     x6, #4
    b.ge    .L1_loop_main

.L1_tail_ge2:
    cmp     x6, #2
    b.lt    .L1_tail_ge1

    prfm    pldl1keep, [x1, #8192]

    ldrh    w12, [x2], #2

    mov     x8, x1
    add     x9, x1, #0x400

    // row0 byte0
    and     w4, w12, #0xff
    ldr     s0, [x8, x4, lsl #2]
    fadd    s16, s16, s0

    // row1 byte1
    lsr     w12, w12, #8
    and     w4, w12, #0xff
    ldr     s0, [x9, x4, lsl #2]
    fadd    s16, s16, s0

    add     x1, x1, #0x800
    subs    x6, x6, #2

.L1_tail_ge1:
    cbz     x6, .L1_ret

    ldrb    w12, [x2]
    ldr     s0, [x1, x12, lsl #2]
    fadd    s16, s16, s0

.L1_ret:
    fmov    s0, s16
    ret
.size DistanceCodesN8x1S, .-DistanceCodesN8x1S
