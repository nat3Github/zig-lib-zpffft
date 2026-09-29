const std = @import("std");
const assert = std.debug.assert;
const simd_math = @import("simd_math.zig");

pub fn Passes(comptime T: type) type {
    return PassesL(T, 4);
}

/// Radix passes run an independent FFT per lane, so only the vector type
/// depends on L.
pub fn PassesL(comptime T: type, comptime L: comptime_int) type {
    const s = simd_math.SIMDL(T, L);

    return struct {
        pub fn rfftf1_ps(n: i32, input: s.CVP, work1: s.VP, work2: s.VP, wa: [*]const T, ifac: []const i32) s.VP {
            var in = @constCast(input);
            var out = if (in == work2) work1 else work2;
            const nf = @as(usize, @intCast(ifac[1]));
            var l2: i32 = n;
            var iw: i32 = n - 1;
            var k1: usize = 1;
            while (k1 <= nf) : (k1 += 1) {
                const kh = nf - k1;
                const ip = ifac[kh + 2];
                const l1 = @divExact(l2, ip);
                const ido = @divExact(n, l2);
                iw -= (ip - 1) * ido;
                switch (ip) {
                    5 => {
                        const ix2 = iw + ido;
                        const ix3 = ix2 + ido;
                        const ix4 = ix3 + ido;
                        radf5_ps(ido, l1, in, out, wa + @as(usize, @intCast(iw)), wa + @as(usize, @intCast(ix2)), wa + @as(usize, @intCast(ix3)), wa + @as(usize, @intCast(ix4)));
                    },
                    4 => {
                        const ix2 = iw + ido;
                        const ix3 = ix2 + ido;
                        radf4_ps(ido, l1, in, out, wa + @as(usize, @intCast(iw)), wa + @as(usize, @intCast(ix2)), wa + @as(usize, @intCast(ix3)));
                    },
                    3 => {
                        const ix2 = iw + ido;
                        radf3_ps(ido, l1, in, out, wa + @as(usize, @intCast(iw)), wa + @as(usize, @intCast(ix2)));
                    },
                    2 => {
                        radf2_ps(ido, l1, in, out, wa + @as(usize, @intCast(iw)));
                    },
                    else => std.debug.panic("rfftf1_ps: unreachable ip={}", .{ip}),
                }
                l2 = l1;
                if (out == work2) {
                    out = work1;
                    in = work2;
                } else {
                    out = work2;
                    in = work1;
                }
            }
            return in;
        }

        pub fn rfftb1_ps(n: i32, input: s.CVP, work1: s.VP, work2: s.VP, wa: [*]const T, ifac: []const i32) s.VP {
            var in = @constCast(input);
            var out = if (in == work2) work1 else work2;
            const nf = @as(usize, @intCast(ifac[1]));
            var l1: i32 = 1;
            var iw: i32 = 0;
            var k1: usize = 1;
            while (k1 <= nf) : (k1 += 1) {
                const ip = ifac[k1 + 1];
                const l2 = ip * l1;
                const ido = @divExact(n, l2);
                switch (ip) {
                    5 => {
                        const ix2 = iw + ido;
                        const ix3 = ix2 + ido;
                        const ix4 = ix3 + ido;
                        radb5_ps(ido, l1, in, out, wa + @as(usize, @intCast(iw)), wa + @as(usize, @intCast(ix2)), wa + @as(usize, @intCast(ix3)), wa + @as(usize, @intCast(ix4)));
                    },
                    4 => {
                        const ix2 = iw + ido;
                        const ix3 = ix2 + ido;
                        radb4_ps(ido, l1, in, out, wa + @as(usize, @intCast(iw)), wa + @as(usize, @intCast(ix2)), wa + @as(usize, @intCast(ix3)));
                    },
                    3 => {
                        const ix2 = iw + ido;
                        radb3_ps(ido, l1, in, out, wa + @as(usize, @intCast(iw)), wa + @as(usize, @intCast(ix2)));
                    },
                    2 => {
                        radb2_ps(ido, l1, in, out, wa + @as(usize, @intCast(iw)));
                    },
                    else => std.debug.panic("rfftb1_ps: unreachable ip={}", .{ip}),
                }
                l1 = l2;
                iw += (ip - 1) * ido;
                if (out == work2) {
                    out = work1;
                    in = work2;
                } else {
                    out = work2;
                    in = work1;
                }
            }
            return in;
        }

        pub fn cfftf1_ps(n: i32, input: s.CVP, work1: s.VP, work2: s.VP, wa: [*]const T, ifac: []const i32, isign: i32) s.VP {
            var in = @constCast(input);
            var out = if (in == work2) work1 else work2;
            const nf = @as(usize, @intCast(ifac[1]));
            var l1: i32 = 1;
            var iw: i32 = 0;
            var k1: usize = 2;
            const fsign: T = if (isign == -1) -1.0 else 1.0;
            while (k1 <= nf + 1) : (k1 += 1) {
                const ip = ifac[k1];
                const l2 = ip * l1;
                const ido = @divExact(n, l2);
                const idot = ido + ido;
                switch (ip) {
                    5 => {
                        const ix2 = iw + idot;
                        const ix3 = ix2 + idot;
                        const ix4 = ix3 + idot;
                        passf5_ps(idot, l1, in, out, wa + @as(usize, @intCast(iw)), wa + @as(usize, @intCast(ix2)), wa + @as(usize, @intCast(ix3)), wa + @as(usize, @intCast(ix4)), fsign);
                    },
                    4 => {
                        const ix2 = iw + idot;
                        const ix3 = ix2 + idot;
                        passf4_ps(idot, l1, in, out, wa + @as(usize, @intCast(iw)), wa + @as(usize, @intCast(ix2)), wa + @as(usize, @intCast(ix3)), fsign);
                    },
                    3 => {
                        const ix2 = iw + idot;
                        passf3_ps(idot, l1, in, out, wa + @as(usize, @intCast(iw)), wa + @as(usize, @intCast(ix2)), fsign);
                    },
                    2 => {
                        passf2_ps(idot, l1, in, out, wa + @as(usize, @intCast(iw)), fsign);
                    },
                    else => std.debug.panic("cfftf1_ps: unreachable ip={}", .{ip}),
                }
                l1 = l2;
                iw += (ip - 1) * idot;
                if (out == work2) {
                    out = work1;
                    in = work2;
                } else {
                    out = work2;
                    in = work1;
                }
            }
            return in;
        }

        // --- MERGED COMPLEX PASSES ---
        fn passf2_ps(ido: i32, l1: i32, cc: s.CVP, ch: s.VP, wa1: [*]const T, fsign: T) void {
            const l1ido = l1 * ido;
            if (ido <= 2) {
                var k: i32 = 0;
                while (k < l1ido) : (k += ido) {
                    const pc = cc + @as(usize, @intCast(2 * k));
                    const ph = ch + @as(usize, @intCast(k));
                    ph[0] = s.vadd(pc[0], pc[@as(usize, @intCast(ido))]);
                    ph[@as(usize, @intCast(l1ido))] = s.vsub(pc[0], pc[@as(usize, @intCast(ido))]);
                    ph[1] = s.vadd(pc[1], pc[@as(usize, @intCast(ido + 1))]);
                    ph[@as(usize, @intCast(l1ido + 1))] = s.vsub(pc[1], pc[@as(usize, @intCast(ido + 1))]);
                }
            } else {
                var k: i32 = 0;
                while (k < l1ido) : (k += ido) {
                    const pc = cc + @as(usize, @intCast(2 * k));
                    const ph = ch + @as(usize, @intCast(k));
                    var i: i32 = 0;
                    while (i < ido - 1) : (i += 2) {
                        var tr2 = s.vsub(pc[@as(usize, @intCast(i))], pc[@as(usize, @intCast(i + ido))]);
                        var ti2 = s.vsub(pc[@as(usize, @intCast(i + 1))], pc[@as(usize, @intCast(i + ido + 1))]);
                        const wr = wa1[@as(usize, @intCast(i))];
                        const wi = fsign * wa1[@as(usize, @intCast(i + 1))];
                        ph[@as(usize, @intCast(i))] = s.vadd(pc[@as(usize, @intCast(i))], pc[@as(usize, @intCast(i + ido))]);
                        ph[@as(usize, @intCast(i + 1))] = s.vadd(pc[@as(usize, @intCast(i + 1))], pc[@as(usize, @intCast(i + ido + 1))]);
                        const vwr = s.vld1(wr);
                        const vwi = s.vld1(wi);
                        s.vcplxmul(&tr2, &ti2, vwr, vwi);
                        ph[@as(usize, @intCast(i + l1ido))] = tr2;
                        ph[@as(usize, @intCast(i + l1ido + 1))] = ti2;
                    }
                }
            }
        }

        fn passf3_ps(ido: i32, l1: i32, cc: s.CVP, ch: s.VP, wa1: [*]const T, wa2: [*]const T, fsign: T) void {
            const taur: T = -0.5;
            const taui: T = 0.866025403784439 * fsign;
            const l1ido = l1 * ido;
            var k: i32 = 0;
            while (k < l1ido) : (k += ido) {
                const pc = cc + @as(usize, @intCast(3 * k));
                const ph = ch + @as(usize, @intCast(k));
                var i: i32 = 0;
                while (i < ido - 1) : (i += 2) {
                    const tr2 = s.vadd(pc[@as(usize, @intCast(i + ido))], pc[@as(usize, @intCast(i + 2 * ido))]);
                    const cr2 = s.vmadd(s.vld1(taur), tr2, pc[@as(usize, @intCast(i))]);
                    ph[@as(usize, @intCast(i))] = s.vadd(pc[@as(usize, @intCast(i))], tr2);
                    const ti2 = s.vadd(pc[@as(usize, @intCast(i + ido + 1))], pc[@as(usize, @intCast(i + 2 * ido + 1))]);
                    const ci2 = s.vmadd(s.vld1(taur), ti2, pc[@as(usize, @intCast(i + 1))]);
                    ph[@as(usize, @intCast(i + 1))] = s.vadd(pc[@as(usize, @intCast(i + 1))], ti2);
                    // cr3/ci3 = taui * d{r,i}, folded into the adds below.
                    const dr = s.vsub(pc[@as(usize, @intCast(i + ido))], pc[@as(usize, @intCast(i + 2 * ido))]);
                    const di = s.vsub(pc[@as(usize, @intCast(i + ido + 1))], pc[@as(usize, @intCast(i + 2 * ido + 1))]);
                    var dr2 = s.vnmsub(s.vld1(taui), di, cr2);
                    var dr3 = s.vmadd(s.vld1(taui), di, cr2);
                    var di2 = s.vmadd(s.vld1(taui), dr, ci2);
                    var di3 = s.vnmsub(s.vld1(taui), dr, ci2);
                    const wr1 = wa1[@as(usize, @intCast(i))];
                    const wi1 = fsign * wa1[@as(usize, @intCast(i + 1))];
                    const wr2 = wa2[@as(usize, @intCast(i))];
                    const wi2 = fsign * wa2[@as(usize, @intCast(i + 1))];
                    const vwr1 = s.vld1(wr1);
                    const vwi1 = s.vld1(wi1);
                    const vwr2 = s.vld1(wr2);
                    const vwi2 = s.vld1(wi2);
                    s.vcplxmul(&dr2, &di2, vwr1, vwi1);
                    ph[@as(usize, @intCast(i + l1ido))] = dr2;
                    ph[@as(usize, @intCast(i + l1ido + 1))] = di2;
                    s.vcplxmul(&dr3, &di3, vwr2, vwi2);
                    ph[@as(usize, @intCast(i + 2 * l1ido))] = dr3;
                    ph[@as(usize, @intCast(i + 2 * l1ido + 1))] = di3;
                }
            }
        }

        fn passf4_ps(ido: i32, l1: i32, cc: s.CVP, ch: s.VP, wa1: [*]const T, wa2: [*]const T, wa3: [*]const T, fsign: T) void {
            const l1ido = l1 * ido;
            if (ido == 2) {
                var k: i32 = 0;
                while (k < l1ido) : (k += ido) {
                    const pc = cc + @as(usize, @intCast(4 * k));
                    const ph = ch + @as(usize, @intCast(k));
                    const tr1 = s.vsub(pc[0], pc[@as(usize, @intCast(2 * ido))]);
                    const tr2 = s.vadd(pc[0], pc[@as(usize, @intCast(2 * ido))]);
                    const ti1 = s.vsub(pc[1], pc[@as(usize, @intCast(2 * ido + 1))]);
                    const ti2 = s.vadd(pc[1], pc[@as(usize, @intCast(2 * ido + 1))]);
                    // ti4/tr4 = fsign * d{i,r}4, folded into the adds below.
                    const di4 = s.vsub(pc[@as(usize, @intCast(ido))], pc[@as(usize, @intCast(3 * ido))]);
                    const dr4 = s.vsub(pc[@as(usize, @intCast(3 * ido + 1))], pc[@as(usize, @intCast(ido + 1))]);
                    const tr3 = s.vadd(pc[@as(usize, @intCast(ido))], pc[@as(usize, @intCast(3 * ido))]);
                    const ti3 = s.vadd(pc[@as(usize, @intCast(ido + 1))], pc[@as(usize, @intCast(3 * ido + 1))]);
                    ph[0] = s.vadd(tr2, tr3);
                    ph[1] = s.vadd(ti2, ti3);
                    ph[@as(usize, @intCast(l1ido))] = s.vmadd(s.vld1(fsign), dr4, tr1);
                    ph[@as(usize, @intCast(l1ido + 1))] = s.vmadd(s.vld1(fsign), di4, ti1);
                    ph[@as(usize, @intCast(2 * l1ido))] = s.vsub(tr2, tr3);
                    ph[@as(usize, @intCast(2 * l1ido + 1))] = s.vsub(ti2, ti3);
                    ph[@as(usize, @intCast(3 * l1ido))] = s.vnmsub(s.vld1(fsign), dr4, tr1);
                    ph[@as(usize, @intCast(3 * l1ido + 1))] = s.vnmsub(s.vld1(fsign), di4, ti1);
                }
            } else {
                var k: i32 = 0;
                while (k < l1ido) : (k += ido) {
                    const pc = cc + @as(usize, @intCast(4 * k));
                    const ph = ch + @as(usize, @intCast(k));
                    var i: i32 = 0;
                    while (i < ido - 1) : (i += 2) {
                        const tr1 = s.vsub(pc[@as(usize, @intCast(i))], pc[@as(usize, @intCast(i + 2 * ido))]);
                        const tr2 = s.vadd(pc[@as(usize, @intCast(i))], pc[@as(usize, @intCast(i + 2 * ido))]);
                        const ti1 = s.vsub(pc[@as(usize, @intCast(i + 1))], pc[@as(usize, @intCast(i + 2 * ido + 1))]);
                        const ti2 = s.vadd(pc[@as(usize, @intCast(i + 1))], pc[@as(usize, @intCast(i + 2 * ido + 1))]);
                        // tr4/ti4 = fsign * d{r,i}4, folded into the adds below.
                        const dr4 = s.vsub(pc[@as(usize, @intCast(i + 3 * ido + 1))], pc[@as(usize, @intCast(i + ido + 1))]);
                        const di4 = s.vsub(pc[@as(usize, @intCast(i + ido))], pc[@as(usize, @intCast(i + 3 * ido))]);
                        const tr3 = s.vadd(pc[@as(usize, @intCast(i + ido))], pc[@as(usize, @intCast(i + 3 * ido))]);
                        const ti3 = s.vadd(pc[@as(usize, @intCast(i + ido + 1))], pc[@as(usize, @intCast(i + 3 * ido + 1))]);
                        ph[@as(usize, @intCast(i))] = s.vadd(tr2, tr3);
                        var cr3 = s.vsub(tr2, tr3);
                        ph[@as(usize, @intCast(i + 1))] = s.vadd(ti2, ti3);
                        var ci3 = s.vsub(ti2, ti3);
                        var cr2 = s.vmadd(s.vld1(fsign), dr4, tr1);
                        var cr4 = s.vnmsub(s.vld1(fsign), dr4, tr1);
                        var ci2 = s.vmadd(s.vld1(fsign), di4, ti1);
                        var ci4 = s.vnmsub(s.vld1(fsign), di4, ti1);
                        const wr1 = wa1[@as(usize, @intCast(i))];
                        const wi1 = fsign * wa1[@as(usize, @intCast(i + 1))];
                        const wr2 = wa2[@as(usize, @intCast(i))];
                        const wi2 = fsign * wa2[@as(usize, @intCast(i + 1))];
                        const wr3 = wa3[@as(usize, @intCast(i))];
                        const wi3 = fsign * wa3[@as(usize, @intCast(i + 1))];
                        
                        // Load twiddles as pre-broadcasted vectors
                        const vwr1 = s.vld1(wr1);
                        const vwi1 = s.vld1(wi1);
                        const vwr2 = s.vld1(wr2);
                        const vwi2 = s.vld1(wi2);
                        const vwr3 = s.vld1(wr3);
                        const vwi3 = s.vld1(wi3);

                        s.vcplxmul(&cr2, &ci2, vwr1, vwi1);
                        ph[@as(usize, @intCast(i + l1ido))] = cr2;
                        ph[@as(usize, @intCast(i + l1ido + 1))] = ci2;
                        s.vcplxmul(&cr3, &ci3, vwr2, vwi2);
                        ph[@as(usize, @intCast(i + 2 * l1ido))] = cr3;
                        ph[@as(usize, @intCast(i + 2 * l1ido + 1))] = ci3;
                        s.vcplxmul(&cr4, &ci4, vwr3, vwi3);
                        ph[@as(usize, @intCast(i + 3 * l1ido))] = cr4;
                        ph[@as(usize, @intCast(i + 3 * l1ido + 1))] = ci4;
                    }
                }
            }
        }

        fn passf5_ps(ido: i32, l1: i32, cc: s.CVP, ch: s.VP, wa1: [*]const T, wa2: [*]const T, wa3: [*]const T, wa4: [*]const T, fsign: T) void {
            const tr11: T = 0.309016994374947;
            const ti11: T = 0.951056516295154 * fsign;
            const tr12: T = -0.809016994374947;
            const ti12: T = 0.587785252292473 * fsign;
            const l1ido = l1 * ido;
            var k: i32 = 0;
            while (k < l1) : (k += 1) {
                const pc = cc + @as(usize, @intCast(k * 5 * ido));
                const ph = ch + @as(usize, @intCast(k * ido));
                var i: i32 = 0;
                while (i < ido - 1) : (i += 2) {
                    const trV2 = s.vadd(pc[@as(usize, @intCast(i + ido))], pc[@as(usize, @intCast(i + 4 * ido))]);
                    const tiV2 = s.vadd(pc[@as(usize, @intCast(i + 1 + ido))], pc[@as(usize, @intCast(i + 1 + 4 * ido))]);
                    const trV5 = s.vsub(pc[@as(usize, @intCast(i + ido))], pc[@as(usize, @intCast(i + 4 * ido))]);
                    const tiV5 = s.vsub(pc[@as(usize, @intCast(i + 1 + ido))], pc[@as(usize, @intCast(i + 1 + 4 * ido))]);
                    const trV3 = s.vadd(pc[@as(usize, @intCast(i + 2 * ido))], pc[@as(usize, @intCast(i + 3 * ido))]);
                    const tiV3 = s.vadd(pc[@as(usize, @intCast(i + 1 + 2 * ido))], pc[@as(usize, @intCast(i + 1 + 3 * ido))]);
                    const trV4 = s.vsub(pc[@as(usize, @intCast(i + 2 * ido))], pc[@as(usize, @intCast(i + 3 * ido))]);
                    const tiV4 = s.vsub(pc[@as(usize, @intCast(i + 1 + 2 * ido))], pc[@as(usize, @intCast(i + 1 + 3 * ido))]);
                    
                    ph[@as(usize, @intCast(i))] = s.vadd(pc[@as(usize, @intCast(i))], s.vadd(trV2, trV3));
                    ph[@as(usize, @intCast(i + 1))] = s.vadd(pc[@as(usize, @intCast(i + 1))], s.vadd(tiV2, tiV3));
                    
                    const crV2 = s.vmadd(s.vld1(tr12), trV3, s.vmadd(s.vld1(tr11), trV2, pc[@as(usize, @intCast(i))]));
                    const ciV2 = s.vmadd(s.vld1(tr12), tiV3, s.vmadd(s.vld1(tr11), tiV2, pc[@as(usize, @intCast(i + 1))]));
                    const crV3 = s.vmadd(s.vld1(tr11), trV3, s.vmadd(s.vld1(tr12), trV2, pc[@as(usize, @intCast(i))]));
                    const ciV3 = s.vmadd(s.vld1(tr11), tiV3, s.vmadd(s.vld1(tr12), tiV2, pc[@as(usize, @intCast(i + 1))]));

                    const crV5 = s.vmadd(s.vld1(ti11), trV5, s.vmul(s.vld1(ti12), trV4));
                    const ciV5 = s.vmadd(s.vld1(ti11), tiV5, s.vmul(s.vld1(ti12), tiV4));
                    const crV4 = s.vnmsub(s.vld1(ti11), trV4, s.vmul(s.vld1(ti12), trV5));
                    const ciV4 = s.vnmsub(s.vld1(ti11), tiV4, s.vmul(s.vld1(ti12), tiV5));
                    
                    var drV3 = s.vsub(crV3, ciV4); var drV4 = s.vadd(crV3, ciV4);
                    var diV3 = s.vadd(ciV3, crV4); var diV4 = s.vsub(ciV3, crV4);
                    var drV5 = s.vadd(crV2, ciV5); var drV2 = s.vsub(crV2, ciV5);
                    var diV5 = s.vsub(ciV2, crV5); var diV2 = s.vadd(ciV2, crV5);
                    
                    const wr1 = wa1[@as(usize, @intCast(i))]; const wi1 = fsign * wa1[@as(usize, @intCast(i + 1))];
                    const wr2 = wa2[@as(usize, @intCast(i))]; const wi2 = fsign * wa2[@as(usize, @intCast(i + 1))];
                    const wr3 = wa3[@as(usize, @intCast(i))]; const wi3 = fsign * wa3[@as(usize, @intCast(i + 1))];
                    const wr4 = wa4[@as(usize, @intCast(i))]; const wi4 = fsign * wa4[@as(usize, @intCast(i + 1))];
                    
                    const vwr1 = s.vld1(wr1); const vwi1 = s.vld1(wi1);
                    const vwr2 = s.vld1(wr2); const vwi2 = s.vld1(wi2);
                    const vwr3 = s.vld1(wr3); const vwi3 = s.vld1(wi3);
                    const vwr4 = s.vld1(wr4); const vwi4 = s.vld1(wi4);
                    
                    s.vcplxmul(&drV2, &diV2, vwr1, vwi1);
                    ph[@as(usize, @intCast(i + l1ido))] = drV2; ph[@as(usize, @intCast(i + 1 + l1ido))] = diV2;
                    s.vcplxmul(&drV3, &diV3, vwr2, vwi2);
                    ph[@as(usize, @intCast(i + 2 * l1ido))] = drV3; ph[@as(usize, @intCast(i + 1 + 2 * l1ido))] = diV3;
                    s.vcplxmul(&drV4, &diV4, vwr3, vwi3);
                    ph[@as(usize, @intCast(i + 3 * l1ido))] = drV4; ph[@as(usize, @intCast(i + 1 + 3 * l1ido))] = diV4;
                    s.vcplxmul(&drV5, &diV5, vwr4, vwi4);
                    ph[@as(usize, @intCast(i + 4 * l1ido))] = drV5; ph[@as(usize, @intCast(i + 1 + 4 * l1ido))] = diV5;
                }
            }
        }

        // --- REAL PASSES ---
        pub fn radf2_ps(ido: i32, l1: i32, cc: s.CVP, ch: s.VP, wa1: [*]const T) void {
            const l1ido = l1 * ido;
            var k: i32 = 0;
            while (k < l1ido) : (k += ido) {
                const v_a = cc[@as(usize, @intCast(k))];
                const v_b = cc[@as(usize, @intCast(k + l1ido))];
                ch[@as(usize, @intCast(2 * k))] = s.vadd(v_a, v_b);
                ch[@as(usize, @intCast(2 * (k + ido) - 1))] = s.vsub(v_a, v_b);
            }
            if (ido < 2) return;
            if (ido != 2) {
                k = 0;
                while (k < l1ido) : (k += ido) {
                    var i: i32 = 2;
                    while (i < ido) : (i += 2) {
                        var tr2 = cc[@as(usize, @intCast(i - 1 + k + l1ido))];
                        var ti2 = cc[@as(usize, @intCast(i + k + l1ido))];
                        s.vcplxmulconj(&tr2, &ti2, s.vld1(wa1[@as(usize, @intCast(i - 2))]), s.vld1(wa1[@as(usize, @intCast(i - 1))]));
                        ch[@as(usize, @intCast(i + 2 * k))] = s.vadd(cc[@as(usize, @intCast(i + k))], ti2);
                        ch[@as(usize, @intCast(2 * (k + ido) - i))] = s.vsub(ti2, cc[@as(usize, @intCast(i + k))]);
                        ch[@as(usize, @intCast(i - 1 + 2 * k))] = s.vadd(cc[@as(usize, @intCast(i - 1 + k))], tr2);
                        ch[@as(usize, @intCast(2 * (k + ido) - i - 1))] = s.vsub(cc[@as(usize, @intCast(i - 1 + k))], tr2);
                    }
                }
                if (@rem(ido, 2) == 1) return;
            }
            k = 0;
            while (k < l1ido) : (k += ido) {
                ch[@as(usize, @intCast(2 * k + ido))] = -cc[@as(usize, @intCast(ido - 1 + k + l1ido))];
                ch[@as(usize, @intCast(2 * k + ido - 1))] = cc[@as(usize, @intCast(k + ido - 1))];
            }
        }

        pub fn radb2_ps(ido: i32, l1: i32, cc: s.CVP, ch: s.VP, wa1: [*]const T) void {
            const l1ido = l1 * ido;
            var k: i32 = 0;
            while (k < l1ido) : (k += ido) {
                const a = cc[@as(usize, @intCast(2 * k))];
                const b = cc[@as(usize, @intCast(2 * (k + ido) - 1))];
                ch[@as(usize, @intCast(k))] = s.vadd(a, b);
                ch[@as(usize, @intCast(k + l1ido))] = s.vsub(a, b);
            }
            if (ido < 2) return;
            if (ido != 2) {
                k = 0;
                while (k < l1ido) : (k += ido) {
                    var i: i32 = 2;
                    while (i < ido) : (i += 2) {
                        const a = cc[@as(usize, @intCast(i - 1 + 2 * k))];
                        const b = cc[@as(usize, @intCast(2 * (k + ido) - i - 1))];
                        const c = cc[@as(usize, @intCast(i + 2 * k))];
                        const d = cc[@as(usize, @intCast(2 * (k + ido) - i))];
                        ch[@as(usize, @intCast(i - 1 + k))] = s.vadd(a, b);
                        var tr2 = s.vsub(a, b);
                        ch[@as(usize, @intCast(i + k))] = s.vsub(c, d);
                        var ti2 = s.vadd(c, d);
                        const vwa_r = s.vld1(wa1[@as(usize, @intCast(i - 2))]);
                        const vwa_i = s.vld1(wa1[@as(usize, @intCast(i - 1))]);
                        s.vcplxmul(&tr2, &ti2, vwa_r, vwa_i);
                        ch[@as(usize, @intCast(i - 1 + k + l1ido))] = tr2;
                        ch[@as(usize, @intCast(i + k + l1ido))] = ti2;
                    }
                }
                if (@rem(ido, 2) == 1) return;
            }
            k = 0;
            while (k < l1ido) : (k += ido) {
                ch[@as(usize, @intCast(ido - 1 + k))] = s.vadd(cc[@as(usize, @intCast(2 * k + ido - 1))], cc[@as(usize, @intCast(2 * k + ido - 1))]);
                ch[@as(usize, @intCast(ido - 1 + k + l1ido))] = s.vmul(s.vld1(@as(T, -2.0)), cc[@as(usize, @intCast(2 * k + ido))]);
            }
        }

        pub fn radf3_ps(ido: i32, l1: i32, cc: s.CVP, ch: s.VP, wa1: [*]const T, wa2: [*]const T) void {
            const taur: T = -0.5;
            const taui: T = 0.866025403784439;
            const l1ido = l1 * ido;
            {
                var cc_ptr = cc;
                var ch_ptr = ch;
                const cc_end = cc + @as(usize, @intCast(l1ido));
                while (@intFromPtr(cc_ptr) < @intFromPtr(cc_end)) {
                    const cr2 = s.vadd(cc_ptr[@as(usize, @intCast(l1ido))], cc_ptr[@as(usize, @intCast(2 * l1ido))]);
                    ch_ptr[0] = s.vadd(cc_ptr[0], cr2);
                    ch_ptr[@as(usize, @intCast(2 * ido))] = s.vmul(s.vld1(taui), s.vsub(cc_ptr[@as(usize, @intCast(2 * l1ido))], cc_ptr[@as(usize, @intCast(l1ido))]));
                    ch_ptr[@as(usize, @intCast(ido - 1 + ido))] = s.vmadd(s.vld1(taur), cr2, cc_ptr[0]);
                    cc_ptr += @as(usize, @intCast(ido));
                    ch_ptr += @as(usize, @intCast(3 * ido));
                }
            }
            if (ido == 1) return;
            var k: i32 = 0;
            while (k < l1ido) : (k += ido) {
                const pc = cc + @as(usize, @intCast(k));
                var i: i32 = 2;
                while (i < ido) : (i += 2) {
                    const ic = ido - i;
                    var dr2 = pc[@as(usize, @intCast(i - 1 + l1ido))];
                    var di2 = pc[@as(usize, @intCast(i + l1ido))];
                    s.vcplxmulconj(&dr2, &di2, s.vld1(wa1[@as(usize, @intCast(i - 2))]), s.vld1(wa1[@as(usize, @intCast(i - 1))]));
                    var dr3 = pc[@as(usize, @intCast(i - 1 + 2 * l1ido))];
                    var di3 = pc[@as(usize, @intCast(i + 2 * l1ido))];
                    s.vcplxmulconj(&dr3, &di3, s.vld1(wa2[@as(usize, @intCast(i - 2))]), s.vld1(wa2[@as(usize, @intCast(i - 1))]));
                    const cr2 = s.vadd(dr2, dr3);
                    const ci2 = s.vadd(di2, di3);
                    ch[@as(usize, @intCast(i - 1 + 3 * k))] = s.vadd(pc[@as(usize, @intCast(i - 1))], cr2);
                    ch[@as(usize, @intCast(i + 3 * k))] = s.vadd(pc[@as(usize, @intCast(i))], ci2);
                    const tr2 = s.vmadd(s.vld1(taur), cr2, pc[@as(usize, @intCast(i - 1))]);
                    const ti2 = s.vmadd(s.vld1(taur), ci2, pc[@as(usize, @intCast(i))]);
                    // tr3/ti3 = taui * d{i,r}, folded into the adds below.
                    const di = s.vsub(di2, di3);
                    const dr = s.vsub(dr3, dr2);
                    ch[@as(usize, @intCast(i - 1 + 3 * k + 2 * ido))] = s.vmadd(s.vld1(taui), di, tr2);
                    ch[@as(usize, @intCast(ic - 1 + 3 * k + 1 * ido))] = s.vnmsub(s.vld1(taui), di, tr2);
                    ch[@as(usize, @intCast(i + 3 * k + 2 * ido))] = s.vmadd(s.vld1(taui), dr, ti2);
                    ch[@as(usize, @intCast(ic + 3 * k + 1 * ido))] = s.vmsub(s.vld1(taui), dr, ti2);
                }
            }
        }

        pub fn radb3_ps(ido: i32, l1: i32, cc: s.CVP, ch: s.VP, wa1: [*]const T, wa2: [*]const T) void {
            const taur: T = -0.5;
            const taui: T = 0.866025403784439;
            const taui_2: T = taui * 2.0;
            const l1ido = l1 * ido;
            {
                var cc_ptr = cc;
                var ch_ptr = ch;
                const ch_end = ch + @as(usize, @intCast(l1ido));
                while (@intFromPtr(ch_ptr) < @intFromPtr(ch_end)) {
                    var tr2 = cc_ptr[@as(usize, @intCast(ido - 1 + ido))];
                    tr2 = s.vadd(tr2, tr2);
                    const cr2 = s.vmadd(s.vld1(taur), tr2, cc_ptr[0]);
                    ch_ptr[0] = s.vadd(cc_ptr[0], tr2);
                    const d = cc_ptr[@as(usize, @intCast(2 * ido))];
                    ch_ptr[@as(usize, @intCast(l1ido))] = s.vnmsub(s.vld1(taui_2), d, cr2);
                    ch_ptr[@as(usize, @intCast(2 * l1ido))] = s.vmadd(s.vld1(taui_2), d, cr2);
                    cc_ptr += @as(usize, @intCast(3 * ido));
                    ch_ptr += @as(usize, @intCast(ido));
                }
            }
            if (ido == 1) return;
            var k: i32 = 0;
            while (k < l1ido) : (k += ido) {
                const pc = cc + @as(usize, @intCast(3 * k));
                var ph = ch + @as(usize, @intCast(k));
                var i: i32 = 2;
                while (i < ido) : (i += 2) {
                    const ic = ido - i;
                    const tr2 = s.vadd(pc[@as(usize, @intCast(i - 1 + 2 * ido))], pc[@as(usize, @intCast(ic - 1 + ido))]);
                    const cr2 = s.vmadd(s.vld1(taur), tr2, pc[@as(usize, @intCast(i - 1))]);
                    ph[@as(usize, @intCast(i - 1))] = s.vadd(pc[@as(usize, @intCast(i - 1))], tr2);
                    const ti2 = s.vsub(pc[@as(usize, @intCast(i + 2 * ido))], pc[@as(usize, @intCast(ic + ido))]);
                    const ci2 = s.vmadd(s.vld1(taur), ti2, pc[@as(usize, @intCast(i))]);
                    ph[@as(usize, @intCast(i))] = s.vadd(pc[@as(usize, @intCast(i))], ti2);
                    // cr3/ci3 = taui * d{r,i}, folded into the adds below.
                    const dr = s.vsub(pc[@as(usize, @intCast(i - 1 + 2 * ido))], pc[@as(usize, @intCast(ic - 1 + ido))]);
                    const di = s.vadd(pc[@as(usize, @intCast(i + 2 * ido))], pc[@as(usize, @intCast(ic + ido))]);
                    var dr2 = s.vnmsub(s.vld1(taui), di, cr2);
                    var dr3 = s.vmadd(s.vld1(taui), di, cr2);
                    var di2 = s.vmadd(s.vld1(taui), dr, ci2);
                    var di3 = s.vnmsub(s.vld1(taui), dr, ci2);
                    const vwa1_r = s.vld1(wa1[@as(usize, @intCast(i - 2))]);
                    const vwa1_i = s.vld1(wa1[@as(usize, @intCast(i - 1))]);
                    s.vcplxmul(&dr2, &di2, vwa1_r, vwa1_i);
                    ph[@as(usize, @intCast(i - 1 + l1ido))] = dr2;
                    ph[@as(usize, @intCast(i + l1ido))] = di2;
                    const vwa2_r = s.vld1(wa2[@as(usize, @intCast(i - 2))]);
                    const vwa2_i = s.vld1(wa2[@as(usize, @intCast(i - 1))]);
                    s.vcplxmul(&dr3, &di3, vwa2_r, vwa2_i);
                    ph[@as(usize, @intCast(i - 1 + 2 * l1ido))] = dr3;
                    ph[@as(usize, @intCast(i + 2 * l1ido))] = di3;
                }
            }
        }

        pub fn radf4_ps(ido: i32, l1: i32, cc: s.CVP, ch: s.VP, wa1: [*]const T, wa2: [*]const T, wa3: [*]const T) void {
            const minus_hsqt2: T = -0.7071067811865475;
            const l1ido = l1 * ido;
            {
                var cc_ptr = cc;
                var ch_ptr = ch;
                const cc_end = cc + @as(usize, @intCast(l1ido));
                while (@intFromPtr(cc_ptr) < @intFromPtr(cc_end)) {
                    // this loop represents between 25% and 40% of total radf4_ps cost !
                    const a0 = cc_ptr[0];
                    const a1 = cc_ptr[@as(usize, @intCast(l1ido))];
                    const a2 = cc_ptr[@as(usize, @intCast(2 * l1ido))];
                    const a3 = cc_ptr[@as(usize, @intCast(3 * l1ido))];
                    const tr1 = s.vadd(a1, a3);
                    const tr2 = s.vadd(a0, a2);
                    ch_ptr[@as(usize, @intCast(2 * ido - 1))] = s.vsub(a0, a2);
                    ch_ptr[@as(usize, @intCast(2 * ido))] = s.vsub(a3, a1);
                    ch_ptr[0] = s.vadd(tr1, tr2);
                    ch_ptr[@as(usize, @intCast(4 * ido - 1))] = s.vsub(tr2, tr1);
                    cc_ptr += @as(usize, @intCast(ido));
                    ch_ptr += @as(usize, @intCast(4 * ido));
                }
            }
            if (ido < 2) return;
            if (ido != 2) {
                var k: i32 = 0;
                while (k < l1ido) : (k += ido) {
                    var pc = cc + @as(usize, @intCast(1 + k));
                    var i: i32 = 2;
                    while (i < ido) : (i += 2) {
                        const ic = ido - i;
                        var cr2 = pc[@as(usize, @intCast(1 * l1ido))];
                        var ci2 = pc[@as(usize, @intCast(1 * l1ido + 1))];
                        const vwa1_r = s.vld1(wa1[@as(usize, @intCast(i - 2))]);
                        const vwa1_i = s.vld1(wa1[@as(usize, @intCast(i - 1))]);
                        s.vcplxmulconj(&cr2, &ci2, vwa1_r, vwa1_i);

                        var cr3 = pc[@as(usize, @intCast(2 * l1ido))];
                        var ci3 = pc[@as(usize, @intCast(2 * l1ido + 1))];
                        const vwa2_r = s.vld1(wa2[@as(usize, @intCast(i - 2))]);
                        const vwa2_i = s.vld1(wa2[@as(usize, @intCast(i - 1))]);
                        s.vcplxmulconj(&cr3, &ci3, vwa2_r, vwa2_i);

                        var cr4 = pc[@as(usize, @intCast(3 * l1ido))];
                        var ci4 = pc[@as(usize, @intCast(3 * l1ido + 1))];
                        const vwa3_r = s.vld1(wa3[@as(usize, @intCast(i - 2))]);
                        const vwa3_i = s.vld1(wa3[@as(usize, @intCast(i - 1))]);
                        s.vcplxmulconj(&cr4, &ci4, vwa3_r, vwa3_i);

                        const tr1 = s.vadd(cr2, cr4);
                        const tr4 = s.vsub(cr4, cr2);
                        const tr2 = s.vadd(pc[0], cr3);
                        const tr3 = s.vsub(pc[0], cr3);
                        ch[@as(usize, @intCast(i - 1 + 4 * k))] = s.vadd(tr1, tr2);
                        ch[@as(usize, @intCast(ic - 1 + 4 * k + 3 * ido))] = s.vsub(tr2, tr1);
                        const ti1 = s.vadd(ci2, ci4);
                        const ti4 = s.vsub(ci2, ci4);
                        ch[@as(usize, @intCast(i - 1 + 4 * k + 2 * ido))] = s.vadd(ti4, tr3);
                        ch[@as(usize, @intCast(ic - 1 + 4 * k + 1 * ido))] = s.vsub(tr3, ti4);
                        const ti2 = s.vadd(pc[1], ci3);
                        const ti3 = s.vsub(pc[1], ci3);
                        ch[@as(usize, @intCast(i + 4 * k))] = s.vadd(ti1, ti2);
                        ch[@as(usize, @intCast(ic + 4 * k + 3 * ido))] = s.vsub(ti1, ti2);
                        ch[@as(usize, @intCast(i + 4 * k + 2 * ido))] = s.vadd(tr4, ti3);
                        ch[@as(usize, @intCast(ic + 4 * k + 1 * ido))] = s.vsub(tr4, ti3);
                        pc += 2;
                    }
                }
                if (@rem(ido, 2) == 1) return;
            }
            var k: i32 = 0;
            while (k < l1ido) : (k += ido) {
                const a = cc[@as(usize, @intCast(ido - 1 + k + l1ido))];
                const b = cc[@as(usize, @intCast(ido - 1 + k + 3 * l1ido))];
                const c = cc[@as(usize, @intCast(ido - 1 + k))];
                const d = cc[@as(usize, @intCast(ido - 1 + k + 2 * l1ido))];
                const ti1 = s.vmul(s.vld1(minus_hsqt2), s.vadd(a, b));
                const tr1 = s.vmul(s.vld1(minus_hsqt2), s.vsub(b, a));
                ch[@as(usize, @intCast(ido - 1 + 4 * k))] = s.vadd(tr1, c);
                ch[@as(usize, @intCast(ido - 1 + 4 * k + 2 * ido))] = s.vsub(c, tr1);
                ch[@as(usize, @intCast(4 * k + 1 * ido))] = s.vsub(ti1, d);
                ch[@as(usize, @intCast(4 * k + 3 * ido))] = s.vadd(ti1, d);
            }
        }

        pub fn radb4_ps(ido: i32, l1: i32, cc: s.CVP, ch: s.VP, wa1: [*]const T, wa2: [*]const T, wa3: [*]const T) void {
            const minus_sqrt2: T = -1.414213562373095;
            const two: T = 2.0;
            const l1ido = l1 * ido;
            {
                var cc_ptr = cc;
                var ch_ptr = ch;
                const ch_end = ch + @as(usize, @intCast(l1ido));
                while (@intFromPtr(ch_ptr) < @intFromPtr(ch_end)) {
                    const a = cc_ptr[0];
                    const b = cc_ptr[@as(usize, @intCast(4 * ido - 1))];
                    const c = cc_ptr[@as(usize, @intCast(2 * ido))];
                    const d = cc_ptr[@as(usize, @intCast(2 * ido - 1))];
                    // tr3/tr4 = 2*d, 2*c, folded into the adds below.
                    const tr2 = s.vadd(a, b);
                    const tr1 = s.vsub(a, b);
                    ch_ptr[0] = s.vmadd(s.vld1(two), d, tr2);
                    ch_ptr[@as(usize, @intCast(2 * l1ido))] = s.vnmsub(s.vld1(two), d, tr2);
                    ch_ptr[@as(usize, @intCast(1 * l1ido))] = s.vnmsub(s.vld1(two), c, tr1);
                    ch_ptr[@as(usize, @intCast(3 * l1ido))] = s.vmadd(s.vld1(two), c, tr1);
                    cc_ptr += @as(usize, @intCast(4 * ido));
                    ch_ptr += @as(usize, @intCast(ido));
                }
            }
            if (ido < 2) return;
            if (ido != 2) {
                var k: i32 = 0;
                while (k < l1ido) : (k += ido) {
                    const pc = cc + @as(usize, @intCast(4 * k));
                    var ph = ch + @as(usize, @intCast(k + 1));
                    var i: i32 = 2;
                    while (i < ido) : (i += 2) {
                        const tr1 = s.vsub(pc[@as(usize, @intCast(i - 1))], pc[@as(usize, @intCast(4 * ido - i - 1))]);
                        const tr2 = s.vadd(pc[@as(usize, @intCast(i - 1))], pc[@as(usize, @intCast(4 * ido - i - 1))]);
                        const ti4 = s.vsub(pc[@as(usize, @intCast(2 * ido + i - 1))], pc[@as(usize, @intCast(2 * ido - i - 1))]);
                        const tr3 = s.vadd(pc[@as(usize, @intCast(2 * ido + i - 1))], pc[@as(usize, @intCast(2 * ido - i - 1))]);
                        ph[0] = s.vadd(tr2, tr3);
                        var cr3 = s.vsub(tr2, tr3);

                        const ti3 = s.vsub(pc[@as(usize, @intCast(2 * ido + i))], pc[@as(usize, @intCast(2 * ido - i))]);
                        const tr4 = s.vadd(pc[@as(usize, @intCast(2 * ido + i))], pc[@as(usize, @intCast(2 * ido - i))]);
                        var cr2 = s.vsub(tr1, tr4);
                        var cr4 = s.vadd(tr1, tr4);

                        const ti1 = s.vadd(pc[@as(usize, @intCast(i))], pc[@as(usize, @intCast(4 * ido - i))]);
                        const ti2 = s.vsub(pc[@as(usize, @intCast(i))], pc[@as(usize, @intCast(4 * ido - i))]);

                        ph[1] = s.vadd(ti2, ti3);
                        ph += @as(usize, @intCast(l1ido));
                        var ci3 = s.vsub(ti2, ti3);
                        var ci2 = s.vadd(ti1, ti4);
                        var ci4 = s.vsub(ti1, ti4);

                        s.vcplxmul(&cr2, &ci2, s.vld1(wa1[@as(usize, @intCast(i - 2))]), s.vld1(wa1[@as(usize, @intCast(i - 1))]));
                        ph[0] = cr2;
                        ph[1] = ci2;
                        ph += @as(usize, @intCast(l1ido));

                        s.vcplxmul(&cr3, &ci3, s.vld1(wa2[@as(usize, @intCast(i - 2))]), s.vld1(wa2[@as(usize, @intCast(i - 1))]));
                        ph[0] = cr3;
                        ph[1] = ci3;
                        ph += @as(usize, @intCast(l1ido));

                        s.vcplxmul(&cr4, &ci4, s.vld1(wa3[@as(usize, @intCast(i - 2))]), s.vld1(wa3[@as(usize, @intCast(i - 1))]));
                        ph[0] = cr4;
                        ph[1] = ci4;
                        ph -= @as(usize, @intCast(3 * l1ido));
                        ph += 2;
                    }
                }
                if (@rem(ido, 2) == 1) return;
            }
            var k: i32 = 0;
            while (k < l1ido) : (k += ido) {
                const idx0 = @as(usize, @intCast(4 * k + ido));
                const c = cc[idx0 - 1];
                const d = cc[idx0 + @as(usize, @intCast(2 * ido - 1))];
                const a = cc[idx0 + 0];
                const b = cc[idx0 + @as(usize, @intCast(2 * ido + 0))];
                const tr1 = s.vsub(c, d);
                const tr2 = s.vadd(c, d);
                const ti1 = s.vadd(b, a);
                const ti2 = s.vsub(b, a);
                const ph = ch + @as(usize, @intCast(k + ido - 1));
                ph[0] = s.vadd(tr2, tr2);
                ph[@as(usize, @intCast(2 * l1ido))] = s.vadd(ti2, ti2);
                ph[@as(usize, @intCast(1 * l1ido))] = s.vmul(s.vld1(minus_sqrt2), s.vsub(ti1, tr1));
                ph[@as(usize, @intCast(3 * l1ido))] = s.vmul(s.vld1(minus_sqrt2), s.vadd(ti1, tr1));
            }
        }

        pub fn radf5_ps(ido: i32, l1: i32, cc: s.CVP, ch: s.VP, wa1: [*]const T, wa2: [*]const T, wa3: [*]const T, wa4: [*]const T) void {
            const tr11: T = 0.309016994374947;
            const ti11: T = 0.951056516295154;
            const tr12: T = -0.809016994374947;
            const ti12: T = 0.587785252292473;


            var k: i32 = 0;
            while (k < l1) : (k += 1) {
                const cr2 = s.vadd(cc[@as(usize, @intCast((k + 4 * l1) * ido))], cc[@as(usize, @intCast((k + l1) * ido))]);
                const ci5 = s.vsub(cc[@as(usize, @intCast((k + 4 * l1) * ido))], cc[@as(usize, @intCast((k + l1) * ido))]);
                const cr3 = s.vadd(cc[@as(usize, @intCast((k + 3 * l1) * ido))], cc[@as(usize, @intCast((k + 2 * l1) * ido))]);
                const ci4 = s.vsub(cc[@as(usize, @intCast((k + 3 * l1) * ido))], cc[@as(usize, @intCast((k + 2 * l1) * ido))]);
                ch[@as(usize, @intCast(5 * k * ido))] = s.vadd(cc[@as(usize, @intCast(k * ido))], s.vadd(cr2, cr3));
                ch[@as(usize, @intCast(ido - 1 + (5 * k + 1) * ido))] = s.vmadd(s.vld1(tr12), cr3, s.vmadd(s.vld1(tr11), cr2, cc[@as(usize, @intCast(k * ido))]));
                ch[@as(usize, @intCast((5 * k + 2) * ido))] = s.vmadd(s.vld1(ti11), ci5, s.vmul(s.vld1(ti12), ci4));
                ch[@as(usize, @intCast(ido - 1 + (5 * k + 3) * ido))] = s.vmadd(s.vld1(tr11), cr3, s.vmadd(s.vld1(tr12), cr2, cc[@as(usize, @intCast(k * ido))]));
                ch[@as(usize, @intCast((5 * k + 4) * ido))] = s.vnmsub(s.vld1(ti11), ci4, s.vmul(s.vld1(ti12), ci5));
            }
            if (ido == 1) return;
            k = 0;
            while (k < l1) : (k += 1) {
                var i: i32 = 2;
                while (i < ido) : (i += 2) {
                    const ic = ido - i;
                    var dr2 = s.vld1(wa1[@as(usize, @intCast(i - 2))]);
                    var di2 = s.vld1(wa1[@as(usize, @intCast(i - 1))]);
                    var dr3 = s.vld1(wa2[@as(usize, @intCast(i - 2))]);
                    var di3 = s.vld1(wa2[@as(usize, @intCast(i - 1))]);
                    var dr4 = s.vld1(wa3[@as(usize, @intCast(i - 2))]);
                    var di4 = s.vld1(wa3[@as(usize, @intCast(i - 1))]);
                    var dr5 = s.vld1(wa4[@as(usize, @intCast(i - 2))]);
                    var di5 = s.vld1(wa4[@as(usize, @intCast(i - 1))]);

                    s.vcplxmulconj(&dr2, &di2, cc[@as(usize, @intCast(i - 1 + (k + l1) * ido))], cc[@as(usize, @intCast(i + (k + l1) * ido))]);
                    s.vcplxmulconj(&dr3, &di3, cc[@as(usize, @intCast(i - 1 + (k + 2 * l1) * ido))], cc[@as(usize, @intCast(i + (k + 2 * l1) * ido))]);
                    s.vcplxmulconj(&dr4, &di4, cc[@as(usize, @intCast(i - 1 + (k + 3 * l1) * ido))], cc[@as(usize, @intCast(i + (k + 3 * l1) * ido))]);
                    s.vcplxmulconj(&dr5, &di5, cc[@as(usize, @intCast(i - 1 + (k + 4 * l1) * ido))], cc[@as(usize, @intCast(i + (k + 4 * l1) * ido))]);

                    const cr2 = s.vadd(dr2, dr5); const ci5 = s.vsub(dr5, dr2);
                    const cr5 = s.vsub(di2, di5); const ci2 = s.vadd(di2, di5);
                    const cr3 = s.vadd(dr3, dr4); const ci4 = s.vsub(dr4, dr3);
                    const cr4 = s.vsub(di3, di4); const ci3 = s.vadd(di3, di4);

                    ch[@as(usize, @intCast(i - 1 + 5 * k * ido))] = s.vadd(cc[@as(usize, @intCast(i - 1 + k * ido))], s.vadd(cr2, cr3));
                    ch[@as(usize, @intCast(i + 5 * k * ido))] = s.vsub(cc[@as(usize, @intCast(i + k * ido))], s.vadd(ci2, ci3));
                    const tr2 = s.vmadd(s.vld1(tr12), cr3, s.vmadd(s.vld1(tr11), cr2, cc[@as(usize, @intCast(i - 1 + k * ido))]));
                    const ti2 = s.vnmsub(s.vld1(tr12), ci3, s.vnmsub(s.vld1(tr11), ci2, cc[@as(usize, @intCast(i + k * ido))]));
                    const tr3 = s.vmadd(s.vld1(tr11), cr3, s.vmadd(s.vld1(tr12), cr2, cc[@as(usize, @intCast(i - 1 + k * ido))]));
                    const ti3 = s.vnmsub(s.vld1(tr11), ci3, s.vnmsub(s.vld1(tr12), ci2, cc[@as(usize, @intCast(i + k * ido))]));
                    const tr5 = s.vmadd(s.vld1(ti11), cr5, s.vmul(s.vld1(ti12), cr4));
                    const ti5 = s.vmadd(s.vld1(ti11), ci5, s.vmul(s.vld1(ti12), ci4));
                    const tr4 = s.vnmsub(s.vld1(ti11), cr4, s.vmul(s.vld1(ti12), cr5));
                    const ti4 = s.vnmsub(s.vld1(ti11), ci4, s.vmul(s.vld1(ti12), ci5));

                    ch[@as(usize, @intCast(i - 1 + (5 * k + 2) * ido))] = s.vsub(tr2, tr5);
                    ch[@as(usize, @intCast(ic - 1 + (5 * k + 1) * ido))] = s.vadd(tr2, tr5);
                    ch[@as(usize, @intCast(i + (5 * k + 2) * ido))] = s.vadd(ti2, ti5);
                    ch[@as(usize, @intCast(ic + (5 * k + 1) * ido))] = s.vsub(ti5, ti2);
                    ch[@as(usize, @intCast(i - 1 + (5 * k + 4) * ido))] = s.vsub(tr3, tr4);
                    ch[@as(usize, @intCast(ic - 1 + (5 * k + 3) * ido))] = s.vadd(tr3, tr4);
                    ch[@as(usize, @intCast(i + (5 * k + 4) * ido))] = s.vadd(ti3, ti4);
                    ch[@as(usize, @intCast(ic + (5 * k + 3) * ido))] = s.vsub(ti4, ti3);
                }
            }
        }

        pub fn radb5_ps(ido: i32, l1: i32, cc: s.CVP, ch: s.VP, wa1: [*]const T, wa2: [*]const T, wa3: [*]const T, wa4: [*]const T) void {
            const tr11: T = 0.309016994374947;
            const ti11: T = 0.951056516295154;
            const tr12: T = -0.809016994374947;
            const ti12: T = 0.587785252292473;


            var k: i32 = 0;
            while (k < l1) : (k += 1) {
                const ti5 = s.vadd(cc[@as(usize, @intCast((5 * k + 2) * ido))], cc[@as(usize, @intCast((5 * k + 2) * ido))]);
                const ti4 = s.vadd(cc[@as(usize, @intCast((5 * k + 4) * ido))], cc[@as(usize, @intCast((5 * k + 4) * ido))]);
                const tr2 = s.vadd(cc[@as(usize, @intCast(ido - 1 + (5 * k + 1) * ido))], cc[@as(usize, @intCast(ido - 1 + (5 * k + 1) * ido))]);
                const tr3 = s.vadd(cc[@as(usize, @intCast(ido - 1 + (5 * k + 3) * ido))], cc[@as(usize, @intCast(ido - 1 + (5 * k + 3) * ido))]);
                ch[@as(usize, @intCast(k * ido))] = s.vadd(cc[@as(usize, @intCast(5 * k * ido))], s.vadd(tr2, tr3));
                const cr2 = s.vmadd(s.vld1(tr12), tr3, s.vmadd(s.vld1(tr11), tr2, cc[@as(usize, @intCast(5 * k * ido))]));
                const cr3 = s.vmadd(s.vld1(tr11), tr3, s.vmadd(s.vld1(tr12), tr2, cc[@as(usize, @intCast(5 * k * ido))]));
                const ci5 = s.vmadd(s.vld1(ti11), ti5, s.vmul(s.vld1(ti12), ti4));
                const ci4 = s.vnmsub(s.vld1(ti11), ti4, s.vmul(s.vld1(ti12), ti5));
                ch[@as(usize, @intCast((k + l1) * ido))] = s.vsub(cr2, ci5);
                ch[@as(usize, @intCast((k + 2 * l1) * ido))] = s.vsub(cr3, ci4);
                ch[@as(usize, @intCast((k + 3 * l1) * ido))] = s.vadd(cr3, ci4);
                ch[@as(usize, @intCast((k + 4 * l1) * ido))] = s.vadd(cr2, ci5);
            }
            if (ido == 1) return;
            k = 0;
            while (k < l1) : (k += 1) {
                var i: i32 = 2;
                while (i < ido) : (i += 2) {
                    const ic = ido - i;
                    const ti5 = s.vadd(cc[@as(usize, @intCast(i + (5 * k + 2) * ido))], cc[@as(usize, @intCast(ic + (5 * k + 1) * ido))]);
                    const ti2 = s.vsub(cc[@as(usize, @intCast(i + (5 * k + 2) * ido))], cc[@as(usize, @intCast(ic + (5 * k + 1) * ido))]);
                    const ti4 = s.vadd(cc[@as(usize, @intCast(i + (5 * k + 4) * ido))], cc[@as(usize, @intCast(ic + (5 * k + 3) * ido))]);
                    const ti3 = s.vsub(cc[@as(usize, @intCast(i + (5 * k + 4) * ido))], cc[@as(usize, @intCast(ic + (5 * k + 3) * ido))]);
                    const tr5 = s.vsub(cc[@as(usize, @intCast(i - 1 + (5 * k + 2) * ido))], cc[@as(usize, @intCast(ic - 1 + (5 * k + 1) * ido))]);
                    const tr2 = s.vadd(cc[@as(usize, @intCast(i - 1 + (5 * k + 2) * ido))], cc[@as(usize, @intCast(ic - 1 + (5 * k + 1) * ido))]);
                    const tr4 = s.vsub(cc[@as(usize, @intCast(i - 1 + (5 * k + 4) * ido))], cc[@as(usize, @intCast(ic - 1 + (5 * k + 3) * ido))]);
                    const tr3 = s.vadd(cc[@as(usize, @intCast(i - 1 + (5 * k + 4) * ido))], cc[@as(usize, @intCast(ic - 1 + (5 * k + 3) * ido))]);
                    ch[@as(usize, @intCast(i - 1 + k * ido))] = s.vadd(cc[@as(usize, @intCast(i - 1 + 5 * k * ido))], s.vadd(tr2, tr3));
                    ch[@as(usize, @intCast(i + k * ido))] = s.vadd(cc[@as(usize, @intCast(i + 5 * k * ido))], s.vadd(ti2, ti3));
                    const cr2 = s.vmadd(s.vld1(tr12), tr3, s.vmadd(s.vld1(tr11), tr2, cc[@as(usize, @intCast(i - 1 + 5 * k * ido))]));
                    const ci2 = s.vmadd(s.vld1(tr12), ti3, s.vmadd(s.vld1(tr11), ti2, cc[@as(usize, @intCast(i + 5 * k * ido))]));
                    const cr3 = s.vmadd(s.vld1(tr11), tr3, s.vmadd(s.vld1(tr12), tr2, cc[@as(usize, @intCast(i - 1 + 5 * k * ido))]));
                    const ci3 = s.vmadd(s.vld1(tr11), ti3, s.vmadd(s.vld1(tr12), ti2, cc[@as(usize, @intCast(i + 5 * k * ido))]));
                    const cr5 = s.vmadd(s.vld1(ti11), tr5, s.vmul(s.vld1(ti12), tr4));
                    const ci5 = s.vmadd(s.vld1(ti11), ti5, s.vmul(s.vld1(ti12), ti4));
                    const cr4 = s.vnmsub(s.vld1(ti11), tr4, s.vmul(s.vld1(ti12), tr5));
                    const ci4 = s.vnmsub(s.vld1(ti11), ti4, s.vmul(s.vld1(ti12), ti5));
                    var dr3 = s.vsub(cr3, ci4); var dr4 = s.vadd(cr3, ci4);
                    var di3 = s.vadd(ci3, cr4); var di4 = s.vsub(ci3, cr4);
                    var dr5 = s.vadd(cr2, ci5); var dr2 = s.vsub(cr2, ci5);
                    var di5 = s.vsub(ci2, cr5); var di2 = s.vadd(ci2, cr5);
                    s.vcplxmul(&dr2, &di2, s.vld1(wa1[@as(usize, @intCast(i - 2))]), s.vld1(wa1[@as(usize, @intCast(i - 1))]));
                    s.vcplxmul(&dr3, &di3, s.vld1(wa2[@as(usize, @intCast(i - 2))]), s.vld1(wa2[@as(usize, @intCast(i - 1))]));
                    s.vcplxmul(&dr4, &di4, s.vld1(wa3[@as(usize, @intCast(i - 2))]), s.vld1(wa3[@as(usize, @intCast(i - 1))]));
                    s.vcplxmul(&dr5, &di5, s.vld1(wa4[@as(usize, @intCast(i - 2))]), s.vld1(wa4[@as(usize, @intCast(i - 1))]));
                    ch[@as(usize, @intCast(i - 1 + (k + l1) * ido))] = dr2; ch[@as(usize, @intCast(i + (k + l1) * ido))] = di2;
                    ch[@as(usize, @intCast(i - 1 + (k + 2 * l1) * ido))] = dr3; ch[@as(usize, @intCast(i + (k + 2 * l1) * ido))] = di3;
                    ch[@as(usize, @intCast(i - 1 + (k + 3 * l1) * ido))] = dr4; ch[@as(usize, @intCast(i + (k + 3 * l1) * ido))] = di4;
                    ch[@as(usize, @intCast(i - 1 + (k + 4 * l1) * ido))] = dr5; ch[@as(usize, @intCast(i + (k + 4 * l1) * ido))] = di5;
                }
            }
        }
    };
}
