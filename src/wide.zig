//! Lane-generic glue around the radix passes (see docs/wide-simd-design.md).
//! Used for L = 8, 16; the 4-lane path keeps its hand-written code in zpfft.zig.

const std = @import("std");
const simd_math = @import("simd_math.zig");
const setup_mod = @import("setup.zig");

pub fn Wide(comptime T: type, comptime L: comptime_int) type {
    const s = simd_math.SIMDL(T, L);
    const Vec = s.Vec;
    const VP = s.VP;
    const CVP = s.CVP;
    const H = L / 2;

    return struct {
        inline fn splat(x: comptime_float) Vec {
            return @splat(@as(T, x));
        }

        /// (tr + i·ti) · e^{i·sgn·2πk/n}, constants folded at comptime.
        inline fn rotate(comptime k: comptime_int, comptime n: comptime_int, comptime sgn: comptime_float, tr: *Vec, ti: *Vec) void {
            const ang = sgn * 2.0 * std.math.pi * @as(comptime_float, k) / @as(comptime_float, n);
            const c = @cos(ang);
            const sn = @sin(ang);
            if (8 * k == n or 8 * k == 3 * n) {
                // |c| == |s|: 2 adds + 2 muls.
                const r = tr.*;
                const i = ti.*;
                if ((c > 0) == (sn > 0)) {
                    tr.* = (r - i) * splat(c);
                    ti.* = (r + i) * splat(c);
                } else {
                    tr.* = (r + i) * splat(c);
                    ti.* = (i - r) * splat(c);
                }
            } else {
                s.vcplxmul(tr, ti, splat(c), splat(sn));
            }
        }

        /// Forward DFT (e^{-2πi jq/n}), decimation in time, natural order.
        /// conj_hi: return conj of outputs n/2..n-1 (free: flips the last subs).
        inline fn dftFwd(comptime n: comptime_int, comptime conj_hi: bool, re: [n]Vec, im: [n]Vec) [2][n]Vec {
            if (n == 1) return .{ re, im };
            var er: [n / 2]Vec = undefined;
            var ei: [n / 2]Vec = undefined;
            var or_: [n / 2]Vec = undefined;
            var oi: [n / 2]Vec = undefined;
            inline for (0..n / 2) |i| {
                er[i] = re[2 * i];
                ei[i] = im[2 * i];
                or_[i] = re[2 * i + 1];
                oi[i] = im[2 * i + 1];
            }
            const E = dftFwd(n / 2, false, er, ei);
            const O = dftFwd(n / 2, false, or_, oi);
            var yr: [n]Vec = undefined;
            var yi: [n]Vec = undefined;
            inline for (0..n / 2) |k| {
                var tr = O[0][k];
                var ti = O[1][k];
                if (4 * k == n) {
                    // t · (-i) = (ti, -tr): fold the sign into the butterfly.
                    yr[k] = E[0][k] + ti;
                    yi[k] = E[1][k] - tr;
                    yr[k + n / 2] = E[0][k] - ti;
                    yi[k + n / 2] = if (conj_hi) -(E[1][k] + tr) else E[1][k] + tr;
                    continue;
                }
                if (k != 0) rotate(k, n, -1.0, &tr, &ti);
                yr[k] = E[0][k] + tr;
                yi[k] = E[1][k] + ti;
                yr[k + n / 2] = E[0][k] - tr;
                yi[k + n / 2] = if (conj_hi) ti - E[1][k] else E[1][k] - ti;
            }
            return .{ yr, yi };
        }

        /// Unnormalized inverse DFT (e^{+2πi jq/n}), decimation in frequency,
        /// natural order. conj_hi: conjugate inputs n/2..n-1 first (free).
        inline fn dftInv(comptime n: comptime_int, comptime conj_hi: bool, re: [n]Vec, im: [n]Vec) [2][n]Vec {
            if (n == 1) return .{ re, im };
            var ar: [n / 2]Vec = undefined;
            var ai: [n / 2]Vec = undefined;
            var br: [n / 2]Vec = undefined;
            var bi: [n / 2]Vec = undefined;
            inline for (0..n / 2) |k| {
                const xr = re[k];
                const xi = im[k];
                const hr = re[k + n / 2];
                const hi = im[k + n / 2];
                ar[k] = xr + hr;
                ai[k] = if (conj_hi) xi - hi else xi + hi;
                var tr = xr - hr;
                var ti = if (conj_hi) xi + hi else xi - hi;
                if (4 * k == n) {
                    // · (+i) = (-ti, tr): swap the sub operands instead of negating.
                    br[k] = if (conj_hi) -(xi + hi) else hi - xi;
                    bi[k] = tr;
                    continue;
                }
                if (k != 0) rotate(k, n, 1.0, &tr, &ti);
                br[k] = tr;
                bi[k] = ti;
            }
            const A = dftInv(n / 2, false, ar, ai);
            const B = dftInv(n / 2, false, br, bi);
            var yr: [n]Vec = undefined;
            var yi: [n]Vec = undefined;
            inline for (0..n / 2) |r| {
                yr[2 * r] = A[0][r];
                yi[2 * r] = A[1][r];
                yr[2 * r + 1] = B[0][r];
                yi[2 * r + 1] = B[1][r];
            }
            return .{ yr, yi };
        }

        // ── complex ─────────────────────────────────────────────────────────

        pub fn cplxFinalize(Ncvec: i32, in: CVP, out: VP, e: CVP) void {
            @setEvalBranchQuota(20_000);
            for (0..@as(usize, @intCast(@divExact(Ncvec, L)))) |blk| {
                const src = in + 2 * L * blk;
                const dst = out + 2 * L * blk;
                const eb = e + 2 * (L - 1) * blk;
                var re: [L]Vec = undefined;
                var im: [L]Vec = undefined;
                inline for (0..L) |r| {
                    re[r] = src[2 * r];
                    im[r] = src[2 * r + 1];
                }
                s.transposeL(&re);
                s.transposeL(&im);
                inline for (1..L) |j| s.vcplxmul(&re[j], &im[j], eb[2 * (j - 1)], eb[2 * (j - 1) + 1]);
                const y = dftFwd(L, false, re, im);
                inline for (0..L) |q| {
                    dst[2 * q] = y[0][q];
                    dst[2 * q + 1] = y[1][q];
                }
            }
        }

        pub fn cplxPreprocess(Ncvec: i32, in: CVP, out: VP, e: CVP) void {
            @setEvalBranchQuota(20_000);
            for (0..@as(usize, @intCast(@divExact(Ncvec, L)))) |blk| {
                const src = in + 2 * L * blk;
                const dst = out + 2 * L * blk;
                const eb = e + 2 * (L - 1) * blk;
                var re: [L]Vec = undefined;
                var im: [L]Vec = undefined;
                inline for (0..L) |q| {
                    re[q] = src[2 * q];
                    im[q] = src[2 * q + 1];
                }
                var y = dftInv(L, false, re, im);
                inline for (1..L) |j| s.vcplxmulconj(&y[0][j], &y[1][j], eb[2 * (j - 1)], eb[2 * (j - 1) + 1]);
                s.transposeL(&y[0]);
                s.transposeL(&y[1]);
                inline for (0..L) |r| {
                    dst[2 * r] = y[0][r];
                    dst[2 * r + 1] = y[1][r];
                }
            }
        }

        // ── real ────────────────────────────────────────────────────────────

        /// cos/sin(π t / L), t in 0..2L; exact at multiples of π/2.
        const roots = blk: {
            var c: [2 * L]T = undefined;
            var sn: [2 * L]T = undefined;
            for (0..2 * L) |t| {
                if ((2 * t) % L == 0) {
                    c[t] = .{ 1, 0, -1, 0 }[2 * t / L];
                    sn[t] = .{ 0, 1, 0, -1 }[2 * t / L];
                } else {
                    const a = std.math.pi * @as(comptime_float, t) / @as(comptime_float, L);
                    c[t] = @cos(a);
                    sn[t] = @sin(a);
                }
            }
            break :blk .{ c, sn };
        };

        /// acc + x·k with k comptime; 0 and ±1 cost nothing / one add.
        /// (Accumulators start at -0.0 so the first add folds away.)
        inline fn madd(acc: T, x: T, comptime k: T) T {
            return if (k == 0) acc else if (k == 1) acc + x else if (k == -1) acc - x else acc + x * k;
        }

        /// Rows (m0..m0+L-1) → twiddle → DFT_L → slots 2q = Y_q, 2q'-1 = conj Y_{L-q'}.
        inline fn realFinalizeBlock(re_in: [L]Vec, im_in: [L]Vec, eb: CVP, dst: VP) void {
            var re = re_in;
            var im = im_in;
            s.transposeL(&re);
            s.transposeL(&im);
            inline for (1..L) |j| s.vcplxmul(&re[j], &im[j], eb[2 * (j - 1)], eb[2 * (j - 1) + 1]);
            const y = dftFwd(L, true, re, im);
            inline for (0..H) |q| {
                dst[4 * q] = y[0][q];
                dst[4 * q + 1] = y[1][q];
            }
            inline for (1..H + 1) |qp| {
                dst[2 * (2 * qp - 1)] = y[0][L - qp];
                dst[2 * (2 * qp - 1) + 1] = y[1][L - qp];
            }
        }

        pub fn realFinalize(Ncvec: i32, in: CVP, out: VP, e: CVP) void {
            @setEvalBranchQuota(20_000);
            const nc: usize = @intCast(Ncvec);
            const cr: [L]T = in[0];
            const ci: [L]T = in[2 * nc - 1];

            // X_j[m] sits at in[2m-1], in[2m]; row m = 0 is zero in block 0.
            var re: [L]Vec = undefined;
            var im: [L]Vec = undefined;
            re[0] = @splat(0);
            im[0] = @splat(0);
            inline for (1..L) |r| {
                re[r] = in[2 * r - 1];
                im[r] = in[2 * r];
            }
            realFinalizeBlock(re, im, e, out);
            for (1..nc / L) |blk| {
                const src = in + 2 * L * blk - 1;
                inline for (0..L) |r| {
                    re[r] = src[2 * r];
                    im[r] = src[2 * r + 1];
                }
                realFinalizeBlock(re, im, e + 2 * (L - 1) * blk, out + 2 * L * blk);
            }

            // Lane 0 of block 0 (m = 0): A[q] = Σ cr_j W_L^{jq} at bins Mq,
            // B[p] = Σ ci_j e^{-iπj(2p+1)/L} at bins Mp + M/2.
            const fout: [*]T = @ptrCast(out);
            inline for (0..H + 1) |q| {
                var ar: T = -0.0;
                var ai: T = -0.0;
                inline for (0..L) |j| {
                    const t = (2 * j * q) % (2 * L);
                    ar = madd(ar, cr[j], roots[0][t]);
                    ai = madd(ai, cr[j], -roots[1][t]);
                }
                if (q == 0) {
                    fout[0] = ar;
                } else if (q == H) {
                    fout[L] = ar;
                } else {
                    fout[4 * q * L] = ar;
                    fout[(4 * q + 1) * L] = ai;
                }
            }
            inline for (0..H) |p| {
                var br: T = -0.0;
                var bi: T = -0.0;
                inline for (0..L) |j| {
                    const t = (j * (2 * p + 1)) % (2 * L);
                    br = madd(br, ci[j], roots[0][t]);
                    bi = madd(bi, ci[j], -roots[1][t]);
                }
                fout[(4 * p + 2) * L] = br;
                fout[(4 * p + 3) * L] = bi;
            }
        }

        /// Inverse of realFinalizeBlock: rows scaled by L.
        inline fn realPreprocessBlock(src: CVP, eb: CVP) [2][L]Vec {
            var re: [L]Vec = undefined;
            var im: [L]Vec = undefined;
            inline for (0..H) |q| {
                re[q] = src[4 * q];
                im[q] = src[4 * q + 1];
            }
            inline for (1..H + 1) |qp| {
                re[L - qp] = src[2 * (2 * qp - 1)];
                im[L - qp] = src[2 * (2 * qp - 1) + 1];
            }
            var y = dftInv(L, true, re, im);
            inline for (1..L) |j| s.vcplxmulconj(&y[0][j], &y[1][j], eb[2 * (j - 1)], eb[2 * (j - 1) + 1]);
            s.transposeL(&y[0]);
            s.transposeL(&y[1]);
            return y;
        }

        pub fn realPreprocess(Ncvec: i32, in: CVP, out: VP, e: CVP) void {
            @setEvalBranchQuota(20_000);
            const nc: usize = @intCast(Ncvec);
            const fin: [*]const T = @ptrCast(in);

            // Lane 0 of block 0 → per-lane DC (cr) and Nyquist (ci), times L.
            var cr: [L]T = undefined;
            var ci: [L]T = undefined;
            inline for (0..L) |j| {
                // A[0] + (-1)^j A[L/2] + 2 Σ_{q=1}^{L/2-1} Re(A[q] e^{2πijq/L})
                var a: T = -0.0;
                inline for (1..H) |q| {
                    const t = (2 * j * q) % (2 * L);
                    a = madd(a, fin[4 * q * L], roots[0][t]);
                    a = madd(a, fin[(4 * q + 1) * L], -roots[1][t]);
                }
                cr[j] = fin[0] + (if (j % 2 == 0) fin[L] else -fin[L]) + 2 * a;
                // 2 Σ_{p<L/2} Re(B[p] e^{iπj(2p+1)/L})
                var b: T = -0.0;
                inline for (0..H) |p| {
                    const t = (j * (2 * p + 1)) % (2 * L);
                    b = madd(b, fin[(4 * p + 2) * L], roots[0][t]);
                    b = madd(b, fin[(4 * p + 3) * L], -roots[1][t]);
                }
                ci[j] = 2 * b;
            }

            const y0 = realPreprocessBlock(in, e);
            inline for (1..L) |r| {
                out[2 * r - 1] = y0[0][r];
                out[2 * r] = y0[1][r];
            }
            for (1..nc / L) |blk| {
                const y = realPreprocessBlock(in + 2 * L * blk, e + 2 * (L - 1) * blk);
                const dst = out + 2 * L * blk - 1;
                inline for (0..L) |r| {
                    dst[2 * r] = y[0][r];
                    dst[2 * r + 1] = y[1][r];
                }
            }
            out[0] = cr;
            out[2 * nc - 1] = ci;
        }

        // ── reorder ─────────────────────────────────────────────────────────

        /// [~0, L-1, L-2, …, 1]: lane 0 of b, then lanes L-1..1 of a reversed.
        const rev_mask = blk: {
            var m: [L]i32 = undefined;
            m[0] = ~@as(i32, 0);
            for (1..L) |i| m[i] = L - i;
            break :blk m;
        };

        pub fn zreorder(setup: anytype, input: []const T, output: []T, direction: setup_mod.Direction) void {
            std.debug.assert(input.ptr != output.ptr);
            const vin: CVP = @ptrCast(@alignCast(input.ptr));
            const vout: VP = @ptrCast(@alignCast(output.ptr));
            const nc: usize = @intCast(setup.Ncvec);
            const nb = nc / L;
            const fwd = direction == .Forward;

            if (setup.transform == .Complex) {
                for (0..nc) |k| {
                    const kk = k / L + (k % L) * nb;
                    if (fwd) {
                        const r = s.interleave2(vin[2 * k], vin[2 * k + 1]);
                        vout[2 * kk] = r[0];
                        vout[2 * kk + 1] = r[1];
                    } else {
                        const r = s.uninterleave2(vin[2 * kk], vin[2 * kk + 1]);
                        vout[2 * k] = r[0];
                        vout[2 * k + 1] = r[1];
                    }
                }
                return;
            }

            // Real: ordered region s (nb chunks of L bins) holds slot s.
            // Even slots map straight through; odd slots are bin-reversed.
            inline for (0..L) |sl| {
                const region = sl * nb;
                if (sl % 2 == 0) {
                    for (0..nb) |b| {
                        const u = 2 * (L * b + sl); // unordered
                        const o = 2 * (region + b); // ordered
                        if (fwd) {
                            const r = s.interleave2(vin[u], vin[u + 1]);
                            vout[o] = r[0];
                            vout[o + 1] = r[1];
                        } else {
                            const r = s.uninterleave2(vin[o], vin[o + 1]);
                            vout[u] = r[0];
                            vout[u + 1] = r[1];
                        }
                    }
                } else if (fwd) {
                    // chunk c = [slot_j lane 0, slot_{j-1} lanes L-1..1], j = (nb - c) mod nb
                    for (0..nb) |c| {
                        const j = (nb - c) % nb;
                        const jm1 = (2 * nb - c - 1) % nb;
                        const a = 2 * (L * jm1 + sl);
                        const b = 2 * (L * j + sl);
                        const re = @shuffle(T, vin[a], vin[b], rev_mask);
                        const im = @shuffle(T, vin[a + 1], vin[b + 1], rev_mask);
                        const r = s.interleave2(re, im);
                        vout[2 * (region + c)] = r[0];
                        vout[2 * (region + c) + 1] = r[1];
                    }
                } else {
                    // block b = shuffle(chunk[nb-1-b], chunk[(nb-b) mod nb]); carry the latter.
                    var cur = s.uninterleave2(vin[2 * region], vin[2 * region + 1]);
                    for (0..nb) |b| {
                        const c1 = region + nb - 1 - b;
                        const nxt = s.uninterleave2(vin[2 * c1], vin[2 * c1 + 1]);
                        const u = 2 * (L * b + sl);
                        vout[u] = @shuffle(T, nxt[0], cur[0], rev_mask);
                        vout[u + 1] = @shuffle(T, nxt[1], cur[1], rev_mask);
                        cur = nxt;
                    }
                }
            }
        }
    };
}
