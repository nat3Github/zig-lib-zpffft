const std = @import("std");
const simd_math = @import("simd_math.zig");
const setup_mod = @import("setup.zig");
const passes_mod = @import("radix_passes.zig");
const wide_mod = @import("wide.zig");

pub const TransformType = setup_mod.TransformType;
pub const Direction = setup_mod.Direction;
pub const Error = setup_mod.Error;

pub fn ZPFFT(comptime T: type) type {
    const s = simd_math.SIMD(T);
    const Vec = s.Vec;
    const Setup = setup_mod.Setup(T);

    return struct {
        const wide = simd_math.wideLanes(T);
        const W = wide_mod.Wide(T, wide);

        /// 4-lane or wide, per setup.lanes. Returns the buffer holding the result.
        /// Buffers are validated by transformImpl.
        fn transformInternal(setup: *const Setup, finput: [*]const T, foutput: [*]T, scratch: [*]T, direction: Direction, ordered: bool) [*]T {
            return if (wide != 4 and setup.lanes != 4)
                run(wide, setup, finput, foutput, scratch, direction, ordered)
            else
                run(4, setup, finput, foutput, scratch, direction, ordered);
        }

        fn run(comptime L: comptime_int, setup: *const Setup, finput: [*]const T, foutput: [*]T, scratch: [*]T, direction: Direction, ordered: bool) [*]T {
            const S = simd_math.SIMDL(T, L);
            const P = passes_mod.PassesL(T, L);
            const G = if (L == 4) @This() else W;
            const Ncvec = setup.Ncvec;
            const nf_odd = (setup.ifac[1] & 1) != 0;
            const vinput: S.CVP = @ptrCast(@alignCast(finput));
            const voutput: S.VP = @ptrCast(@alignCast(foutput));
            const buff: [2]S.VP = .{ voutput, @ptrCast(@alignCast(scratch)) };
            const e: S.CVP = @ptrCast(@alignCast(setup.e));

            // L=4 fuses the reorder into the finalize / preprocess pass: no extra pass.
            const fused = L == 4;
            const reorder_pass = ordered and !fused;
            var ib: usize = if (nf_odd != reorder_pass) 1 else 0;

            if (direction == .Forward) {
                ib = 1 - ib;
                if (setup.transform == .Real) {
                    const active = P.rfftf1_ps(2 * Ncvec, vinput, buff[ib], buff[1 - ib], setup.twiddle, &setup.ifac);
                    const active_ib: usize = if (active == buff[0]) 0 else 1;
                    if (fused and ordered)
                        realFinalizeOrdered(Ncvec, buff[active_ib], buff[1 - active_ib], e)
                    else
                        G.realFinalize(Ncvec, buff[active_ib], buff[1 - active_ib], e);
                    ib = 1 - active_ib;
                } else {
                    const tmp = buff[ib];
                    var k: usize = 0;
                    while (k < @as(usize, @intCast(Ncvec))) : (k += 1) {
                        const res_ui = S.uninterleave2(vinput[k * 2], vinput[k * 2 + 1]);
                        tmp[k * 2] = res_ui[0]; tmp[k * 2 + 1] = res_ui[1];
                    }
                    const active = P.cfftf1_ps(Ncvec, buff[ib], buff[1 - ib], buff[ib], setup.twiddle, &setup.ifac, -1);
                    const active_ib: usize = if (active == buff[0]) 0 else 1;
                    if (fused and ordered)
                        cplxFinalizeOrdered(Ncvec, buff[active_ib], buff[1 - active_ib], e)
                    else
                        G.cplxFinalize(Ncvec, buff[active_ib], buff[1 - active_ib], e);
                    ib = 1 - active_ib;
                }
                if (reorder_pass) {
                    const src = buff[ib];
                    const dst = buff[1 - ib];
                    const size = if (setup.transform == .Real) setup.N else 2 * setup.N;
                    reorder(setup, @as([*]const T, @ptrCast(src))[0..@intCast(size)], @as([*]T, @ptrCast(dst))[0..@intCast(size)], .Forward);
                    ib = 1 - ib;
                }
            } else {
                var input = vinput;
                // The ordered first pass reads scattered input: never run it in place.
                if (ordered and vinput == buff[ib]) ib = 1 - ib;
                if (reorder_pass) {
                    const size = if (setup.transform == .Real) setup.N else 2 * setup.N;
                    reorder(setup, @as([*]const T, @ptrCast(vinput))[0..@intCast(size)], @as([*]T, @ptrCast(buff[ib]))[0..@intCast(size)], .Backward);
                    input = buff[ib];
                    ib = 1 - ib;
                }
                if (setup.transform == .Real) {
                    if (fused and ordered)
                        realPreprocessOrdered(Ncvec, input, buff[ib], e)
                    else
                        G.realPreprocess(Ncvec, input, buff[ib], e);
                    const active = P.rfftb1_ps(2 * Ncvec, buff[ib], buff[0], buff[1], setup.twiddle, &setup.ifac);
                    ib = if (active == buff[0]) @as(usize, 0) else 1;
                } else {
                    if (fused and ordered)
                        cplxPreprocessOrdered(Ncvec, input, buff[ib], e)
                    else
                        G.cplxPreprocess(Ncvec, input, buff[ib], e);
                    const active = P.cfftf1_ps(Ncvec, buff[ib], buff[0], buff[1], setup.twiddle, &setup.ifac, 1);
                    const active_ib: usize = if (active == buff[0]) 0 else 1;
                    const res = buff[active_ib];
                    var k: usize = 0;
                    while (k < @as(usize, @intCast(Ncvec))) : (k += 1) {
                        const res_i = S.interleave2(res[k * 2], res[k * 2 + 1]);
                        res[k * 2] = res_i[0]; res[k * 2 + 1] = res_i[1];
                    }
                    ib = active_ib;
                }
            }
            return @ptrCast(buff[ib]);
        }

        /// Unordered (PFFFT internal layout) transform. input, output and work each need
        /// setup.bufferLen() floats aligned to setup.alignment(); input == output (in place) is fine,
        /// work must not overlap either.
        pub fn transform(setup: *const Setup, input: []const T, output: []T, work: []T, direction: Direction) Error!void {
            try transformImpl(setup, input, output, work, direction, false);
        }

        /// Like transform, but the spectrum is in canonical order.
        pub fn transformOrdered(setup: *const Setup, input: []const T, output: []T, work: []T, direction: Direction) Error!void {
            try transformImpl(setup, input, output, work, direction, true);
        }

        fn transformImpl(setup: *const Setup, input: []const T, output: []T, work: []T, direction: Direction, ordered: bool) Error!void {
            try checkBuffer(setup, input);
            try checkBuffer(setup, output);
            try checkBuffer(setup, work);
            const size = setup.bufferLen();
            if (overlaps(work[0..size], input[0..size]) or overlaps(work[0..size], output[0..size])) return error.BufferOverlap;
            if (input.ptr != output.ptr and overlaps(input[0..size], output[0..size])) return error.BufferOverlap;

            const res = transformInternal(setup, input.ptr, output.ptr, work.ptr, direction, ordered);
            if (res != output.ptr) @memcpy(output[0..size], res[0..size]);
        }

        fn checkBuffer(setup: *const Setup, buf: []const T) Error!void {
            if (buf.len < setup.bufferLen()) return error.BufferTooSmall;
            if (!std.mem.isAligned(@intFromPtr(buf.ptr), setup.alignment())) return error.Misaligned;
        }

        fn overlaps(a: []const T, b: []const T) bool {
            const a0 = @intFromPtr(a.ptr);
            const b0 = @intFromPtr(b.ptr);
            return a0 < b0 + b.len * @sizeOf(T) and b0 < a0 + a.len * @sizeOf(T);
        }

        /// Converts between the unordered and canonical layouts. Buffers as for transform;
        /// input and output must not overlap.
        pub fn zreorder(setup: *const Setup, input: []const T, output: []T, direction: Direction) Error!void {
            try checkBuffer(setup, input);
            try checkBuffer(setup, output);
            const size = setup.bufferLen();
            if (overlaps(input[0..size], output[0..size])) return error.BufferOverlap;
            reorder(setup, input, output, direction);
        }

        /// zreorder without the checks: buffers are validated by the caller.
        fn reorder(setup: *const Setup, input: []const T, output: []T, direction: Direction) void {
            const Ncvec = setup.Ncvec;
            const vin: [*]const Vec = @ptrCast(@alignCast(input.ptr));
            const vout: [*]Vec = @ptrCast(@alignCast(output.ptr));
            const N = setup.N;
            if (wide != 4 and setup.lanes != 4) return W.zreorder(setup, input, output, direction);

            if (setup.transform == .Real) {
                const dk = @divExact(N, 32);
                if (direction == .Forward) {
                    var k: usize = 0;
                    while (k < @as(usize, @intCast(dk))) : (k += 1) {
                        const res1 = s.interleave2(vin[k * 8 + 0], vin[k * 8 + 1]);
                        vout[2 * (0 * @as(usize, @intCast(dk)) + k) + 0] = res1[0];
                        vout[2 * (0 * @as(usize, @intCast(dk)) + k) + 1] = res1[1];
                        
                        const res2 = s.interleave2(vin[k * 8 + 4], vin[k * 8 + 5]);
                        vout[2 * (2 * @as(usize, @intCast(dk)) + k) + 0] = res2[0];
                        vout[2 * (2 * @as(usize, @intCast(dk)) + k) + 1] = res2[1];
                    }
                    reversedCopy(dk, vin + 2, 8, vout + @divExact(@as(usize, @intCast(N)), 8)); // out + N/2 floats = N/8 vectors
                    reversedCopy(dk, vin + 6, 8, vout + @divExact(@as(usize, @intCast(N)), 4)); // out + N floats = N/4 vectors
                } else {
                    var k: usize = 0;
                    while (k < @as(usize, @intCast(dk))) : (k += 1) {
                        const res1 = s.uninterleave2(vin[2 * (0 * @as(usize, @intCast(dk)) + k) + 0], vin[2 * (0 * @as(usize, @intCast(dk)) + k) + 1]);
                        vout[k * 8 + 0] = res1[0];
                        vout[k * 8 + 1] = res1[1];
                        
                        const res2 = s.uninterleave2(vin[2 * (2 * @as(usize, @intCast(dk)) + k) + 0], vin[2 * (2 * @as(usize, @intCast(dk)) + k) + 1]);
                        vout[k * 8 + 4] = res2[0];
                        vout[k * 8 + 5] = res2[1];
                    }
                    unreversedCopy(dk, vin + @divExact(@as(usize, @intCast(N)), 16), vout + @as(usize, @intCast(@divTrunc(N, 4) - 6)), -8);
                    unreversedCopy(dk, vin + @divExact(@as(usize, @intCast(3 * N)), 16), vout + @as(usize, @intCast(@divTrunc(N, 4) - 2)), -8);
                }
            } else {
                var k: i32 = 0;
                while (k < Ncvec) : (k += 1) {
                    const kk = @divTrunc(k, 4) + @rem(k, 4) * @divExact(Ncvec, 4);
                    const vkk = @as(usize, @intCast(kk));
                    const vk = @as(usize, @intCast(k));
                    if (direction == .Forward) {
                        const res = s.interleave2(vin[vk * 2], vin[vk * 2 + 1]);
                        vout[vkk * 2] = res[0];
                        vout[vkk * 2 + 1] = res[1];
                    } else {
                        const res = s.uninterleave2(vin[vkk * 2], vin[vkk * 2 + 1]);
                        vout[vk * 2] = res[0];
                        vout[vk * 2 + 1] = res[1];
                    }
                }
            }
        }

        fn reversedCopy(n: i32, in: [*]const Vec, stride: i32, out_ptr: [*]Vec) void {
            var vin = in;
            var vout = out_ptr;
            
            const res0 = s.interleave2(vin[0], vin[1]);
            const g0 = res0[0]; var g1 = res0[1];
            vin += @as(usize, @intCast(stride));
            
            vout -= 1;
            vout[0] = s.vswaphl(g0, g1);
            
            var k: i32 = 1;
            while (k < n) : (k += 1) {
                const resk = s.interleave2(vin[0], vin[1]);
                const h0 = resk[0]; const h1 = resk[1];
                vin += @as(usize, @intCast(stride));
                
                vout -= 1;
                vout[0] = s.vswaphl(g1, h0);
                vout -= 1;
                vout[0] = s.vswaphl(h0, h1);
                g1 = h1;
            }
            vout -= 1;
            vout[0] = s.vswaphl(g1, g0);
        }

        fn unreversedCopy(n: i32, in: [*]const Vec, out_ptr: [*]Vec, out_stride: i32) void {
            var vin = in;
            var vout = out_ptr;
            var k: i32 = 1;
            
            const g0 = vin[0]; var g1 = vin[0]; vin += 1;
            while (k < n) : (k += 1) {
                const h0 = vin[0]; const h1 = vin[1]; vin += 2;
                const g1_new = s.vswaphl(g1, h0);
                const h0_new = s.vswaphl(h0, h1);
                const res = s.uninterleave2(h0_new, g1_new);
                vout[0] = res[0]; vout[1] = res[1];
                
                if (out_stride < 0) {
                    vout -= @as(usize, @intCast(-out_stride));
                } else {
                    vout += @as(usize, @intCast(out_stride));
                }
                
                g1 = h1;
            }
            const h0 = vin[0]; const h1 = g0;
            const g1_final = s.vswaphl(g1, h0);
            const h0_final = s.vswaphl(h0, h1);
            const res_final = s.uninterleave2(h0_final, g1_final);
            vout[0] = res_final[0]; vout[1] = res_final[1];
        }


        fn cplxPreprocess(Ncvec: i32, in: [*]const Vec, out: [*]Vec, e: [*]const Vec) void {
            cplxPreprocessImpl(false, Ncvec, in, out, e);
        }

        /// zreorder(.Backward) followed by cplxPreprocess, in one pass.
        fn cplxPreprocessOrdered(Ncvec: i32, in: [*]const Vec, out: [*]Vec, e: [*]const Vec) void {
            cplxPreprocessImpl(true, Ncvec, in, out, e);
        }

        // Ordered: pair j of block k (vectors 8k+2j, +1) lives interleaved at 2(k + j*dk).
        fn cplxPreprocessImpl(comptime ordered: bool, Ncvec: i32, in: [*]const Vec, out: [*]Vec, e: [*]const Vec) void {
            const dk: usize = @intCast(@divExact(Ncvec, 4));
            var k: usize = 0;
            while (k < dk) : (k += 1) {
                var v: [8]Vec = undefined;
                inline for (0..4) |j| {
                    const pr: [2]Vec = if (ordered) s.uninterleave2(in[2 * (k + j * dk)], in[2 * (k + j * dk) + 1]) else .{ in[8 * k + 2 * j], in[8 * k + 2 * j + 1] };
                    v[2 * j] = pr[0];
                    v[2 * j + 1] = pr[1];
                }
                var r0 = v[0]; var im0 = v[1];
                var r1 = v[2]; var im1 = v[3];
                var r2 = v[4]; var im2 = v[5];
                var r3 = v[6]; var im3 = v[7];

                const sr0 = s.vadd(r0, r2); const dr0 = s.vsub(r0, r2);
                const sr1 = s.vadd(r1, r3); const dr1 = s.vsub(r1, r3);
                const si0 = s.vadd(im0, im2); const di0 = s.vsub(im0, im2);
                const si1 = s.vadd(im1, im3); const di1 = s.vsub(im1, im3);
                
                r0 = s.vadd(sr0, sr1); im0 = s.vadd(si0, si1);
                r1 = s.vsub(dr0, di1); im1 = s.vadd(di0, dr1);
                r2 = s.vsub(sr0, sr1); im2 = s.vsub(si0, si1);
                r3 = s.vadd(dr0, di1); im3 = s.vsub(di0, dr1);

                s.vcplxmulconj(&r1, &im1, e[k * 6 + 0], e[k * 6 + 1]);
                s.vcplxmulconj(&r2, &im2, e[k * 6 + 2], e[k * 6 + 3]);
                s.vcplxmulconj(&r3, &im3, e[k * 6 + 4], e[k * 6 + 5]);
                
                s.vtranspose4(&r0, &r1, &r2, &r3);
                s.vtranspose4(&im0, &im1, &im2, &im3);
                
                out[8 * k + 0] = r0; out[8 * k + 1] = im0;
                out[8 * k + 2] = r1; out[8 * k + 3] = im1;
                out[8 * k + 4] = r2; out[8 * k + 5] = im2;
                out[8 * k + 6] = r3; out[8 * k + 7] = im3;
            }
        }

        fn cplxFinalize(Ncvec: i32, in: [*]const Vec, out: [*]Vec, e: [*]const Vec) void {
            cplxFinalizeImpl(false, Ncvec, in, out, e);
        }

        /// cplxFinalize followed by zreorder(.Forward), in one pass.
        fn cplxFinalizeOrdered(Ncvec: i32, in: [*]const Vec, out: [*]Vec, e: [*]const Vec) void {
            cplxFinalizeImpl(true, Ncvec, in, out, e);
        }

        // Ordered: pair j of block k is written interleaved to 2(k + j*dk).
        fn cplxFinalizeImpl(comptime ordered: bool, Ncvec: i32, in: [*]const Vec, out: [*]Vec, e: [*]const Vec) void {
            const dk: usize = @intCast(@divExact(Ncvec, 4));
            var k: usize = 0;
            while (k < dk) : (k += 1) {
                var r0 = in[8 * k + 0]; var im0 = in[8 * k + 1];
                var r1 = in[8 * k + 2]; var im1 = in[8 * k + 3];
                var r2 = in[8 * k + 4]; var im2 = in[8 * k + 5];
                var r3 = in[8 * k + 6]; var im3 = in[8 * k + 7];

                s.vtranspose4(&r0, &r1, &r2, &r3);
                s.vtranspose4(&im0, &im1, &im2, &im3);

                s.vcplxmul(&r1, &im1, e[k * 6 + 0], e[k * 6 + 1]);
                s.vcplxmul(&r2, &im2, e[k * 6 + 2], e[k * 6 + 3]);
                s.vcplxmul(&r3, &im3, e[k * 6 + 4], e[k * 6 + 5]);
                
                const sr0 = s.vadd(r0, r2); const dr0 = s.vsub(r0, r2);
                const sr1 = s.vadd(r1, r3); const dr1 = s.vsub(r1, r3);
                const si0 = s.vadd(im0, im2); const di0 = s.vsub(im0, im2);
                const si1 = s.vadd(im1, im3); const di1 = s.vsub(im1, im3);

                r0 = s.vadd(sr0, sr1); im0 = s.vadd(si0, si1);
                r1 = s.vadd(dr0, di1); im1 = s.vsub(di0, dr1);
                r2 = s.vsub(sr0, sr1); im2 = s.vsub(si0, si1);
                r3 = s.vsub(dr0, di1); im3 = s.vadd(di0, dr1);
                
                const v = [8]Vec{ r0, im0, r1, im1, r2, im2, r3, im3 };
                inline for (0..4) |j| {
                    if (ordered) {
                        const pr = s.interleave2(v[2 * j], v[2 * j + 1]);
                        out[2 * (k + j * dk)] = pr[0];
                        out[2 * (k + j * dk) + 1] = pr[1];
                    } else {
                        out[8 * k + 2 * j] = v[2 * j];
                        out[8 * k + 2 * j + 1] = v[2 * j + 1];
                    }
                }
            }
        }

        pub fn realPreprocess(Ncvec: i32, in: [*]const Vec, out: [*]Vec, e: [*]const Vec) void {
            realPreprocessImpl(false, Ncvec, in, out, e);
        }

        /// zreorder(.Backward) followed by realPreprocess, in one pass.
        pub fn realPreprocessOrdered(Ncvec: i32, in: [*]const Vec, out: [*]Vec, e: [*]const Vec) void {
            realPreprocessImpl(true, Ncvec, in, out, e);
        }

        fn realPreprocessImpl(comptime ordered: bool, Ncvec: i32, in: [*]const Vec, out: [*]Vec, e: [*]const Vec) void {
            const dk: usize = @intCast(@divExact(Ncvec, 4));
            var Xr: s.Union = undefined;
            var Xi: s.Union = undefined;
            const sqrt2 = @as(T, @floatCast(1.4142135623730951));

            // Ordered: each block is gathered into blk, like zreorder(.Backward).
            var blk: [8]Vec = undefined;
            if (ordered) readOrderedBlock(in, 0, dk, &blk);
            const b0: [*]const Vec = if (ordered) &blk else in;
            const fin: [*]const T = @ptrCast(b0);

            var i: usize = 0;
            while (i < 4) : (i += 1) {
                Xr.f[i] = fin[8 * i];
                Xi.f[i] = fin[8 * i + 4];
            }

            realPreprocess4x4(b0, e, out + 1, true);

            var k: usize = 1;
            while (k < dk) : (k += 1) {
                if (ordered) readOrderedBlock(in, k, dk, &blk);
                realPreprocess4x4(if (ordered) &blk else in + 8 * k, e + k * 6, out - 1 + 8 * k, false);
            }

            const uout: [*]s.Union = @ptrCast(out);
            uout[0].f[0] = (Xr.f[0] + Xi.f[0]) + 2 * Xr.f[2];
            uout[0].f[1] = (Xr.f[0] - Xi.f[0]) - 2 * Xi.f[2];
            uout[0].f[2] = (Xr.f[0] + Xi.f[0]) - 2 * Xr.f[2];
            uout[0].f[3] = (Xr.f[0] - Xi.f[0]) + 2 * Xi.f[2];

            uout[@as(usize, @intCast(2 * Ncvec - 1))].f[0] = 2 * (Xr.f[1] + Xr.f[3]);
            uout[@as(usize, @intCast(2 * Ncvec - 1))].f[1] = sqrt2 * (Xr.f[1] - Xr.f[3]) - sqrt2 * (Xi.f[1] + Xi.f[3]);
            uout[@as(usize, @intCast(2 * Ncvec - 1))].f[2] = 2 * (Xi.f[3] - Xi.f[1]);
            uout[@as(usize, @intCast(2 * Ncvec - 1))].f[3] = -sqrt2 * (Xr.f[1] - Xr.f[3]) - sqrt2 * (Xi.f[1] + Xi.f[3]);
        }

        pub fn realFinalize(Ncvec: i32, in: [*]const Vec, out: [*]Vec, e: [*]const Vec) void {
            realFinalizeImpl(false, Ncvec, in, out, e);
        }

        /// realFinalize followed by zreorder(.Forward), in one pass.
        pub fn realFinalizeOrdered(Ncvec: i32, in: [*]const Vec, out: [*]Vec, e: [*]const Vec) void {
            realFinalizeImpl(true, Ncvec, in, out, e);
        }

        fn realFinalizeImpl(comptime ordered: bool, Ncvec: i32, in: [*]const Vec, out: [*]Vec, e: [*]const Vec) void {
            const dk: usize = @intCast(@divExact(Ncvec, 4));
            const cr: s.Union = .{ .f = in[0] };
            const ci: s.Union = .{ .f = in[@as(usize, @intCast(2 * Ncvec - 1))] };
            const hsqrt2 = @as(T, @floatCast(0.7071067811865476));
            const zero = s.vzero();

            // Ordered: each block goes through blk, then scatters like zreorder.
            var blk: [8]Vec = undefined;
            const b0: [*]Vec = if (ordered) &blk else out;
            realFinalize4x4(&zero, &zero, in + 1, e, b0);

            const uout: [*]s.Union = @ptrCast(b0);
            uout[0].f[0] = (cr.f[0] + cr.f[2]) + (cr.f[1] + cr.f[3]);
            uout[1].f[0] = (cr.f[0] + cr.f[2]) - (cr.f[1] + cr.f[3]);
            uout[4].f[0] = (cr.f[0] - cr.f[2]);
            uout[5].f[0] = (cr.f[3] - cr.f[1]);
            uout[2].f[0] = ci.f[0] + hsqrt2 * (ci.f[1] - ci.f[3]);
            uout[3].f[0] = -ci.f[2] - hsqrt2 * (ci.f[1] + ci.f[3]);
            uout[6].f[0] = ci.f[0] - hsqrt2 * (ci.f[1] - ci.f[3]);
            uout[7].f[0] = ci.f[2] - hsqrt2 * (ci.f[1] + ci.f[3]);

            // reversedCopy state for vectors 2,3 (a) and 6,7 (b): output
            // vector j of a reversed half is vswaphl of neighbouring blocks.
            const ra: [2]Vec = if (ordered) s.interleave2(blk[2], blk[3]) else undefined;
            const rb: [2]Vec = if (ordered) s.interleave2(blk[6], blk[7]) else undefined;
            var ga = ra[1];
            var gb = rb[1];
            if (ordered) {
                writeOrderedHalves(&blk, out, 0, dk);
                out[4 * dk - 1] = s.vswaphl(ra[0], ra[1]);
                out[8 * dk - 1] = s.vswaphl(rb[0], rb[1]);
            }

            var save = in[7];
            var k: usize = 1;
            while (k < dk) : (k += 1) {
                const save_next = in[8 * k + 7];
                realFinalize4x4(&save, &in[8 * k], in + 8 * k + 1, e + k * 6, if (ordered) &blk else out + k * 8);
                save = save_next;
                if (ordered) {
                    writeOrderedHalves(&blk, out, k, dk);
                    const ha = s.interleave2(blk[2], blk[3]);
                    out[4 * dk - 2 * k] = s.vswaphl(ga, ha[0]);
                    out[4 * dk - 2 * k - 1] = s.vswaphl(ha[0], ha[1]);
                    ga = ha[1];
                    const hb = s.interleave2(blk[6], blk[7]);
                    out[8 * dk - 2 * k] = s.vswaphl(gb, hb[0]);
                    out[8 * dk - 2 * k - 1] = s.vswaphl(hb[0], hb[1]);
                    gb = hb[1];
                }
            }
            if (ordered) {
                out[2 * dk] = s.vswaphl(ga, ra[0]);
                out[6 * dk] = s.vswaphl(gb, rb[0]);
            }
        }

        /// zreorder(.Forward) of block k's vectors 0,1 and 4,5.
        inline fn writeOrderedHalves(blk: *const [8]Vec, out: [*]Vec, k: usize, dk: usize) void {
            const r1 = s.interleave2(blk[0], blk[1]);
            out[2 * k] = r1[0];
            out[2 * k + 1] = r1[1];
            const r2 = s.interleave2(blk[4], blk[5]);
            out[2 * (2 * dk + k)] = r2[0];
            out[2 * (2 * dk + k) + 1] = r2[1];
        }

        /// Block k of zreorder(.Backward): vectors 0,1 / 4,5 uninterleaved from
        /// 2k / 2(2dk+k); 2,3 / 6,7 are unreversedCopy's output for this block
        /// (halves at 2dk and 6dk, m = 2(dk-k), wrapping to 0 for k = 0).
        inline fn readOrderedBlock(in: [*]const Vec, k: usize, dk: usize, blk: *[8]Vec) void {
            const r1 = s.uninterleave2(in[2 * k], in[2 * k + 1]);
            blk[0] = r1[0];
            blk[1] = r1[1];
            const r2 = s.uninterleave2(in[2 * (2 * dk + k)], in[2 * (2 * dk + k) + 1]);
            blk[4] = r2[0];
            blk[5] = r2[1];
            const m = 2 * (dk - k);
            const wrap = if (k == 0) 0 else m;
            const a = in + 2 * dk;
            const ra = s.uninterleave2(s.vswaphl(a[m - 1], a[wrap]), s.vswaphl(a[m - 2], a[m - 1]));
            blk[2] = ra[0];
            blk[3] = ra[1];
            const b = in + 6 * dk;
            const rb = s.uninterleave2(s.vswaphl(b[m - 1], b[wrap]), s.vswaphl(b[m - 2], b[m - 1]));
            blk[6] = rb[0];
            blk[7] = rb[1];
        }

        fn realPreprocess4x4(in: [*]const Vec, e: [*]const Vec, out: [*]Vec, first: bool) void {
            var r0 = in[0]; var im0 = in[1]; var r1 = in[2]; var im1 = in[3];
            var r2 = in[4]; var im2 = in[5]; var r3 = in[6]; var im3 = in[7];

            const sr0 = s.vadd(r0, r3); const dr0 = s.vsub(r0, r3);
            const sr1 = s.vadd(r1, r2); const dr1 = s.vsub(r1, r2);
            const sim0 = s.vadd(im0, im3); const dim0 = s.vsub(im0, im3);
            const sim1 = s.vadd(im1, im2); const dim1 = s.vsub(im1, im2);

            r0 = s.vadd(sr0, sr1);
            r2 = s.vsub(sr0, sr1);
            r1 = s.vsub(dr0, sim1);
            r3 = s.vadd(dr0, sim1);
            im0 = s.vsub(dim0, dim1);
            im2 = s.vadd(dim0, dim1);
            im1 = s.vsub(sim0, dr1);
            im3 = s.vadd(sim0, dr1);

            s.vcplxmulconj(&r1, &im1, e[0], e[1]);
            s.vcplxmulconj(&r2, &im2, e[2], e[3]);
            s.vcplxmulconj(&r3, &im3, e[4], e[5]);

            s.vtranspose4(&r0, &r1, &r2, &r3);
            s.vtranspose4(&im0, &im1, &im2, &im3);

            var vout = out;
            if (!first) {
                vout[0] = r0; vout[1] = im0; vout += 2;
            }
            vout[0] = r1; vout[1] = im1;
            vout[2] = r2; vout[3] = im2;
            vout[4] = r3; vout[5] = im3;
        }

        fn realFinalize4x4(in0: *const Vec, in1: *const Vec, in_ptr: [*]const Vec, e: [*]const Vec, out: [*]Vec) void {
            var r0 = in0.*; var im0 = in1.*;
            const in = in_ptr;
            var r1 = in[0]; var im1 = in[1];
            var r2 = in[2]; var im2 = in[3];
            var r3 = in[4]; var im3 = in[5];

            s.vtranspose4(&r0, &r1, &r2, &r3);
            s.vtranspose4(&im0, &im1, &im2, &im3);

            s.vcplxmul(&r1, &im1, e[0], e[1]);
            s.vcplxmul(&r2, &im2, e[2], e[3]);
            s.vcplxmul(&r3, &im3, e[4], e[5]);

            const sr0 = s.vadd(r0, r2); const dr0 = s.vsub(r0, r2);
            const sr1 = s.vadd(r1, r3); const dr1 = s.vsub(r3, r1);
            const sim0 = s.vadd(im0, im2); const dim0 = s.vsub(im0, im2);
            const sim1 = s.vadd(im1, im3); const dim1 = s.vsub(im3, im1);

            r0 = s.vadd(sr0, sr1);
            r3 = s.vsub(sr0, sr1);
            im0 = s.vadd(sim0, sim1);
            im3 = s.vsub(sim1, sim0);
            r1 = s.vadd(dr0, dim1);
            r2 = s.vsub(dr0, dim1);
            im1 = s.vsub(dr1, dim0);
            im2 = s.vadd(dr1, dim0);

            out[0] = r0; out[1] = im0; out[2] = r1; out[3] = im1;
            out[4] = r2; out[5] = im2; out[6] = r3; out[7] = im3;
        }
    };
}
