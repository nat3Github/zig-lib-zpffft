const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

/// Fused multiply-add only where the target has it in hardware: on other
/// targets @mulAdd lowers to a (very slow) software fmaf libcall.
pub const use_fma = switch (builtin.cpu.arch) {
    .aarch64, .aarch64_be => true,
    .x86, .x86_64 => std.Target.x86.featureSetHas(builtin.cpu.features, .fma),
    .arm, .armeb, .thumb, .thumbeb => std.Target.arm.featureSetHasAll(builtin.cpu.features, .{ .neon, .vfp4 }),
    .riscv32, .riscv64 => std.Target.riscv.featureSetHas(builtin.cpu.features, .v),
    .powerpc64, .powerpc64le => std.Target.powerpc.featureSetHas(builtin.cpu.features, .vsx),
    // Scalar maebr/madbr are base z/Architecture. (Zig 0.16 forces the vector
    // facility off on s390x, llvm#168992, so a `vector` check would never pass.)
    .s390x => true,
    // wasm: relaxed_madd may or may not fuse, so results would vary by engine.
    else => false,
};

/// Lanes of the wide path for T: -Dsimd_lanes, or (0 = auto) the target's
/// suggested vector length clamped to 4..16. Capped at 512-bit vectors.
/// 4 means no wide path.
pub fn wideLanes(comptime T: type) comptime_int {
    const req = build_options.simd_lanes;
    const l = if (req == 0) std.math.clamp(std.simd.suggestVectorLength(T) orelse 4, 4, 16) else req;
    if (l != 4 and l != 8 and l != 16) @compileError("simd_lanes must be 0 (auto), 4, 8 or 16");
    return @min(l, 64 / @sizeOf(T));
}

pub fn SIMD(comptime T: type) type {
    return SIMDL(T, 4);
}

pub fn SIMDL(comptime T: type, comptime L: comptime_int) type {
    return struct {
        pub const Vec = @Vector(L, T);
        pub const simd_sz = L;
        /// Buffers only need 4-lane alignment (the pffft contract), so wide
        /// vectors are accessed through under-aligned pointers. For L=4 these
        /// are plain [*]Vec.
        pub const A = @alignOf(@Vector(4, T));
        pub const VP = [*]align(A) Vec;
        pub const CVP = [*]align(A) const Vec;

        pub const Union = extern struct {
            f: [L]T,
        };

        pub inline fn fma(a: Vec, b: Vec, c: Vec) Vec {
            return vmadd(a, b, c);
        }

        pub inline fn vzero() Vec {
            return @as(Vec, @splat(0.0));
        }

        pub inline fn vld1(f: T) Vec {
            return @as(Vec, @splat(f));
        }

        pub inline fn vmul(a: Vec, b: Vec) Vec {
            return a * b;
        }

        pub inline fn vadd(a: Vec, b: Vec) Vec {
            return a + b;
        }

        pub inline fn vsub(a: Vec, b: Vec) Vec {
            return a - b;
        }

        pub inline fn vmadd(a: Vec, b: Vec, c: Vec) Vec {
            return if (use_fma) @mulAdd(Vec, a, b, c) else a * b + c;
        }

        pub inline fn vmsub(a: Vec, b: Vec, c: Vec) Vec {
            return if (use_fma) @mulAdd(Vec, a, b, -c) else a * b - c;
        }

        /// c - a*b (fmls on aarch64, vfnmadd on x86).
        pub inline fn vnmsub(a: Vec, b: Vec, c: Vec) Vec {
            return if (use_fma) @mulAdd(Vec, -a, b, c) else c - a * b;
        }

        /// Aligned load: ptr must be @alignOf(Vec)-aligned.
        pub inline fn vld1_ptr(ptr: [*]const T) Vec {
            const p: *const Vec = @ptrCast(@alignCast(ptr));
            return p.*;
        }

        pub inline fn vld1_unaligned(ptr: [*]const T) Vec {
            const p: *const [L]T = @ptrCast(@alignCast(ptr));
            return p.*;
        }

        /// Aligned store: ptr must be @alignOf(Vec)-aligned.
        pub inline fn vstoreA(ptr: [*]T, v: Vec) void {
            const p: *Vec = @ptrCast(@alignCast(ptr));
            p.* = v;
        }

        pub inline fn vstoreU(ptr: [*]T, v: Vec) void {
            const p: *[L]T = @ptrCast(@alignCast(ptr));
            p.* = v;
        }

        pub inline fn interleave2(in1: Vec, in2: Vec) [2]Vec {
            return .{
                @shuffle(T, in1, in2, interleave_mask[0]),
                @shuffle(T, in1, in2, interleave_mask[1]),
            };
        }

        pub inline fn uninterleave2(in1: Vec, in2: Vec) [2]Vec {
            return .{
                @shuffle(T, in1, in2, uninterleave_mask[0]),
                @shuffle(T, in1, in2, uninterleave_mask[1]),
            };
        }

        // L=4: {0,~0,1,~1}/{2,~2,3,~3} and {0,2,~0,~2}/{1,3,~1,~3}.
        const interleave_mask = blk: {
            var m: [2][L]i32 = undefined;
            for (0..2) |h| for (0..L) |i| {
                const src: i32 = h * L / 2 + i / 2;
                m[h][i] = if (i % 2 == 0) src else ~src;
            };
            break :blk m;
        };
        const uninterleave_mask = blk: {
            var m: [2][L]i32 = undefined;
            for (0..2) |h| for (0..L) |i| {
                m[h][i] = if (i < L / 2) 2 * i + h else ~@as(i32, 2 * (i - L / 2) + h);
            };
            break :blk m;
        };

        /// In-place L×L transpose (rows = vectors), bitwise trn: stage g pairs
        /// rows (i, i+g) with i&g == 0. L·log2(L) shuffles; for L=4 the same
        /// masks as vtranspose4.
        pub inline fn transposeL(v: *[L]Vec) void {
            comptime var g = 1;
            inline while (g < L) : (g *= 2) {
                const lo = comptime blk: {
                    var m: [L]i32 = undefined;
                    for (0..L) |c| m[c] = if (c & g == 0) c else ~@as(i32, c - g);
                    break :blk m;
                };
                const hi = comptime blk: {
                    var m: [L]i32 = undefined;
                    for (0..L) |c| m[c] = if (c & g == 0) c + g else ~@as(i32, c);
                    break :blk m;
                };
                inline for (0..L) |i| {
                    if (i & g == 0) {
                        const a = v[i];
                        const b = v[i + g];
                        v[i] = @shuffle(T, a, b, lo);
                        v[i + g] = @shuffle(T, a, b, hi);
                    }
                }
            }
        }

        pub inline fn vuninterleave2(in1: Vec, in2: Vec, out1: *Vec, out2: *Vec) void {
            const res = uninterleave2(in1, in2);
            out1.* = res[0];
            out2.* = res[1];
        }

        pub inline fn vinterleave2(in1: Vec, in2: Vec, out1: *Vec, out2: *Vec) void {
            const res = interleave2(in1, in2);
            out1.* = res[0];
            out2.* = res[1];
        }

        // vtranspose4, vswaphl, vrevS, vrevC: 4-lane only.

        /// trn-then-halves form: every step stays within 128-bit halves or
        /// moves whole halves, so f64 on AVX2 is 8 ops (unpck*pd + vperm2f128)
        /// instead of 16 vpermpd/blend, and RVV avoids widening zip emulation.
        /// Same op count as the zip form on SSE, NEON, wasm and VSX.
        pub inline fn vtranspose4(v0: *Vec, v1: *Vec, v2: *Vec, v3: *Vec) void {
            const t0 = @shuffle(T, v0.*, v1.*, [4]i32{ 0, ~@as(i32, 0), 2, ~@as(i32, 2) });
            const t1 = @shuffle(T, v0.*, v1.*, [4]i32{ 1, ~@as(i32, 1), 3, ~@as(i32, 3) });
            const t2 = @shuffle(T, v2.*, v3.*, [4]i32{ 0, ~@as(i32, 0), 2, ~@as(i32, 2) });
            const t3 = @shuffle(T, v2.*, v3.*, [4]i32{ 1, ~@as(i32, 1), 3, ~@as(i32, 3) });

            v0.* = @shuffle(T, t0, t2, [4]i32{ 0, 1, ~@as(i32, 0), ~@as(i32, 1) });
            v1.* = @shuffle(T, t1, t3, [4]i32{ 0, 1, ~@as(i32, 0), ~@as(i32, 1) });
            v2.* = @shuffle(T, t0, t2, [4]i32{ 2, 3, ~@as(i32, 2), ~@as(i32, 3) });
            v3.* = @shuffle(T, t1, t3, [4]i32{ 2, 3, ~@as(i32, 2), ~@as(i32, 3) });
        }

        pub inline fn vswaphl(a: Vec, b: Vec) Vec {
            return @shuffle(T, b, a, [4]i32{ 0, 1, ~@as(i32, 2), ~@as(i32, 3) });
        }

        pub inline fn vrevS(a: Vec) Vec {
            return @shuffle(T, a, undefined, [4]i32{ 3, 2, 1, 0 });
        }

        pub inline fn vrevC(a: Vec) Vec {
            return @shuffle(T, a, undefined, [4]i32{ 2, 3, 0, 1 });
        }

        pub inline fn vcplxmul(ar: *Vec, ai: *Vec, br: Vec, bi: Vec) void {
            // Written as c -/+ a*b so aarch64 folds the negation into fmls.
            const tr = if (use_fma) @mulAdd(Vec, -ai.*, bi, ar.* * br) else ar.* * br - ai.* * bi;
            const ti = if (use_fma) @mulAdd(Vec, ar.*, bi, ai.* * br) else ar.* * bi + ai.* * br;
            ar.* = tr;
            ai.* = ti;
        }

        pub inline fn vcplxmulconj(ar: *Vec, ai: *Vec, br: Vec, bi: Vec) void {
            const tr = if (use_fma) @mulAdd(Vec, ai.*, bi, ar.* * br) else ar.* * br + ai.* * bi;
            const ti = if (use_fma) @mulAdd(Vec, -ar.*, bi, ai.* * br) else ai.* * br - ar.* * bi;
            ar.* = tr;
            ai.* = ti;
        }
    };
}
