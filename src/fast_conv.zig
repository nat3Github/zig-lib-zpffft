const std = @import("std");
const simd = @import("simd_math.zig");
const setup_mod = @import("setup.zig");

const Setup = setup_mod.Setup(f32);
const wide = simd.wideLanes(f32);

/// ab += a * b * scaling, on unordered spectra from `transform`.
pub fn zconvolveAccumulate(setup: *Setup, a: []const f32, b: []const f32, ab: []f32, scaling: f32) void {
    zconvolve(true, setup, a, b, ab, scaling);
}

/// ab = a * b * scaling, on unordered spectra from `transform`.
pub fn zconvolveNoAccu(setup: *Setup, a: []const f32, b: []const f32, ab: []f32, scaling: f32) void {
    zconvolve(false, setup, a, b, ab, scaling);
}

fn zconvolve(comptime accumulate: bool, setup: *Setup, a: []const f32, b: []const f32, ab: []f32, scaling: f32) void {
    if (wide != 4 and setup.lanes != 4) return zconvolveL(wide, accumulate, setup, a, b, ab, scaling);
    zconvolveL(4, accumulate, setup, a, b, ab, scaling);
}

fn zconvolveL(comptime L: comptime_int, comptime accumulate: bool, setup: *Setup, a: []const f32, b: []const f32, ab: []f32, scaling: f32) void {
    const s = simd.SIMDL(f32, L);
    const Ncvec: usize = @intCast(setup.Ncvec);
    const va: s.CVP = @ptrCast(@alignCast(a.ptr));
    const vb: s.CVP = @ptrCast(@alignCast(b.ptr));
    const vab: s.VP = @ptrCast(@alignCast(ab.ptr));
    const vscal = s.vld1(scaling);

    // Real: lane 0 of pair 0 holds (DC, Nyquist), two real values that must
    // not be complex-multiplied. Run the full loop, then patch that lane.
    const ar = a[0];
    const ai = a[s.simd_sz];
    const abr = ab[0];
    const abi = ab[s.simd_sz];

    for (0..Ncvec) |k| {
        var ar_v = va[2 * k];
        var ai_v = va[2 * k + 1];
        s.vcplxmul(&ar_v, &ai_v, vb[2 * k], vb[2 * k + 1]);
        if (accumulate) {
            vab[2 * k] = s.vmadd(ar_v, vscal, vab[2 * k]);
            vab[2 * k + 1] = s.vmadd(ai_v, vscal, vab[2 * k + 1]);
        } else {
            vab[2 * k] = s.vmul(ar_v, vscal);
            vab[2 * k + 1] = s.vmul(ai_v, vscal);
        }
    }

    if (setup.transform == .Real) {
        ab[0] = (if (accumulate) abr else 0) + ar * b[0] * scaling;
        ab[s.simd_sz] = (if (accumulate) abi else 0) + ai * b[s.simd_sz] * scaling;
    }
}
