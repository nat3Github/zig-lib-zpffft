const std = @import("std");
const math = std.math;
const simd_math = @import("simd_math.zig");

pub const TransformType = enum {
    Real,
    Complex,
};

pub const Direction = enum {
    Forward,
    Backward,
};

/// Errors of transform, transformOrdered, zreorder and the fast_conv functions.
pub const Error = error{
    /// A buffer is shorter than Setup.bufferLen().
    BufferTooSmall,
    /// A buffer is not aligned to Setup.alignment() bytes.
    Misaligned,
    /// Buffers overlap where they must not (work with input/output; zreorder input with output;
    /// transform input with output other than exactly in place).
    BufferOverlap,
};

/// Largest N accepted by Setup.init. Keeps every i32 index expression (up to 3·N) in range.
pub const max_n: i32 = 1 << 29;

/// A positive i32 has at most 30 prime factors, so ifac (n, nf, factors...) never overflows.
const ifac_len = 2 + 30;

pub fn Setup(comptime T: type) type {
    const s = simd_math.SIMD(T);
    const Vec = s.Vec;
    const wide = simd_math.wideLanes(T);

    return struct {
        const Self = @This();

        N: i32,
        /// Complex vectors of `lanes` lanes: N/(2·lanes) real, N/lanes complex.
        Ncvec: i32,
        ifac: [ifac_len]i32,
        transform: TransformType,
        data: []align(64) Vec,
        e: [*]T,
        twiddle: [*]T,
        allocator: std.mem.Allocator,
        /// SIMD lanes of the unordered layout: the wide build lanes when N is
        /// a multiple of 2·L² (real) or L² (complex), else 4.
        lanes: i32,

        pub fn init(allocator: std.mem.Allocator, N: i32, transform: TransformType) !*Self {
            const unit: i32 = if (transform == .Real) 2 else 1;
            if (N <= 0 or N > max_n or @rem(N, unit * 16) != 0) return error.InvalidSize;
            const L: usize = if (wide != 4 and @rem(N, unit * wide * wide) == 0) wide else 4;

            var self = try allocator.create(Self);
            errdefer allocator.destroy(self);
            self.allocator = allocator;
            self.N = N;
            self.transform = transform;
            self.lanes = @intCast(L);
            self.Ncvec = @divExact(N, unit * @as(i32, @intCast(L)));
            const Ncvec: usize = @intCast(self.Ncvec);

            // e (2·Ncvec·(L-1) scalars) then the fftpack twiddles (≤ 2·Ncvec):
            // at most 2·Ncvec·L scalars, within 3·Ncvec 4-lane vectors' worth.
            const data_count = 3 * Ncvec * L / 4;
            self.data = try allocator.alignedAlloc(Vec, .@"64", data_count);
            errdefer allocator.free(self.data);
            self.e = @ptrCast(self.data.ptr);
            self.twiddle = self.e + 2 * Ncvec * (L - 1);

            const PI = math.pi;

            for (0..Ncvec) |k_u| {
                const k = @as(T, @floatFromInt(k_u));
                const i = k_u / L;
                const j = k_u % L;
                for (0..L - 1) |m| {
                    const A = -2.0 * PI * @as(T, @floatFromInt(m + 1)) * k / @as(T, @floatFromInt(N));
                    self.e[(2 * (i * (L - 1) + m) + 0) * L + j] = @cos(A);
                    self.e[(2 * (i * (L - 1) + m) + 1) * L + j] = @sin(A);
                }
            }

            if (transform == .Real) {
                rffti1_ps(T, @divExact(N, self.lanes), self.twiddle, &self.ifac);
            } else {
                cffti1_ps(T, @divExact(N, self.lanes), self.twiddle, &self.ifac);
            }

            var m_sum: i32 = 1;
            var k: usize = 0;
            while (k < @as(usize, @intCast(self.ifac[1]))) : (k += 1) {
                m_sum *= self.ifac[2 + k];
            }
            if (m_sum != @divExact(N, self.lanes)) return error.InvalidFactorization;

            return self;
        }

        /// Floats in each input, output and work buffer: N (real) or 2N (complex).
        pub fn bufferLen(self: *const Self) usize {
            const n: usize = @intCast(self.N);
            return if (self.transform == .Real) n else 2 * n;
        }

        /// Byte alignment every buffer must have: that of the SIMD vector this setup runs on.
        pub fn alignment(self: *const Self) usize {
            return if (wide != 4 and self.lanes != 4) @alignOf(simd_math.SIMDL(T, wide).Vec) else @alignOf(Vec);
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
            self.allocator.destroy(self);
        }
    };
}

fn decompose(n: i32, ifac: *[ifac_len]i32, ntryh: []const i32) i32 {
    var nl = n;
    var nf: i32 = 0;
    var j: usize = 0;
    while (ntryh[j] != 0) : (j += 1) {
        const ntry = ntryh[j];
        while (nl != 1) {
            const nq = @divTrunc(nl, ntry);
            const nr = nl - ntry * nq;
            if (nr == 0) {
                ifac[2 + @as(usize, @intCast(nf))] = ntry;
                nf += 1;
                nl = nq;
                if (ntry == 2 and nf != 1) {
                    var i: i32 = 2;
                    while (i <= nf) : (i += 1) {
                        const ib = nf - i + 2;
                        ifac[@as(usize, @intCast(ib + 1))] = ifac[@as(usize, @intCast(ib))];
                    }
                    ifac[2] = 2;
                }
            } else break;
        }
    }
    ifac[0] = n;
    ifac[1] = nf;
    return nf;
}

fn rffti1_ps(comptime T: type, n: i32, wa: [*]T, ifac: *[ifac_len]i32) void {
    const ntryh = [_]i32{ 4, 2, 3, 5, 0 };
    const nf = decompose(n, ifac, &ntryh);
    const argh = (2.0 * math.pi) / @as(T, @floatFromInt(n));
    var is: i32 = 0;
    const nfm1 = nf - 1;
    var l1: i32 = 1;
    var k1: i32 = 1;
    while (k1 <= nfm1) : (k1 += 1) {
        const ip = ifac[@as(usize, @intCast(k1 + 1))];
        var ld: i32 = 0;
        const l2 = l1 * ip;
        const ido = @divExact(n, l2);
        const ipm = ip - 1;
        var j: i32 = 1;
        while (j <= ipm) : (j += 1) {
            var i = is;
            var fi: T = 0;
            ld += l1;
            const argld = @as(T, @floatFromInt(ld)) * argh;
            var ii: i32 = 3;
            while (ii <= ido) : (ii += 2) {
                i += 2;
                fi += 1;
                wa[@as(usize, @intCast(i - 2))] = @cos(fi * argld);
                wa[@as(usize, @intCast(i - 1))] = @sin(fi * argld);
            }
            is += ido;
        }
        l1 = l2;
    }
}

fn cffti1_ps(comptime T: type, n: i32, wa: [*]T, ifac: *[ifac_len]i32) void {
    const ntryh = [_]i32{ 5, 3, 4, 2, 0 };
    const nf = decompose(n, ifac, &ntryh);
    const argh = (2.0 * math.pi) / @as(T, @floatFromInt(n));
    var i: i32 = 1;
    var l1: i32 = 1;
    var k1: i32 = 1;
    while (k1 <= nf) : (k1 += 1) {
        const ip = ifac[@as(usize, @intCast(k1 + 1))];
        var ld: i32 = 0;
        const l2 = l1 * ip;
        const ido = @divExact(n, l2);
        const idot = ido + ido + 2;
        const ipm = ip - 1;
        var j: i32 = 1;
        while (j <= ipm) : (j += 1) {
            const im1 = i;
            var fi: T = 0;
            wa[@as(usize, @intCast(i - 1))] = 1;
            wa[@as(usize, @intCast(i))] = 0;
            ld += l1;
            const argld = @as(T, @floatFromInt(ld)) * argh;
            var ii: i32 = 4;
            while (ii <= idot) : (ii += 2) {
                i += 2;
                fi += 1;
                wa[@as(usize, @intCast(i - 1))] = @cos(fi * argld);
                wa[@as(usize, @intCast(i))] = @sin(fi * argld);
            }
            if (ip > 5) {
                wa[@as(usize, @intCast(im1 - 1))] = wa[@as(usize, @intCast(i - 1))];
                wa[@as(usize, @intCast(im1))] = wa[@as(usize, @intCast(i))];
            }
        }
        l1 = l2;
    }
}
