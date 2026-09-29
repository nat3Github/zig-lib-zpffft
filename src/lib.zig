const std = @import("std");
pub const setup = @import("setup.zig");
pub const zpfft = @import("zpfft.zig");
pub const fast_conv = @import("fast_conv.zig");
pub const simd = @import("simd_math.zig");
pub const radix_passes = @import("radix_passes.zig");

pub const TransformType = setup.TransformType;
pub const Direction = setup.Direction;

pub const Float32 = struct {
    pub const Setup = setup.Setup(f32);
    pub const ZPFFT = zpfft.ZPFFT(f32);
    
    pub const transform = ZPFFT.transform;
    pub const transformOrdered = ZPFFT.transformOrdered;
    pub const zreorder = ZPFFT.zreorder;
};

pub const Float64 = struct {
    pub const Setup = setup.Setup(f64);
    pub const ZPFFT = zpfft.ZPFFT(f64);
    
    pub const transform = ZPFFT.transform;
    pub const transformOrdered = ZPFFT.transformOrdered;
    pub const zreorder = ZPFFT.zreorder;
};

// Default to f32 for backward compatibility if needed, 
// but it's cleaner to be explicit.
pub const Setup = Float32.Setup;
pub const transform = Float32.transform;
pub const transformOrdered = Float32.transformOrdered;
pub const zreorder = Float32.zreorder;

