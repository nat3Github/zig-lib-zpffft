const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const simd_lanes = b.option(u32, "simd_lanes", "SIMD lanes for sizes that allow it: 4, 8 or 16 (default: from target vector width)") orelse 0;

    const opts = b.addOptions();
    opts.addOption(u32, "simd_lanes", simd_lanes);

    const mod = b.addModule("pffft", .{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    mod.addOptions("build_options", opts);
}
