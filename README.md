# zpffft

Pure-Zig port of [PFFFT](https://github.com/marton78/pffft) (a pretty fast FFT) for Zig 0.16:
real and complex 1D transforms in `f32` and `f64`, SIMD via `@Vector` (4 lanes, plus 8/16-lane
paths on wide targets). No C, no libc, no dependencies.

## Use

```sh
zig fetch --save git+https://github.com/nat3Github/zig-lib-zpffft#<commit>
```

```zig
// build.zig
const zpffft = b.dependency("zpffft", .{ .target = target, .optimize = optimize });
exe.root_module.addImport("pffft", zpffft.module("pffft"));
```

```zig
const pffft = @import("pffft");

const setup = try pffft.Float32.Setup.init(allocator, 1024, .Real);
defer setup.deinit();
// input/output: 1024 floats (64-byte aligned is fastest). work = null allocates a
// scratch buffer per call (panics on OOM); pass a 1024-float (complex: 2048) buffer to avoid it.
pffft.Float32.transformOrdered(setup, input, output, null, .Forward);
```

`transform` produces PFFFT's internal (unordered) layout, which is enough for fast convolution;
`transformOrdered` produces the canonical order. Transforms are unscaled: forward then backward
multiplies by N. Valid sizes are those of PFFFT (N = 2^a·3^b·5^c with a minimum per transform type).

Build option: `-Dsimd_lanes=4|8|16` forces the wide-path lane count (default: target vector width).

## License

BSD-like, see [LICENSE.txt](LICENSE.txt): the Zig port keeps the license of the original PFFFT
(Julien Pommier, Hayati Ayguen, Dario Mambro), which is based on FFTPACKv4 by Paul Swarztrauber
(NCAR) and carries the FFTPACKv5 license of UCAR. Redistributions must retain the notices and
disclaimer in LICENSE.txt.
