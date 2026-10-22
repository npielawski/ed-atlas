"""Writes copies of the trace files that contain the posterior samples only.

The traces written by train_mort.py and train_diag.py also hold the data the models were fitted on
(`observed_data`, `constant_data`) and the sampler statistics. Only `posterior/betas` is needed to make a
prediction, and the fitting data must not be published. For every trace_*.nc of the source folder this writes
a file with the same name to the output folder, holding the group `posterior` with `betas` and its
coordinates (`chain`, `draw`, `dims`) and the attributes of the original. The values are copied unchanged,
and each new file is read back and compared to the original.

    python strip_traces.py SOURCE_DIR OUTPUT_DIR [N_PROCESSES]
"""
import glob
import os
import sys
from multiprocessing import Pool

import netCDF4
import numpy as np


def copy_attrs(src, dst):
    for name in src.ncattrs():
        dst.setncattr(name, src.getncattr(name))


def strip(args):
    source, output = args
    final, output = output, output + ".part"
    with netCDF4.Dataset(source) as src:
        post = src.groups["posterior"]
        betas = post.variables["betas"][:]
        chain = post.variables["chain"][:]
        draw = post.variables["draw"][:]
        dims = [str(x) for x in post.variables["dims"][:]]
        with netCDF4.Dataset(output, "w", format="NETCDF4") as dst:
            copy_attrs(src, dst)
            g = dst.createGroup("posterior")
            copy_attrs(post, g)
            g.createDimension("chain", len(chain))
            g.createDimension("draw", len(draw))
            g.createDimension("dims", len(dims))
            g.createVariable("chain", "i8", ("chain",))[:] = chain
            g.createVariable("draw", "i8", ("draw",))[:] = draw
            labels = g.createVariable("dims", str, ("dims",))
            for i, label in enumerate(dims):
                labels[i] = label
            g.createVariable(
                "betas", "f8", ("chain", "draw", "dims"), zlib=True, complevel=4, fill_value=np.nan
            )[:] = betas
    # Read the new file back and compare it to the original.
    with netCDF4.Dataset(output) as new, netCDF4.Dataset(source) as old:
        assert list(new.groups) == ["posterior"], list(new.groups)
        assert list(new.groups["posterior"].variables) == ["chain", "draw", "dims", "betas"]
        assert np.array_equal(new.groups["posterior"].variables["betas"][:], old.groups["posterior"].variables["betas"][:])
        assert [str(x) for x in new.groups["posterior"].variables["dims"][:]] == dims
        assert np.array_equal(new.groups["posterior"].variables["chain"][:], chain)
        assert np.array_equal(new.groups["posterior"].variables["draw"][:], draw)
    os.replace(output, final)
    return os.path.basename(source), os.path.getsize(source), os.path.getsize(final)


if __name__ == "__main__":
    source_dir, output_dir = sys.argv[1], sys.argv[2]
    n_processes = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    os.makedirs(output_dir, exist_ok=True)
    jobs = [
        (path, os.path.join(output_dir, os.path.basename(path)))
        for path in sorted(glob.glob(os.path.join(source_dir, "trace*.nc")))
        if not os.path.exists(os.path.join(output_dir, os.path.basename(path)))
    ]
    print(f"{len(jobs)} files to write", flush=True)
    before = after = 0
    with Pool(n_processes) as pool:
        for i, (name, size_in, size_out) in enumerate(pool.imap_unordered(strip, jobs), 1):
            before += size_in
            after += size_out
            print(f"{i}/{len(jobs)} {name}: {size_in / 1e6:.0f} MB -> {size_out / 1e6:.0f} MB", flush=True)
    print(f"Done: {before / 1e9:.1f} GB -> {after / 1e9:.1f} GB")
