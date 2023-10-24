import argparse
import functools
import os
import pathlib
from multiprocessing.pool import ThreadPool

from tqdm import tqdm


''' 
Helper function for scripts that iterate over large sets of files. Defines command-line arguments
for operating over a large set of files, then handles setting up a worker queue system to operate
on those files. You need to provide your own process_file_fn.

process_file_fn expected signature:
 (path, output_path)
'''
def do_to_files(process_file_fn):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--glob')
    parser.add_argument('--out')
    parser.add_argument('--resume')
    parser.add_argument('--num_workers')

    args = parser.parse_args()
    src = args.path
    glob = args.glob
    out = args.out
    resume = args.resume
    num_workers = int(args.num_workers)

    path = pathlib.Path(src)
    files = path.rglob(glob)
    files = [str(f) for f in files]
    files = files[resume:]
    pfn = functools.partial(process_file_fn, output_path=out)
    if num_workers > 0:
        with ThreadPool(num_workers) as pool:
            list(tqdm(pool.imap(pfn, files), total=len(files)))
    else:
        for f in tqdm(files):
            pfn(f)
