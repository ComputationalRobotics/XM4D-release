import numpy as np


def save_matrix_to_bin(filename, matrix):
    try:
        with open(filename, "wb") as file:
            rows, cols = matrix.shape
            # 4-byte int header
            file.write(rows.to_bytes(4, byteorder='little'))
            file.write(cols.to_bytes(4, byteorder='little'))

            matrix.T.tofile(file)

            print(f"Matrix saved to {filename}. Rows: {rows}, Cols: {cols}")
    except IOError:
        print("Cannot write to file")

def load_matrix_from_bin(filename):
    try:
        with open(filename, "rb") as file:
            rows = int.from_bytes(file.read(4), byteorder='little')
            cols = int.from_bytes(file.read(4), byteorder='little')

            n = rows

            matrix = np.fromfile(file, dtype=np.double, count=rows * cols)
            matrix = matrix.reshape((rows, cols), order='F')

            print(f"rows: {rows}, cols: {cols}")
            return matrix, n
    except IOError:
        print("Cannot open file")
        return None, None


def savez_compressed_fast(file, compresslevel=1, **arrays):
    """np.savez_compressed with a zlib level (numpy always uses the default 6): the same archive (one zip64
    <key>.npy member per key, object arrays pickled), so np.load reads the same arrays. Level 1 writes the
    float tracks ~4x faster for a ~4% larger file."""
    import os
    import zipfile
    file = os.fspath(file)
    if not file.endswith(".npz"):
        file = file + ".npz"
    with zipfile.ZipFile(file, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=compresslevel,
                         allowZip64=True) as zipf:
        for key, val in arrays.items():
            with zipf.open(key + ".npy", "w", force_zip64=True) as fid:
                np.lib.format.write_array(fid, np.asanyarray(val), allow_pickle=True)