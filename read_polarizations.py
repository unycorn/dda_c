#!/usr/bin/env python3
import sys
import numpy as np

def read_polarizations_binary(filename):
    with open(filename, 'rb') as f:
        # Read N (4-byte integer)
        N = np.fromfile(f, dtype=np.int32, count=1)[0]
        
        # Read frequency (8-byte double)
        freq = np.fromfile(f, dtype=np.float64, count=1)[0]
        
        # Read complex doubles (2 components per point: Ex and Mz)
        # numpy complex128 matches C++ std::complex<double>
        data = np.fromfile(f, dtype=np.complex128, count=2*N)
        
        return N, freq, data.reshape(-1, 2)  # reshape to (N, 2) array

def write_polarizations_csv(data, freq, outfile=sys.stdout):
    # Write header with frequency as comment
    print(f"# Frequency: {freq:.4e} Hz", file=outfile)
    print("Re_px,Im_px,Re_mz,Im_mz", file=outfile)
    
    # Write data in scientific notation with 4 decimal places
    for ex, mz in data:
        print(f"{ex.real:.4e},{ex.imag:.4e},{mz.real:.4e},{mz.imag:.4e}", 
              file=outfile)

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <binary_file>", file=sys.stderr)
        sys.exit(1)
    
    try:
        N, freq, data = read_polarizations_binary(sys.argv[1])
        write_polarizations_csv(data, freq)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()