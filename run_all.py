"""Regenerate every figure and table:  python run_all.py   (add --quick to skip n=80 in Table 1.2)"""
import subprocess
import sys

SCRIPTS = [
    ["fig1_1_pentagon_and_its_average.py"],
    ["fig1_2_repeated_averaging.py"],
    ["fig1_3_normalized_iteration.py"],
    ["fig2_1_eigenvalues_of_M5.py"],
    ["fig3_1_even_and_odd_polygons.py"],
    ["fig4_1_tilted_ellipse.py"],
    ["table1_1_k_delta.py"],
    ["table1_2_k_uncross.py"] + (["--ns", "10", "20", "40"] if "--quick" in sys.argv else []),
    ["table2_1_damping_factor.py"],
    ["check_theorems.py"],
]

if __name__ == "__main__":
    for script in SCRIPTS:
        print(f"\n$ python {' '.join(script)}")
        subprocess.run([sys.executable, *script], check=True)
    print("\nDone: see figures/ and tables/")
