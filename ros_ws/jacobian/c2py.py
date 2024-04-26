import re
import sys
import warnings


def main():
    # convert the C code into Python code
    for line in sys.stdin:

        line = line.strip()

        # Get rid of type declarations.
        line = line.replace("self->", "")
        for s in ["vec", "quat", "mat33"]:
            line = line.replace(f"struct {s} ", "")
        line = line.replace("float ", "")

        # Struct element accesses become individual variables.
        line = line.replace("->", "_")
        # line = line.replace(".", "_")

        # Dual array indices become multidimentional indices.
        # (SymPy weirdness - wouldn't be necesary to generate NumPy code.)
        line = re.sub(
            ".m\[(\d)\]\[(\d)\]",
            lambda m: f"[{m.group(1)}, {m.group(2)}]",
            line
        )

        # Remove "f" suffix on float literals.
        line = re.sub(f"([0-9]*\.[0-9]*)f", lambda m: m.group(1), line)

        # Remove comments and empty lines.
        if line.startswith("//"):
            continue
        if line == "":
            continue

        # Now we are left with only assignment statements.
        assert line[-1] == ";"
        line = line[:-1]
        lhs, rhs = [s.strip() for s in line.split("=")]

        # Create temporary vars to break up jacobians, otherwise SymPy hangs.
        # TODO: investigate if it's needed...
        if lhs == "zdes":
            print(f"zdes_old = {rhs}")
            print("zdes = symvec('zdes')")
            continue

        print(f"{lhs} = {rhs}")


if __name__ == "__main__":
    main()
