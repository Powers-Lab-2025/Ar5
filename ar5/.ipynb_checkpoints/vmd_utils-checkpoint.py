import os
import tqdm


def generate_vmd_hopper_tcl(hopper_results, simtree_root, out_folder, use_dync=True, verbose=False):

    os.makedirs(out_folder, exist_ok=True)

    bar = tqdm.tqdm(hopper_results, desc="Generating hopper TCLs")

    for hopper in bar:
        mol = hopper['molecule']
        temp = hopper['temperature']
        frag_idx = hopper['fragment_index']

        combined_xyz_name = f"{mol}_ramp_combined_dync2.xyz" if use_dync else f"{mol}_ramp_combined.xyz"
        combined_xyz_path = os.path.join(simtree_root, mol, temp, combined_xyz_name)

        tcl_lines = []
        tcl_lines.append(f'mol new "{combined_xyz_path}" type xyz\n')

        tcl_lines.append('mol delrep 0 topz\n')
        tcl_lines.append('mol representation Lines\n')
        tcl_lines.append('mol color ColorID 0\n')
        tcl_lines.append('mol selection all\n')
        tcl_lines.append('mol addrep top\n')

        tcl_lines.append('mol representation Licorice 0.3 12.0 12.0\n')
        tcl_lines.append('mol color Name\n')
        tcl_lines.append(f'mol selection "fragment {frag_idx}"\n')
        tcl_lines.append('mol addrep top\n')

        tcl_filename = f"hopper_{mol}_{temp}_frag{frag_idx}.tcl"
        tcl_path = os.path.join(out_folder, tcl_filename)

        with open(tcl_path, 'w') as f:
            f.writelines(tcl_lines)

        if verbose:
            bar.set_postfix(molecule=mol, temp=temp, frag=frag_idx)