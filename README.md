# 🧬 ThromboGen: Multiscale Blood Clot Generation and Photoacoustic Simulation Framework

**ThromboGen** is an open-source MATLAB-based platform for generating realistic **3D blood clot microstructures** and performing **multiscale photoacoustic (PA) and optical simulations**.  
It integrates *stochastic fibrin–RBC–platelet network modeling*, *Monte Carlo light propagation (ValoMC)*, and *k-Wave acoustic simulations* into a unified computational workflow.

---

## 🔬 Overview

ThromboGen provides a unified pipeline for:
1. **Microstructure generation** — creates fibrin networks, RBCs, and platelets with realistic spatial and statistical properties.  
2. **Microscale optical-acoustic simulation** — computes local photoacoustic responses using voxelized 3D clot models.  
3. **Macroscale simulation** — performs full-sample photoacoustic signal propagation using k-Wave.  
4. **Postprocessing** — visualizes spectra, reconstructs 3D volumes, and extracts compositional and mechanical indicators.

This framework enables data-driven *in-silico* analysis of blood clot composition, morphology, and biomechanics for translational photoacoustic imaging research.

---

## 🧩 Repository Structure

```
ThromboGen/
│
├── Thrombogen.m                     # Main clot generator script
├── calculate_PA_response_micro.m    # Microscale photoacoustic simulation
├── run_macroscale_simulation.m      # k-Wave macroscale PA simulation
├── postprocess_results.m            # Visualization and analysis
│
├── /mex/                            # Compiled C++ MEX functions for speed
├── /data/                           # Sample .mat inputs or example datasets
├── /utils/                          # Helper functions (geometry, I/O, etc.)
├── /results/                        # Saved microstructures, spectra, and volumes
│
└── Multiscale_simulation/           # Complete pipeline integration (optional)
```

---

## ⚙️ Installation Requirements

### 1. MATLAB Toolboxes
Ensure the following MATLAB toolboxes are installed:
- **Image Processing Toolbox**  
- **Statistics and Machine Learning Toolbox**  
- **Parallel Computing Toolbox**  
- **Curve Fitting Toolbox** *(optional, for Gaussian or spline fitting)*

### 2. External Packages
| Package | Description | Link |
|----------|--------------|------|
| **k-Wave Toolbox** | Time-domain acoustic and photoacoustic simulations | [https://www.k-wave.org](https://www.k-wave.org) |
| **ValoMC** | Monte Carlo light transport simulation in voxel grids | [https://github.com/InverseLight/ValoMC](https://github.com/InverseLight/ValoMC) |
| **C++ MEX compiler** | Required for custom geometry and relaxation functions | Run `mex -setup C++` in MATLAB |

### 3. (Optional) Visualization
For 3D visualization and rendering:
- **Blender** ≥ 4.0 (for importing `.ply` models)
- **Paraview** or MATLAB’s plotting tools for 3D field visualization

---

## ⚡ Quick Start

%% Step 1 — Check dependencies
check the dependencies are installed and paths are correct

Open MATLAB and run the following commands step by step:

%% Step 2 — Generate a clot
Thrombogen;

%% Step 3 — Compute microscale photoacoustic response
calculate_PA_response_micro;

%% Step 4 — Run multiscale simulation
Multiscale_simulation;

%% Step 5 — Visualize and postprocess
Simulation_post_process;
```

### Example: Generate multiple random microstructures
```matlab
create_microstructure_bins
```

---

## 🧠 Workflow Description

### **1️⃣ Clot Microstructure Generation**
Run:
```matlab
Thrombogen
```
Creates fibrin–RBC–platelet microstructures and saves:
- `ClotMatrix.mat` — voxelized representation  
- Optional `.ply` meshes for fibrin strands and inclusions  

You can also generate a **pool of clots** by looping with different random seeds.

---

### **2️⃣ Microscale Photoacoustic Simulation**
```matlab
calculate_PA_response_micro
```
Simulates local optical absorption and initial PA pressure using **ValoMC** and **k-Wave**.

---

### **3️⃣ Macroscale (using our multiscale approach) Simulation**
```matlab
Multiscale_simulation.m
```
Uses **k-Wave** to simulate acoustic wave propagation across the entire sample volume.

---

### **4️⃣ Postprocessing**
```matlab
Simulation_post_process.m
```
- Reconstructs 3D PA volumes  
- Calculates frequency spectra  
- Speckle statestic analysis
- Generates visualizations for experimental comparison  

---

## 📊 Output Files

- `ClotMatrix.mat` — voxel-based 3D structure  
- `microscale PA responses` — microscale PA responses of voxel blocks averaged over its surfaces   
- `sensor_data` — macroscale simulation results on a 2D US detector plain
- `.ply` meshes — optional for rendering  
- Plots and figures (frequency-domain, volumetric reconstructions, etc.)
---

## 🧰 Notes & Tips

- Set a fixed random seed using `rng(seedValue)` for reproducible results.  
- The **`filling_factor`** and **`num_inclusions`** parameters control the fibrin-to-RBC ratio.   
- Export `.ply` models to Blender for realistic rendering.

---

## 🧪 Dependency Check Function

You can include this helper function to verify installations:

```matlab
function checkDependencies()
    fprintf('🔍 Checking MATLAB dependencies...\n');

    % MATLAB toolboxes
    requiredToolboxes = {'Image Processing Toolbox', 'Statistics and Machine Learning Toolbox', 'Parallel Computing Toolbox'};
    v = ver;
    installed = {v.Name};

    for i = 1:length(requiredToolboxes)
        if ~ismember(requiredToolboxes{i}, installed)
            warning('❌ %s is not installed.', requiredToolboxes{i});
        else
            fprintf('✅ %s found.\n', requiredToolboxes{i});
        end
    end

    % k-Wave check
    if exist('kspaceFirstOrder3D', 'file')
        fprintf('✅ k-Wave detected.\n');
    else
        warning('❌ k-Wave toolbox not found. Please install from https://www.k-wave.org');
    end

    % ValoMC check
    if exist('valomc', 'file')
        fprintf('✅ ValoMC detected.\n');
    else
        warning('❌ ValoMC not found. Please install from https://github.com/ssit/ValoMC');
    end
end
```

---

## 📖 Citation

If you use or expand this framework, please cite the following article:

> *A Multiscale Framework for In Silico 
Thrombus Generation and 
Photoacoustic Simulations*  
> Authors: [H.Ghodsi et al.]  
> Journal: [Journal of Physics Photonics]  
> Year: [2025]
> doi: *[To be added]*

---

## ⚖️ License

This project is released under the **MIT License**:

```
MIT License
Copyright (c) 2025 Fereidoonnezhad Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🤝 Contributions

Contributions are welcome!  
You may fork this repository, report issues, or submit pull requests to expand ThromboGen.  
For major updates, please open a discussion first.

---

## 📫 Contact

**Fereidoonnezhad Lab**  
📧 [h.ghodsi@tudelft.nl, b.fereidoonnezhad@tudelft.nl, S.Iskander-Rizk@tudelft.nl]  
🏛️ [TU Delft, Faculty of Mechanical Engineering]  
🌐 [https://github.com/FereidoonnezhadLab/ThromboGen](https://github.com/FereidoonnezhadLab/ThromboGen)

---

## 🧱 Acknowledgements

- [MATLAB and MathWorks Toolboxes](https://www.mathworks.com)  
- [k-Wave toolbox](https://www.k-wave.org)  
- [ValoMC](https://github.com/ssit/ValoMC)  
- [Blender](https://www.blender.org) for rendering and visualization  
