<div align="center">
    
# MATLAB Code for PhEVD Data Processing

</div>

---

# 🤓 General information

The `Two_people_machine_PhEVD.m` script is designed to perform signal processing on data from network analyzer measurements, specifically utilizing the s-parameters of a four-port network to analyze and extract various matrix operations. 

The script employs concepts such as the Fast Fourier Transform (FFT), matrix convolutions, and Para-hermitian Eigenvalue Decomposition (PhEVD) to process frequency-domain data.

- **Loading S-Parameter Data**:
        The script loads and processes .s4p (Touchstone) files, which are frequency-domain network parameters for multi-port networks.
        For each of the 320 files (file1), it uses sparameters to extract the s-parameter data from the corresponding file.

- **Matrix Initialization**:
        Arrays H, Hf, and R are initialized to store the processed data in both the time and frequency domains. These are 4x4 matrices with dimensionality based on the frequency points.

- **FFT and Matrix Convolution**:
        Each S-parameter matrix undergoes an Inverse FFT (IFFT) to convert the data from the frequency domain to the time domain.
        A matrix convolution is then applied to generate R matrices, which are further processed using parametric Hermitian operations (ParaHerm and PolyMatConv).

- **Eigenvalue Decomposition (EVD)**:
        The script applies Eigenvalue Decomposition (EVD) to each processed Polynomial matrix. The Eigenvalues are calculated for each frequency bin of the Fourier-transformed matrices.

- **Result Storage**:
        The resulting matrices, including Eigenvalues, are stored for further analysis. Arrays Lambda1, Lambda2, and EigVals hold these results.


---

# 💻 How to run the code

- Ensure all necessary .s4p files (numbered 1.s4p to 320.s4p) are available in the working directory.
  
- Ensure the directory contains: `ParaHerm.m`, `ParaHermDFT.m`, `ParaHermIDFT.m`, and `PolyMatConv.m`.
  
- Run `Two_people_machine_PhEVD.m` in MATLAB.
  
- The script processes all files and computes the required matrices and eigenvalues.
  
- The script generates a variety of matrices (H, Hf, R, Rf) and computes the eigenvalues, which are stored in EigVals.
    - These can be used for further analysis or visualized to interpret the system's behavior across the frequency spectrum.
    - **Notes**:
        - Modify the num variable if you are working with a different number of .s4p files.
        - The script uses complex matrix operations and requires sufficient computational resources, particularly memory, to handle large datasets.

---

# 🪛 Key Functions Used

- sparameters: Extracts S-parameters from Touchstone files.
  
- ifft: Performs Inverse Fast Fourier Transform.

- PolyMatConv: A custom function for polynomial matrix convolution.

- ParaHerm: Likely a custom function to apply Hermitian processing to matrices.

- eig: MATLAB’s built-in function for eigenvalue decomposition.

- ParaHermDFT: A function for parametric Hermitian-based Discrete Fourier Transform (DFT).



