import torch
import torch.nn.functional as Fun
import math

# --- Reference Implementation using torch.stft/torch.istft ---
class STFTISTFTReference:
    def __init__(self, n_fft, hop_length, win_length, window):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window # Shape: [n_fft] or [win_length]

    def _stft(self, x):
        # x: [B, 1, T] -> squeeze channel dim for torch.stft
        x_squeezed = x.squeeze(1) # [B, T]
        spec = torch.stft(
            x_squeezed,
            self.n_fft,
            self.hop_length,
            self.win_length,
            window=self.window.to(x.device),
            return_complex=True
        )
        # spec: [B, F, T]
        spec = torch.view_as_real(spec)  # [B, F, T, 2]
        return spec[..., 0], spec[..., 1] # real: [B, F, T], imag: [B, F, T]

    def _istft(self, magnitude, phase):
        # magnitude: [B, F, T]
        # phase: [B, F, T]
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        complex_spec = torch.complex(real, img) # [B, F, T]
        # complex_spec: [B, F, T]
        inverse_transform = torch.istft(
            complex_spec,
            self.n_fft,
            self.hop_length,
            self.win_length,
            window=self.window.to(magnitude.device),
            length=24000 # Specify length to match original
        )
        return inverse_transform # [B, 24000]

class STFTISTFTReplacerManualFFT:
    def __init__(self, n_fft, hop_length, win_length, window):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        # Store window, ensure it's the right shape [1, 1, n_fft] for broadcasting
        # We assume window is 1D [win_length] and win_length == n_fft
        self.window = window.view(1, 1, -1) # Shape: [1, 1, n_fft]

        # --- Precompute DFT matrices for manual FFT/IFFT ---
        # 1. DFT matrix for rfft (only positive frequencies)
        # W_rfft[k, n] = exp(-2j * pi * k * n / N) for k=0..N//2, n=0..N-1
        N = n_fft
        k_indices = torch.arange(N // 2 + 1).view(-1, 1) # [N//2 + 1, 1]
        n_indices = torch.arange(N).view(1, -1)          # [1, N]
        freqs = 2j * math.pi * k_indices * n_indices / N
        self.W_rfft = torch.exp(-freqs) # [N//2 + 1, N] complex64

        # Pre-calculate constants for ISTFT
        self.pad_amount = n_fft // 2
        self.target_length = 24000 # Known fixed length from problem statement


    def _stft(self, x):
        """Manual STFT implementation for fixed parameters."""
        print("x",x.shape)
        B, C, T = x.shape
        n_fft = self.n_fft
        hop_length = self.hop_length
        window = self.window.to(x.device) # [1, 1, n_fft]
        W_rfft = self.W_rfft.to(x.device) # [F, n_fft] complex64

        # 1. Padding for center=True (constant padding)
        pad_amount = self.pad_amount
        x_padded = Fun.pad(x, (pad_amount, pad_amount), mode='constant', value=0) # [B, C, T + 2*pad]

        # 2. Unfold to create frames
        frames = x_padded.unfold(-1, n_fft, hop_length) # [B, C, T_frames, n_fft]

        # 3. Apply window (broadcasting)
        windowed_frames = frames * window # [B, C, T_frames, n_fft]

        # 4. Apply manual RFFT using precomputed DFT matrix
        BCT_frames = B * C * frames.shape[2]
        windowed_frames_flat = windowed_frames.view(BCT_frames, n_fft) # [B*C*T_frames, n_fft] (real)
        windowed_frames_flat_complex = torch.complex(windowed_frames_flat, torch.zeros_like(windowed_frames_flat)) # [B*C*T_frames, n_fft] (complex)
        stft_flat = torch.matmul(windowed_frames_flat_complex, W_rfft.t()) # [B*C*T_frames, F] (complex)
        stft_result = stft_flat.view(B, C, frames.shape[2], n_fft // 2 + 1) # [B, C, T_frames, F] (complex)

        # 5. Transpose to match torch.stft output format [B, C, F, T] and squeeze channel
        stft_result = stft_result.transpose(-1, -2) # [B, C, F, T_frames] (complex)
        stft_result_squeezed = stft_result.squeeze(1) # [B, F, T_frames] (complex)

        # 6. Separate real and imaginary parts
        real_part = stft_result_squeezed.real # [B, F, T_frames] (real)
        imag_part = stft_result_squeezed.imag # [B, F, T_frames] (real)

        return real_part, imag_part

    def _istft(self, magnitude, phase):
        """
        Manual ISTFT implementation matching torch.istft logic for fixed parameters,
        WITHOUT using torch.fft operators. Uses basic PyTorch ops only.
        Input magnitude: [B, F, T_frames] = [1, 9, 6001]
        Input phase: [B, F, T_frames] = [1, 9, 6001]
        Output: [B, T] = [1, 24000]
        """
        B, F, T_frames = magnitude.shape
        n_fft = self.n_fft
        hop_length = self.hop_length
        # window: [1, 1, n_fft] -> Squeeze for easier handling in calculations
        window = self.window.squeeze().to(magnitude.device) # [n_fft]
        N = n_fft
        target_length = self.target_length # 24000
        pad_amount = self.pad_amount # n_fft // 2 = 8

        # --- 1. Reconstruct complex single-sided spectrogram ---
        magnitude = torch.clip(magnitude, max=1e2)
        real_part = magnitude * torch.cos(phase)       # [B, F, T_frames]
        imag_part = magnitude * torch.sin(phase)       # [B, F, T_frames]
        # complex_spec_single = torch.complex(real_part, imag_part) # [B, F, T_frames]
        # We avoid torch.complex, work directly with real/imag parts

        # --- 2. Extend spectrum to full (for manual IDFT) ---
        # complex_spec_single: [B, F, T_frames], F = N//2 + 1 = 9
        if F != N // 2 + 1:
            raise ValueError(f"Input F ({F}) does not match expected N//2+1 ({N//2+1}) for onesided=True")

        # Construct full spectrum [B, N, T_frames] from single-sided [B, F, T_frames]
        # Using conjugate symmetry: X_full[k] = conj(X_full[N-k]) for k = 1, ..., N//2-1
        # X_full[0] = X_single[0] (DC - real)
        # X_full[k] = X_single[k] for k = 1, ..., N//2-1 (Positive frequencies)
        # X_full[N//2] = X_single[N//2] (Nyquist - real)
        # X_full[k] = conj(X_single[N-k]) for k = N//2+1, ..., N-1 (Negative frequencies)
        # Note: X_single indices [0, 1, 2, 3, 4, 5, 6, 7, 8] correspond to frequencies [0, 1, 2, 3, 4, 5, 6, 7, 8]
        # We need to place conj(X_single[7]), ..., conj(X_single[1]) at indices [9, 10, 11, 12, 13, 14, 15]

        # Initialize full spectrum tensors for real and imaginary parts
        X_full_real = torch.zeros(B, N, T_frames, dtype=real_part.dtype, device=real_part.device) # [B, N, T_frames]
        X_full_imag = torch.zeros(B, N, T_frames, dtype=imag_part.dtype, device=imag_part.device) # [B, N, T_frames]

        # Copy the single-sided spectrum (frequencies 0 to N//2) to positive frequencies
        X_full_real[:, :F, :] = real_part # [B, 9, T_frames] -> indices [0..8]
        X_full_imag[:, :F, :] = imag_part # [B, 9, T_frames] -> indices [0..8]

        # Copy the conjugate symmetric part (frequencies N//2+1 to N-1)
        # Take positive frequencies 1 to N//2-1 (indices 1 to 7 in single-sided input)
        # Their indices in single-sided input are 1 to F-2 = 1 to 7
        # Their conjugates (negating imaginary part) go to indices N-1 down to N//2+1
        # Which are indices 15 down to 9 in the full spectrum (N=16)
        # conj(X[k]) = (real(X[k]), -imag(X[k]))
        if F > 2:
            # --- CORRECTED APPROACH: Avoid negative step in slice for assignment ---
            # 1. Identify the slice of positive frequencies (1 to F-2) in the single-sided input
            pos_freq_indices = slice(1, F-1) # slice(1, 8) -> indices 1, 2, ..., 7

            # 2. Extract the corresponding real and imaginary parts
            # real_part[:, pos_freq_indices, :] -> [B, F-2, T_frames] = [B, 7, T_frames]
            # imag_part[:, pos_freq_indices, :] -> [B, F-2, T_frames] = [B, 7, T_frames]
            pos_real_part = real_part[:, pos_freq_indices, :]   # [B, 7, T_frames]
            pos_imag_part = imag_part[:, pos_freq_indices, :]   # [B, 7, T_frames]

            # 3. Flip these parts along the frequency dimension to match the order of negative frequencies
            # Flipping [X[1], X[2], ..., X[7]] -> [X[7], X[6], ..., X[1]]
            flipped_pos_real_part = torch.flip(pos_real_part, dims=[1]) # [B, 7, T_frames]
            flipped_pos_imag_part = torch.flip(pos_imag_part, dims=[1]) # [B, 7, T_frames]

            # 4. Identify the slice of negative frequencies in the full spectrum
            # Indices N-1 down to N//2+1 -> Indices 15 down to 9 for N=16
            # We need to assign to indices [9, 10, ..., 15] which is slice(9, 16) or slice(N//2 + 1, N)
            neg_freq_indices_full = slice(N // 2 + 1, N) # slice(9, 16) -> indices 9, 10, ..., 15

            # 5. Assign the flipped (conjugated) parts to the negative frequency slots
            # Real part stays the same under conjugation
            X_full_real[:, neg_freq_indices_full, :] = flipped_pos_real_part # [B, 7, T_frames] assigned to indices [9..15]
            # Imaginary part is negated under conjugation
            X_full_imag[:, neg_freq_indices_full, :] = -flipped_pos_imag_part # [B, 7, T_frames] assigned to indices [9..15]

        # --- 3. Apply manual IDFT to each frame ---
        # IDFT formula: x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(2j*pi*k*n/N)
        # For real output, x[n] = (1/N) * sum_{k=0}^{N-1} (X_real[k] + j*X_imag[k]) * (cos(2pi*k*n/N) + j*sin(2pi*k*n/N))
        # Real{x[n]} = (1/N) * sum_{k=0}^{N-1} [ X_real[k]*cos(...) - X_imag[k]*sin(...) ]
        # We compute the real part of the IDFT result directly.
        # Precompute IDFT twiddle factors (real part only, as we expect real output)
        # W_idft_real[k, n] = cos(2*pi*k*n/N)
        # W_idft_imag[k, n] = sin(2*pi*k*n/N) # Needed for the full complex multiplication
        k_indices_idft = torch.arange(N, dtype=torch.float32, device=X_full_real.device).view(N, 1) # [N, 1]
        n_indices_idft = torch.arange(N, dtype=torch.float32, device=X_full_real.device).view(1, N) # [1, N]
        angle = 2 * torch.pi * k_indices_idft * n_indices_idft / N # [N, N]
        W_idft_real = torch.cos(angle) # [N, N]
        W_idft_imag = torch.sin(angle) # [N, N]

        # Reshape for batched matrix multiplication
        # X_full_real: [B, N, T_frames] -> [B * T_frames, N]
        # X_full_imag: [B, N, T_frames] -> [B * T_frames, N]
        BT_frames = B * T_frames
        X_full_real_flat = X_full_real.transpose(1, 2).contiguous().view(BT_frames, N) # [B*T_frames, N]
        X_full_imag_flat = X_full_imag.transpose(1, 2).contiguous().view(BT_frames, N) # [B*T_frames, N]

        # Perform IDFT calculation: real part of (X * W_idft)
        # (X_real + j*X_imag) * (W_real + j*W_imag) = (X_real*W_real - X_imag*W_imag) + j*(...)
        # We only need the real part: (X_real*W_real - X_imag*W_imag)
        # Matrix multiply: [BT, N] @ [N, N] -> [BT, N]
        time_frames_real_part1 = torch.matmul(X_full_real_flat, W_idft_real.t()) # [BT, N]
        time_frames_real_part2 = torch.matmul(X_full_imag_flat, W_idft_imag.t()) # [BT, N]
        time_frames_real_flat = (time_frames_real_part1 - time_frames_real_part2) / N # [BT, N]
        # Reshape back to [B, T_frames, N] and then transpose to [B, N, T_frames]
        time_frames_real = time_frames_real_flat.view(B, T_frames, N).transpose(1, 2) # [B, N, T_frames]

        # --- 4. Apply window ---
        # time_frames_real: [B, N, T_frames]
        # window: [n_fft]
        windowed_time_frames = time_frames_real * window.unsqueeze(0).unsqueeze(-1) # [B, N, T_frames]

        # --- 5. Overlap-Add using fold ---
        # windowed_time_frames: [B, N, T_frames]
        # Prepare for fold: [B, N, T_frames] is already the correct shape [B, C*kernel_size[1], L]
        # where C=1 (implicitly), kernel_size[1]=N, L=T_frames.
        # output_size = (1, padded_output_length)
        # kernel_size = (1, N)
        # stride = (1, hop_length)
        fold_input = windowed_time_frames # [B, N, T_frames]

        # Calculate padded output length (before unpadding)
        # This is the length of the signal after overlap-add, before removing center padding.
        # It corresponds to the range covered by all frames.
        # Last frame starts at (T_frames - 1) * hop_length.
        # It spans indices [(T_frames - 1) * hop_length, ..., (T_frames - 1) * hop_length + N - 1].
        # So the maximum index written is (T_frames - 1) * hop_length + N - 1.
        # The total length (0-based indexing) needed is (T_frames - 1) * hop_length + N.
        padded_output_length = (T_frames - 1) * hop_length + N # e.g., (6001-1)*4 + 16 = 24016

        # Perform overlap-add with fold
        signal_padded = Fun.fold(
            fold_input, # [B, N, T_frames]
            output_size=(1, padded_output_length), # e.g., (1, 24016)
            kernel_size=(1, N),                   # e.g., (1, 16)
            stride=(1, hop_length)                # e.g., (1, 4)
        ) # Output shape: [B, 1, 1, padded_output_length]
        signal_padded = signal_padded.squeeze(2) # [B, 1, padded_output_length]

        # --- 6. Calculate and apply NOLA normalization (using direct summation) ---
        # Calculate the NOLA sum: sum of squared window values for each time point
        # We simulate the overlap-add process used by fold, but just summing window^2 contributions.
        nola_sum_padded = torch.zeros(padded_output_length, dtype=torch.float32, device=window.device) # [padded_output_length]
        window_sq = window.pow(2) # [N]

        # Iterate through each frame and add its window^2 contribution to the NOLA sum
        for t in range(T_frames):
            start_idx = t * hop_length
            end_idx = start_idx + N
            # Add window_sq to the corresponding slice of nola_sum_padded
            if end_idx <= padded_output_length:
                nola_sum_padded[start_idx:end_idx] += window_sq
            else:
                # Handle case where frame extends beyond (shouldn't happen with correct calc, but safe)
                nola_sum_padded[start_idx:] += window_sq[:padded_output_length - start_idx]

        # Add small epsilon to avoid division by zero and reshape for broadcasting
        nola_sum_shaped = nola_sum_padded.view(1, 1, -1) + 1e-12 # [1, 1, padded_output_length] e.g., [1, 1, 24016]

        # Apply NOLA normalization
        signal_normalized_padded = signal_padded / nola_sum_shaped # [B, 1, padded_output_length]

        # --- 7. Remove padding (center=True) to match specified length ---
        # The original signal length was target_length (24000).
        # Padding of pad_amount (n_fft//2 = 8) was added to both sides during the conceptual STFT.
        # So, the valid signal in signal_normalized_padded is from index pad_amount
        # to index pad_amount + target_length - 1.
        # We slice to get exactly target_length points.
        if pad_amount > 0:
            # Slice from index pad_amount to pad_amount + target_length
            inverse_transform = signal_normalized_padded[:, :, pad_amount:pad_amount + target_length] # [B, 1, 24000]
        else:
            inverse_transform = signal_normalized_padded[:, :, :target_length]

        return inverse_transform.squeeze(1) # [B, 24000] - Final output shape

if __name__ == "__main__":

    # --- Example Usage ---
    # (Assuming the classes STFTISTFTReference and the test code are defined elsewhere or similar)
    # Define fixed parameters matching your original setup
    n_fft = 16
    hop_length = 4
    win_length = 16
    # Create a sample window (e.g., Hann window)
    window = torch.hann_window(win_length) # Shape: [16]

    # Create instances
    manual_impl = STFTISTFTReplacerManualFFT(n_fft, hop_length, win_length, window)

    # Dummy input signal (Batch=1, Channel=1, Time=24000)
    x = torch.randn(1, 1, 24000)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    window = window.to(device) # Move window to the same device
    manual_impl.window = window.view(1, 1, -1) # Update window in instance

    # --- Test _istft ---
    print("Testing _istft...")
    # --- Use the manual STFT result to test ISTFTs ---
    real_part_manual, imag_part_manual = manual_impl._stft(x)
    complex_spec_manual = torch.complex(real_part_manual, imag_part_manual) # [B, F, T] = [1, 9, 6001]
    magnitude_test = torch.abs(complex_spec_manual)  # [B, F, T] = [1, 9, 6001]
    phase_test = torch.angle(complex_spec_manual)    # [B, F, T] = [1, 9, 6001]

    # --- Run Manual ISTFT ---
    reconstructed_x_manual = manual_impl._istft(magnitude_test, phase_test)
    print(f"Manual _istft output shape: {reconstructed_x_manual.shape}")
    # Expected: [1, 24000]

    # --- Compare with original signal (Full Cycle Test) ---
    diff_full_cycle = torch.abs(x.squeeze(1) - reconstructed_x_manual).mean() # x is [B, 1, T], reconstructed is [B, T]
    print(f"Mean absolute difference (Full Cycle - Original vs Reconstructed): {diff_full_cycle.item()}")
    print("This shows the error introduced by the full manual pipeline.")
    print("-" * 30)

    # --- If you have the reference implementation for comparison ---
    # diff_recon_vs_ref = torch.abs(reconstructed_x_manual - reconstructed_x_ref).mean() # Assuming reconstructed_x_ref exists
    # print(f"Mean absolute difference (Manual ISTFT vs Reference ISTFT): {diff_recon_vs_ref.item()}")



    # --- Example Usage ---
    # Define fixed parameters matching your original setup
    n_fft = 16
    hop_length = 4
    win_length = 16
    # Create a sample window (e.g., Hann window) - Ensure it's on the correct device later
    window = torch.hann_window(win_length) # Shape: [16]

    # Create instances
    reference_impl = STFTISTFTReference(n_fft, hop_length, win_length, window)
    manual_impl = STFTISTFTReplacerManualFFT(n_fft, hop_length, win_length, window)

    # Dummy input signal (Batch=1, Channel=1, Time=24000)
    x = torch.randn(1, 1, 24000)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    window = window.to(device) # Move window to the same device

    # Update instances with device-corrected window
    reference_impl.window = window
    manual_impl.window = window.view(1, 1, -1)

    print("Testing _stft...")
    # --- Test _stft ---
    real_part_ref, imag_part_ref = reference_impl._stft(x)
    print(f"Reference _stft output shapes: real={real_part_ref.shape}, imag={imag_part_ref.shape}")
    # Expected: [1, 9, 6001], [1, 9, 6001]

    real_part_manual, imag_part_manual = manual_impl._stft(x)
    print(f"Manual _stft output shapes: real={real_part_manual.shape}, imag={imag_part_manual.shape}")
    # Expected: [1, 9, 6001], [1, 9, 6001]

    # Calculate difference in STFT outputs
    diff_real_stft = torch.abs(real_part_manual - real_part_ref).mean()
    diff_imag_stft = torch.abs(imag_part_manual - imag_part_ref).mean()
    print(f"Mean absolute difference (STFT Real part): {diff_real_stft.item()}")
    print(f"Mean absolute difference (STFT Imaginary part): {diff_imag_stft.item()}")
    print("If differences are small (e.g., < 1e-4), manual STFT is likely correct.")
    print("-" * 30)

    # --- Test _istft ---
    print("Testing _istft...")
    # --- CORRECT WAY: Use the reference STFT result to test ISTFTs ---
    # First, get the complex spectrogram from reference STFT
    complex_spec_ref = torch.complex(real_part_ref, imag_part_ref) # [B, F, T] = [1, 9, 6001]
    # Then, calculate the correct magnitude and phase
    magnitude_test = torch.abs(complex_spec_ref)  # [B, F, T] = [1, 9, 6001]
    phase_test = torch.angle(complex_spec_ref)    # [B, F, T] = [1, 9, 6001]
    print(f"Correct magnitude shape: {magnitude_test.shape}")
    print(f"Correct phase shape: {phase_test.shape}")

    # --- Run Reference ISTFT ---
    reconstructed_x_ref = reference_impl._istft(magnitude_test, phase_test)
    print(f"Reference _istft output shape: {reconstructed_x_ref.shape}")
    # Expected: [1, 24000]

    # --- Run Manual ISTFT ---
    reconstructed_x_manual = manual_impl._istft(magnitude_test, phase_test)
    print(f"Manual _istft output shape: {reconstructed_x_manual.shape}")
    # Expected: [1, 24000]

    # --- Compare ISTFT outputs ---
    diff_recon = torch.abs(reconstructed_x_manual - reconstructed_x_ref).mean()
    print(f"Mean absolute difference (ISTFT Reconstructed signal): {diff_recon.item()}")
    # This should now be much smaller.
    print("If difference is small (e.g., < 1e-3), manual ISTFT matches the reference implementation.")
    print("-" * 30)

    # --- Optional: Full Cycle Test (STFT -> ISTFT) ---
    print("Testing full cycle (Manual STFT -> Manual ISTFT)...")
    # Get magnitude and phase from manual STFT
    mag_manual = torch.abs(torch.complex(real_part_manual, imag_part_manual))
    phs_manual = torch.angle(torch.complex(real_part_manual, imag_part_manual))
    # Reconstruct using manual ISTFT
    x_reconstructed_full_cycle = manual_impl._istft(mag_manual, phs_manual)
    # Compare with original input
    diff_full_cycle = torch.abs(x - x_reconstructed_full_cycle.unsqueeze(1)).mean() # x is [B, 1, T]
    print(f"Mean absolute difference (Full Cycle - Original vs Reconstructed): {diff_full_cycle.item()}")
    print("This shows the error introduced by the full manual pipeline.")
    print("-" * 30)