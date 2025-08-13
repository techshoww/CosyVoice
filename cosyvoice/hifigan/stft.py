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
        # Precompute real and imaginary parts for efficiency (avoids repeated .real/.imag calls)
        self.W_rfft_real = self.W_rfft.real # [N//2 + 1, N] (real)
        self.W_rfft_imag = self.W_rfft.imag # [N//2 + 1, N] (real)

        # Pre-calculate constants for ISTFT
        self.pad_amount = n_fft // 2
        self.target_length = 24000 # Known fixed length from problem statement


    def _stft(self, x):
        """Manual STFT implementation for fixed parameters, using indexing instead of unfold, avoiding torch.complex."""
        # print("x", x.shape)
        B, C, T = x.shape
        n_fft = self.n_fft
        hop_length = self.hop_length
        window = self.window.to(x.device)  # [1, 1, n_fft]
        # W_rfft = self.W_rfft.to(x.device)  # [F, n_fft] complex64
        # Precompute real and imaginary parts of the DFT matrix for manual multiplication
        W_rfft_real = self.W_rfft_real.to(x.device) # [F, n_fft] (real)
        W_rfft_imag = self.W_rfft_imag.to(x.device) # [F, n_fft] (real)

        # 1. Padding for center=True (constant padding)
        pad_amount = self.pad_amount
        x_padded = Fun.pad(x, (pad_amount, pad_amount), mode='reflect')  # [B, C, T_padded = 24000 + 16 = 24016]
        T_padded = x_padded.shape[-1]
        # print(f"After padding: {x_padded.shape}")

        # 2. Manually create frames using advanced indexing (replacing unfold)
        # Calculate number of frames based on torch.stft's logic with center=True
        T_padded = x_padded.shape[-1]
        T_frames = 1 + (T_padded - n_fft) // hop_length
        # print(f"Number of frames (T_frames): {T_frames}")

        # --- Use advanced indexing ---
        # We want to gather elements for each frame.
        # Frame t takes elements from x_padded at indices [t*hop, t*hop+1, ..., t*hop+n_fft-1]
        # We can create index tensors for each dimension of x_padded: [B, C, T_padded]

        # Dimension 0 (Batch): indices are 0, 1, ..., B-1, repeated for all C and n_fft elements in a frame
        batch_indices = torch.arange(B, device=x.device).view(B, 1, 1, 1) # [B, 1, 1, 1]
        batch_indices = batch_indices.expand(B, C, T_frames, n_fft)      # [B, C, T_frames, n_fft]

        # Dimension 1 (Channel): indices are 0, 1, ..., C-1, repeated for all B and n_fft elements in a frame
        channel_indices = torch.arange(C, device=x.device).view(1, C, 1, 1) # [1, C, 1, 1]
        channel_indices = channel_indices.expand(B, C, T_frames, n_fft)    # [B, C, T_frames, n_fft]

        # Dimension 2 (Time): indices are [t*hop, t*hop+1, ..., t*hop+n_fft-1] for each frame t
        # Create base offsets for each frame: [0, hop, 2*hop, ..., (T_frames-1)*hop]
        frame_starts = torch.arange(T_frames, device=x.device) * hop_length # [T_frames]
        # Create offsets within a window: [0, 1, 2, ..., n_fft-1]
        window_offsets = torch.arange(n_fft, device=x.device)              # [n_fft]
        # Combine them to get indices for the time dimension: [T_frames, n_fft]
        # Add a dimension to frame_starts to allow broadcasting: [T_frames, 1] + [n_fft] -> [T_frames, n_fft]
        time_indices_base = frame_starts.view(T_frames, 1) + window_offsets # [T_frames, n_fft]
        # Expand to match batch and channel dimensions: [1, 1, T_frames, n_fft] -> [B, C, T_frames, n_fft]
        time_indices = time_indices_base.unsqueeze(0).unsqueeze(0).expand(B, C, T_frames, n_fft) # [B, C, T_frames, n_fft]

        # Now use advanced indexing to gather the frames
        # x_padded: [B, C, T_padded]
        # batch_indices, channel_indices, time_indices: [B, C, T_frames, n_fft]
        frames = x_padded[batch_indices, channel_indices, time_indices] # [B, C, T_frames, n_fft]
        # print(f"Frames shape (after manual indexing): {frames.shape}")
        # --- END manual unfold ---

        # 3. Apply window (broadcasting)
        # Ensure window has the correct shape for broadcasting [1, 1, 1, n_fft]
        window_for_broadcast = window.view(1, 1, 1, n_fft) # [1, 1, 1, n_fft]
        windowed_frames = frames * window_for_broadcast  # [B, C, T_frames, n_fft]
        # print(f"Windowed frames shape: {windowed_frames.shape}")

        # 4. Apply manual RFFT using precomputed DFT matrix (WITHOUT torch.complex)
        # windowed_frames_flat: [B*C*T_frames, n_fft] (real)
        # W_rfft: [F, n_fft] (complex) -> Precomputed as self.W_rfft = torch.exp(-1j * ...)
        # We need to compute matmul(windowed_frames_flat_complex, W_rfft.t())
        # where windowed_frames_flat_complex is windowed_frames_flat + 0j.
        # This is a complex matrix multiplication: (a + 0j) * (c + dj) = (a*c) + j(a*d)
        # We avoid creating the complex tensor and compute real/imag parts directly.
        BCT_frames = B * C * T_frames
        windowed_frames_flat = windowed_frames.view(BCT_frames, n_fft)  # [BCT, n_fft] (real)

        # Precompute real and imaginary parts of the DFT matrix W_rfft
        # W_rfft[k, n] = exp(-2j * pi * k * n / N)
        # self.W_rfft is already computed in __init__ as complex: torch.exp(-2j * pi * k * n / N)
        # For manual multiplication, extract real and imag parts.
        # It's more efficient to do this once in __init__, let's assume we have W_rfft_real and W_rfft_imag
        # Or compute them here from self.W_rfft if not stored.
        # Let's compute them here for clarity, assuming self.W_rfft is complex.
        W_rfft_full = self.W_rfft.to(x.device)  # [F, n_fft] complex64
        W_rfft_real = W_rfft_full.real         # [F, n_fft] (real)
        W_rfft_imag = W_rfft_full.imag         # [F, n_fft] (real)

        # Reshape windowed_frames_flat for matrix multiplication: [BCT, 1, n_fft]
        windowed_frames_flat_mm = windowed_frames_flat.unsqueeze(1) # [BCT, 1, n_fft]

        # Perform the real part of the complex multiplication: (a + 0j) * (c + dj) = (a*c) + j(a*d)
        # Real part of result: a * c - 0 * d = a * c
        # Imaginary part of result: a * d + 0 * c = a * d
        # We compute both parts.
        # Matrix multiply [BCT, 1, n_fft] @ [n_fft, F] -> [BCT, 1, F] -> squeeze(1) -> [BCT, F]
        # Multiply windowed_frames_flat (real) with W_rfft_real (real) for the real part of STFT
        stft_result_real_flat = torch.matmul(windowed_frames_flat_mm, W_rfft_real.t()).squeeze(1) # [BCT, F]
        # Multiply windowed_frames_flat (real) with W_rfft_imag (imag) for the imag part of STFT
        stft_result_imag_flat = torch.matmul(windowed_frames_flat_mm, W_rfft_imag.t()).squeeze(1) # [BCT, F]

        # Combine real and imaginary parts into the final STFT result shape
        # Reshape back to [B, C, T_frames, F] for both real and imaginary parts
        stft_result_real = stft_result_real_flat.view(B, C, T_frames, n_fft // 2 + 1) # [B, C, T_frames, F]
        stft_result_imag = stft_result_imag_flat.view(B, C, T_frames, n_fft // 2 + 1) # [B, C, T_frames, F]

        # 5. Transpose to match torch.stft output format [B, C, F, T] and squeeze channel
        # We need to combine real/imag for transpose/squeeze or do them separately.
        # It's simpler to temporarily create a complex tensor for structural ops if allowed,
        # but since the goal is no torch.complex, let's do transpose/squeeze on real/imag separately.
        stft_result_real = stft_result_real.transpose(-1, -2)  # [B, C, F, T_frames] (real)
        stft_result_imag = stft_result_imag.transpose(-1, -2)  # [B, C, F, T_frames] (real)
        stft_result_squeezed_real = stft_result_real.squeeze(1)  # [B, F, T_frames] (real)
        stft_result_squeezed_imag = stft_result_imag.squeeze(1)  # [B, F, T_frames] (real)

        # 6. Separate real and imaginary parts (already separated)
        real_part = stft_result_squeezed_real  # [B, F, T_frames] (real)
        imag_part = stft_result_squeezed_imag  # [B, F, T_frames] (real)


        # --- Alternative: Keep real/imag completely separate throughout (more cumbersome) ---
        # If we want to avoid creating the temporary complex tensor:
        # stft_result_real = stft_result_real.transpose(-1, -2) # [B, C, F, T_frames]
        # stft_result_imag = stft_result_imag.transpose(-1, -2) # [B, C, F, T_frames]
        # stft_result_squeezed_real = stft_result_real.squeeze(1) # [B, F, T_frames]
        # stft_result_squeezed_imag = stft_result_imag.squeeze(1) # [B, F, T_frames]
        # real_part = stft_result_squeezed_real
        # imag_part = stft_result_squeezed_imag
        # --- END Alternative ---

        return real_part, imag_part

    def _istft(self, magnitude, phase):
        """
        Manual ISTFT implementation matching torch.istft logic for fixed parameters,
        WITHOUT using torch.fft operators or torch.complex. Uses basic PyTorch ops only.
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
        target_length = self.target_length # 24000 (Assumed attribute)
        pad_amount = self.pad_amount # n_fft // 2 = 8 (Assumed attribute)

        # --- 1. Reconstruct complex single-sided spectrogram (using real/imag parts) ---
        magnitude = torch.clip(magnitude, max=1e2)
        real_part = magnitude * torch.cos(phase)       # [B, F, T_frames]
        imag_part = magnitude * torch.sin(phase)       # [B, F, T_frames]
        # Avoid torch.complex: work directly with real_part and imag_part

        # --- 2. Extend spectrum to full (for manual IDFT) ---
        # complex_spec_single represented by real_part [B, F, T_frames] and imag_part [B, F, T_frames]
        # F = N//2 + 1 = 9
        if F != N // 2 + 1:
            raise ValueError(f"Input F ({F}) does not match expected N//2+1 ({N//2+1}) for onesided=True")

        # Construct full spectrum tensors for real and imaginary parts
        X_full_real = torch.zeros(B, N, T_frames, dtype=real_part.dtype, device=real_part.device) # [B, N, T_frames]
        X_full_imag = torch.zeros(B, N, T_frames, dtype=imag_part.dtype, device=imag_part.device) # [B, N, T_frames]

        # Copy the single-sided spectrum (frequencies 0 to N//2) to positive frequencies
        X_full_real[:, :F, :] = real_part # [B, 9, T_frames] -> indices [0..8]
        X_full_imag[:, :F, :] = imag_part # [B, 9, T_frames] -> indices [0..8]

        # Copy the conjugate symmetric part (frequencies N//2+1 to N-1)
        if F > 2: # Need positive frequencies to have conjugates
            # 1. Identify the slice of positive frequencies (1 to F-2) in the single-sided input
            pos_freq_indices = slice(1, F-1) # slice(1, 8) -> indices 1, 2, ..., 7
            # 2. Extract the corresponding real and imaginary parts
            pos_real_part = real_part[:, pos_freq_indices, :]   # [B, 7, T_frames]
            pos_imag_part = imag_part[:, pos_freq_indices, :]   # [B, 7, T_frames]
            # 3. Flip these parts along the frequency dimension
            flipped_pos_real_part = torch.flip(pos_real_part, dims=[1]) # [B, 7, T_frames]
            flipped_pos_imag_part = torch.flip(pos_imag_part, dims=[1]) # [B, 7, T_frames]
            # 4. Identify the slice of negative frequencies in the full spectrum
            # Indices N-1 down to N//2+1 -> Indices 15 down to 9 for N=16
            neg_freq_indices_full = slice(N // 2 + 1, N) # slice(9, 16) -> indices 9, 10, ..., 15
            # 5. Assign the flipped (conjugated) parts to the negative frequency slots
            # Real part stays the same under conjugation
            X_full_real[:, neg_freq_indices_full, :] = flipped_pos_real_part # [B, 7, T_frames]
            # Imaginary part is negated under conjugation
            X_full_imag[:, neg_freq_indices_full, :] = -flipped_pos_imag_part # [B, 7, T_frames]

        # --- 3. Apply manual IDFT to each frame ---
        # IDFT formula: x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(2j*pi*k*n/N)
        # Compute the real part of the IDFT result directly.
        # Precompute IDFT twiddle factors
        k_indices_idft = torch.arange(N, dtype=torch.float32, device=X_full_real.device).view(N, 1) # [N, 1]
        n_indices_idft = torch.arange(N, dtype=torch.float32, device=X_full_real.device).view(1, N) # [1, N]
        angle = 2 * math.pi * k_indices_idft * n_indices_idft / N # [N, N]
        W_idft_real = torch.cos(angle) # [N, N]
        W_idft_imag = torch.sin(angle) # [N, N]

        # Reshape for batched matrix multiplication
        BT_frames = B * T_frames
        X_full_real_flat = X_full_real.transpose(1, 2).contiguous().view(BT_frames, N) # [B*T_frames, N]
        X_full_imag_flat = X_full_imag.transpose(1, 2).contiguous().view(BT_frames, N) # [B*T_frames, N]

        # Perform IDFT calculation: real part of (X * W_idft)
        # Real part: (X_real * W_real) - (X_imag * W_imag)
        time_frames_real_part1 = torch.matmul(X_full_real_flat, W_idft_real.t()) # [BT, N]
        time_frames_real_part2 = torch.matmul(X_full_imag_flat, W_idft_imag.t()) # [BT, N]
        time_frames_real_flat = (time_frames_real_part1 - time_frames_real_part2) / N # [BT, N]
        # Reshape back to [B, T_frames, N] and then transpose to [B, N, T_frames]
        time_frames_real = time_frames_real_flat.view(B, T_frames, N).transpose(1, 2) # [B, N, T_frames]

        # --- 4. Apply window ---
        # time_frames_real: [B, N, T_frames]
        # window: [n_fft]
        windowed_time_frames = time_frames_real * window.unsqueeze(0).unsqueeze(-1) # [B, N, T_frames]

        # --- 5. Overlap-Add using manual indexing (replacing Fun.fold) ---
        # windowed_time_frames: [B, N, T_frames] (This is the fold_input)
        # Calculate padded output length (before unpadding)
        padded_output_length = (T_frames - 1) * hop_length + N # e.g., (6001-1)*4 + 16 = 24016

        # --- Manual Overlap-Add using indexing ---
        # Initialize the output tensor with zeros
        signal_padded = torch.zeros(B, 1, padded_output_length, dtype=windowed_time_frames.dtype, device=windowed_time_frames.device) # [B, 1, padded_output_length]

        # Iterate through each frame and add its contribution to the output signal
        for t in range(T_frames):
            start_idx = t * hop_length
            end_idx = start_idx + N
            # Extract the t-th frame's contribution [B, N] and reshape for broadcasting [B, 1, N]
            frame_contribution = windowed_time_frames[:, :, t].unsqueeze(1) # [B, 1, N]
            # Add the frame contribution to the corresponding slice of signal_padded
            if end_idx <= padded_output_length:
                signal_padded[:, :, start_idx:end_idx] += frame_contribution
            else:
                # Handle potential edge case (shouldn't occur with correct length calc)
                signal_padded[:, :, start_idx:] += frame_contribution[:, :, :padded_output_length - start_idx]
        # signal_padded now has shape [B, 1, padded_output_length]
        # This replaces the output of Fun.fold(...).squeeze(2)
        # --- END Manual Overlap-Add (replacing Fun.fold) ---

        # --- 6. Calculate and apply NOLA normalization (using direct summation) ---
        nola_sum_padded = torch.zeros(padded_output_length, dtype=torch.float32, device=window.device) # [padded_output_length]
        window_sq = window.pow(2) # [N]

        # Iterate through each frame and add its window^2 contribution to the NOLA sum
        for t in range(T_frames):
            start_idx = t * hop_length
            end_idx = start_idx + N
            if end_idx <= padded_output_length:
                nola_sum_padded[start_idx:end_idx] += window_sq
            else:
                # Handle potential edge case
                nola_sum_padded[start_idx:] += window_sq[:padded_output_length - start_idx]

        # Add small epsilon to avoid division by zero and reshape for broadcasting
        nola_sum_shaped = nola_sum_padded.view(1, 1, -1) + 1e-12 # [1, 1, padded_output_length]

        # Apply NOLA normalization
        signal_normalized_padded = signal_padded / nola_sum_shaped # [B, 1, padded_output_length]

        # --- 7. Remove padding (center=True) to match specified length ---
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