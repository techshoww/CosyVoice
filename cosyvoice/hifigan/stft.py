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

# from https://github.com/DakeQQ/STFT-ISTFT-ONNX.git
class STFTISTFTReplacerManualFFT(torch.nn.Module):
    def __init__(self, max_frames, n_fft, hop_length, win_length, window):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        # Store window, ensure it's the right shape [1, 1, n_fft] for broadcasting
        # We assume window is 1D [win_length] and win_length == n_fft
        self.window = window.view(1, 1, -1) # Shape: [1, 1, n_fft]

        self.pad_mode = "constant" #"reflect"
        # self.register_buffer('padding_zero', torch.zeros(1, 1, self.n_fft//2, dtype=torch.float32))
        self.padding_zero = torch.zeros(1, 1, self.n_fft//2, dtype=torch.float32)
        

        t  = torch.arange(n_fft).float().unsqueeze(0)
        f  = torch.arange(self.n_fft//2 + 1).float().unsqueeze(1)
        omega = 2 * torch.pi * f * t / n_fft
        # self.register_buffer(
        #     'cos_kernel',
        #     (torch.cos(omega) * window.unsqueeze(0)).unsqueeze(1)
        # )
        self.cos_kernel = (torch.cos(omega) * window.unsqueeze(0)).unsqueeze(1)
        # self.register_buffer(
        #     'sin_kernel',
        #     (-torch.sin(omega) * window.unsqueeze(0)).unsqueeze(1)
        # )
        self.sin_kernel = (-torch.sin(omega) * window.unsqueeze(0)).unsqueeze(1)
        fourier_basis = torch.fft.fft(torch.eye(n_fft, dtype=torch.float32))
        fourier_basis = torch.vstack([
            torch.real(fourier_basis[:self.n_fft//2 + 1]),
            torch.imag(fourier_basis[:self.n_fft//2 + 1])
        ]).float()

        self.forward_basis = window * fourier_basis.unsqueeze(1)
        self.inverse_basis = window * torch.linalg.pinv(
            (fourier_basis * n_fft) / hop_length
        ).T.unsqueeze(1)

        # max_frames = 27840 //hop_length + 1
        # overlap-add weighting
        n          = n_fft + hop_length * (max_frames - 1)
        window_sum = torch.zeros(n, dtype=torch.float32)

        orig_win = window
        wn = orig_win / orig_win.abs().max()

        if win_length < n_fft:
            pl = (n_fft - win_length) // 2
            pr = n_fft - win_length - pl
            win_sq = torch.nn.functional.pad(wn ** 2, (pl, pr))
        else:
            win_sq = wn ** 2

        for i in range(max_frames):
            s = i * hop_length
            window_sum[s:s + n_fft] += win_sq[:max(0, min(n_fft, n - s))]

        # self.register_buffer('forward_basis', forward_basis)
        # self.register_buffer('inverse_basis', inverse_basis)
        # self.register_buffer('window_sum_inv', n_fft / (window_sum * hop_length + 1e-7))
        self.window_sum_inv = n_fft / (window_sum * hop_length + 1e-7)

    def _pad_input(self, x):
        if self.pad_mode == 'reflect':
            return torch.nn.functional.pad(x, (self.n_fft//2, self.n_fft//2), mode='reflect')
        return torch.cat((self.padding_zero.to(x.device), x, self.padding_zero.to(x.device)), dim=-1)

    def _stft(self, x):
        """Manual STFT implementation for fixed parameters, using indexing instead of unfold, avoiding torch.complex."""
        x_padded = self._pad_input(x)
        real_part = torch.nn.functional.conv1d(x_padded, self.cos_kernel.to(x_padded.device), stride=self.hop_length)
        imag_part = torch.nn.functional.conv1d(x_padded, self.sin_kernel.to(x_padded.device), stride=self.hop_length)
        return real_part, imag_part

    def _istft(self, magnitude, phase):
        """
        Manual ISTFT implementation matching torch.istft logic for fixed parameters,
        WITHOUT using torch.fft operators or torch.complex. Uses basic PyTorch ops only.
        Input magnitude: [B, F, T_frames] = [1, 9, 6001]
        Input phase: [B, F, T_frames] = [1, 9, 6001]
        Output: [B, T] = [1, 24000]
        """
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)

        inp = torch.cat((real, imag), dim=1)  # == cat(real, imag)
        inv = torch.nn.functional.conv_transpose1d(inp, self.inverse_basis.to(magnitude.device), stride=self.hop_length)
        s, e = self.n_fft//2, inv.size(-1) - self.n_fft//2
        ret =  inv[:, :, s:e] * self.window_sum_inv[s:e].to(magnitude.device)
        return ret.squeeze(1)

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