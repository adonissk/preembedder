import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def weighted_pearson_correlation(y_true: torch.Tensor, y_pred: torch.Tensor, weights: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Calculates the weighted Pearson correlation coefficient using PyTorch.

    Args:
        y_true: Ground truth target values (1D Tensor).
        y_pred: Predicted values (1D Tensor).
        weights: Weights for each sample (1D Tensor).
        epsilon: Small value to prevent division by zero.

    Returns:
        The weighted Pearson correlation coefficient (scalar Tensor).
    """
    if y_true.shape != y_pred.shape or y_true.shape != weights.shape:
        raise ValueError(f"Input tensors must have the same shape. Got y_true: {y_true.shape}, y_pred: {y_pred.shape}, weights: {weights.shape}")
    if y_true.ndim != 1:
         raise ValueError(f"Input tensors must be 1D. Got {y_true.ndim} dimensions.")
    if torch.sum(weights) <= 0:
        logging.warning("Sum of weights is zero or negative. Correlation is undefined. Returning NaN.")
        return torch.tensor(float('nan'), device=y_true.device)

    # Ensure weights are positive and normalize them (optional, but good practice)
    # weights = torch.clamp(weights, min=0) # Ensure non-negative weights
    weights = weights / torch.sum(weights) # Normalize weights to sum to 1

    # Calculate weighted means
    mean_true = torch.sum(weights * y_true)
    mean_pred = torch.sum(weights * y_pred)

    # Calculate centered values
    centered_true = y_true - mean_true
    centered_pred = y_pred - mean_pred

    # Calculate weighted covariance
    covariance = torch.sum(weights * centered_true * centered_pred)

    # Calculate weighted standard deviations
    variance_true = torch.sum(weights * centered_true**2)
    variance_pred = torch.sum(weights * centered_pred**2)

    std_true = torch.sqrt(variance_true)
    std_pred = torch.sqrt(variance_pred)

    # Calculate correlation
    correlation = covariance / (std_true * std_pred + epsilon)

    # Clamp result to [-1, 1] to handle potential floating point inaccuracies
    correlation = torch.clamp(correlation, -1.0, 1.0)

    # Handle cases where standard deviation is zero (constant input)
    if std_true < epsilon or std_pred < epsilon:
        logging.debug("One or both input series have near-zero weighted standard deviation. Correlation is NaN or 0.")
        # If both are constant, correlation is undefined (NaN). If only one, it's 0.
        # However, the division by epsilon often handles this reasonably. Let's return the computed value (likely near 0 or NaN if cov is also 0).
        # Alternatively, explicitly return NaN or 0:
        # if std_true < epsilon and std_pred < epsilon:
        #     return torch.tensor(float('nan'), device=y_true.device)
        # else:
        #     return torch.tensor(0.0, device=y_true.device)
        # Sticking with the epsilon approach for now.

    return correlation

# Alias for convenience
wcorr = weighted_pearson_correlation

# Example usage (optional, for testing)
if __name__ == '__main__':
    print("Testing Weighted Pearson Correlation...")

    # Test case 1: Perfect positive correlation
    y_t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_p = torch.tensor([2.0, 4.0, 6.0, 8.0])
    w = torch.tensor([1.0, 1.0, 1.0, 1.0])
    corr1 = wcorr(y_t, y_p, w)
    print(f"Test 1 (Perfect Positive): Correlation = {corr1.item():.4f} (Expected: 1.0)")
    assert torch.isclose(corr1, torch.tensor(1.0)), "Test 1 Failed"

    # Test case 2: Perfect negative correlation
    y_p_neg = torch.tensor([8.0, 6.0, 4.0, 2.0])
    corr2 = wcorr(y_t, y_p_neg, w)
    print(f"Test 2 (Perfect Negative): Correlation = {corr2.item():.4f} (Expected: -1.0)")
    assert torch.isclose(corr2, torch.tensor(-1.0)), "Test 2 Failed"

    # Test case 3: No correlation
    y_p_zero = torch.tensor([1.0, -1.0, 1.0, -1.0])
    corr3 = wcorr(y_t, y_p_zero, w)
    print(f"Test 3 (No Correlation): Correlation = {corr3.item():.4f} (Expected: near 0.0)")
    # Value might not be exactly 0 due to floating point, check if close
    assert torch.isclose(corr3, torch.tensor(0.0), atol=1e-5), "Test 3 Failed"

    # Test case 4: Weighted correlation
    y_t_w = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_p_w = torch.tensor([1.5, 2.5, 2.5, 4.5])
    w_w = torch.tensor([1.0, 1.0, 5.0, 1.0]) # Weight 3rd point heavily
    corr4 = wcorr(y_t_w, y_p_w, w_w)
    # Manually calculate expected (approximate calculation for sanity check)
    # Weighted means: mt=(1+2+15+4)/8=2.75, mp=(1.5+2.5+12.5+4.5)/8 = 2.625
    # Cov: sum(w*(yt-mt)*(yp-mp))/sum(w) = (1*(-1.75)*(-1.125) + 1*(-0.75)*(-0.125) + 5*(0.25)*(-0.125) + 1*(1.25)*(1.875)) / 8
    #      = (1.96875 + 0.09375 - 0.15625 + 2.34375) / 8 = 4.25 / 8 = 0.53125
    # VarT: sum(w*(yt-mt)^2)/sum(w) = (1*(-1.75)^2 + 1*(-0.75)^2 + 5*(0.25)^2 + 1*(1.25)^2) / 8
    #       = (3.0625 + 0.5625 + 0.3125 + 1.5625) / 8 = 5.5 / 8 = 0.6875 => stdT = 0.829
    # VarP: sum(w*(yp-mp)^2)/sum(w) = (1*(-1.125)^2 + 1*(-0.125)^2 + 5*(-0.125)^2 + 1*(1.875)^2) / 8
    #       = (1.265625 + 0.015625 + 0.078125 + 3.515625) / 8 = 4.875 / 8 = 0.609375 => stdP = 0.7806
    # Corr = 0.53125 / (0.829 * 0.7806) ~= 0.53125 / 0.6471 ~= 0.8209
    print(f"Test 4 (Weighted): Correlation = {corr4.item():.4f} (Expected: ~0.8209)")
    assert torch.isclose(corr4, torch.tensor(0.8209), atol=1e-4), "Test 4 Failed"

    # Test case 5: Constant input
    y_t_const = torch.tensor([2.0, 2.0, 2.0, 2.0])
    y_p_const = torch.tensor([3.0, 3.0, 3.0, 3.0])
    corr5a = wcorr(y_t_const, y_p, w)
    corr5b = wcorr(y_t, y_p_const, w)
    corr5c = wcorr(y_t_const, y_p_const, w)
    print(f"Test 5a (Constant True): Correlation = {corr5a.item()}") # Should be 0 or NaN
    print(f"Test 5b (Constant Pred): Correlation = {corr5b.item()}") # Should be 0 or NaN
    print(f"Test 5c (Both Constant): Correlation = {corr5c.item()}") # Should be NaN
    assert corr5a.isnan() or torch.isclose(corr5a, torch.tensor(0.0)), "Test 5a Failed"
    assert corr5b.isnan() or torch.isclose(corr5b, torch.tensor(0.0)), "Test 5b Failed"
    assert corr5c.isnan(), "Test 5c Failed"

    # Test case 6: Zero weights
    w_zero = torch.tensor([0.0, 0.0, 0.0, 0.0])
    corr6 = wcorr(y_t, y_p, w_zero)
    print(f"Test 6 (Zero Weights): Correlation = {corr6.item()}") # Should be NaN
    assert corr6.isnan(), "Test 6 Failed"

    print("\nAll tests passed!")
