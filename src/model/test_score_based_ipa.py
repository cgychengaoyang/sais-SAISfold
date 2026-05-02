"""Tests for score-based IPA model."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from openfold.utils.rigid_utils import Rigid, Rotation

from src.model.score_based_ipa import (
    BackboneUpdate,
    MinimalIPA,
    Transition,
    EdgeTransition,
    DyneTrionScoreNet,
    DyneTrionScoreLoss,
    calc_rot_score,
    calc_trans_score,
)


class TestBackboneUpdate:
    """Test BackboneUpdate module."""
    
    def test_output_shape(self):
        """Test output shape is [B, N, 6]."""
        module = BackboneUpdate(c_s=256)
        s = torch.randn(2, 30, 256)
        output = module(s)
        assert output.shape == (2, 30, 6)
    
    def test_initialization(self):
        """Test that weights are initialized to zero."""
        module = BackboneUpdate(c_s=256)
        assert torch.allclose(module.linear.weight, torch.zeros_like(module.linear.weight))
        assert torch.allclose(module.linear.bias, torch.zeros_like(module.linear.bias))
    
    def test_zeros_input(self):
        """Test that zero input produces zero output (due to zero init)."""
        module = BackboneUpdate(c_s=256)
        s = torch.zeros(2, 30, 256)
        output = module(s)
        assert torch.allclose(output, torch.zeros_like(output))


class TestMinimalIPA:
    """Test MinimalIPA module."""
    
    def test_output_shape(self):
        """Test output shape matches input."""
        ipa = MinimalIPA(c_s=256, c_z=128)
        s = torch.randn(2, 30, 256)
        z = torch.randn(2, 30, 30, 128)
        r = Rigid.identity(shape=(2, 30))
        mask = torch.ones(2, 30)
        
        output = ipa(s, z, r, mask)
        assert output.shape == s.shape
    
    def test_without_edge_features(self):
        """Test IPA works without edge features."""
        ipa = MinimalIPA(c_s=256, c_z=128)
        s = torch.randn(2, 30, 256)
        r = Rigid.identity(shape=(2, 30))
        mask = torch.ones(2, 30)
        
        output = ipa(s, None, r, mask)
        assert output.shape == s.shape
    
    def test_masking(self):
        """Test that masking works correctly."""
        ipa = MinimalIPA(c_s=256, c_z=128)
        s = torch.randn(2, 30, 256)
        z = torch.randn(2, 30, 30, 128)
        r = Rigid.identity(shape=(2, 30))
        mask = torch.ones(2, 30)
        mask[:, 15:] = 0  # Mask second half
        
        output = ipa(s, z, r, mask)
        assert output.shape == s.shape


class TestTransition:
    """Test Transition module."""
    
    def test_output_shape(self):
        """Test output shape matches input."""
        trans = Transition(c=256)
        s = torch.randn(2, 30, 256)
        output = trans(s)
        assert output.shape == s.shape
    
    def test_residual_connection(self):
        """Test that output is different from input (residual works)."""
        trans = Transition(c=256)
        s = torch.randn(2, 30, 256)
        output = trans(s)
        assert not torch.allclose(output, s)


class TestEdgeTransition:
    """Test EdgeTransition module."""
    
    def test_output_shape(self):
        """Test output shape."""
        edge_trans = EdgeTransition(c_s=256, c_z=128)
        s = torch.randn(2, 30, 256)
        z = torch.randn(2, 30, 30, 128)
        
        output = edge_trans(s, z)
        assert output.shape == z.shape


class TestDyneTrionScoreNet:
    """Test DyneTrionScoreNet module."""
    
    def test_forward(self):
        """Test forward pass."""
        model = DyneTrionScoreNet(
            c_s_input=384,
            c_z_input=128,
            c_s=256,
            c_z=128,
            num_blocks=4,
        )
        
        node_repr = torch.randn(2, 30, 384)
        edge_repr = torch.randn(2, 30, 30, 128)
        rigids = model.init_random_rigids(2, 30)
        
        output = model(node_repr, edge_repr, rigids)
        
        assert 'rigids' in output
        assert 'init_rigids' in output
        assert 'node_repr' in output
        assert 'edge_repr' in output
        assert output['rigids'].shape == (2, 30, 7)
    
    def test_rigids_normalized(self):
        """Test that output quaternions are normalized."""
        model = DyneTrionScoreNet(
            c_s_input=384,
            c_z_input=128,
            c_s=256,
            c_z=128,
            num_blocks=2,
        )
        
        node_repr = torch.randn(2, 30, 384)
        edge_repr = torch.randn(2, 30, 30, 128)
        rigids = torch.randn(2, 30, 7)
        
        output = model(node_repr, edge_repr, rigids)
        
        quats = output['rigids'][..., :4]
        quat_norms = quats.norm(dim=-1)
        assert torch.allclose(quat_norms, torch.ones_like(quat_norms), atol=1e-5)
    
    def test_sample_structure_random_init(self):
        """Test sampling with random initialization."""
        model = DyneTrionScoreNet(
            c_s_input=384,
            c_z_input=128,
            c_s=256,
            c_z=128,
            num_blocks=2,
        )
        
        node_repr = torch.randn(2, 30, 384)
        edge_repr = torch.randn(2, 30, 30, 128)
        
        # Don't provide init_rigids - should use random init
        final_rigids = model.sample_structure(
            node_repr, edge_repr, init_rigids=None, num_steps=1
        )
        
        assert final_rigids.shape == (2, 30, 7)
    
    def test_with_mask(self):
        """Test forward pass with mask."""
        model = DyneTrionScoreNet(
            c_s_input=384,
            c_z_input=128,
            c_s=256,
            c_z=128,
            num_blocks=2,
        )
        
        node_repr = torch.randn(2, 30, 384)
        edge_repr = torch.randn(2, 30, 30, 128)
        rigids = model.init_random_rigids(2, 30)
        mask = torch.ones(2, 30)
        mask[:, 15:] = 0
        
        output = model(node_repr, edge_repr, rigids, mask)
        assert output['rigids'].shape == (2, 30, 7)


class TestScoreFunctions:
    """Test score calculation functions."""
    
    def test_calc_rot_score(self):
        """Test rotation score calculation."""
        B, N = 2, 30
        
        # Create two different rotations
        quats_0 = torch.randn(B, N, 4)
        quats_0 = quats_0 / quats_0.norm(dim=-1, keepdim=True)
        quats_t = torch.randn(B, N, 4)
        quats_t = quats_t / quats_t.norm(dim=-1, keepdim=True)
        
        rot_0 = Rotation(quats=quats_0)
        rot_t = Rotation(quats=quats_t)
        t = torch.rand(B)
        
        score = calc_rot_score(rot_0, rot_t, t)
        
        assert score.shape == (B, N, 3)
        assert not torch.allclose(score, torch.zeros_like(score))
    
    def test_calc_trans_score(self):
        """Test translation score calculation."""
        B, N = 2, 30
        
        trans_0 = torch.randn(B, N, 3)
        trans_t = trans_0 + torch.randn(B, N, 3) * 0.5  # Add noise
        t = torch.rand(B)
        
        score = calc_trans_score(trans_0, trans_t, t, trans_scale=1.0)
        
        assert score.shape == (B, N, 3)
        assert not torch.allclose(score, torch.zeros_like(score))
    
    def test_calc_trans_score_zero_time(self):
        """Test translation score at t=0 (should be large)."""
        B, N = 2, 30
        
        trans_0 = torch.randn(B, N, 3)
        trans_t = trans_0 + torch.randn(B, N, 3) * 0.5
        t = torch.zeros(B)  # t=0
        
        score = calc_trans_score(trans_0, trans_t, t, trans_scale=1.0)
        
        # At t=0, score should be large (division by small number)
        assert score.abs().max() > 1e3


class TestDyneTrionScoreLoss:
    """Test DyneTrionScoreLoss module."""
    
    def test_loss_computation(self):
        """Test that loss is computed correctly."""
        loss_fn = DyneTrionScoreLoss(trans_scale=1.0)
        
        # Create model output
        B, N = 2, 30
        model_out = {
            'rigids': torch.randn(B, N, 7),
            'init_rigids': torch.randn(B, N, 7),
        }
        # Normalize quaternions
        model_out['rigids'][..., :4] = model_out['rigids'][..., :4] / model_out['rigids'][..., :4].norm(dim=-1, keepdim=True)
        model_out['init_rigids'][..., :4] = model_out['init_rigids'][..., :4] / model_out['init_rigids'][..., :4].norm(dim=-1, keepdim=True)
        
        gt_rigids = torch.randn(B, N, 7)
        gt_rigids[..., :4] = gt_rigids[..., :4] / gt_rigids[..., :4].norm(dim=-1, keepdim=True)
        
        t = torch.rand(B)
        
        losses = loss_fn(model_out, gt_rigids, t)
        
        assert 'rot_loss' in losses
        assert 'trans_loss' in losses
        assert 'total_loss' in losses
        
        # Loss should be non-negative
        assert losses['rot_loss'].item() >= 0
        assert losses['trans_loss'].item() >= 0
        assert losses['total_loss'].item() >= 0
    
    def test_loss_with_mask(self):
        """Test loss computation with mask."""
        loss_fn = DyneTrionScoreLoss(trans_scale=1.0)
        
        B, N = 2, 30
        model_out = {
            'rigids': torch.randn(B, N, 7),
            'init_rigids': torch.randn(B, N, 7),
        }
        model_out['rigids'][..., :4] = model_out['rigids'][..., :4] / model_out['rigids'][..., :4].norm(dim=-1, keepdim=True)
        model_out['init_rigids'][..., :4] = model_out['init_rigids'][..., :4] / model_out['init_rigids'][..., :4].norm(dim=-1, keepdim=True)
        
        gt_rigids = torch.randn(B, N, 7)
        gt_rigids[..., :4] = gt_rigids[..., :4] / gt_rigids[..., :4].norm(dim=-1, keepdim=True)
        
        t = torch.rand(B)
        mask = torch.ones(B, N)
        mask[:, 15:] = 0
        
        losses = loss_fn(model_out, gt_rigids, t, mask)
        
        assert losses['rot_loss'].item() >= 0
        assert losses['trans_loss'].item() >= 0
    
    def test_loss_non_zero(self):
        """Test that loss is non-zero when init and final are different."""
        loss_fn = DyneTrionScoreLoss(trans_scale=1.0)
        
        B, N = 2, 30
        
        # Make init and final significantly different
        final_rigids = torch.randn(B, N, 7)
        final_rigids[..., :4] = final_rigids[..., :4] / final_rigids[..., :4].norm(dim=-1, keepdim=True)
        final_rigids[..., 4:] = torch.randn(B, N, 3) * 10  # Large translation
        
        init_rigids = torch.randn(B, N, 7)
        init_rigids[..., :4] = init_rigids[..., :4] / init_rigids[..., :4].norm(dim=-1, keepdim=True)
        init_rigids[..., 4:] = torch.randn(B, N, 3) * 10 + 5.0  # Different translation
        
        model_out = {
            'rigids': final_rigids,
            'init_rigids': init_rigids,
        }
        
        gt_rigids = torch.randn(B, N, 7)
        gt_rigids[..., :4] = gt_rigids[..., :4] / gt_rigids[..., :4].norm(dim=-1, keepdim=True)
        
        t = torch.rand(B)
        
        losses = loss_fn(model_out, gt_rigids, t)
        
        # Loss should be non-zero
        assert losses['total_loss'].item() > 1e-6, f"Expected non-zero loss, got {losses['total_loss'].item()}"


def test_end_to_end():
    """End-to-end test of the full pipeline."""
    # Create model
    model = DyneTrionScoreNet(
        c_s_input=384,
        c_z_input=128,
        c_s=256,
        c_z=128,
        num_blocks=2,
    )
    loss_fn = DyneTrionScoreLoss(trans_scale=1.0)
    
    # Create data
    B, N = 2, 30
    node_repr = torch.randn(B, N, 384)
    edge_repr = torch.randn(B, N, N, 128)
    init_rigids = model.init_random_rigids(B, N, trans_scale=10.0)
    
    # Ground truth
    gt_rigids = torch.randn(B, N, 7)
    gt_rigids[..., :4] = gt_rigids[..., :4] / gt_rigids[..., :4].norm(dim=-1, keepdim=True)
    t = torch.rand(B)
    
    # Forward pass
    output = model(node_repr, edge_repr, init_rigids)
    
    # Compute loss
    losses = loss_fn(output, gt_rigids, t)
    
    # Backward pass
    losses['total_loss'].backward()
    
    # Check gradients exist
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "No gradients computed"
    
    print("✓ End-to-end test passed!")


if __name__ == '__main__':
    # Run tests
    print("Running tests...")
    
    # Simple run without pytest
    test_classes = [
        TestBackboneUpdate,
        TestMinimalIPA,
        TestTransition,
        TestEdgeTransition,
        TestDyneTrionScoreNet,
        TestScoreFunctions,
        TestDyneTrionScoreLoss,
    ]
    
    for test_class in test_classes:
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    method = getattr(instance, method_name)
                    method()
                    print(f"✓ {test_class.__name__}.{method_name}")
                except Exception as e:
                    print(f"✗ {test_class.__name__}.{method_name}: {e}")
    
    # Run end-to-end test
    try:
        test_end_to_end()
    except Exception as e:
        print(f"✗ test_end_to_end: {e}")
    
    print("\nAll tests completed!")
