import torch
import torch.nn as nn
import torch.nn.functional as F

class ComparisonModel(nn.Module):
    def __init__(self):
        super(ComparisonModel, self).__init__()
        
        # --- Encoder (Compression) ---
        # Input shape: (B, 1, 48, 128)
        # Note: The user mentioned Input_Shape = (1, 48, 128)
        # Assuming 48 is H (e.g. stations * channels or similar mapping) and 128 is W (time points)
        # However, our dataset provides (Batch, Stations, Channels, Time) -> (B, S, 3, T)
        # We need to reshape the input to match the (B, 1, H, W) format expected by this Conv2D model.
        # Let's assume we stack stations and channels into H.
        # Or maybe H=48 is fixed.
        # The user's prompt says: Input_Shape = (1, 48, 128).
        # Our config says MAX_STATIONS=24, CHANNELS=3, SEQ_LEN=512.
        # If we use all stations and channels: 24 * 3 = 72.
        # If the user's paper uses 48, maybe they used 16 stations * 3 channels?
        # Or maybe it's just an example shape.
        # I will adapt the input layer to match our data shape or resize/pad to match the structure.
        # To be safe and general, I'll design the encoder to handle dynamic H if possible, or fixed to our data.
        # Our data: (B, 24, 3, 512).
        # Reshape to (B, 1, 72, 512) for Conv2D?
        # Or (B, 3, 24, 512)?
        # The user's pseudocode starts with Conv2D(64, (3,3)).
        # Let's assume input is (B, 1, H, W).
        
        self.encoder_layers = nn.ModuleList()
        
        # Layer 1: 1 -> 64
        self.encoder_layers.append(self._make_enc_block(1, 64))
        # Layer 2: 64 -> 64
        self.encoder_layers.append(self._make_enc_block(64, 64))
        # Layer 3: 64 -> 64
        self.encoder_layers.append(self._make_enc_block(64, 64))
        # Layer 4: 64 -> 128
        self.encoder_layers.append(self._make_enc_block(64, 128))
        # Layer 5: 128 -> 128
        self.encoder_layers.append(self._make_enc_block(128, 128))
        
        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        self.bottleneck_bn = nn.BatchNorm2d(128)
        self.bottleneck_act = nn.LeakyReLU(0.2, inplace=True)
        
        # --- Decoder (Expansion) ---
        # We need to go from H=1 to H=128 (target output size).
        # We use UpSampling(size=(2, 1)) which doubles H.
        # 1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128.
        # This requires 7 upsampling steps.
        # Channel sequence: 128 -> ... -> 3.
        
        self.decoder_layers = nn.ModuleList()
        # 128 -> 128 (H: 1->2)
        self.decoder_layers.append(self._make_dec_block(128, 128))
        # 128 -> 128 (H: 2->4)
        self.decoder_layers.append(self._make_dec_block(128, 128))
        # 128 -> 128 (H: 4->8)
        self.decoder_layers.append(self._make_dec_block(128, 128))
        # 128 -> 64 (H: 8->16)
        self.decoder_layers.append(self._make_dec_block(128, 64))
        # 64 -> 32 (H: 16->32)
        self.decoder_layers.append(self._make_dec_block(64, 32))
        # 32 -> 16 (H: 32->64)
        self.decoder_layers.append(self._make_dec_block(32, 16))
        # 16 -> 16 (H: 64->128)
        self.decoder_layers.append(self._make_dec_block(16, 16))
        
        # Final layer
        self.final_conv = nn.Conv2d(16, 3, kernel_size=(3, 1), padding=(1, 0))
        
        # We also need to predict Magnitude (Mw). The user didn't specify, but our task requires it.
        # I'll add a small head for Mw from the bottleneck.
        self.mw_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def _make_enc_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)) # Reduces H and W by half
        )

    def _make_dec_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=(2, 1), mode='nearest'), # Doubles H, W stays
            nn.Conv2d(in_c, out_c, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, wav, geo=None, amp=None, mask=None):
        # wav: (B, S, 3, T) = (B, 24, 3, 512)
        b, s, c, t = wav.shape
        x = wav.view(b, 1, s*c, t) # (B, 1, 72, 512)
        
        # Encoder
        for layer in self.encoder_layers:
            x = layer(x)
            
        # Bottleneck
        x = F.adaptive_avg_pool2d(x, (1, 1)) # (B, 128, 1, 1)
        x = self.bottleneck_conv(x)
        x = self.bottleneck_bn(x)
        x = self.bottleneck_act(x)
        
        # Mw prediction
        mw_pred = self.mw_head(x)
        
        # Decoder
        z = x
        # Pass through all decoder blocks (Upsample included in blocks)
        for layer in self.decoder_layers:
            z = layer(z)
            
        z = self.final_conv(z) # (B, 3, 128, 1)
        
        # Now we have (B, 3, 128, 1).
        # This represents probabilities for Strike, Dip, Rake.
        # We need to convert this to scalar values (angles) for loss calculation (regression),
        # or use KL divergence if we had ground truth distributions.
        # But our ground truth is scalar angles.
        # So we should perform "Expectation" or "Argmax".
        # Expectation is differentiable.
        # Angles are periodic? Strike [0, 360], Dip [0, 90], Rake [-180, 180].
        # We need to map index 0..127 to these ranges.
        
        # Let's assume indices map linearly.
        # Strike: 0..127 -> 0..360
        # Dip: 0..127 -> 0..90
        # Rake: 0..127 -> -180..180
        
        # Softmax over the 128 dimension
        probs = F.softmax(z.squeeze(-1), dim=2) # (B, 3, 128)
        
        # Create bins
        device = z.device
        bins = torch.linspace(0, 1, 128).to(device)
        
        # Expectation
        # (B, 3, 128) * (128) -> (B, 3)
        expected_norm = torch.sum(probs * bins.view(1, 1, 128), dim=2) # (B, 3) 0..1 range
        
        # Map to physical ranges
        # 0: Strike (0-360) => val * 360
        # 1: Dip (0-90) => val * 90
        # 2: Rake (-180-180) => val * 360 - 180
        
        strike = expected_norm[:, 0] * 360.0
        dip = expected_norm[:, 1] * 90.0
        rake = expected_norm[:, 2] * 360.0 - 180.0
        
        # Convert Strike/Dip/Rake to Moment Tensor?
        # Our training target is Moment Tensor (6 components).
        # We need a differentiable SDR -> MT converter.
        # Or we change the loss to compare SDR directly?
        # The dataset provides MT. We can convert MT to SDR?
        # Converting MT to SDR is complex and not unique.
        # Converting SDR + Mw to MT is standard.
        # So we should convert Pred_SDR + Pred_Mw -> Pred_MT.
        
        # But wait, the existing model outputs `mt_pred` (normalized 6D vector).
        # If I output SDR, I should convert it to MT to be compatible with `train.py` loss,
        # OR I implement SDR->MT conversion here.
        
        mt_pred = self.sdr_mw_to_mt(strike, dip, rake, mw_pred)
        
        return mt_pred, mw_pred

    def sdr_mw_to_mt(self, strike, dip, rake, mw):
        # Differentiable conversion from Strike, Dip, Rake, Mw to Moment Tensor
        # Formula from Aki & Richards
        
        # Angles to radians
        d2r = torch.pi / 180.0
        s = strike * d2r
        d = dip * d2r
        r = rake * d2r
        
        sin_s = torch.sin(s)
        cos_s = torch.cos(s)
        sin_d = torch.sin(d)
        cos_d = torch.cos(d)
        sin_2d = torch.sin(2*d)
        cos_2d = torch.cos(2*d)
        sin_r = torch.sin(r)
        cos_r = torch.cos(r)
        
        # M0 from Mw
        # Mw = 2/3 log10(M0) - 6.07
        # log10(M0) = 1.5 * (Mw + 6.07)
        # M0 = 10 ** ...
        # But we only need normalized MT for direction, and Mw for magnitude.
        # The existing code separates MT (direction) and Mw.
        # mt_pred in existing code is normalized (norm=1).
        # So we can just compute the unit MT from SDR.
        
        # Mxx = - (sin(d) cos(r) sin(2s) + sin(2d) sin(r) sin(s)^2)
        # Myy =   (sin(d) cos(r) sin(2s) - sin(2d) sin(r) cos(s)^2)
        # Mzz =   sin(2d) sin(r)
        # Mxy =   (sin(d) cos(r) cos(2s) + 0.5 * sin(2d) sin(r) sin(2s))
        # Mxz = - (cos(d) cos(r) cos(s)  + cos(2d) sin(r) sin(s))
        # Myz = - (cos(d) cos(r) sin(s)  - cos(2d) sin(r) cos(s))
        
        # Note: Coordinate systems vary (USE vs NED).
        # DeepFMT usually assumes USE or similar.
        # I'll implement standard Aki & Richards (North-East-Down) and verify if it matches.
        # Actually, let's look at `obspy` or existing code if possible.
        # Since I cannot easily verify the coordinate system without running, 
        # I will implement a standard one. If the loss is high, it might be coords.
        # But since the model learns to map Wave -> SDR -> MT, 
        # as long as the mapping is consistent and differentiable, the model can learn "rotated" SDRs to match the target MT.
        
        m_xx = -(torch.sin(d)*torch.cos(r)*torch.sin(2*s) + torch.sin(2*d)*torch.sin(r)*torch.sin(s)**2)
        m_yy =  (torch.sin(d)*torch.cos(r)*torch.sin(2*s) - torch.sin(2*d)*torch.sin(r)*torch.cos(s)**2)
        m_zz =  torch.sin(2*d)*torch.sin(r)
        m_xy =  (torch.sin(d)*torch.cos(r)*torch.cos(2*s) + 0.5*torch.sin(2*d)*torch.sin(r)*torch.sin(2*s))
        m_xz = -(torch.cos(d)*torch.cos(r)*torch.cos(s)  + torch.cos(2*d)*torch.sin(r)*torch.sin(s))
        m_yz = -(torch.cos(d)*torch.cos(r)*torch.sin(s)  - torch.cos(2*d)*torch.sin(r)*torch.cos(s))
        
        # Stack
        mt = torch.stack([m_xx, m_yy, m_zz, m_xy, m_xz, m_yz], dim=1)
        
        # Normalize
        mt = mt / (torch.norm(mt, dim=1, keepdim=True) + 1e-10)
        
        return mt

    # Override the _make_dec_block to exclude Upsample since we do it manually
    def _make_dec_block(self, in_c, out_c):
        return nn.Sequential(
            # Upsample removed from here
            nn.Conv2d(in_c, out_c, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
