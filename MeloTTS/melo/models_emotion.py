import math
import torch
from torch import nn
from torch.nn import functional as F

from melo import commons
from melo import modules
from melo import attentions

from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from melo.commons import init_weights, get_padding
import melo.monotonic_align as monotonic_align

# 기존 모델들 import
from melo.models import (
    DurationDiscriminator,
    TransformerCouplingBlock,
    StochasticDurationPredictor,
    DurationPredictor,
    TextEncoder,
    ResidualCouplingBlock,
    PosteriorEncoder,
    Generator,
    DiscriminatorP,
    DiscriminatorS,
    MultiPeriodDiscriminator,
    ReferenceEncoder
)


class EmotionEmbedding(nn.Module):
    """감정 임베딩 레이어"""
    def __init__(self, n_emotions, emotion_channels=256):
        super().__init__()
        self.n_emotions = n_emotions
        self.emotion_channels = emotion_channels
        self.emb_e = nn.Embedding(n_emotions, emotion_channels)
        
    def forward(self, emotion_ids):
        """
        Args:
            emotion_ids: [batch_size] 감정 ID 텐서
        Returns:
            emotion_emb: [batch_size, emotion_channels] 감정 임베딩
        """
        return self.emb_e(emotion_ids)


class SynthesizerTrnEmotion(nn.Module):
    """
    감정 지원 Synthesizer for Training
    """

    def __init__(
        self,
        n_vocab,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=256,
        gin_channels=256,
        n_emotions=7,  # 감정 개수 추가
        emotion_channels=256,  # 감정 임베딩 차원
        use_sdp=True,
        n_flow_layer=4,
        n_layers_trans_flow=6,
        flow_share_parameter=False,
        use_transformer_flow=True,
        use_vc=False,
        num_languages=None,
        num_tones=None,
        norm_refenc=False,
        **kwargs
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.n_emotions = n_emotions
        self.emotion_channels = emotion_channels
        self.n_layers_trans_flow = n_layers_trans_flow
        self.use_spk_conditioned_encoder = kwargs.get(
            "use_spk_conditioned_encoder", True
        )
        self.use_sdp = use_sdp
        self.use_noise_scaled_mas = kwargs.get("use_noise_scaled_mas", False)
        self.mas_noise_scale_initial = kwargs.get("mas_noise_scale_initial", 0.01)
        self.noise_scale_delta = kwargs.get("noise_scale_delta", 2e-6)
        self.current_mas_noise_scale = self.mas_noise_scale_initial
        
        # 감정 임베딩 추가
        self.emotion_embedding = EmotionEmbedding(n_emotions, emotion_channels)
        
        # 통합된 gin_channels (speaker + emotion)
        total_gin_channels = gin_channels + emotion_channels
        
        if self.use_spk_conditioned_encoder and total_gin_channels > 0:
            self.enc_gin_channels = total_gin_channels
        else:
            self.enc_gin_channels = 0
            
        self.enc_p = TextEncoder(
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=self.enc_gin_channels,
            num_languages=num_languages,
            num_tones=num_tones,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=total_gin_channels,  # 통합된 gin_channels 사용
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=total_gin_channels,  # 통합된 gin_channels 사용
        )
        if use_transformer_flow:
            self.flow = TransformerCouplingBlock(
                inter_channels,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers_trans_flow,
                5,
                p_dropout,
                n_flow_layer,
                gin_channels=total_gin_channels,  # 통합된 gin_channels 사용
                share_parameter=flow_share_parameter,
            )
        else:
            self.flow = ResidualCouplingBlock(
                inter_channels,
                hidden_channels,
                5,
                1,
                n_flow_layer,
                gin_channels=total_gin_channels,  # 통합된 gin_channels 사용
            )
        self.sdp = StochasticDurationPredictor(
            hidden_channels, 192, 3, 0.5, 4, gin_channels=total_gin_channels
        )
        self.dp = DurationPredictor(
            hidden_channels, 256, 3, 0.5, gin_channels=total_gin_channels
        )

        if n_speakers > 0:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
        else:
            self.ref_enc = ReferenceEncoder(spec_channels, gin_channels, layernorm=norm_refenc)
        self.use_vc = use_vc

    def forward(self, x, x_lengths, y, y_lengths, sid, emotion_ids, tone, language, bert, ja_bert):
        """
        Args:
            x: 텍스트 입력
            x_lengths: 텍스트 길이
            y: 스펙트로그램
            y_lengths: 스펙트로그램 길이
            sid: 화자 ID
            emotion_ids: 감정 ID [batch_size]
            tone: 톤
            language: 언어
            bert: BERT 임베딩
            ja_bert: 일본어 BERT 임베딩
        """
        # 화자 임베딩
        if self.n_speakers > 0:
            speaker_g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            speaker_g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
        
        # 감정 임베딩
        emotion_g = self.emotion_embedding(emotion_ids).unsqueeze(-1)  # [b, emotion_channels, 1]
        
        # 화자 + 감정 임베딩 결합
        g = torch.cat([speaker_g, emotion_g], dim=1)  # [b, gin_channels + emotion_channels, 1]
        
        if self.use_vc:
            g_p = None
        else:
            g_p = g
            
        x, m_p, logs_p, x_mask = self.enc_p(
            x, x_lengths, tone, language, bert, ja_bert, g=g_p
        )
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            neg_cent1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent2 = torch.matmul(
                -0.5 * (z_p**2).transpose(1, 2), s_p_sq_r
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(
                z_p.transpose(1, 2), (m_p * s_p_sq_r)
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent4 = torch.sum(
                -0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            if self.use_noise_scaled_mas:
                epsilon = (
                    torch.std(neg_cent)
                    * torch.randn_like(neg_cent)
                    * self.current_mas_noise_scale
                )
                neg_cent = neg_cent + epsilon

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = (
                monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1))
                .unsqueeze(1)
                .detach()
            )

        w = attn.sum(2)

        l_length_sdp = self.sdp(x, x_mask, w, g=g)
        l_length_sdp = l_length_sdp / torch.sum(x_mask)

        logw_ = torch.log(w + 1e-6) * x_mask
        logw = self.dp(x, x_mask, g=g)
        l_length_dp = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(
            x_mask
        )  # for averaging

        l_length = l_length_dp + l_length_sdp

        # expand prior
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        o = self.dec(z_slice, g=g)
        return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(
        self,
        x,
        x_lengths,
        sid,
        emotion_ids,
        tone,
        language,
        bert,
        ja_bert,
        noise_scale=0.667,
        length_scale=1,
        noise_scale_w=0.8,
        max_len=None,
        sdp_ratio=0,
        y=None,
        g=None,
    ):
        # 화자 임베딩
        if self.n_speakers > 0:
            speaker_g = self.emb_g(sid).unsqueeze(-1)
        else:
            speaker_g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
        
        # 감정 임베딩
        emotion_g = self.emotion_embedding(emotion_ids).unsqueeze(-1)
        
        # 화자 + 감정 임베딩 결합
        g = torch.cat([speaker_g, emotion_g], dim=1)
        
        x, m_p, logs_p, x_mask = self.enc_p(
            x, x_lengths, tone, language, bert, ja_bert, g=g
        )
        logw = self.sdp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w) * (
            sdp_ratio
        ) + self.dp(x, x_mask, g=g) * (1 - sdp_ratio)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)

    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt, emotion_ids_src, emotion_ids_tgt, tau=1.0):
        """
        음성 변환 (화자 + 감정)
        """
        # 소스 화자 + 감정 임베딩
        if self.n_speakers > 0:
            speaker_g_src = self.emb_g(sid_src).unsqueeze(-1)
        else:
            speaker_g_src = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
        emotion_g_src = self.emotion_embedding(emotion_ids_src).unsqueeze(-1)
        g_src = torch.cat([speaker_g_src, emotion_g_src], dim=1)
        
        # 타겟 화자 + 감정 임베딩
        if self.n_speakers > 0:
            speaker_g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
        else:
            speaker_g_tgt = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
        emotion_g_tgt = self.emotion_embedding(emotion_ids_tgt).unsqueeze(-1)
        g_tgt = torch.cat([speaker_g_tgt, emotion_g_tgt], dim=1)
        
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.dec(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat) 