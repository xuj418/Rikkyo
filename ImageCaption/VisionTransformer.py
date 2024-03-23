import tensorflow as tf
import numpy as np


class PatchEmbed(tf.keras.layers.Layer):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.embed_dim = embed_dim
        self.img_size = (img_size, img_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = tf.keras.layers.Conv2D(filters=embed_dim, kernel_size=patch_size,
                                           strides=patch_size, padding='SAME',
                                           kernel_initializer=tf.keras.initializers.LecunNormal(),
                                           bias_initializer=tf.keras.initializers.Zeros())

    def call(self, inputs, **kwargs):
        batch_size, height, width, channel = inputs.shape
        # print("inputs.shape:", inputs.shape) #(64, 224, 224, 3)
        assert height == self.img_size[0] and width == self.img_size[1], \
            f"Input image size ({height}*{width}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(inputs)
        # [B, H, W, C] -> [B, H*W, C]
        x = tf.reshape(x, [batch_size, self.num_patches, self.embed_dim])
        return x


class Attention(tf.keras.layers.Layer):
    k_ini = tf.keras.initializers.GlorotUniform()
    b_ini = tf.keras.initializers.Zeros()

    def __init__(self,
                 dim,
                 num_heads,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 name=None):
        super(Attention, self).__init__(name=name)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = tf.keras.layers.Dense(dim * 3, use_bias=qkv_bias, name="qkv",
                                         kernel_initializer=self.k_ini, bias_initializer=self.b_ini)
        self.attn_drop = tf.keras.layers.Dropout(attn_drop_ratio)
        self.proj = tf.keras.layers.Dense(dim, name="out",
                                          kernel_initializer=self.k_ini, bias_initializer=self.b_ini)
        self.proj_drop = tf.keras.layers.Dropout(proj_drop_ratio)

    def call(self, inputs, training=None):
        # [batch_size, num_patches + 1, total_embed_dim]
        batch_size, num_patches, embed_dim = inputs.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        qkv = self.qkv(inputs)
        # print(C // self.num_heads)

        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        qkv = tf.reshape(qkv, [batch_size, num_patches, 3, self.num_heads, embed_dim // self.num_heads])

        # transpose: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])

        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = tf.matmul(a=q, b=k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        # multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        x = tf.matmul(attn, v)
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        x = tf.transpose(x, [0, 2, 1, 3])
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = tf.reshape(x, [batch_size, num_patches, embed_dim])

        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x


class MLP(tf.keras.layers.Layer):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    k_ini = tf.keras.initializers.GlorotUniform()
    b_ini = tf.keras.initializers.RandomNormal(stddev=1e-6)

    def __init__(self, in_features, mlp_ratio=4.0, drop=0., name=None):
        super(MLP, self).__init__(name=name)
        self.fc1 = tf.keras.layers.Dense(int(in_features * mlp_ratio), name="Dense_0",
                                         kernel_initializer=self.k_ini, bias_initializer=self.b_ini)
        self.act = tf.keras.layers.Activation("gelu")
        self.fc2 = tf.keras.layers.Dense(in_features, name="Dense_1",
                                         kernel_initializer=self.k_ini, bias_initializer=self.b_ini)
        self.drop = tf.keras.layers.Dropout(drop)

    def call(self, inputs, training=None):
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 name=None):
        super(EncoderLayer, self).__init__(name=name)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="LayerNorm_0")

        self.attn = Attention(dim, num_heads=num_heads,
                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio,
                              name="MultiHeadAttention")

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = tf.keras.layers.Dropout(rate=drop_path_ratio, noise_shape=(None, 1, 1)) \
            if drop_path_ratio > 0. \
            else tf.keras.layers.Activation("linear")
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="LayerNorm_1")
        self.mlp = MLP(dim, drop=drop_ratio, name="MlpBlock")

    def call(self, inputs, training=None, **kwargs):
        norm = self.norm1(inputs)
        attn = self.attn(norm)
        x = inputs + self.drop_path(attn, training=training)

        x = x + self.drop_path(self.mlp(self.norm2(x)), training=training)
        return x


class EncoderModel(tf.keras.layers.Layer):
    def __init__(self, num_layers_encoder, num_heads_encoder, 
                 img_size=224, patch_size=16, embed_dim=768,
                 qkv_bias=True, qk_scale=None,
                 drop_ratio=0.1, attn_drop_ratio=0., drop_path_ratio=0.1,
                 representation_size=None, name="ViT-B/16"):
        super(EncoderModel, self).__init__(name=name)
        self.embed_dim = embed_dim
        self.depth = num_layers_encoder
        self.qkv_bias = qkv_bias

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.pos = positional_encoding(300, embed_dim)
        #self.cls_token_pos_embed = ConcatClassTokenAddPosEmbed(embed_dim=embed_dim,
        #                                                       num_patches=num_patches,
        #                                                       name="cls_pos")

        self.pos_drop = tf.keras.layers.Dropout(drop_ratio)

        dpr = np.linspace(0., drop_path_ratio, num_layers_encoder)  # stochastic depth decay rule
        self.blocks = [EncoderLayer(dim=embed_dim, num_heads=num_heads_encoder, qkv_bias=qkv_bias,
                                    qk_scale=qk_scale, drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                                    drop_path_ratio=dpr[i], name="encoder_block_{}".format(i))
                       for i in range(num_layers_encoder)]

        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="encoder_norm")

        #if representation_size:
        #    self.has_logits = True
        #    self.pre_logits = tf.keras.layers.Dense(representation_size, activation="tanh", name="pre_logits")
        #else:
        #    self.has_logits = False
        #    self.pre_logits = tf.keras.layers.Activation("linear")

        #self.head = tf.keras.layers.Dense(num_classes, name="head", kernel_initializer=tf.keras.initializers.Zeros())

    def call(self, inputs, training=None):
        # [B, H, W, C] -> [B, num_patches, embed_dim]
        x = self.patch_embed(inputs)  # [B, 196, 768]
        #x = self.cls_token_pos_embed(x)  # [B, 176, 768]
        
        # 初期
        # x.shape: (batch_size, sq_len)
        seq_len = tf.shape(x)[1]
        # print("x.shape:",x.shape)

        # x = x * tf.math.sqrt(tf.cast(self.embed_dim, dtype=tf.float32))

        # lay2: 位置エンコーディング
        # x.shape: (batch_size, sq_len, d_model)
        positional = self.pos[:, :seq_len, :]
        # print("positional:",positional.shape)
        x = x + positional
        # print("x2:", x.shape)
        
        x = self.pos_drop(x, training=training)

        for block in self.blocks:
            x = block(x, training=training)

        # print("vtx1.shape:", x.shape)
        x = self.norm(x)
        # print("vtx2.shape:", x.shape)
        # x = self.pre_logits(x[:, 0])
        # print("vtx3.shape:", x.shape)
        # x = self.head(x)
        # print("vtx4.shape:", x.shape)

        return x


# 位置エンコーディング
def get_angle(pos, i, d_model):
    angle = 1 / (np.power(10000, (2 * (i // 2)) / d_model))
    return pos * angle


def positional_encoding(position, d_model):
    # (position, 1) * (1 * d_model) = (position, d_model)
    pos = get_angle(np.arange(position)[:, np.newaxis],
                    np.arange(d_model)[np.newaxis, :],
                    d_model)

    pos[:, 0::2] = np.sin(pos[:, 0::2])
    pos[:, 1::2] = np.cos(pos[:, 1::2])

    # (1, position, d_model)
    pos = pos[np.newaxis, :]

    return pos


# padマスク、0の値は利用しない
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), dtype=tf.float32)
    # shape:(batch_size, 1, 1, seq_len)
    return seq[:, tf.newaxis, tf.newaxis, :]


# padマスク、0の値は利用しない
def create_padding_mask_image(img):
    img = tf.cast(tf.math.equal(img, 0), dtype=tf.float32)
    # shape:(batch_size, 1, seq_len)
    return img[:, tf.newaxis, :]


# decoder_layer用、未来の値を利用しない
# num_lower: <0 対角線下の数値を保留
# num_upper: <0 対角線上の数値を保留
def create_look_ahead_attention(size):
    # shape:(seq_len, seq_len)
    return 1 - tf.linalg.band_part(tf.ones(size), -1, 0)


# スケーリングドットアテンション
# q.shape : (..., seq_len_q, depth)
# k.shape : (..., seq_len_k, depth)
# v.shape : (..., seq_len_v, depth)
# seq_len_k = seq_len_v
def scaled_dot_product_attention(q, k, v, mask):
    # matmul_qk.shape : (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    attention_logics = matmul_qk / (tf.math.sqrt(dk))

    if mask is not None:
        # マスクした値には、attention_weight計算しない
        # softmax後、該当セルは０
        attention_logics = attention_logics + (mask * -1e9)

    # attention_weight.shape : (..., seq_len_q, seq_len_k)
    attention_weight = tf.nn.softmax(attention_logics, axis=-1)

    # output.shape : (..., seq_len_q, depth_v)
    output = tf.matmul(attention_weight, v)

    return output, attention_weight


# マルチヘッドアテンション
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name=None):
        super(MultiHeadAttention, self).__init__(name=name)

        self.d_model = d_model
        self.num_heads = num_heads

        # evenly divisible、均等に割り切れる
        assert self.d_model % self.num_heads == 0

        self.depth = tf.cast(d_model / self.num_heads, tf.int64)

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    # マルチヘッダ、d_model⇨(num_heads, depth)
    def split_heads(self, x, batch_size):
        # x.shape.     : (batch_size, seq_len, d_model)
        # →→ reshape.  : (batch_size, seq_len, num_heads, depth)
        # →→ transpose : (batch_size, num_heads, seq_len, depth)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask, **kwargs):

        batch_size = tf.cast(tf.shape(q)[0], tf.int64)

        # q.shape : (batch_size, seq_len_q, d_model)
        # k.shape : (batch_size, seq_len_k, d_model)
        # v.shape : (batch_size, seq_len_v, d_model)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # q.shape : (batch_size, num_heads, seq_len_q, depth)
        # k.shape : (batch_size, num_heads, seq_len_k, depth)
        # v.shape : (batch_size, num_heads, seq_len_v, depth)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # attention_weight.shape : (batch_size, num_heads, seq_len_q, seq_len_k)
        # output.shape           : (batch_size, num_heads, seq_len_q, depth)
        scaled_attention, attention_weight = scaled_dot_product_attention(q, k, v, mask)

        # output.shape           : (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # output.shape           : (batch_size, seq_len_q, d_model)
        scaled_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(scaled_attention)

        return output, attention_weight


def point_wise_feed_forward_network(d_model, diff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(units=diff, activation="relu"),
        tf.keras.layers.Dense(units=d_model)
    ])


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff):
        super(DecoderLayer, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff

        # Muti-Head Attention
        self.mha1 = MultiHeadAttention(self.d_model, self.num_heads)
        self.mha2 = MultiHeadAttention(self.d_model, self.num_heads)

        # Feed Forward Network
        self.ffn = point_wise_feed_forward_network(self.d_model, self.dff)

        # LayerNormalization
        self.layNorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layNorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layNorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout
        self.dropout1 = tf.keras.layers.Dropout(rate=0.1)
        self.dropout2 = tf.keras.layers.Dropout(rate=0.1)
        self.dropout3 = tf.keras.layers.Dropout(rate=0.1)

    def call(self, x, encoding_input, combined_mask, enc_dec_padding_mask):
        # lay1: Masked Muti-Head Attention
        mha_out1, attention_weight1 = self.mha1(x, x, x, mask=combined_mask)
        mha_out1 = self.dropout1(mha_out1)
        # print("mha_out1:", mha_out1.shape)

        # lay2: Add & Norm
        lay_norm_out1 = self.layNorm1(x + mha_out1)

        # lay3: Muti-Head Attention
        # v,k:encoding_input
        # q  :layNorm_out1
        mha_out2, attention_weight2 = self.mha2(
            encoding_input, encoding_input, lay_norm_out1, mask=enc_dec_padding_mask)
        # print("mha_out2:", mha_out2.shape)
        mha_out2 = self.dropout2(mha_out2)

        # lay4: Add & Norm
        lay_norm_out2 = self.layNorm2(lay_norm_out1 + mha_out2)

        # Lay5: Feed Forward
        ffn_out = self.ffn(lay_norm_out2)
        # print("ffn_out:", ffn_out.shape)
        ffn_out = self.dropout3(ffn_out)

        # lay6: Add & Norm
        lay_norm_out3 = self.layNorm3(lay_norm_out2 + ffn_out)

        return lay_norm_out3


class DecoderModel(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, max_position_len,
                 num_layers, num_heads, dff):
        super(DecoderModel, self).__init__()

        self.d_model = d_model
        self.max_position_len = max_position_len
        self.num_layers = num_layers

        self.emd = tf.keras.layers.Embedding(vocab_size, d_model)

        self.pos = positional_encoding(max_position_len, d_model)

        self.decoder = [DecoderLayer(d_model, num_heads, dff) for _ in range(num_layers)]

    def call(self, x, encoder_out, combined_mask, enc_dec_padding_mask):
        # 初期
        # x.shape: (batch_size, sq_len)
        seq_len = tf.shape(x)[1]
        # print("x.shape:",x.shape)

        # lay1: 埋め込み
        # x.shape: (batch_size, sq_len, d_model)
        x = self.emd(x)
        # print("x1:", x.shape)

        x = x * tf.math.sqrt(tf.cast(self.d_model, dtype=tf.float32))

        # lay2: 位置エンコーディング
        # x.shape: (batch_size, sq_len, d_model)
        positional = self.pos[:, :seq_len, :]
        # print("positional:",positional.shape)
        x = x + positional
        # print("x2:", x.shape)

        # lay3: エンコーディング
        # x.shape: (batch_size, sq_len, d_model)
        for i in range(self.num_layers):
            x = self.decoder[i](x, encoder_out, combined_mask, enc_dec_padding_mask)

        # (batch_size, tar_seq_len, d_model)
        return x


class Transformer(tf.keras.Model):
    def __init__(self, target_vocab_size, target_seq_len, d_model, 
                 num_heads_encoder, num_layers_encoder, num_heads_decoder, num_layers_decoder, dff):
        super().__init__()

        # Encoding
        self.encoder_model = EncoderModel(num_layers_encoder, num_heads_encoder)
        #self.encoder_model = EncoderModel()

        # Decoding
        self.decoder_model = DecoderModel(target_vocab_size, d_model, target_seq_len,
                                          num_layers_decoder, num_heads_decoder, dff)

        # 全結合層
        self.linear = tf.keras.layers.Dense(target_vocab_size)

    def call(self, input_target):
        # pairとして入力
        x_input, y_target = input_target
        # print("x_input:", x_input)
        # print("y_target:", y_target)

        # マスク
        enc_padding_mask, combined_mask, enc_dec_padding_mask = self.create_mask(y_target)

        # encode.shape:(batch_size, seq_len, d_model)
        encoder_output = self.encoder_model(x_input)

        # decode.shape:(batch_size, seq_len, d_model)
        decoder_output = self.decoder_model(y_target, encoder_output,
                                            combined_mask=combined_mask,
                                            enc_dec_padding_mask=enc_dec_padding_mask)

        # (batch_size, seq_len, vocab_size)
        output = self.linear(decoder_output)

        # output = tf.nn.softmax(output)

        return output

    def create_mask(self, tar):
        # マスク１：エンコーディングのMulti-Head Attention用
        # enc_padding_mask = create_padding_mask(inp)

        # マスク２：デンコーディングのMulti-Head Attention用
        # エンコーディングのINPUT
        # enc_dec_padding_mask = create_padding_mask_image(inp)

        # マクス３：デンコーディングのLAY1用
        size = (tar.shape[1], tar.shape[1])
        look_ahead_mask = create_look_ahead_attention(size)
        dec_padding_mask = create_padding_mask(tar)

        combined_mask = tf.maximum(look_ahead_mask, dec_padding_mask)

        return None, combined_mask, None
