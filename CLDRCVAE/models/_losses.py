import tensorflow as tf
from keras import backend as K
from keras import losses

from ._utils import compute_mmd, _nelem, _nan2zero, _nan2inf, _reduce_mean


def kl_recon_mse(mu, log_var, alpha=0.1, eta=1.0):

    def kl_recon_loss(y_true, y_pred):
        kl_loss = 0.5 * K.mean(K.exp(log_var) + K.square(mu) - 1. - log_var, 1)
        recon_loss = 0.5 * losses.mean_squared_error(y_true, y_pred)

        return eta * recon_loss + alpha * kl_loss

    return kl_recon_loss


def kl_recon_sse(mu, log_var, alpha=0.1, eta=1.0):
    def kl_recon_loss(y_true, y_pred):
        # 计算KL散度损失
        kl_loss = 0.5 * K.mean(K.exp(log_var) + K.square(mu) - 1. - log_var, 1)
        # 计算SSE重构损失
        recon_loss = 0.5 * K.sum(K.square((y_true - y_pred)), axis=-1)

        # 加权组合KL和SSE损失
        return eta * recon_loss + alpha * kl_loss

    return kl_recon_loss


def pure_kl_loss(mu, log_var):

    def kl_loss(y_true, y_pred):
        kl_div = K.mean(K.exp(log_var) + K.square(mu) - 1. - log_var, 1)
        return kl_div

    kl_loss.__name__ = "kl"

    return kl_loss


def sse_loss(y_true, y_pred):
    return K.sum(K.square((y_true - y_pred)), axis=-1)


def mse_loss(y_true, y_pred):
    return losses.mean_squared_error(y_true, y_pred)


def mmd(n_conditions, beta, kernel_method='multi-scale-rbf', computation_method="general"):
    def mmd_loss(real_labels, y_pred):
        with tf.variable_scope("mmd_loss", reuse=tf.AUTO_REUSE):
            real_labels = K.reshape(K.cast(real_labels, 'int32'), (-1,))
            conditions_mmd = tf.dynamic_partition(y_pred, real_labels, num_partitions=n_conditions)
            loss = 0.0
            if computation_method.isdigit():
                boundary = int(computation_method)
                for i in range(boundary):
                    for j in range(boundary, n_conditions):
                        loss += _nan2zero(compute_mmd(conditions_mmd[i], conditions_mmd[j], kernel_method))
            else:
                for i in range(len(conditions_mmd)):
                    for j in range(i):
                        loss += _nan2zero(compute_mmd(conditions_mmd[i], conditions_mmd[j], kernel_method))
            if n_conditions == 1:
                loss = _nan2zero(tf.zeros(shape=(1,)))
            return beta * loss

    return mmd_loss


# NB loss and ZINB are taken from https://github.com/theislab/dca, thanks to @gokceneraslan

class NB(object):
    def __init__(self, theta, masking=False, scope='nbinom_loss/', scale_factor=1.0):

        # for numerical stability
        self.eps = 1e-8
        self.scale_factor = scale_factor
        self.scope = scope
        self.masking = masking
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps

        with tf.name_scope(self.scope):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32) * scale_factor

            if self.masking:
                nelem = _nelem(y_true)
                y_true = _nan2zero(y_true)

            # Clip theta
            theta = tf.minimum(self.theta, 1e6)

            t1 = tf.lgamma(theta + eps) + tf.lgamma(y_true + 1.0) - tf.lgamma(y_true + theta + eps)
            t2 = (theta + y_true) * tf.log(1.0 + (y_pred / (theta + eps))) + (
                    y_true * (tf.log(theta + eps) - tf.log(y_pred + eps)))
            final = t1 + t2

            final = _nan2inf(final)

            if mean:
                if self.masking:
                    final = tf.divide(tf.reduce_sum(final), nelem)
                else:
                    final = tf.reduce_mean(final)
            else:
                final = tf.reduce_sum(final)

        return final


class ZINB(NB):
    def __init__(self, pi, ridge_lambda=0.0, scope='zinb_loss/', **kwargs):
        super().__init__(scope=scope, **kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps

        with tf.name_scope(self.scope):
            # reuse existing NB neg.log.lik.
            # mean is always False here, because everything is calculated
            # element-wise. we take the mean only in the end
            nb_case = super().loss(y_true, y_pred, mean=False) - tf.log(1.0 - self.pi + eps)

            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32) * scale_factor
            theta = tf.minimum(self.theta, 1e6)

            zero_nb = tf.pow(theta / (theta + y_pred + eps), theta)
            zero_case = -tf.log(self.pi + ((1.0 - self.pi) * zero_nb) + eps)
            result = tf.where(tf.less(y_true, 1e-8), zero_case, nb_case)
            ridge = self.ridge_lambda * tf.square(self.pi)
            result += ridge

            if mean:
                if self.masking:
                    result = _reduce_mean(result)
                else:
                    result = tf.reduce_mean(result)

            result = _nan2inf(result)

        return result

def nb_kl_loss(disp, mu, log_var, scale_factor=1.0, alpha=0.1, eta=1.0):

    kl = pure_kl_loss(mu, log_var)

    def nb(y_true, y_pred):
        nb_obj = NB(theta=disp, masking=False, scale_factor=scale_factor)
        nb_loss_value = eta * nb_obj.loss(y_true, y_pred, mean=True)

        return nb_loss_value + alpha * kl(y_true, y_pred)

    nb.__name__ = 'nb_kl'
    return nb


def nb_loss(disp, scale_factor=1.0, eta=1.0):
    def nb(y_true, y_pred):
        nb_obj = NB(theta=disp, masking=False, scale_factor=scale_factor)
        return eta * nb_obj.loss(y_true, y_pred, mean=True)

    nb.__name__ = 'nb'
    return nb


def zinb_kl_loss(pi, disp, mu, log_var, ridge=0.1, alpha=0.1, eta=1.0):
    kl = pure_kl_loss(mu, log_var)

    def zinb(y_true, y_pred):
        zinb_obj = ZINB(pi, theta=disp, ridge_lambda=ridge)
        zinb_loss_value = eta * zinb_obj.loss(y_true, y_pred)

        return zinb_loss_value + alpha * kl(y_true, y_pred)

    zinb.__name__ = 'zinb_kl'
    return zinb


def zinb_loss(pi, disp, ridge=0.1, eta=1.0):
    def zinb(y_true, y_pred):
        zinb_obj = ZINB(pi, theta=disp, ridge_lambda=ridge)
        return eta * zinb_obj.loss(y_true, y_pred)

    zinb.__name__ = 'zinb'
    return zinb


def cce_loss(gamma):
    def cce(y_true, y_pred):
        return gamma * K.categorical_crossentropy(y_true, y_pred)

    cce.__name__ = "cce"
    return cce


def accuracy(y_true, y_pred):
    y_true = K.argmax(y_true, axis=1)
    y_pred = K.argmax(y_pred, axis=1)
    return K.mean(K.equal(y_true, y_pred))


def contrastive_loss_fn(contrastive_lambda, z, contrastive_pos_z, contrastive_neg_z):

    def contrastive_loss(y_true, y_pred):
        # 对 z、正样本和负样本进行 l2 归一化
        z_normalized = tf.nn.l2_normalize(z, axis=-1)
        pos_z_normalized = tf.nn.l2_normalize(contrastive_pos_z, axis=-1)
        neg_z_normalized = tf.nn.l2_normalize(contrastive_neg_z, axis=-1)

        # 计算正样本的余弦相似度
        positive_component = tf.reduce_mean(tf.reduce_sum(z_normalized * pos_z_normalized, axis=-1))

        # 计算负样本的余弦相似度
        negative_component = tf.reduce_mean(tf.reduce_sum(z_normalized * neg_z_normalized, axis=-1))

        # 使用对比损失公式，最大化正样本相似度，最小化负样本相似度
        contrastive_loss = -tf.math.log(
            tf.exp(positive_component) / (tf.exp(positive_component) + 0.5 * tf.exp(negative_component))
        )
        return contrastive_lambda * contrastive_loss

    return contrastive_loss


def second_contrastive_loss_fn(z_common, z_specific, pos_z_common, pos_z_specific, second_contrastive_lambda, margin=1.0):

    def second_contrastive_loss(y_true, y_pred):
        # 共性部分的相似性约束 (L_A)
        def compute_L_A(z_common, pos_z_common):
            """
            计算共性部分的相似性约束，使 z_common 和 pos_z_common 之间尽可能接近。
            """
            loss = tf.reduce_mean(tf.square(z_common - pos_z_common))
            return loss

        # 特异性部分的分离性约束 (L_B)
        def compute_L_B(z_specific, pos_z_specific, margin):
            """
            计算特异性部分的分离性约束，使 z_specific 和 pos_z_specific 之间尽可能不同。
            使用带边界的欧氏距离约束。
            """
            distance = tf.reduce_mean(tf.square(z_specific - pos_z_specific))
            loss = tf.reduce_mean(tf.maximum(0.0, margin - distance))
            return loss

        # 协方差约束 (L_cov)
        def compute_L_cov(z_common, z_specific):
            """
            计算 z_common 和 z_specific 之间的协方差约束，最小化它们之间的线性相关性。
            """
            batch_size = tf.shape(z_common)[0]

            # 对 z_common 和 z_specific 进行中心化
            z_common_centered = z_common - tf.reduce_mean(z_common, axis=0, keepdims=True)
            z_specific_centered = z_specific - tf.reduce_mean(z_specific, axis=0, keepdims=True)

            # 计算协方差矩阵
            cov_matrix = tf.matmul(tf.transpose(z_common_centered), z_specific_centered) / tf.cast(batch_size, tf.float32)

            # 计算协方差约束损失（平方和）
            loss = tf.reduce_sum(tf.square(cov_matrix))
            return loss

        # 计算各个损失项
        L_A = compute_L_A(z_common, pos_z_common)
        L_B = compute_L_B(z_specific, pos_z_specific, margin)
        L_cov = compute_L_cov(z_common, z_specific)

        # 合并损失项
        total_loss = 1.0 * L_A + 1.0 * L_B + 0.5 * L_cov

        return second_contrastive_lambda * total_loss

    return second_contrastive_loss

def poisson_recon(y_true, y_pred):
    return tf.reduce_mean(y_pred - y_true * tf.math.log(y_pred + 1e-8))

def poisson_kl_loss(mu, log_var, alpha=0.1, eta=1.0):
    kl = pure_kl_loss(mu, log_var)
    def loss(y_true, y_pred):
        return eta * poisson_recon(y_true, y_pred) + alpha * kl(y_true, y_pred)
    return loss

LOSSES = {
    "mse": kl_recon_mse,                  
    "sse": kl_recon_sse,                   
    "mmd": mmd,                            
    "nb": nb_kl_loss,                      
    "zinb": zinb_kl_loss,                    
    "cce": cce_loss,                  
    "kl": pure_kl_loss,                      
    "sse_recon": sse_loss,                     
    "mse_recon": mse_loss,                     
    'contrastive': contrastive_loss_fn,         
    "second_contrastive": second_contrastive_loss_fn, 
    "nb_wo_kl": nb_loss,                        
    "zinb_wo_kl": zinb_loss,                    
    "acc": accuracy,                           

    "poisson": poisson_kl_loss,          
    "poisson_recon": poisson_recon,      
}

