import os

import anndata
import keras
import numpy as np
import tensorflow as tf

from keras.callbacks import EarlyStopping, History, ReduceLROnPlateau, LambdaCallback
from keras.engine.saving import model_from_json
from keras.layers import Dense, BatchNormalization, Dropout, Lambda, Input, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from scipy import sparse

from trvae.models._activations import ACTIVATIONS
from trvae.models._layers import LAYERS
from trvae.models._losses import LOSSES
from trvae.models._utils import print_progress, sample_z
from trvae.utils import label_encoder, train_test_split, remove_sparsity

class CLDRCVAE(object):
    
    def __init__(self, gene_size: int, conditions: list, cell_types: list, n_topic=10, **kwargs):
        self.gene_size = gene_size   
        self.n_topic = n_topic  

        self.conditions = sorted(conditions)
        self.n_conditions = len(self.conditions)  
        self.cell_types = sorted(cell_types)  
        self.n_cell_types = len(self.cell_types)  
        self.cell_type_key = kwargs.get('cell_type_key', 'cell_type')

        self.lr = kwargs.get("learning_rate", 0.001)  
        self.alpha = kwargs.get("alpha", 0.0001) 
        self.eta = kwargs.get("eta", 50.0)  
        self.dr_rate = kwargs.get("dropout_rate", 0.1)  
        self.model_path = kwargs.get("model_path", "./models/CLDRCVAE/") 
        self.loss_fn = kwargs.get("loss_fn", 'mse') 
        self.ridge = kwargs.get('ridge', 0.1) 
        self.scale_factor = kwargs.get("scale_factor", 1.0) 
        self.clip_value = kwargs.get('clip_value', 3.0) 
        self.epsilon = kwargs.get('epsilon', 0.01) 
        self.output_activation = kwargs.get("output_activation", 'linear') 
        self.use_batchnorm = kwargs.get("use_batchnorm", True) 

        self.architecture = kwargs.get("architecture", [128, 128]) 
        self.size_factor_key = kwargs.get("size_factor_key", 'size_factors')  
        self.device = kwargs.get("device", "gpu") if len(K.tensorflow_backend._get_available_gpus()) > 0 else 'cpu'

        self.gene_names = kwargs.get("gene_names", None)
        self.model_name = kwargs.get("model_name", "CLDRCVAE")
        self.class_name = kwargs.get("class_name", 'CLDRCVAE')

        self.x = Input(shape=(self.gene_size,), name="data")
        self.size_factor = Input(shape=(1,), name='size_factor')
        self.encoder_labels = Input(shape=(self.n_conditions,), name="encoder_labels")
        self.decoder_labels = Input(shape=(self.n_conditions,), name="decoder_labels")
        self.cell_type_labels = Input(shape=(self.n_cell_types,), name="cell_type_labels")
        self.x_hat = tf.random.normal([1, gene_size])  
        self.z = Input(shape=(self.n_topic,), name="latent_data")

        self.topk = kwargs.pop("topk", 5) 
        self.contrastive_lambda = kwargs.pop("contrastive_lambda", 10.0) 
        self.contrastive_x = Input(shape=(self.gene_size,), name="contrastive_data") 
        self.contrastive_labels = Input(shape=(self.n_conditions,), name="contrastive_labels") 
        self.contrastive_z = Input(shape=(self.n_topic,), name="contrastive_z") 
        self.gamma_cov = kwargs.get('gamma_cov', 0.1)  
        self.second_contrastive_lambda= kwargs.pop("second_contrastive_lambda", 10.0)  
        self.margin = kwargs.get('margin', 1.0) 
        self.common_dim = kwargs.get('common_dim', 25) 

        self.condition_encoder = kwargs.get("condition_encoder", None)

        self.disp = tf.Variable(
            initial_value=np.ones(self.gene_size),
            dtype=tf.float32,
            name="disp"
        )

        self.network_kwargs = {
            "gene_size": self.gene_size,
            "n_topic": self.n_topic,
            "conditions": self.conditions,
            "dropout_rate": self.dr_rate,
            "loss_fn": self.loss_fn,
            "output_activation": self.output_activation,
            "size_factor_key": self.size_factor_key,
            "architecture": self.architecture,
            "use_batchnorm": self.use_batchnorm,
            "gene_names": self.gene_names,
            "condition_encoder": self.condition_encoder,
            "train_device": self.device,
        }

        self.training_kwargs = {
            "learning_rate": self.lr,
            "alpha": self.alpha,
            "eta": self.eta,
            "ridge": self.ridge,
            "scale_factor": self.scale_factor,
            "clip_value": self.clip_value,
            "model_path": self.model_path,
        }

        self.beta = kwargs.get('beta', 50.0)
        self.mmd_computation_method = kwargs.pop("mmd_computation_method", "general")

        kwargs.update({"model_name": "cvae", "class_name": "CLDRCVAE"})

        self.network_kwargs.update({
            "mmd_computation_method": self.mmd_computation_method,
            "contrastive_lambda": self.contrastive_lambda,
        })

        self.training_kwargs.update({
            "beta": self.beta,
            "contrastive_lambda": self.contrastive_lambda,
        })

        self.init_w = keras.initializers.glorot_normal()

        if kwargs.get("construct_model", True):
            self.construct_network()

        if kwargs.get("construct_model", True) and kwargs.get("compile_model", True):
            self.compile_models()

        print_summary = kwargs.get("print_summary", False)
        if print_summary:
            self.encoder_model.summary()
            self.decoder_model.summary()
            self.cvae_model.summary()


    @classmethod
    def from_config(cls, config_path, new_params=None, compile=True, construct=True):
       
        import json

        with open(config_path, 'rb') as f:
            class_config = json.load(f)

        class_config['construct_model'] = construct
        class_config['compile_model'] = compile

        if new_params:
            class_config.update(new_params)

        return cls(**class_config)


    def _encoder(self, name="encoder"):
        
        h = concatenate([self.x, self.encoder_labels], axis=1)

        for idx, n_neuron in enumerate(self.architecture):
            h = Dense(n_neuron, kernel_initializer=self.init_w, use_bias=False)(h)
            if self.use_batchnorm:
                h = BatchNormalization()(h)
            h = LeakyReLU()(h)
            if self.dr_rate > 0:
                h = Dropout(self.dr_rate)(h)  

        mean = Dense(self.n_topic, kernel_initializer=self.init_w)(h)
        log_var = Dense(self.n_topic, kernel_initializer=self.init_w)(h)

        z = Lambda(sample_z, output_shape=(self.n_topic,))([mean, log_var])

        model = Model(inputs=[self.x, self.encoder_labels], outputs=[mean, log_var, z], name=name)

        return mean, log_var, model


    def _output_decoder(self, h):
    
        h = Dense(self.gene_size, activation=None,
                  kernel_initializer=self.init_w,
                  use_bias=True)(h)
        h = ACTIVATIONS[self.output_activation](h)

        model_inputs = [self.z, self.decoder_labels]
        model_outputs = [h]

        return model_inputs, model_outputs


    def _decoder(self, name="decoder"):
        
        h = concatenate([self.z, self.decoder_labels], axis=1)

        for idx, n_neuron in enumerate(self.architecture[::-1]):
            h = Dense(n_neuron, kernel_initializer=self.init_w, use_bias=False)(h)
            if self.use_batchnorm:
                h = BatchNormalization()(h)
            h = LeakyReLU()(h)
           
            if idx == 0:
                h_mmd = h
            if self.dr_rate > 0:
                h = Dropout(self.dr_rate)(h)

        model_inputs, model_outputs = self._output_decoder(h)

        model = Model(inputs=model_inputs, outputs=model_outputs, name=name)
        
        mmd_model = Model(inputs=model_inputs, outputs=h_mmd, name='mmd_decoder')
        return model, mmd_model

    def construct_network(self):
        
        self.mu, self.log_var, self.encoder_model = self._encoder(name="encoder")
        self.decoder_model, self.decoder_mmd_model = self._decoder(name="decoder")

        inputs = [self.x, self.encoder_labels, self.decoder_labels, self.cell_type_labels]
        self.z = self.encoder_model(inputs[:2])[2]
        self.z_common = self.z[:, :self.common_dim]  
        self.z_specific = self.z[:, self.common_dim:self.n_topic]  
        decoder_inputs = [self.z, self.decoder_labels]

        self.decoder_outputs = self.decoder_model(decoder_inputs)
        decoder_mmd_outputs = self.decoder_mmd_model(decoder_inputs)

        reconstruction_output = Lambda(lambda x: x, name="reconstruction")(self.decoder_outputs)  
        mmd_output = Lambda(lambda x: x, name="mmd")(decoder_mmd_outputs)  
        contrastive_output = Lambda(lambda x: x, name="contrastive")(self.z)
        second_contrastive_output = Lambda(lambda x: x, name="second_contrastive")(self.z)

        self.cvae_model = Model(inputs=inputs,
                                outputs=[reconstruction_output, mmd_output, contrastive_output,
                                         second_contrastive_output],
                                name="cvae")

        self.custom_objects = {'mean_activation': ACTIVATIONS['mean_activation'],
                               'disp_activation': ACTIVATIONS['disp_activation'],
                               'SliceLayer': LAYERS['SliceLayer'],
                               'ColwiseMultLayer': LAYERS['ColWiseMultLayer'],
                               }

        get_custom_objects().update(self.custom_objects)
        print(f"{self.class_name}' network has been successfully constructed!")


    def _calculate_loss(self):

        mmd_loss = LOSSES['mmd'](self.n_conditions, self.beta) 
        kl_loss = LOSSES['kl'](self.mu, self.log_var)

        if self.loss_fn == 'nb':
            
            recon_loss = LOSSES['nb_wo_kl'](self.disp) 
        elif self.loss_fn == 'zinb':
            recon_loss = LOSSES['zinb_wo_kl']
        else:
            recon_loss = LOSSES[f'{self.loss_fn}_recon']

        contrastive_pos_x, contrastive_pos_labels, contrastive_neg_x, contrastive_neg_labels = \
            self.generate_contrastive_samples(self.x, self.encoder_labels, self.cell_type_labels)
        self._contrastive_learning(contrastive_pos_x, contrastive_pos_labels, contrastive_neg_x, contrastive_neg_labels)
        contrastive_loss = LOSSES['contrastive'](self.contrastive_lambda, self.z, self.contrastive_pos_z,
                                                 self.contrastive_neg_z)

        contrastive_pos_x, contrastive_pos_labels = self.select_samples_for_second_contrastive(self.x, self.encoder_labels, self.cell_type_labels)
        pos_z_common, pos_z_specific = self._contrastive_learning_second(contrastive_pos_x, contrastive_pos_labels)
        second_contrastive_loss = LOSSES['second_contrastive'](
            self.z_common,
            self.z_specific,
            pos_z_common,
            pos_z_specific,
            self.second_contrastive_lambda,
            margin=self.margin
        )

        loss = LOSSES[self.loss_fn](self.mu, self.log_var, self.alpha, self.eta)

        return loss, recon_loss, mmd_loss, kl_loss, contrastive_loss, second_contrastive_loss


    def compile_models(self):

        optimizer = keras.optimizers.Adam(lr=self.lr, clipvalue=self.clip_value, epsilon=self.epsilon)

        loss, recon_loss, mmd_loss, kl_loss, contrastive_loss, second_contrastive_loss = self._calculate_loss()

        self.cvae_model.compile(optimizer=optimizer,
                                loss=[loss, mmd_loss, contrastive_loss, second_contrastive_loss],
                                metrics={self.cvae_model.outputs[0].name: loss,
                                         self.cvae_model.outputs[1].name: mmd_loss,
                                         self.cvae_model.outputs[2].name: contrastive_loss,
                                         self.cvae_model.outputs[3].name: second_contrastive_loss}
                                )
        print("CLDRCVAE's network has been successfully compiled!")

    @tf.function
    def generate_contrastive_samples(self, X, encoder_labels, cell_type_labels):

        def calculate_top_k_genes(cell_type):
            cell_type = tf.cast(cell_type, dtype=cell_type_labels.dtype)
            indices = tf.where(tf.equal(cell_type_labels, cell_type))[:, 0]
            cell_type_data = tf.gather(X, indices)

            variance_per_gene = tf.math.reduce_variance(cell_type_data, axis=0)
            top_k_genes = tf.argsort(variance_per_gene, direction='DESCENDING')[:self.topk]
            return top_k_genes

        unique_cell_types = tf.cast(
            tf.reduce_any(tf.equal(tf.reshape(cell_type_labels, [-1, 1]),
                                   tf.range(self.n_cell_types, dtype=cell_type_labels.dtype)), axis=0),
            dtype=tf.float32
        )
        unique_cell_types = tf.reshape(unique_cell_types, [1, -1])

        top_k_genes_list = tf.map_fn(
            lambda ct: calculate_top_k_genes(ct) if unique_cell_types[0, ct] == 1 else tf.zeros(self.topk, dtype=tf.int32),
            tf.range(self.n_cell_types),
            dtype=tf.int32
        )

        def process_sample(i):
            condition_label_i = tf.expand_dims(tf.gather(encoder_labels, i), axis=0)
            cell_type_label_i = tf.expand_dims(tf.gather(cell_type_labels, i), axis=0)

            same_condition_indices = tf.where(
                (tf.reduce_all(tf.equal(encoder_labels, condition_label_i), axis=1)) &
                (tf.reduce_all(tf.equal(cell_type_labels, cell_type_label_i), axis=1))
            )[:, 0]

            if tf.size(same_condition_indices) > 0:
                chosen_index = tf.random.shuffle(same_condition_indices)[0]
                contrastive_pos_sample = X[chosen_index]
            else:
                contrastive_pos_sample = X[i]

            exists_in_unique = tf.reduce_any(tf.reduce_all(tf.equal(unique_cell_types, cell_type_label_i), axis=1))
            if not exists_in_unique:
                return contrastive_pos_sample, encoder_labels[i], X[i], encoder_labels[i]

            cell_type_index = tf.where(tf.reduce_all(tf.equal(unique_cell_types, cell_type_label_i), axis=1))[0][0]
            top_k_genes = top_k_genes_list[cell_type_index]

            contrastive_neg_sample = tf.identity(X[i])
            contrastive_neg_sample = tf.tensor_scatter_nd_update(
                contrastive_neg_sample,
                indices=tf.reshape(top_k_genes, [-1, 1]),
                updates=tf.gather(self.x_hat[i], top_k_genes)
            )

            return contrastive_pos_sample, encoder_labels[i], contrastive_neg_sample, encoder_labels[i]

        contrastive_pos_X, contrastive_pos_labels, contrastive_neg_X, contrastive_neg_labels = tf.map_fn(
            lambda i: process_sample(i),
            elems=tf.range(tf.shape(X)[0]),
            dtype=(tf.float32, tf.float32, tf.float32, tf.float32)
        )

        return contrastive_pos_X, contrastive_pos_labels, contrastive_neg_X, contrastive_neg_labels

    def _contrastive_learning(self, contrastive_pos_x, contrastive_pos_labels, contrastive_neg_x, contrastive_neg_labels):

        contrastive_pos_encoder_inputs = [contrastive_pos_x, contrastive_pos_labels]
        contrastive_neg_encoder_inputs = [contrastive_neg_x, contrastive_neg_labels]

        self.contrastive_pos_z = self.encoder_model(contrastive_pos_encoder_inputs)
        self.contrastive_neg_z = self.encoder_model(contrastive_neg_encoder_inputs)


    def select_samples_for_second_contrastive(self, X, encoder_labels, cell_type_labels):

        def process_sample(i):
            same_condition_indices = \
            np.where((encoder_labels != encoder_labels[i]) & (cell_type_labels == cell_type_labels[i]))[0]

            same_type_indices = same_condition_indices[same_condition_indices != i]

            if len(same_condition_indices) > 0:
                chosen_index = np.random.choice(same_type_indices, 1)[0]
                contrastive_sample = X[chosen_index]
                return contrastive_sample, encoder_labels[i]
            else:
                return X[i], encoder_labels[i]

        pos_samples, pos_labels = tf.map_fn(
            lambda i: process_sample(i),
            elems=tf.range(tf.shape(X)[0]),
            dtype=(X.dtype, encoder_labels.dtype)
        )

        return pos_samples, pos_labels


    def _contrastive_learning_second(self, contrastive_x, contrastive_labels):

        contrastive_encoder_inputs = [contrastive_x, contrastive_labels]

        contrastive_z = self.encoder_model(contrastive_encoder_inputs)[2] 
        contrastive_z_common = contrastive_z[:, :self.common_dim] 
        contrastive_z_specific = contrastive_z[:, self.common_dim:self.n_topic] 

        return contrastive_z_common, contrastive_z_specific


    def to_mmd_layer(self, adata, batch_key, cell_type_labels):

        adata = remove_sparsity(adata)

        encoder_labels, _ = label_encoder(adata, self.condition_encoder, batch_key)
        decoder_labels, _ = label_encoder(adata, self.condition_encoder, batch_key)

        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_conditions)
        decoder_labels = to_categorical(decoder_labels, num_classes=self.n_conditions)

        cvae_inputs = [adata.X, encoder_labels, decoder_labels, cell_type_labels]

        mmd = self.cvae_model.predict(cvae_inputs)[1]
        mmd = np.nan_to_num(mmd, nan=0.0, posinf=0.0, neginf=0.0)

        adata_mmd = anndata.AnnData(X=mmd)
        adata_mmd.obs = adata.obs.copy(deep=True)

        return adata_mmd


    def to_z_latent(self, adata, batch_key, cell_type_labels):
        
        if sparse.issparse(adata.X):
            adata.X = adata.X.A

        encoder_labels, _ = label_encoder(adata, self.condition_encoder, batch_key)
        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_conditions)
        
        if cell_type_labels is not None and len(self.encoder_model.inputs) == 3:
            latent = self.encoder_model.predict([adata.X, encoder_labels, cell_type_labels])[2]
        else:
            latent = self.encoder_model.predict([adata.X, encoder_labels])[2]  

        latent = np.nan_to_num(latent)

        latent_adata = anndata.AnnData(X=latent)
        latent_adata.obs = adata.obs.copy(deep=True)

        return latent_adata


    def get_latent(self, adata, batch_key, return_z=True):
        
        if set(self.gene_names).issubset(set(adata.var_names)):
            adata = adata[:, self.gene_names]
        else:
            raise Exception("set of gene names in train adata are inconsistent with trVAE'sgene_names")

        if self.beta == 0:
            return_z = True

        le = LabelEncoder()
        integer_encoded_labels = le.fit_transform(adata.obs[self.cell_type_key])
        cell_type_labels = to_categorical(integer_encoded_labels, num_classes=self.n_cell_types)

        if return_z or self.beta == 0:
            return self.to_z_latent(adata, batch_key, cell_type_labels)
        else:
            return self.to_mmd_layer(adata, batch_key, cell_type_labels)


    def predict(self, adata, condition_key, target_condition=None):
        
        adata = remove_sparsity(adata)

        encoder_labels, _ = label_encoder(adata, self.condition_encoder, condition_key)
        if target_condition is not None:
            decoder_labels = np.zeros_like(encoder_labels) + self.condition_encoder[target_condition]
        else:
            decoder_labels, _ = label_encoder(adata, self.condition_encoder, condition_key)

        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_conditions)
        decoder_labels = to_categorical(decoder_labels, num_classes=self.n_conditions)

        if hasattr(self, 'cell_type_encoder'):
            new_labels = set(adata.obs[self.cell_type_key]) - set(self.cell_type_encoder.classes_)

            if new_labels:
                self.cell_type_encoder.classes_ = np.append(self.cell_type_encoder.classes_, list(new_labels))

            integer_encoded_labels = self.cell_type_encoder.transform(adata.obs[self.cell_type_key])
        else:
            self.cell_type_encoder = LabelEncoder()
            integer_encoded_labels = self.cell_type_encoder.fit_transform(adata.obs[self.cell_type_key])

        cell_type_labels = to_categorical(integer_encoded_labels, num_classes=self.n_cell_types)

        x_hat = self.cvae_model.predict([adata.X, encoder_labels, decoder_labels, cell_type_labels])[0]

        adata_pred = anndata.AnnData(X=x_hat)
        adata_pred.obs = adata.obs
        adata_pred.var_names = adata.var_names

        return adata_pred


    def restore_model_weights(self, compile=True):

        if os.path.exists(os.path.join(self.model_path, f"{self.model_name}.h5")):
            self.cvae_model.load_weights(os.path.join(self.model_path, f'{self.model_name}.h5'))

            self.encoder_model = self.cvae_model.get_layer("encoder")
            self.decoder_model = self.cvae_model.get_layer("decoder")

            if compile:
                self.compile_models()
            print(f"{self.model_name}'s weights has been successfully restored!")
            return True
        return False


    def restore_model_config(self, compile=True):
        
        if os.path.exists(os.path.join(self.model_path, f"{self.model_name}.json")):
            json_file = open(os.path.join(self.model_path, f"{self.model_name}.json"), 'rb')
            loaded_model_json = json_file.read()
            self.cvae_model = model_from_json(loaded_model_json)
            self.encoder_model = self.cvae_model.get_layer("encoder")
            self.decoder_model = self.cvae_model.get_layer("decoder")

            if compile:
                self.compile_models()

            print(f"{self.model_name}'s network's config has been successfully restored!")
            return True
        else:
            return False


    def restore_class_config(self, compile_and_consturct=True):
        
        import json
        if os.path.exists(os.path.join(self.model_path, f"{self.class_name}.json")):
            with open(os.path.join(self.model_path, f"{self.class_name}.json"), 'rb') as f:
                trVAE_config = json.load(f)

            for key, value in trVAE_config.items():
                if key in self.network_kwargs.keys():
                    self.network_kwargs[key] = value
                elif key in self.training_kwargs.keys():
                    self.training_kwargs[key] = value

            for key, value in trVAE_config.items():
                setattr(self, key, value)

            if compile_and_consturct:
                self.construct_network()
                self.compile_models()

            print(f"{self.class_name}'s config has been successfully restored!")
            return True
        else:
            return False


    def save(self, make_dir=True):

        if make_dir:
            os.makedirs(self.model_path, exist_ok=True)

        if os.path.exists(self.model_path):
            self.save_model_weights(make_dir)
            self.save_model_config(make_dir)
            self.save_class_config(make_dir)
            print(f"\n{self.class_name} has been successfully saved in {self.model_path}.")
            return True
        else:
            return False


    def save_model_weights(self, make_dir=True):
        
        if make_dir:
            os.makedirs(self.model_path, exist_ok=True)

        if os.path.exists(self.model_path):
            self.cvae_model.save_weights(os.path.join(self.model_path, f"{self.model_name}.h5"),
                                         overwrite=True)
            return True
        else:
            return False


    def save_model_config(self, make_dir=True):
        
        if make_dir:
            os.makedirs(self.model_path, exist_ok=True)

        if os.path.exists(self.model_path):
            model_json = self.cvae_model.to_json()
            with open(os.path.join(self.model_path, f"{self.model_name}.json"), 'w') as file:
                file.write(model_json)
            return True
        else:
            return False


    def save_class_config(self, make_dir=True):
       
        import json

        if make_dir:
            os.makedirs(self.model_path, exist_ok=True)

        if os.path.exists(self.model_path):
            config = {"gene_size": self.gene_size,
                      "n_topic": self.n_topic,
                      "n_conditions": self.n_conditions,
                      "condition_encoder": self.condition_encoder,
                      "gene_names": self.gene_names}
            all_configs = dict(list(self.network_kwargs.items()) +
                               list(self.training_kwargs.items()) +
                               list(config.items()))
            with open(os.path.join(self.model_path, f"{self.class_name}.json"), 'w') as f:
                json.dump(all_configs, f)

            return True
        else:
            return False


    def _fit(self, adata,
             condition_key, train_size=0.8,
             n_epochs=300, batch_size=512,
             early_stop_limit=10, lr_reducer=7,
             save=True, retrain=True, verbose=3):
        train_adata, valid_adata = train_test_split(adata, train_size)

        if self.gene_names is None:
            self.gene_names = train_adata.var_names.tolist()
        else:
            if set(self.gene_names).issubset(set(train_adata.var_names)):
                train_adata = train_adata[:, self.gene_names]
            else:
                raise Exception("set of gene names in train adata are inconsistent with class' gene_names")

            if set(self.gene_names).issubset(set(valid_adata.var_names)):
                valid_adata = valid_adata[:, self.gene_names]
            else:
                raise Exception("set of gene names in valid adata are inconsistent with class' gene_names")

        train_expr = train_adata.X.A if sparse.issparse(train_adata.X) else train_adata.X
        valid_expr = valid_adata.X.A if sparse.issparse(valid_adata.X) else valid_adata.X

        train_conditions_encoded, self.condition_encoder = label_encoder(train_adata, le=self.condition_encoder,
                                                                         condition_key=condition_key)

        valid_conditions_encoded, self.condition_encoder = label_encoder(valid_adata, le=self.condition_encoder,
                                                                         condition_key=condition_key)

        if not retrain and os.path.exists(os.path.join(self.model_path, f"{self.model_name}.h5")):
            self.restore_model_weights()
            return

        callbacks = [
            History(),
        ]

        if verbose > 2:
            callbacks.append(
                LambdaCallback(on_epoch_end=lambda epoch, logs: print_progress(epoch, logs, n_epochs)))
            fit_verbose = 0
        else:
            fit_verbose = verbose

        if early_stop_limit > 0:
            callbacks.append(EarlyStopping(patience=early_stop_limit, monitor='val_loss'))

        if lr_reducer > 0:
            callbacks.append(ReduceLROnPlateau(monitor='val_loss', patience=lr_reducer))

        train_conditions_onehot = to_categorical(train_conditions_encoded, num_classes=self.n_conditions)
        valid_conditions_onehot = to_categorical(valid_conditions_encoded, num_classes=self.n_conditions)

        x_train = [train_expr, train_conditions_onehot, train_conditions_onehot]
        x_valid = [valid_expr, valid_conditions_onehot, valid_conditions_onehot]

        y_train = [train_expr, train_conditions_encoded]
        y_valid = [valid_expr, valid_conditions_encoded]

        self.cvae_model.fit(x=x_train,
                            y=y_train,
                            validation_data=(x_valid, y_valid),
                            epochs=n_epochs,
                            batch_size=batch_size,
                            verbose=fit_verbose,
                            callbacks=callbacks,
                            )
        if save:
            self.save(make_dir=True)


    def _train_on_batch(self, adata,
                        condition_key, train_size=0.8,
                        n_epochs=300, batch_size=512,
                        early_stop_limit=10, lr_reducer=7,
                        save=True, retrain=True, verbose=3):
        train_adata, valid_adata = train_test_split(adata, train_size)

        if self.gene_names is None:
            self.gene_names = train_adata.var_names.tolist()
        else:
            if set(self.gene_names).issubset(set(train_adata.var_names)):
                train_adata = train_adata[:, self.gene_names]
            else:
                raise Exception("set of gene names in train adata are inconsistent with class' gene_names")
            if set(self.gene_names).issubset(set(valid_adata.var_names)):
                valid_adata = valid_adata[:, self.gene_names]
            else:
                raise Exception("set of gene names in valid adata are inconsistent with class' gene_names")

        train_conditions_encoded, self.condition_encoder = label_encoder(train_adata, le=self.condition_encoder,
                                                                         condition_key=condition_key)
        valid_conditions_encoded, self.condition_encoder = label_encoder(valid_adata, le=self.condition_encoder,
                                                                         condition_key=condition_key)

        if not retrain and os.path.exists(os.path.join(self.model_path, f"{self.model_name}.h5")):
            self.restore_model_weights()
            return

        train_conditions_onehot = to_categorical(train_conditions_encoded, num_classes=self.n_conditions)
        valid_conditions_onehot = to_categorical(valid_conditions_encoded, num_classes=self.n_conditions)

        if sparse.issparse(train_adata.X):
            is_sparse = True
        else:
            is_sparse = False

        train_expr = train_adata.X
        valid_expr = valid_adata.X.A if is_sparse else valid_adata.X

        valid_cell_type_encoded, _ = label_encoder(valid_adata, condition_key=self.cell_type_key)
        valid_cell_type_onehot = to_categorical(valid_cell_type_encoded, num_classes=self.n_cell_types)

        x_valid = [valid_expr, valid_conditions_onehot, valid_conditions_onehot, valid_cell_type_onehot]

        if self.loss_fn in ['nb', 'zinb']:
            x_valid.append(valid_adata.obs[self.size_factor_key].values)
            contrastive_placeholder = np.zeros((valid_expr.shape[0], self.n_topic))
            y_valid = [valid_adata.raw.X.A if sparse.issparse(valid_adata.raw.X) else valid_adata.raw.X,
                       valid_conditions_encoded, contrastive_placeholder, contrastive_placeholder]
        else:
            contrastive_placeholder = np.zeros((valid_expr.shape[0], self.n_topic))
            y_valid = [valid_expr, valid_conditions_encoded, contrastive_placeholder, contrastive_placeholder]

        es_patience, best_val_loss = 0, 1e10
        for i in range(n_epochs):
            train_loss = train_recon_loss = train_mmd_loss = train_contrastive_loss = train_second_contrastive_loss = 0.0
            for j in range(min(200, train_adata.shape[0] // batch_size)):
                batch_indices = np.random.choice(train_adata.shape[0], batch_size)
                batch_expr = train_expr[batch_indices, :].A if is_sparse else train_expr[batch_indices, :]

                batch_cell_type_encoded = label_encoder(train_adata, le=None, condition_key=self.cell_type_key)[0][
                    batch_indices]
                batch_cell_type_onehot = to_categorical(batch_cell_type_encoded, num_classes=self.n_cell_types)

                x_train = [batch_expr, train_conditions_onehot[batch_indices], train_conditions_onehot[batch_indices],
                           batch_cell_type_onehot]

                if self.loss_fn in ['nb', 'zinb']:
                    x_train.append(train_adata.obs[self.size_factor_key].values[batch_indices])
                    contrastive_placeholder = np.zeros((batch_size, self.n_topic))
                    y_train = [train_adata.raw.X[batch_indices].A if sparse.issparse(
                        train_adata.raw.X[batch_indices]) else train_adata.raw.X[batch_indices],
                               train_conditions_encoded[batch_indices], contrastive_placeholder, contrastive_placeholder]
                else:
                    contrastive_placeholder = np.random.uniform(low=0.01, high=0.1, size=(batch_size, self.n_topic))
                    y_train = [batch_expr, train_conditions_encoded[batch_indices], contrastive_placeholder, contrastive_placeholder]

                batch_loss, batch_recon_loss, batch_kl_loss, batch_contrastive_loss, batch_second_contrastive_loss = self.cvae_model.train_on_batch(x_train, y_train)

                train_loss += batch_loss / batch_size
                train_recon_loss += batch_recon_loss / batch_size
                train_mmd_loss += batch_kl_loss / batch_size
                train_contrastive_loss += batch_contrastive_loss / batch_size
                train_second_contrastive_loss += batch_second_contrastive_loss / batch_size

            valid_loss, valid_recon_loss, valid_mmd_loss, valid_contrastive_loss, valid_second_contrastive_loss = self.cvae_model.evaluate(x_valid, y_valid, verbose=0)

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                es_patience = 0
            else:
                es_patience += 1
                if es_patience == early_stop_limit:
                    print("Training stopped with Early Stopping")
                    break

            logs = {"loss": train_loss, "recon_loss": train_recon_loss, "mmd_loss": train_mmd_loss,
                    "contrastive_loss": train_contrastive_loss,
                    "second_contrastive_loss": train_second_contrastive_loss,
                    "val_loss": valid_loss, "val_recon_loss": valid_recon_loss, "val_mmd_loss": valid_mmd_loss,
                    "val_contrastive_loss": valid_contrastive_loss,
                    "val_second_contrastive_loss": valid_second_contrastive_loss}
            print_progress(i, logs, n_epochs)

        if save:
            self.save(make_dir=True)


    def train(self, adata,
              condition_key, train_size=0.8,
              n_epochs=200, batch_size=128,
              early_stop_limit=10, lr_reducer=8,
              save=True, retrain=True, verbose=3):

        if self.device == 'gpu':
            return self._fit(adata, condition_key, train_size, n_epochs, batch_size, early_stop_limit,
                             lr_reducer, save, retrain, verbose)
        else:
            return self._train_on_batch(adata, condition_key, train_size, n_epochs, batch_size,
                                        early_stop_limit, lr_reducer, save, retrain,
                                        verbose)