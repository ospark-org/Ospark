from . import *


class CocaTrainer(Trainer):

    def train_step(self, train_data: Dict[str, tf.Tensor], target_data: tf.Tensor):

        with tf.GradientTape() as tape:

            logits, cls_image_embedding, cls_embedding = self.model.pipeline(images=train_data["image"],
                                                                             text=train_data["text"])
            contrastive_loss = self.loss_function["contrastive_loss"](prediction=cls_image_embedding,
                                                                      target_data=cls_embedding)
            caption_loss = self.loss_function["caption_loss"](prediction=logits,
                                                              target_data=target_data)
            weights    = self.weights_operator.collect_weights()
            tape.watch(weights)
            loss_value = contrastive_loss + caption_loss
        gradients = tape.gradient(loss_value, weights)
        self.optimizer.apply_gradients(zip(gradients, weights))
        return loss_value
