import torch


def eliminate_error(x):
    # Not Recommend
    x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
    x = torch.where(torch.isinf(x), torch.zeros_like(x), x)
    return x


class TrainModelSurrogate:
    def __init__(self):
        self.train_num = 0
        self.batch_size = 8
        self.lamda = 3.0
        self.d_loss_sum = 0
        self.s_loss_sum = 0
        self.forward_loss_lamda = 0.05  # 0.0005  # 0.005 will be fine
        self.backward_loss_lamda = 0.01  # 0.0005  # 0.0005 will be fine
        self.backward_loss_grad_coefficient = 0.01
        self.max_grad_clip = 0.1  # 0.0001 is fine

    def get_lamda(self):
        # Adaptive gamma and lambda
        if self.train_num > 50:
            lamda2 = self.s_loss_sum / self.d_loss_sum  # Use history s_loss sum and d_loss sum, compute lamda2
            self.lamda = self.lamda * 0.9 + lamda2 * 0.1  # Update lamda with lamda2 using momentum
        else:
            self.lamda = 3.0

    def lifelong_forward_loss(self, surrogate_model, optimizer, history_batch, current_batch):
        # batch: [images, logits, labels]
        surrogate_model.train()
        surrogate_model.zero_grad()
        for _i, batch in enumerate([history_batch, current_batch]):
            images, target_logits, labels = batch[0], batch[1], batch[2]
            images = images.detach().clone()
            images.requires_grad = True
            if isinstance(labels, int):
                labels = torch.tensor(labels).cuda()
                labels = labels.view(-1).repeat(images.shape[0])

            surrogate_logits = surrogate_model(images)
            surrogate_prob = torch.nn.functional.softmax(surrogate_logits, dim=1)
            s_score = surrogate_prob.gather(1, labels.reshape(-1, 1))
            target_score = target_logits.gather(1, labels.reshape(-1, 1))
            forward_loss = torch.nn.MSELoss()(s_score, target_score.detach()) * self.forward_loss_lamda
            lamda = self.lamda if _i == 0 else 1
            loss = forward_loss * lamda
            loss *= 0.01
            loss.backward()

        torch.nn.utils.clip_grad_norm_(surrogate_model.parameters(), self.max_grad_clip)
        optimizer.step()
        surrogate_model.eval()

    def forward_loss(self, surrogate_model, optimizer, images, target_logits, labels):
        surrogate_model.train()
        images = images.detach().clone()
        images.requires_grad = True
        if isinstance(labels, int):
            labels = torch.tensor(labels).cuda()
            labels = labels.view(-1).repeat(images.shape[0])

        surrogate_logits = surrogate_model(images)
        surrogate_prob = torch.nn.functional.softmax(surrogate_logits, dim=1)
        # since attack image is the correct one, other max logit must be the top 2
        s_score = surrogate_prob.gather(1, labels.reshape(-1, 1))
        # s_score = surrogate_prob.topk(2, dim=1)[0]

        target_score = target_logits.gather(1, labels.reshape(-1, 1))
        # target_score = target_prob.topk(2, dim=1)[0]
        forward_loss = torch.nn.MSELoss()(s_score, target_score.detach()) * self.forward_loss_lamda
        surrogate_model.zero_grad()
        forward_loss.backward()
        torch.nn.utils.clip_grad_norm_(surrogate_model.parameters(), self.max_grad_clip)
        optimizer.step()
        surrogate_model.eval()
        del images

