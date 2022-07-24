import torch


def margin_loss_interface(model, class_num):
    threshold = 5.0

    def loss_function(x, label, targeted):
        with torch.no_grad():
            logits = model(x)
            one_hot = torch.zeros([x.shape[0], class_num], dtype=torch.uint8).cuda()
            label_tensor = torch.tensor(label).reshape(-1, 1).cuda()
            one_hot.scatter_(1, label_tensor, 1)
            one_hot = one_hot.bool()

            if not targeted:
                diff = logits[one_hot] - torch.max(logits[~one_hot].view(len(logits), -1), dim=1)[0]
            else:
                diff = torch.max(logits[~one_hot].view(len(logits), -1), dim=1)[0] - logits[one_hot]
            margin = torch.nn.functional.relu(diff + threshold, True) - threshold

        output = {
            'margin': margin,
            'logits': torch.nn.functional.softmax(logits.clone().detach(), dim=1)
        }
        if targeted:
            logits = torch.nn.functional.softmax(logits, dim=1)
            cross_entropy_loss = torch.sum(-torch.log(logits[:, label]))
            output['loss'] = cross_entropy_loss
        else:
            output['loss'] = margin
        return output

    return loss_function


class BaseAttack:
    def __init__(
            self,
            dataset_name,
            max_query,
            targeted,
            class_num,
            linf=0.05,
    ):
        self.max_query = max_query
        self.targeted = targeted
        self.class_num = class_num
        self.linf = linf
        self.dataset = dataset_name

        if dataset_name in ['imagenet', 'openimage']:
            self.c, self.h, self.w = 3, 224, 224
        elif dataset_name in ['cifar10', 'cifar100']:
            self.c, self.h, self.w = 3, 32, 32
        else:
            raise NotImplementedError

    def attack(self, loss, x, y, init=None, buffer=None, **kwargs):
        """
        :param loss: Target Black-box model loss function
        :param x: Benign image
        :param y: Label
        :param init: Initialization of adversarial perturbation
        :param buffer: Attack list buffer for meta-updating
        :return:
        """
        raise NotImplementedError
