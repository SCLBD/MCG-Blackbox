import torch
import random


class AttackListBuffer:
    def __init__(self, attack_method, uplimit=3, batch_limit=10, batch_size=5):
        self.images_list = []
        self.labels_list = []
        self.logits_list = []
        self.clean_images = []
        self.clean_logits = []
        self.uplimit = uplimit
        self.batch_id = 0
        self.batch_limit = batch_limit
        self.batch_size = batch_size
        if attack_method in ['cgattack']:
            self.add = self.batch_add
        else:
            self.add = self.single_add

    def batch_add(self, images, logits):
        self.images_list.append(images[:self.uplimit].cpu())
        self.logits_list.append(logits[:self.uplimit].cpu())

    def new(self):
        self.images_list.append([])
        self.logits_list.append([])

    def single_add(self, images, logits):
        id = len(self.images_list) - 1
        if len(self.images_list[id]) < self.uplimit:
            self.images_list[id].append(images[0].cpu())
            self.logits_list[id].append(logits[0].cpu())

    def add_clean(self, clean_images, clean_logits, labels):
        id = len(self.images_list) - 1
        if id == -1:
            return -1
        if len(self.images_list[id]) == self.uplimit:
            self.clean_images.append(clean_images[0].cpu())
            self.clean_logits.append(clean_logits.cpu())
            self.labels_list.append(labels)
        else:
            self.images_list.pop(id)
            self.logits_list.pop(id)
        if len(self.images_list) > self.batch_limit:
            self.images_list = self.images_list[1:]
            self.labels_list = self.labels_list[1:]
            self.logits_list = self.logits_list[1:]
            self.clean_images = self.clean_images[1:]
            self.clean_logits = self.clean_logits[1:]
        return len(self.labels_list)

    def make_clean_batch(self):
        # return batch_images, batch_logits, batch_labels
        return torch.stack(self.clean_images, dim=0).cuda(), \
               torch.stack(self.clean_logits, dim=0).cuda(), \
               torch.tensor(self.labels_list).cuda()

    def sample_batch(self, batch_size):
        id = self.batch_id
        self.batch_id = (self.batch_id + 1) % self.uplimit
        length = self.length()
        sample_list = [random.randint(0, length - 1) for _ in range(batch_size)]
        return torch.stack([self.images_list[sample][id] for sample in sample_list], dim=0).cuda(), \
               torch.stack([self.logits_list[sample][id] for sample in sample_list], dim=0).cuda(), \
               torch.tensor([self.labels_list[sample] for sample in sample_list]).cuda()

    def sample_clean_batch(self):
        length = self.length()
        sample_list = [random.randint(0, length - 1) for _ in range(self.batch_size)]
        return torch.stack([self.clean_images[sample] for sample in sample_list], dim=0).cuda(), \
               torch.stack([self.clean_logits[sample] for sample in sample_list], dim=0).cuda(), \
               torch.tensor([self.labels_list[sample] for sample in sample_list]).cuda()

    def make_batch(self):
        id = self.batch_id
        self.batch_id = (self.batch_id + 1) % self.uplimit
        # return batch_images, batch_logits, batch_labels
        return torch.stack([images[id] for images in self.images_list], dim=0).cuda(), \
               torch.stack([logits[id] for logits in self.logits_list], dim=0).cuda(), \
               torch.tensor(self.labels_list).cuda()
        # return current_images, last_images, current_logits, last_logits, labels
        # return torch.stack([images[self.batch_id - 1] for images in self.images_list], dim=0).cuda(), \
        #        torch.stack(self.clean_images, dim=0).cuda(), \
        #        torch.stack([logits[self.batch_id - 1] for logits in self.logits_list], dim=0).cuda(), \
        #        torch.stack(self.clean_logits, dim=0).cuda(), \
        #        torch.tensor(self.labels_list).cuda()

    def length(self):
        return len(self.images_list)

    def clear(self):
        self.images_list = []
        self.labels_list = []
        self.logits_list = []
        self.clean_images = []
        self.clean_logits = []
        self.batch_id = 0


class AttackBuffer:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.clean_images = []
        self.clean_logits = []
        self.labels = []
        self.scores = []

    def add(self, image, label, logits=None, score=None):
        self.clean_images.append(image[0].cpu())
        self.labels.append(label)
        if logits is not None:
            self.clean_logits.append(logits[0].cpu())
        if score is not None:
            self.scores.append(score)
        return len(self.clean_images) == self.batch_size

    def get_item(self, id):
        return self.clean_images[id], self.labels[id]

    def sample_batch(self, batch_size):
        length = self.length()
        # for i, logits in enumerate(self.clean_logits):
        #     print(torch.max(logits))  # right
        #     print(logits[self.labels[i]])

        sample_list = [random.randint(0, length - 1) for _ in range(batch_size)]
        # logits = torch.stack([self.clean_logits[sample] for sample in sample_list], dim=0).cuda()
        # labels = torch.tensor([self.labels[sample] for sample in sample_list]).cuda()
        # for i, logit in enumerate(logits):
        #     print(logit[labels[i]])
        #     print(labels[i])
        return torch.stack([self.clean_images[sample] for sample in sample_list], dim=0).cuda(), \
               torch.stack([self.clean_logits[sample] for sample in sample_list], dim=0).cuda(), \
               torch.tensor([self.labels[sample] for sample in sample_list]).cuda()

    def make_batch(self):
        # return current_images, last_images, current_logits, last_logits, labels
        return torch.stack(self.clean_images, dim=0).cuda(), \
               torch.stack(self.clean_logits, dim=0).cuda(), \
               torch.tensor(self.labels).cuda()

    def sort(self):
        assert len(self.clean_images) == len(self.scores)
        # for i in range(len(self.clean_logits)):
        #     print(self.clean_logits[i][self.labels[i]])
        self.scores, self.labels, self.clean_images, self.clean_logits = zip(
            *sorted(zip(self.scores, self.labels, self.clean_images, self.clean_logits), key=lambda pair: pair[0]))
        # for i in range(len(self.clean_logits)):
        #     print(self.clean_logits[i][self.labels[i]])

    def length(self):
        return len(self.clean_images)

    def clear(self):
        self.clean_images = []
        self.clean_logits = []
        self.labels = []
        self.scores = []