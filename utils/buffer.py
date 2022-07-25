import torch
import random


class AttackListBuffer:
    def __init__(self, attack_method, uplimit=1, buffer_limit=200, batch_size=20):
        self.images_list = []
        self.labels_list = []
        self.logits_list = []

        self.clean_images = []
        self.clean_logits = []

        self.uplimit = uplimit
        self.buffer_limit = buffer_limit
        self.batch_size = batch_size

        # if attack_method in ['cgattack']:
        #     self.add = self.batch_add
        # else:
        #     self.add = self.single_add

    def new(self):
        self.images_list.append([])
        self.logits_list.append([])

    # def batch_add(self, images, logits):
    #     self.images_list.append(images[:self.uplimit].cpu())
    #     self.logits_list.append(logits[:self.uplimit].cpu())

    def add(self, images, logits):
        i = len(self.images_list) - 1
        if len(self.images_list[i]) < self.uplimit:
            self.images_list[i].append(images[0].cpu())
            self.logits_list[i].append(logits[0].cpu())

    def check(self):
        correct = True
        length_list = []
        for i in range(len(self.clean_images)):
            length_list.append(len(self.images_list[i]))
            if len(self.images_list[i]) == 0:
                correct = False
        print(length_list)
        assert correct

    def add_clean(self, clean_images, clean_logits, labels):
        last_i = len(self.images_list) - 1
        if last_i >= 0 and len(self.images_list[last_i]) == 0:
            # First attack successful, no need to accumulate the history
            self.clean_images.pop(last_i)
            self.clean_logits.pop(last_i)
            self.labels_list.pop(last_i)
            self.images_list.pop(last_i)
            self.logits_list.pop(last_i)

        self.clean_images.append(clean_images[0].cpu())
        self.clean_logits.append(clean_logits.cpu())
        self.labels_list.append(labels)

        assert len(self.images_list) == len(self.clean_images) - 1, \
            f'images_list length: {len(self.images_list)}, clean_images length: {len(self.clean_images)}'

        self.new()

        if len(self.clean_images) > self.buffer_limit:
            self.images_list = self.images_list[1:]
            self.labels_list = self.labels_list[1:]
            self.logits_list = self.logits_list[1:]
            self.clean_images = self.clean_images[1:]
            self.clean_logits = self.clean_logits[1:]
        return len(self.labels_list)

    def sample_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        length = self.length()
        sample_list_x = random.sample(range(length - 1), batch_size)
        sample_images = []
        sample_logits = []
        sample_labels = []
        for x in sample_list_x:
            assert len(self.images_list[x]) > 0, f'Index {x} images with no adversarial memory'
            y = random.randint(0, len(self.images_list[x]) - 1)
            sample_images.append(self.images_list[x][y])
            sample_logits.append(self.logits_list[x][y])
            sample_labels.append(self.labels_list[x])
        sample_images = torch.stack(sample_images, dim=0).cuda()
        sample_logits = torch.stack(sample_logits, dim=0).cuda()
        sample_labels = torch.tensor(sample_labels).cuda()
        return sample_images, sample_logits, sample_labels

    def sample_clean_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        length = self.length()
        sample_list = random.sample(range(length), batch_size)
        sample_images = torch.stack([self.clean_images[x] for x in sample_list], dim=0).cuda()
        sample_logits = torch.stack([self.clean_logits[x] for x in sample_list], dim=0).cuda()
        sample_labels = torch.tensor([self.labels_list[x] for x in sample_list]).cuda()
        return sample_images, sample_logits, sample_labels

    def length(self):
        return len(self.images_list)

    def clear(self):
        self.images_list = []
        self.labels_list = []
        self.logits_list = []
        self.clean_images = []
        self.clean_logits = []


class ImageBuffer:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.clean_images = []
        self.clean_logits = []
        self.labels = []
        self.scores = []

    def add(self, image, label, logits=None, score=None):
        for _i in range(len(image)):
            self.clean_images.append(image[_i].cpu())
            if isinstance(label, int) or isinstance(label, float):
                self.labels.append(label)
            else:
                self.labels.append(label[_i])
            if logits is not None:
                self.clean_logits.append(logits[_i].cpu())
            if score is not None:
                if isinstance(label, int) or isinstance(label, float):
                    self.scores.append(score)
                else:
                    self.scores.append(score[_i])
        return len(self.clean_images) == self.batch_size

    def get_item(self, ix):
        return self.clean_images[ix], self.labels[ix]

    def sample_batch(self, batch_size):
        length = self.length()
        sample_list = random.sample(range(length), batch_size)
        sample_images = torch.stack([self.clean_images[sample] for sample in sample_list], dim=0).cuda()
        sample_logits = torch.stack([self.clean_logits[sample] for sample in sample_list], dim=0).cuda()
        sample_labels = torch.tensor([self.labels[sample] for sample in sample_list]).cuda()
        return sample_images, sample_logits, sample_labels

    def make_batch(self):
        # return current_images, last_images, current_logits, labels
        return torch.stack(self.clean_images, dim=0).cuda(), \
               torch.stack(self.clean_logits, dim=0).cuda(), \
               torch.tensor(self.labels).cuda()

    def sort(self):
        assert len(self.clean_images) == len(self.scores)
        self.scores, self.labels, self.clean_images, self.clean_logits = zip(
            *sorted(zip(self.scores, self.labels, self.clean_images, self.clean_logits), key=lambda pair: pair[0]))

    def length(self):
        return len(self.clean_images)

    def clear(self):
        self.clean_images = []
        self.clean_logits = []
        self.labels = []
        self.scores = []
