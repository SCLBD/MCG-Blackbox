import torch
# Note pytorch here defaults on cuda device


def pred_interface(flow_model, target_model, image, latent_operation, max_batch_size, linf):
    images = image.repeat(max_batch_size, 1, 1, 1)

    def pred(latent):
        with torch.no_grad():
            latent, latent_vec = latent_operation(latent, reverse=True)
            real_batch_size = latent.size(0)
            real_images = images[:real_batch_size]
            perturbation, _ = flow_model.flow.decode(real_images, latent, zs=latent_vec)
            # perturbed_img = torch.clamp(original_img + torch.clamp(perturbation, -8./255, 8./255), 0, 1)
            # perturbed_img = torch.clamp(original_img + torch.sign(perturbation) *  8./255, 0, 1)
            perturbed_img = torch.clamp(real_images + torch.sign(perturbation) * linf, 0, 1)
            logits = target_model(perturbed_img)
            # logits = torch.nn.functional.softmax(logits, dim=1)
        return logits, perturbed_img

    return pred


def criterion_interface(targeted, label, lib_device, dataset):
    # assert label.size(0) == 1
    threshold = 5.0
    dataset = dataset
    selected_class = [864, 394, 776, 911, 430, 41, 265, 988, 523, 497]

    def criterion(logits):
        if dataset == 'imagenet' and targeted:
            label_ = selected_class.index(int(label))
            logits_ = logits[:, selected_class]
        else:
            logits_ = logits
            label_ = label
        logits_other = torch.cat((logits_[:, :label_], logits_[:, (label_ + 1):]), dim=1)
        if not targeted:
            diff = logits_[:, label_] - torch.max(logits_other, dim=1)[0]
            label_loss = logits_[:, label_]
        else:
            diff = torch.max(logits_other, dim=1)[0] - logits_[:, label_]
            label_loss = -logits_[:, label_]
        margin = torch.nn.functional.relu(diff + threshold, True) - threshold
        # TODO here used for hard transfer case

        if lib_device == 'torch':
            return margin, label_loss
        elif lib_device == 'numpy':
            return margin.cpu().data.numpy().reshape(-1).tolist(), label_loss.cpu().data.numpy().reshape(-1).tolist()

    return criterion


def compound_interface(flow_model, target_model, image, label, targeted, latent_operation, max_batch_size, lib_device, dataset, linf):
    assert lib_device in ['numpy', 'torch']
    pred = pred_interface(flow_model, target_model, image, latent_operation, max_batch_size, linf)
    criterion = criterion_interface(targeted, label, lib_device, dataset)

    def compound(latent, return_adv_image=False, return_logits=False, return_label_loss=False):
        if lib_device == 'numpy':
            latent = torch.FloatTensor(latent).cuda()
        logits, perturbed_img = pred(latent)
        margin, label_loss = criterion(logits)
        if return_label_loss:
            return margin, perturbed_img, logits, label_loss
        if return_logits:
            return margin, perturbed_img, logits
        elif return_adv_image:
            return margin, perturbed_img
        else:
            return margin

    return compound


def compound_directly_optimize_pertubation_interface(target_model, image, label, targeted, lib_device):
    assert lib_device in ['numpy', 'torch']
    criterion = criterion_interface(targeted, label, lib_device)

    def compound(latent):
        if lib_device == 'numpy':
            latent = torch.FloatTensor(latent).cuda()

        perturbation = latent.reshape([-1] + list(image.shape[1:]))
        perturbed_img = torch.clamp(image + torch.sign(perturbation) * 0.05, 0, 1)
        logits = target_model(perturbed_img)
        return criterion(logits)

    return compound

