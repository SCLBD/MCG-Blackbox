import torch
import copy
import cma
import numpy as np
from graphviz import Digraph
import torch
from torch.autograd import Variable, Function


def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)

def register_hooks(var):
    fn_dict = {}
    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb)

    def is_bad_grad(grad_output):
        if grad_output is None:
            return True
        grad_output = grad_output.data
        return grad_output.ne(grad_output).any() or grad_output.gt(1e6).any()

    def make_dot():
        node_attr = dict(style='filled',
                        shape='box',
                        align='left',
                        fontsize='12',
                        ranksep='0.1',
                        height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '('+(', ').join(map(str, size))+')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):
                u = fn.variable
                node_name = 'Variable\n ' + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor='lightblue')
            else:
                # assert fn in fn_dict, fn
                # fillcolor = 'white'
                # if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                #     fillcolor = 'red'
                fillcolor = 'red'
                dot.node(str(id(fn)), str(type(fn).__name__), fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))
        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot


def margin_loss(model, y, label, targeted, class_num=1000):
    with torch.no_grad():
        threshold = 5.0
        one_hot = torch.zeros([y.shape[0], class_num], dtype=torch.uint8).cuda()
        label_tensor = torch.tensor(label).reshape(-1, 1).cuda()
        one_hot.scatter_(1, label_tensor, 1)
        one_hot = one_hot.bool()
        logits = model(y)
        if not targeted:
            diff = logits[one_hot] - torch.max(logits[~one_hot].view(len(logits), -1), dim=1)[0]
        else:
            diff = torch.max(logits[~one_hot].view(len(logits), -1), dim=1)[0] - logits[one_hot]
        margin = torch.nn.functional.relu(diff + threshold, True) - threshold

        logits = torch.nn.functional.softmax(logits, dim=1)

    if targeted:
        cross_entropy_loss = torch.sum(-torch.log(logits[:, label]))
        return margin, logits, cross_entropy_loss
    else:
        return margin, logits, margin


def adv_loss(adv_models, y, label, targeted, class_num=1000, coefficient=None):
    loss = 0.
    threshold = 5.0
    one_hot = torch.zeros([y.shape[0], class_num], dtype=torch.uint8).cuda()
    label = torch.tensor(label).reshape(-1, 1).cuda()
    one_hot.scatter_(1, label, 1)
    one_hot = one_hot.bool()
    for i, adv_model in enumerate(adv_models):
        logits = adv_model(y)
        # print('adv_loss: label logits: ', torch.nn.functional.softmax(logits)[one_hot])
        if not targeted:
            diff = logits[one_hot] - torch.max(logits[~one_hot].view(len(logits), -1), dim=1)[0]
        else:
            # diff = torch.max(logits, dim=1)[0] - logits[one_hot]
            # diff = torch.max(logits[~one_hot].view(len(logits), -1), dim=1)[0] - logits[one_hot]
            diff = - logits[one_hot]
        margin = torch.nn.functional.relu(diff + threshold, True) - threshold
        if coefficient is None:
            loss += margin.mean()
        else:
            loss += margin.mean() * coefficient[i]
    if coefficient is None:
        loss /= len(adv_models)
    return loss


def finetune_glow_with_coefficient(glow, adv_models, coefficient, image, label, args):
    print('finetune glow')
    glow.train()
    optim = torch.optim.Adam(glow.parameters(), lr=0.0002, betas=(0.9, 0.9999), weight_decay=0.0)
    for i in range(1):
        y = glow.decode(x=image)
        # loss use surrogate models
        loss = adv_loss(adv_models, torch.clamp(torch.clamp(y, -8. / 255., 8. / 255.) + image, 0, 1), label, args.targeted, class_num=args.num_classes, coefficient=coefficient)
        optim.zero_grad()
        loss.backward()
        # operate grad
        if args.max_grad_clip > 0:
            torch.nn.utils.clip_grad_value_(glow.parameters(), args.max_grad_clip)
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(glow.parameters(), args.max_grad_norm)
        # step
        optim.step()
    glow.eval()
    return


def finetune_glow(glow, adv_models, image, label, args, iteration=4):
    print('finetune glow')
    glow.train()
    optim = torch.optim.Adam(glow.parameters(), lr=0.0005, betas=(0.9, 0.9999), weight_decay=0.0)
    for i in range(iteration):
        y = glow.decode(x=image)
        # loss use surrogate models
        loss = adv_loss(adv_models, torch.clamp(torch.clamp(y, -0.05, 0.05) + image, 0, 1), label, args.targeted, class_num=args.num_classes)
        optim.zero_grad()
        loss.backward()
        # operate grad
        if args.max_grad_clip > 0:
            torch.nn.utils.clip_grad_value_(glow.parameters(), args.max_grad_clip)
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(glow.parameters(), args.max_grad_norm)
        # step
        optim.step()
    glow.eval()
    return


def meta_finetune_glow(glow, adv_models, image, label, args, meta_iteration=4, latent=None, latent_operation=None, linf=0.05):
    print('Meta finetune glow')
    glow.train()
    # optim = torch.optim.Adam(glow.parameters(), lr=3e-4, betas=(0.9, 0.9999), weight_decay=0.0)
    optim = torch.optim.Adam(glow.parameters(), lr=0.001, betas=(0.9, 0.9999), weight_decay=0.0)
    meta_state = copy.deepcopy(glow.state_dict())
    if latent is not None:
        latent, latent_vec = latent_operation()

    for i in range(meta_iteration):
        if latent is None:
            y, logdet = glow.decode(x=image, return_prob=True)
        else:
            pass
            # y, logdet =
        # loss_prob, loss_cls = torch.mean(logdet), adv_loss(adv_models, torch.clamp(y * 0.05 + image, 0, 1), label, args.targeted, class_num=args.num_classes)
        loss_prob, loss_cls = torch.mean(logdet), adv_loss(adv_models, torch.clamp(torch.clamp(y, -linf, linf) + image, 0, 1), label, args.targeted, class_num=args.num_classes)
        # loss_prob, loss_cls = torch.mean(logdet), adv_loss(adv_models, torch.clamp(torch.clamp(y, -8. / 255., 8. / 255.) + image, 0, 1), label, args.targeted, class_num=args.num_classes)
        loss = loss_cls
        # loss = 0.5 * loss_prob + loss_cls
        optim.zero_grad()
        loss.backward()
        if args.max_grad_clip > 0:
            torch.nn.utils.clip_grad_value_(glow.parameters(), args.max_grad_clip)
        # step
        # for name, parms in glow.flow.named_parameters():
        #     grad = torch.sum(parms.grad)
        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', grad)
        # assert False
        optim.step()
    # meta_outer_update  0.3 is fine
    lr = 0.005  # 0.3  # 3e-3

    dic = glow.state_dict()
    keys = list(dic.keys())

    for key in keys:
        dic[key] = meta_state[key] + lr * (dic[key] - meta_state[key])
    glow.eval()
    return


from utils.flow_latent_operation import latent_op_full, latent_op_full_full, encode_decode_initialization, local_initialization
def finetune_latent(glow, adv_models, image, label, args, iteration=10, lr=0.5, linf=0.05):
    glow.eval()
    adv_models.eval()
    latent_operation = latent_op_full
    # if latent is None:
    with torch.no_grad():
        init_pert, decode_logdet = glow.decode(image, return_prob=True, no_norm=True)
        latent, encode_logdet, latent_vec = glow.flow.encode(image, init_pert, return_z=True)
    # else:
    #     latent, latent_vec = latent_operation(latent, latent_vec, reverse=True)
    latent = latent.clone().detach().float().cuda()
    latent.requires_grad_()

    optim = torch.optim.Adam([latent], lr=lr, betas=(0.9, 0.9999), weight_decay=0.0)

    # scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)
    # scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[40], gamma=0.2)
    for i in range(iteration):
    # while loss > 0.5:
        # glow.zero_grad()
        perturbation, _ = glow.flow.decode(image, latent, zs=latent_vec)
        # perturbed_img = torch.clamp(image + torch.sign(perturbation) * 0.05, 0, 1)
        # logits = adv_models(perturbed_img)
        logits = adv_models(torch.clamp(torch.clamp(perturbation, -linf, linf) + image, 0., 1.))
        # logits = adv_models(0.05 * perturbation / torch.max(perturbation) + image)
        # logits = adv_models(0.05 * torch.sign(perturbation) + image)
        logits = torch.nn.functional.softmax(logits, dim=1)
        logits_ = logits
        # label_ = label
        # logits_other = torch.cat((logits_[:, :label_], logits_[:, (label_ + 1):]), dim=1)
        # diff = logits_[:, label_] - torch.max(logits_other, dim=1)[0]
        loss = torch.mean(logits_[:, label])
        # loss = torch.nn.functional.relu(diff + 5, True) - 5
        optim.zero_grad()
        # if i % 2 == 0:
        # print(loss, 'loss')
        # if loss > 5:
        #     lr = 0.6
        # elif loss > 2.6:
        #     lr = 0.2
        # elif loss > 2.3:
        #     lr = 0.1
        # elif loss > 1.8:
        #     lr = 0.01
        # else:
        #     lr = 0.05
        # for g in optim.param_groups:
        #     g['lr'] = lr

        # for name, parms in glow.flow.named_parameters():
        #     grad = torch.sum(parms.grad)
        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', grad)
        loss.backward()
        optim.step()
        latent.grad.zero_()
        if loss < -1:
            break
        # scheduler1.step()
        # scheduler2.step()

    latent_base = latent.clone()
    latent_vec = [lat.detach() for lat in latent_vec]
    return latent_operation(latent_base, latent_vec)


# def finetune_latent(latent, compound):
#     print('finetune latent')
#     es = cma.CMAEvolutionStrategy(latent.squeeze(0).cpu().data.numpy().reshape(-1), 1e15, {'seed': 666, 'maxfevals': 200, 'popsize': 20, 'ftarget': 0, 'tolfun': 1e-10})
#     while not es.stop() and es.best.f > 0:
#         X = es.ask()  # get list of new solutions
#         fit = compound(np.array(X))
#         es.tell(X, fit)  # feed values
#     return np.array(es.best.x)

