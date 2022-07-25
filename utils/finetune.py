import torch
import copy
from graphviz import Digraph
import torch
from models.flow_latent import latent_operate


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
            return '(' + (', ').join(map(str, size)) + ')'

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


def adv_loss(adv_models, y, label, targeted, class_num):
    loss = 0.
    threshold = 5.0
    one_hot = torch.zeros([y.shape[0], class_num], dtype=torch.uint8).cuda()
    label = torch.tensor(label).reshape(-1, 1).cuda()
    one_hot.scatter_(1, label, 1)
    one_hot = one_hot.bool()
    for i, adv_model in enumerate(adv_models):
        logits = adv_model(y)
        if not targeted:
            diff = logits[one_hot] - torch.max(logits[~one_hot].view(len(logits), -1), dim=1)[0]
        else:
            diff = - logits[one_hot]
        margin = torch.nn.functional.relu(diff + threshold, True) - threshold
        loss += margin.mean()
    loss /= len(adv_models)
    return loss


def meta_finetune(generator, adv_models, images, labels, latent, args, meta_iteration=3):
    # Meta finetune generator (c-generator) to adapt to the current task
    # Note here finetune need gradients from the surrogate models
    generator.train()
    optimizer = torch.optim.Adam(generator.parameters(), lr=0.001, betas=(0.9, 0.9999), weight_decay=0.0)
    meta_state = copy.deepcopy(generator.state_dict())
    assert latent is not None

    for i in range(meta_iteration):
        # Meta inner update
        latent, latent_vec = latent_operate(latent, reverse=True)
        perturbation, logdet = generator.flow.decode(images, latent, zs=latent_vec)
        perturbation = torch.clamp(perturbation, min=-args.linf, max=args.linf)

        loss_prob = torch.mean(logdet)
        loss_adv = adv_loss(adv_models, torch.clamp(images + perturbation, 0, 1), labels,
                            targeted=args.targeted, class_num=args.num_classes)
        loss = loss_adv

        optimizer.zero_grad()
        loss.backward()
        if args.max_grad_clip > 0:
            torch.nn.utils.clip_grad_value_(generator.parameters(), args.max_grad_clip)
        optimizer.step()

    # Meta outer update
    lr = 0.005
    dic = generator.state_dict()
    for key in list(dic.keys()):
        dic[key] = meta_state[key] + lr * (dic[key] - meta_state[key])
    generator.eval()
    return


def finetune_latent(generator, adv_models, images, labels, latent, args, iteration=10, lr=0.01):
    generator.eval()

    latent, latent_vec = latent_operate(latent, reverse=True)
    latent = latent.clone().detach().float().cuda()
    latent.requires_grad_()
    optimizer = torch.optim.Adam([latent], lr=lr, betas=(0.9, 0.9999), weight_decay=0.0)

    for i in range(iteration):
        perturbation, _ = generator.flow.decode(images, latent, zs=latent_vec)
        perturbation = torch.clamp(perturbation, min=-args.linf, max=args.linf)
        loss = adv_loss(adv_models, torch.clamp(images + perturbation, 0, 1), labels,
                            targeted=args.targeted, class_num=args.num_classes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        latent.grad.zero_()

        if loss < -1:
            break

    latent_base = latent.clone()
    latent_vec = [lat.detach() for lat in latent_vec]
    return latent_operate(latent_base, latent_vec)