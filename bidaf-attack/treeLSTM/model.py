import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("..")


# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, rel_dim, rel_emb=None, device="cpu"):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.device = device
        self.mem_dim = mem_dim
        self.rel_emb = rel_emb
        self.rel_dim = rel_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim + self.rel_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim + self.rel_dim, self.mem_dim)

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        # print(child_h_sum.shape)
        # print(inputs.shape)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c[:, :self.mem_dim])

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, tree, inputs):
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)

        if tree.num_children == 0:
            if self.rel_emb is not None:
                child_c = inputs[0].detach().new(1, self.mem_dim + self.rel_dim).fill_(0.).requires_grad_()
                child_h = inputs[0].detach().new(1, self.mem_dim + self.rel_dim).fill_(0.).requires_grad_()
            else:
                child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
                child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_c, child_h = zip(*map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)
            if self.rel_emb is not None:
                rel_idx = torch.tensor(tree.relation, dtype=torch.long, device=self.device)
                rel_emb = self.rel_emb(rel_idx)
                child_c = torch.cat((child_c, rel_emb), dim=1)
                child_h = torch.cat((child_h, rel_emb), dim=1)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        return tree.state


# putting the whole model together
class TreeLSTM(nn.Module):
    def __init__(self, vocab_size, in_dim, mem_dim, hidden_dim, sparsity,
                 rel_emb=None, rel_dim=0, device="cpu"):
        super(TreeLSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size, in_dim, sparse=sparsity)
        self.device = device
        self.mem_dim = mem_dim
        self.emb.weight.requires_grad = False
        self.childsumtreelstm = ChildSumTreeLSTM(in_dim, mem_dim, rel_dim, rel_emb, device=device)
        # self.wh = nn.Linear(mem_dim, hidden_dim)
        # self.wp = nn.Linear(hidden_dim, num_classes)

    def forward(self, sents, trees, masks=None):
        length = len(list(sents))
        # print(length)
        prediction = torch.zeros(1, 2).to(self.device)
        # prediction = prediction.cuda()

        # print(prediction)
        # print(type(prediction))
        # print(prediction.type())
        hiddens = []
        if masks is None:
            masks = [True] * len(sents)
        # print(len(sents))
        # print(sents)
        # print(len(trees))
        # print(trees)
        for (sent, tree, mask) in zip(sents, trees, masks):
            # print('SENT:', sent)
            # print('TREE:', tree)
            if tree is None or mask is False:
                hiddens.append(torch.zeros(1, self.mem_dim).to(self.device))
                continue
            sent = self.emb(sent)
            state, hidden = self.childsumtreelstm(tree, sent)
            hiddens.append(hidden)
            # print('STATE', state)
            # print('HIDDEN', hidden)
            # pred = F.sigmoid(self.wh(hidden))
            # pred = F.log_softmax(self.wp(pred), dim=1)
            # print(pred)
            # print(type(pred))
            # print(pred.type())
            # todo: change to cuda version
            # prediction = torch.add(prediction, pred).cuda()
            # prediction = torch.add(prediction, pred)
        # prediction = torch.div(prediction, length)
        # print('Prediction', prediction)
        return hiddens, prediction
