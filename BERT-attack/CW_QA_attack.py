import os
import sys
import torch
import numpy as np
from torch import optim
from torch.nn.functional import softmax

from util import args


class CarliniL2_qa:

    def __init__(self, targeted=True, search_steps=None, max_steps=None, cuda=True, debug=False):
        self.debug = debug
        self.targeted = targeted
        self.num_classes = 384
        self.confidence = args.confidence  # FIXME need to find a good value for this, 0 value used in paper not doing much...
        self.initial_const = args.const  # bumped up from default of .01 in reference code
        self.binary_search_steps = search_steps or 1
        self.repeat = self.binary_search_steps >= 10
        self.max_steps = max_steps or args.max_steps
        self.abort_early = True
        self.cuda = cuda
        self.mask = None
        self.batch_info = None
        self.wv = None
        self.inputs = None
        self.init_rand = False  # an experiment, does a random starting point help?

    def _compare(self, start_output, start_target, end_output, end_target):
        return start_output == start_target and end_output == end_target

    def _compare_untargeted(self, start_output, start_target, end_output, end_target):
        return abs(start_output - start_target) < 10 or abs(end_output - end_target) < 10

    def _loss(self, start_output, end_output, start_target, end_target, dist, scale_const):
        # compute the probability of the label class versus the maximum other
        real1 = (start_target * start_output).sum(1)
        other1 = ((1. - start_target) * start_output - start_target * 10000.).max(1)[0]
        # other1 = (torch.topk((1. - start_target) * start_output, 10, dim=1)[0]).sum(1)
        real2 = (end_target * end_output).sum(1)
        other2 = ((1. - end_target) * end_output - end_target * 10000.).max(1)[0]
        # other2 = (torch.topk((1. - end_target) * end_output, 10, dim=1)[0]).sum(1)
        if self.targeted:
            # if targeted, optimize for making the other class most likely
            loss1 = torch.clamp(other1 - real1 + self.confidence, min=0.) + \
                    torch.clamp(other2 - real2 + self.confidence, min=0.)  # equiv to max(..., 0.)
        else:
            # if non-targeted, optimize for making this class least likely.
            loss1 = torch.clamp(real1 - other1 + self.confidence, min=0.) + \
                    torch.clamp(real2 - other2 + self.confidence, min=0.)  # equiv to max(..., 0.)
        loss1 = torch.sum(scale_const * loss1)
        loss2 = dist.sum()
        if args.debugging:
            print("loss 1:", loss1.item(), "   loss 2:", loss2.item())
        loss = loss1 + loss2
        return loss

    def _optimize(self, optimizer, model, input_var, modifier_var, start_target_var, end_target_var, scale_const_var,
                  input_token=None):
        # apply modifier and clamp resulting image to keep bounded from clip_min to clip_max
        batch_adv_sent = []
        if self.mask is None:
            # not word-level attack
            input_adv = modifier_var + input_var
            outputs = model(input_adv)
            input_adv = model.get_embedding()
            input_var = input_token
            seqback = model.get_seqback()
            batch_adv_sent = seqback.adv_sent.copy()
            seqback.adv_sent = []
            # input_adv = self.itereated_var = modifier_var + self.itereated_var
        else:
            # word level attack
            input_adv = modifier_var * self.mask + self.itereated_var
            # input_adv = modifier_var * self.mask + input_var
            for i in range(input_adv.size(0)):
                # for batch size
                new_word_list = []
                add_start = self.batch_info['add_start'][i]
                add_end = self.batch_info['add_end'][i]
                for j in range(add_start, add_end):
                    new_placeholder = input_adv[i, j].data
                    temp_place = new_placeholder.expand_as(self.wv)
                    new_dist = torch.norm(temp_place - self.wv.data, 2, -1)
                    _, new_word = torch.min(new_dist, 0)
                    new_word_list.append(new_word.item())
                    # input_adv.data[i, j] = self.wv[new_word.item()].data
                    input_adv.data[i, j] = self.itereated_var.data[i, j] = self.wv[new_word.item()].data
                    del temp_place
                batch_adv_sent.append(new_word_list)

            self.inputs['perturbed'] = input_adv
            outputs = model(**self.inputs)
        start_logits = outputs[0]
        end_logits = outputs[1]

        def reduce_sum(x, keepdim=True):
            # silly PyTorch, when will you get proper reducing sums/means?
            for a in reversed(range(1, x.dim())):
                x = x.sum(a, keepdim=keepdim)
            return x

        def l2_dist(x, y, keepdim=True):
            d = (x - y) ** 2
            return reduce_sum(d, keepdim=keepdim)

        # distance to the original input data
        dist = l2_dist(input_adv, input_var, keepdim=False)
        loss = self._loss(start_logits, end_logits, start_target_var, end_target_var, dist, scale_const_var)
        if args.debugging:
            print(loss)
        optimizer.zero_grad()
        if input_token is None:
            loss.backward()
        else:
            loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_([modifier_var], args.clip)
        optimizer.step()
        # modifier_var.data -= 2 * modifier_var.grad.data
        # modifier_var.grad.data.zero_()

        loss_np = loss.item()
        dist_np = dist.data.cpu().numpy()
        start_output_np = start_logits.data.cpu().numpy()
        end_output_np = end_logits.data.cpu().numpy()
        input_adv_np = input_adv.data.cpu().numpy()
        return loss_np, dist_np, start_output_np, end_output_np, input_adv_np, batch_adv_sent

    def run(self, model, input, targets, batch_idx=0, batch_size=None, input_token=None):
        if batch_size is None:
            batch_size = input.size(0)  # ([length, batch_size, nhim])
        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        scale_const = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        # python/numpy placeholders for the overall best l2, label score, and adversarial image
        o_best_l2 = [1e10] * batch_size
        o_start_best_score = [-1] * batch_size
        o_end_best_score = [-1] * batch_size
        if input_token is None:
            best_attack = input.cpu().detach().numpy()
            o_best_attack = input.cpu().detach().numpy()
        else:
            best_attack = input_token.cpu().detach().numpy()
            o_best_attack = input_token.cpu().detach().numpy()
        self.o_best_sent = {}
        self.best_sent = {}

        # setup input (image) variable, clamp/scale as necessary
        input_var = torch.tensor(input, requires_grad=False)
        self.itereated_var = torch.tensor(input_var)
        # setup the target variable, we need it to be in one-hot form for the loss function
        start_target = targets[0]
        end_target = targets[1]
        start_target_onehot = torch.zeros(start_target.size() + (self.num_classes,))
        end_target_onehot = torch.zeros(end_target.size() + (self.num_classes,))
        if self.cuda:
            start_target_onehot = start_target_onehot.cuda()
            end_target_onehot = end_target_onehot.cuda()
        start_target_onehot.scatter_(1, start_target.unsqueeze(1), 1.)
        end_target_onehot.scatter_(1, end_target.unsqueeze(1), 1.)
        start_target_var = torch.tensor(start_target_onehot, requires_grad=False)
        end_target_var = torch.tensor(end_target_onehot, requires_grad=False)

        # setup the modifier variable, this is the variable we are optimizing over
        modifier = torch.zeros(input_var.size()).float().cuda()
        if self.cuda:
            modifier = modifier.cuda()
        modifier_var = torch.tensor(modifier, requires_grad=True)

        optimizer = optim.Adam([modifier_var], lr=args.lr)

        for search_step in range(self.binary_search_steps):
            if args.debugging:
                print('Batch: {0:>3}, search step: {1}'.format(batch_idx, search_step))
            if self.debug:
                print('Const:')
                for i, x in enumerate(scale_const):
                    print(i, x)
            best_l2 = [1e10] * batch_size
            start_best_score = [-1] * batch_size
            end_best_score = [-1] * batch_size
            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and search_step == self.binary_search_steps - 1:
                scale_const = upper_bound

            scale_const_tensor = torch.from_numpy(scale_const).float()
            if self.cuda:
                scale_const_tensor = scale_const_tensor.cuda()
            scale_const_var = torch.tensor(scale_const_tensor, requires_grad=False)

            for step in range(self.max_steps):
                # perform the attack
                if self.mask is None:
                    if args.decreasing_temp:
                        cur_temp = args.temp - (args.temp - 0.1) / (self.max_steps - 1) * step
                        model.set_temp(cur_temp)
                        if args.debugging:
                            print("temp:", cur_temp)
                    else:
                        model.set_temp(args.temp)

                loss, dist, start_output, end_output, adv_img, adv_sents = self._optimize(
                    optimizer,
                    model,
                    input_var,
                    modifier_var,
                    start_target_var,
                    end_target_var,
                    scale_const_var,
                    input_token)

                # update best result found
                for i in range(batch_size):
                    start_target_label = start_target[i].item()
                    end_target_label = end_target[i].item()
                    start_output_logits = start_output[i]
                    end_output_logits = end_output[i]
                    start_output_label = np.argmax(start_output_logits)
                    end_output_label = np.argmax(end_output_logits)
                    di = dist[i]
                    if di < best_l2[i] and self._compare_untargeted(start_output_label, start_target_label,
                                                                    end_output_label, end_target_label):
                        best_l2[i] = di
                        start_best_score[i] = start_output_label
                        end_best_score[i] = end_output_label
                        best_attack[i] = adv_img[i]
                        self.best_sent[i] = adv_sents[i]
                    if di < o_best_l2[i] and self._compare(start_output_label, start_target_label,
                                                           end_output_label, end_target_label):
                        o_best_l2[i] = di
                        o_start_best_score[i] = start_output_label
                        o_end_best_score[i] = end_output_label
                        o_best_attack[i] = adv_img[i]
                        self.o_best_sent[i] = adv_sents[i]
                sys.stdout.flush()
                # end inner step loop

            # adjust the constants
            batch_failure = 0
            batch_success = 0
            for i in range(batch_size):
                # if self._compare(best_score[i], target[i]) and best_score[i] != -1:
                #     # successful, do binary search and divide const by two
                #     upper_bound[i] = min(upper_bound[i], scale_const[i])
                #     if upper_bound[i] < 1e9:
                #         scale_const[i] = (lower_bound[i] + upper_bound[i]) / 2
                #     if self.debug:
                #         print('{0:>2} successful attack, lowering const to {1:.3f}'.format(
                #             i, scale_const[i]))
                # else:
                #     # failure, multiply by 10 if no solution found
                #     # or do binary search with the known upper bound
                #     lower_bound[i] = max(lower_bound[i], scale_const[i])
                #     if upper_bound[i] < 1e9:
                #         scale_const[i] = (lower_bound[i] + upper_bound[i]) / 2
                #     else:
                #         scale_const[i] *= 10
                #     if self.debug:
                #         print('{0:>2} failed attack, raising const to {1:.3f}'.format(
                #             i, scale_const[i]))

                if self._compare(o_start_best_score[i], start_target[i].item(), o_end_best_score, end_target[i].item()) \
                        and o_start_best_score[i] != -1 and o_end_best_score != -1:
                    batch_success += 1
                elif self._compare_untargeted(start_best_score[i], start_target[i].item(),
                                              end_best_score[i], end_target[i].item()) \
                        and start_best_score[i] != -1 and end_best_score != -1:
                    o_best_l2[i] = best_l2[i]
                    o_start_best_score[i] = start_best_score[i]
                    o_end_best_score[i] = end_best_score[i]
                    o_best_attack[i] = best_attack[i]
                    self.o_best_sent[i] = self.best_sent[i]
                    batch_success += 1
                else:
                    batch_failure += 1

            print('Num failures: {0:2d}, num successes: {1:2d}\n'.format(batch_failure, batch_success))
            sys.stdout.flush()
            # end outer search loop

        return o_best_attack
