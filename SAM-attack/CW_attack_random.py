import os
import sys
import torch
import numpy as np
from torch import optim
from torch import autograd

from util import args, logger


class CarliniL2_random:

    def __init__(self, targeted=True, search_steps=None, max_steps=None, cuda=True, debug=False):
        self.debug = debug
        self.targeted = targeted
        self.num_classes = 5
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
        self.init_rand = False  # an experiment, does a random starting point help?

    def _compare(self, output, target):
        if not isinstance(output, (float, int, np.int64)):
            output = np.copy(output)
            # if self.targeted:
            #     output[target] -= self.confidence
            # else:
            #     output[target] += self.confidence
            output = np.argmax(output)
        if self.targeted:
            return output == target
        else:
            return output != target

    def _compare_untargeted(self, output, target):
        if not isinstance(output, (float, int, np.int64)):
            output = np.copy(output)
            # if self.targeted:
            #     output[target] -= self.confidence
            # else:
            #     output[target] += self.confidence
            output = np.argmax(output)
        if self.targeted:
            return output == target + 1 or output == target - 1
        else:
            return output != target

    def _loss(self, output, target, dist, scale_const):
        # compute the probability of the label class versus the maximum other
        real = (target * output).sum(1)
        other = ((1. - target) * output - target * 10000.).max(1)[0]
        if self.targeted:
            # if targeted, optimize for making the other class most likely
            loss1 = torch.clamp(other - real + self.confidence, min=0.)  # equiv to max(..., 0.)
        else:
            # if non-targeted, optimize for making this class least likely.
            loss1 = torch.clamp(real - other + self.confidence, min=0.)  # equiv to max(..., 0.)
        loss1 = torch.sum(scale_const * loss1)
        loss2 = dist.sum()
        if args.debugging:
            print("loss 1:", loss1.item(), "   loss 2:", loss2.item())
        loss = loss1 + loss2
        return loss

    def _optimize(self, optimizer, model, input_var, modifier_var, target_var, scale_const_var, input_token=None):
        # apply modifier and clamp resulting image to keep bounded from clip_min to clip_max
        batch_adv_sent = []
        if self.mask is None:
            # not word-level attack
            input_adv = modifier_var + input_var
            # input_adv = self.itereated_var = modifier_var + self.itereated_var
        else:
            # word level attack
            input_adv = modifier_var * self.mask + self.itereated_var
            # input_adv = modifier_var * self.mask + input_var
            for i in range(input_adv.size(1)):
                # for batch size
                new_word_list = []
                allow_idx = self.batch_info['allow_idx'][i]
                for j in allow_idx:
                    new_placeholder = input_adv[j, i].data
                    temp_place = new_placeholder.expand_as(self.wv)
                    new_dist = torch.norm(temp_place - self.wv.data, 2, -1)
                    _, new_word = torch.min(new_dist, 0)
                    new_word_list.append(new_word.item())
                    # input_adv.data[j, i] = self.wv[new_word.item()].data
                    input_adv.data[j, i] = self.itereated_var.data[j, i] = self.wv[new_word.item()].data
                    del temp_place
                batch_adv_sent.append(new_word_list)

        # print("Has nan in embedding: ", torch.sum(torch.isnan(input_var.float())))
        output, attention = model(input_adv)
        if input_token is not None:
            input_adv = model.get_embedding()
            input_var = input_token
            seqback = model.get_seqback()
            batch_adv_sent = seqback.adv_sent.copy()
            seqback.adv_sent = []

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
        loss = self._loss(output, target_var, dist, scale_const_var)
        if args.debugging:
            print(loss)
        optimizer.zero_grad()
        loss.backward()
        # loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_([modifier_var], args.clip)
        optimizer.step()
        # modifier_var.data -= 2 * modifier_var.grad.data
        # modifier_var.grad.data.zero_()

        loss_np = loss.item()
        dist_np = dist.data.cpu().numpy()
        output_np = output.data.cpu().numpy()
        input_adv_np = input_adv.data.cpu().numpy()
        return loss_np, dist_np, output_np, input_adv_np, batch_adv_sent

    def run(self, model, input, target, batch_idx=0, batch_size=None, input_token=None):
        if batch_size is None:
            batch_size = input.size(1)  # ([length, batch_size, nhim])
        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        scale_const = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        # python/numpy placeholders for the overall best l2, label score, and adversarial image
        o_best_l2 = [1e10] * batch_size
        o_best_score = [-1] * batch_size
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
        target_onehot = torch.zeros(target.size() + (self.num_classes,))
        if self.cuda:
            target_onehot = target_onehot.cuda()
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        target_var = torch.tensor(target_onehot, requires_grad=False)

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
            best_score = [-1] * batch_size
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

                loss, dist, output, adv_img, adv_sents = self._optimize(
                    optimizer,
                    model,
                    input_var,
                    modifier_var,
                    target_var,
                    scale_const_var,
                    input_token)

                # if step % 100 == 0 or step == self.max_steps - 1:
                # if args.debugging:
                #     print('Step: {0:>4}, loss: {1:6.4f}, dist: {2:8.5f}, modifier mean: {3:.5e}'.format(
                #         step, loss, dist.mean(), modifier_var.data.mean()))

                # if self.abort_early and step % (self.max_steps // 10) == 0:
                #     if loss > prev_loss * .9999:
                #         print('Aborting early...', loss , '>', prev_loss)
                #         break
                #     prev_loss = loss

                # update best result found
                for i in range(batch_size):
                    target_label = target[i]
                    output_logits = output[i]
                    output_label = np.argmax(output_logits)
                    di = dist[i]
                    if self.debug:
                        if step % 100 == 0:
                            print('{0:>2} dist: {1:.5f}, output: {2:>3}, {3:5.3}, target {4:>3}'.format(
                                i, di, output_label, output_logits[output_label], target_label))
                    if di < best_l2[i] and self._compare_untargeted(output_logits, target_label):
                        # if self._compare(output_logits, target_label):
                        if self.debug:
                            print('{0:>2} best step,  prev dist: {1:.5f}, new dist: {2:.5f}'.format(
                                i, best_l2[i], di))
                        best_l2[i] = di
                        best_score[i] = output_label
                        best_attack[:, i] = adv_img[:, i]
                        self.best_sent[i] = adv_sents[i]
                    if di < o_best_l2[i] and self._compare(output_logits, target_label):
                        # if self._compare(output_logits, target_label):
                        if self.debug:
                            print('{0:>2} best total, prev dist: {1:.5f}, new dist: {2:.5f}'.format(
                                i, o_best_l2[i], di))
                        o_best_l2[i] = di
                        o_best_score[i] = output_label
                        o_best_attack[:, i] = adv_img[:, i]
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
                if self._compare(o_best_score[i], target[i]) and o_best_score[i] != -1:
                    batch_success += 1
                elif self._compare_untargeted(best_score[i], target[i]) and best_score[i] != -1:
                    o_best_l2[i] = best_l2[i]
                    o_best_score[i] = best_score[i]
                    o_best_attack[:, i] = best_attack[:, i]
                    self.o_best_sent[i] = self.best_sent[i]
                    batch_success += 1
                else:
                    batch_failure += 1

            logger.info('Num failures: {0:2d}, num successes: {1:2d}\n'.format(batch_failure, batch_success))
            sys.stdout.flush()
            # end outer search loop

        return o_best_attack
